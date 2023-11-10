#!/usr/bin/env python3

from dotenv import load_dotenv

load_dotenv()

from typing import Optional, BinaryIO, Union, NamedTuple

import os
import io
import sys
import optparse
import termios
# import sounddevice
# import wavio

from os import get_terminal_size
from signal import signal, SIGWINCH
from openai import OpenAI
from pygments import format as colorize
from pygments.lexer import Lexer
from pygments.token import _TokenType, Token
from pygments.lexers import find_lexer_class_for_filename
from pygments.formatter import Formatter
from pygments.formatters import Terminal256Formatter
from wcwidth import wcswidth

CTRL = [1, 5]
PAGE_UP = (-1, 126, [5])
PAGE_DOWN = (-1, 126, [6])

LEVEL_INFO    = 0
LEVEL_WARNING = 1
LEVEL_ERROR   = 2

MAX_TOKENS = 4096
SYSTEM_MESSAGE_CONTENT = """\
You're a programming accessibility tool. You will get natural language describing code and answer with valid {lang}. The natural language input comes from voice recognition and thus will omit a lot of special characters which you will have to fill in. Be sure to always echo back the whole edited file. Keep any messages about what you where doing very short.

The name of the current file is:
{filename}

The user is currently viewing line {start_lineno} to {end_lineno} of this file.

This is the content of the file:
{code}
"""

class UnbufferedInput:
    __slots__ = 'old_attrs',

    old: Optional[list]

    def __init__(self) -> None:
        self.old_attrs = None

    def __enter__(self) -> None:
        # canonical mode, no echo
        self.old_attrs = termios.tcgetattr(sys.stdin)
        new = termios.tcgetattr(sys.stdin)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new)

    def __exit__(self, *args) -> None:
        # restore terminal to previous state
        if self.old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attrs)

class EchoedInput:
    __slots__ = 'old_attrs',

    old: Optional[list]

    def __init__(self) -> None:
        self.old_attrs = None

    def __enter__(self) -> None:
        # canonical mode, echo
        self.old_attrs = termios.tcgetattr(sys.stdin)
        new = termios.tcgetattr(sys.stdin)
        new[3] = new[3] | termios.ICANON | termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new)
        # show cursor
        sys.stdout.write('\x1b[?25h')

    def __exit__(self, *args) -> None:
        # restore terminal to previous state
        if self.old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attrs)
        # hide cursor
        sys.stdout.write('\x1b[?25l')

class Response(NamedTuple):
    code: Optional[str] = None
    message: Optional[str] = None
    message_level: int = LEVEL_INFO

def parse_response(message: str) -> Response:
    index = 0 if message.startswith('```') else message.find('\n```')
    if index == -1:
        return Response(message=message)

    start_index = message.find('\n', index)
    if start_index == -1:
        return Response(message=message)

    start_index += 1

    end_index = message.find('\n```\n', start_index)

    if end_index == -1:
        if message.endswith('\n```'):
            end_index = len(message) - 3
        else:
            return Response(message=message)
    else:
        end_index += 1

    new_message = message[:index].strip()
    tail = message[end_index + 3:].strip()

    if tail:
        if new_message:
            new_message = f'{new_message}\n{tail}'
        else:
            new_message = tail

    code = message[start_index:end_index]

    return Response(code=code, message=new_message or None)

def read_ansi(stdin: BinaryIO) -> tuple[Union[bytes, bytearray], Optional[tuple[int, int, list[int]]]]:
    byte = stdin.read(1)
    if not byte or byte[0] != 27:
        return byte, None

    res = bytearray(byte)

    # TODO: timeout for when user pressed ESC
    byte = stdin.read(1)
    if not byte:
        return res, (-1, -1, [])

    res.extend(byte)
    ibyte = byte[0]
    if ibyte == 91: # '['
        ibyte = -1
        num = 0
        args: list[int] = []
        has_num = False
        first = True
        prefix = -1
        while True:
            byte = stdin.read(1)
            if not byte:
                break

            res.extend(byte)
            ibyte = byte[0]
            if first:
                first = False
                if ibyte == 63: # '?'
                    prefix = ibyte
                    continue

            if ibyte >= 48 and ibyte <= 57: # '0'-'9'
                num = num * 10 + ibyte - 48
                has_num = True
            elif ibyte == 59: # ';'
                if not has_num:
                    break
                args.append(num)
                num = 0
                has_num = False
            else:
                break
        suffix = ibyte

        if has_num:
            args.append(num)

        return res, (prefix, suffix, args)

    return res, (-1, ibyte, [])

def tokenized_lines(content: str, lexer: Optional[Lexer]) -> list[list[tuple[_TokenType, str]]]:
    if lexer:
        line: list[tuple[_TokenType, str]] = []
        lines: list[list[tuple[_TokenType, str]]] = [line]
        for tok_type, tok_data in lexer.get_tokens(content):
            if tok_type is Token.Text and tok_data == '\n':
                line = []
                lines.append(line)
            else:
                prev = 0
                tok_data_len = len(tok_data)
                while prev < tok_data_len:
                    index = tok_data.find('\n', prev)
                    if index == -1:
                        line.append((tok_type, tok_data[prev:]))
                        break

                    if prev != index:
                        line.append((tok_type, tok_data[prev:index]))

                    line = []
                    lines.append(line)

                    prev = index + 1
    else:
        lines = [[(Token.Text, line)] for line in content.split('\n')]
    return lines

def slice_token_line(tokens: list[tuple[_TokenType, str]], offset: int, length: int) -> list[tuple[_TokenType, str]]:
    line: list[tuple[_TokenType, str]] = []
    current_offset = 0
    max_offset = offset + length

    for tok in tokens:
        tok_type, tok_data = tok
        w = wcswidth(tok_data)
        next_offset = current_offset + w
        if next_offset > offset:
            if current_offset < offset:
                for index in range(len(tok_data)):
                    tok_data_prefix = tok_data[:index]
                    w = wcswidth(tok_data_prefix)
                    if current_offset + w >= offset:
                        tok_data = tok_data[index:]
                        break
            if next_offset >= max_offset:
                # TODO: don't cut in the middle of graphemes
                for index in range(len(tok_data), -1, -1):
                    new_tok_data = tok_data[:index]
                    w = wcswidth(new_tok_data)
                    next_offset = current_offset + w
                    if next_offset < max_offset:
                        line.append((tok_type, new_tok_data))
                        return line
                return line
            else:
                line.append((tok_type, tok_data))
        current_offset = next_offset

    return line

class VoiceCoder:
    __slots__ = (
        'scroll_xoffset', 'scroll_yoffset', 'filename', 'stdin', 'term_size',
        'lexer', 'content', 'saved', 'lines', 'formatter', 'max_line_width',
        'message', 'message_level', 'openai'
    )

    scroll_xoffset: int
    scroll_yoffset: int
    filename: str
    stdin: BinaryIO
    term_size: os.terminal_size
    lexer: Optional[Lexer]
    formatter: Formatter
    content: str
    lines: list[list[tuple[_TokenType, str]]]
    saved: bool
    max_line_width: int
    message: Optional[str]
    message_level: int
    openai: OpenAI

    def __init__(self, filename: str) -> None:
        self.scroll_xoffset = 0
        self.scroll_yoffset = 0
        self.filename = filename
        lexer_class = find_lexer_class_for_filename(filename)
        self.lexer = lexer_class() if lexer_class else None
        self.formatter = Terminal256Formatter()
        self.saved = True
        self.message = None
        self.message_level = LEVEL_INFO
        self.openai = OpenAI()

        with open(self.filename, 'rt') as fp:
            self.content = fp.read()

        self.lines = tokenized_lines(self.content, self.lexer)
        self.max_line_width = max(
            wcswidth(''.join(tok_data for _, tok_data in line))
            for line in self.lines
        )

        self.stdin = io.open(sys.stdin.fileno(), 'rb', closefd=False, buffering=False)
        self.term_size = get_terminal_size()

    @property
    def max_yoffset(self) -> int:
        return max(len(self.lines) - self.term_size.lines, 0)

    @property
    def max_xoffset(self) -> int:
        return max(self.max_line_width - (self.term_size.columns - len(str(len(self.lines))) - 2), 0)

    def start(self) -> None:
        signal(SIGWINCH, self._handle_sigwinch)
        with UnbufferedInput():
            try:
                # hide cursor
                sys.stdout.write('\x1b[?25l')
                sys.stdout.flush()

                while True:
                    try:
                        self.redraw()
                        raw, decoded = read_ansi(self.stdin)
                        #print(raw, decoded)

                        self.message = None
                        self.message_level = LEVEL_INFO

                        if raw == b'q':
                            # TODO: ask save
                            break
                        elif raw == b'w':
                            try:
                                with open(self.filename, 'w') as fp:
                                    fp.write(self.content)
                            except Exception as exc:
                                self.message_level = LEVEL_ERROR
                                self.message = str(exc)
                            else:
                                self.saved = True
                                self.message = 'Written to file: ' + self.filename
                                self.message_level = LEVEL_INFO
                        elif raw == b' ':
                            self.message = "Please speak now..."
                            self.message_level = LEVEL_INFO
                            # TODO: record voice in a thread until space is pressed again
                        elif raw == b':':
                            try:
                                # edit via text input
                                sys.stdout.write('\x1b[%dE' % (self.term_size.lines))
                                sys.stdout.write('\x1b[0K\r')
                                sys.stdout.flush()
                                with EchoedInput():
                                    text = input(':')

                                self.message = "Wating for OpenAI..."
                                self.message_level = LEVEL_INFO
                                self.redraw()

                                if self.lexer:
                                    lang = self.lexer.name + ' code'
                                else:
                                    lang = 'code'

                                # TODO: functions for saving, loading, scrolling etc.
                                # TODO: off-load this stuff to a thread
                                start_lineno = self.scroll_yoffset + 1
                                end_lineno = max(start_lineno + self.term_size.lines + 1, 0)
                                response = self.openai.chat.completions.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": SYSTEM_MESSAGE_CONTENT.format(
                                                lang=lang,
                                                code=self.content,
                                                start_lineno=start_lineno,
                                                end_lineno=end_lineno,
                                                filename=self.filename,
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": text
                                        }
                                    ],
                                    max_tokens=MAX_TOKENS
                                )

                                choice = response.choices[0]
                                message = choice.message.content or ''
                                res = parse_response(message)

                                if res.message:
                                    self.message = res.message
                                    self.message_level = res.message_level
                                else:
                                    self.message = None
                                    self.message_level = LEVEL_INFO

                                if res.code is not None:
                                    self.content = res.code

                                    self.lines = tokenized_lines(self.content, self.lexer)
                                    self.max_line_width = max(
                                        wcswidth(''.join(tok_data for _, tok_data in line))
                                        for line in self.lines
                                    )

                                    max_xoffset = self.max_xoffset
                                    max_yoffset = self.max_yoffset

                                    if self.scroll_xoffset > max_xoffset:
                                        self.scroll_xoffset = max_xoffset

                                    if self.scroll_yoffset > max_yoffset:
                                        self.scroll_yoffset = max_yoffset

                                    self.saved = False

                                # DEBUG:
                                #with open("/tmp/response.json", "w") as fp:
                                #    fp.write(choice.message.json())

                            except Exception as exc:
                                self.message_level = LEVEL_ERROR
                                self.message = str(exc)
                        elif decoded:
                            prefix, suffix, args = decoded
                            if prefix == -1:
                                if suffix == 65: # UP
                                    if self.scroll_yoffset > 0:
                                        self.scroll_yoffset -= 1
                                elif suffix == 66: # DOWN
                                    self.scroll_yoffset = min(self.scroll_yoffset + 1, self.max_yoffset)
                                elif suffix == 67: # RIGHT
                                    if self.scroll_xoffset < self.max_xoffset:
                                        self.scroll_xoffset += 1
                                elif suffix == 68: # LEFT
                                    if self.scroll_xoffset > 0:
                                        self.scroll_xoffset -= 1
                                elif suffix == 70: # END
                                    if args == CTRL: # CTRL+END
                                        self.scroll_yoffset = self.max_yoffset
                                    else:
                                        self.scroll_xoffset = self.max_xoffset
                                elif suffix == 72: # HOME
                                    if args == CTRL: # CTRL+HOME
                                        self.scroll_yoffset = 0
                                    else:
                                        self.scroll_xoffset = 0
                                elif decoded == PAGE_UP:
                                    self.scroll_yoffset = max(self.scroll_yoffset - self.term_size.lines, 0)
                                elif decoded == PAGE_DOWN:
                                    self.scroll_yoffset = min(self.scroll_yoffset + self.term_size.lines, self.max_yoffset)
                    except KeyboardInterrupt:
                        self.message = "Keyboard Interrrupt, use q for quit"
                        self.message_level = LEVEL_WARNING
                    except Exception as exc:
                        self.message = str(exc)
                        self.message_level = LEVEL_ERROR
            finally:
                # show cursor
                sys.stdout.write('\x1b[?25h')
                sys.stdout.write('\n')
                sys.stdout.flush()

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        self.stdin.close()

    def _handle_sigwinch(self, sig: int, frame) -> None:
        self.term_size = get_terminal_size()
        max_xoffset = self.max_xoffset
        max_yoffset = self.max_yoffset

        if self.scroll_xoffset > max_xoffset:
            self.scroll_xoffset = max_xoffset

        if self.scroll_yoffset > max_yoffset:
            self.scroll_yoffset = max_yoffset

        self.redraw()

    def redraw(self) -> None:
        # clear screen
        # sys.stdout.write('\x1b[2J')
        # move cursor to upper left corner
        sys.stdout.write('\x1b[H')

        columns = self.term_size.columns
        scroll_xoffset = self.scroll_xoffset
        scroll_yoffset = self.scroll_yoffset
        lineno = scroll_yoffset + 1
        max_yoffset = scroll_yoffset + self.term_size.lines
        message = self.message
        if message is not None:
            max_yoffset = max(max_yoffset - 1, 0)

        max_lineno_len = len(str(max_yoffset))
        avail_columns = max(columns - max_lineno_len - 1, 0)
        for line in self.lines[scroll_yoffset:max_yoffset]:
            sys.stdout.write(str(lineno).rjust(max_lineno_len) + ' ')
            tokens = slice_token_line(line, scroll_xoffset, avail_columns)
            colorize(tokens, self.formatter, sys.stdout)
            # clear to end of line
            sys.stdout.write('\x1b[0K')
            if lineno < max_yoffset:
                sys.stdout.write('\n')
            lineno += 1

        if message is not None:
            sys.stdout.write('\n')
            if self.message_level == LEVEL_WARNING:
                sys.stdout.write('\x1b[33m')
            elif self.message_level == LEVEL_ERROR:
                sys.stdout.write('\x1b[31m')

            sys.stdout.write(message)

            if self.message_level != LEVEL_INFO:
                sys.stdout.write('\x1b[0m')

        # clear to end of screen
        sys.stdout.write('\x1b[0J')
        sys.stdout.flush()

def main() -> None:
    optparser = optparse.OptionParser()
    opts, args = optparser.parse_args()

    if len(args) != 1:
        raise ValueError("expected exactly one file argument")

    filename = args[0]
    coder = VoiceCoder(filename)
    coder.start()

if __name__ == '__main__':
    main()
