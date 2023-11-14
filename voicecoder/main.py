#!/usr/bin/env python3

from dotenv import load_dotenv

load_dotenv()

from typing import Optional, BinaryIO, TextIO, Union, NamedTuple, Literal, List, Pattern

import re
import os
import io
import sys
import json
import optparse
import termios
import platform
import logging
# import xdelta3 # XXX: broken?
import sounddevice
#import wavio
import lameenc
import numpy as np

from os import get_terminal_size
from os.path import join as join_path, expanduser, dirname
from signal import signal, SIGWINCH, SIG_DFL
from queue import Queue
from threading import Thread, Semaphore
from enum import Enum
from datetime import datetime
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai._types import NOT_GIVEN
from pygments import format as colorize
from pygments.lexer import Lexer
from pygments.token import _TokenType, Token
from pygments.lexers import TextLexer, find_lexer_class_for_filename
from pygments.formatter import Formatter
from pygments.formatters import TerminalFormatter
from wcwidth import wcswidth
#from pydantic import BaseModel

logger = logging.getLogger(__name__)

if platform.system() == 'Windows':
    def get_app_data_dir() -> str:
        appdata_dir = os.getenv('LOCALAPPDATA')
        if appdata_dir is None:
            appdata_dir = expanduser('~\\AppData\\Local')

        return join_path(appdata_dir, 'VoiceCoder')
else:
    def get_app_data_dir() -> str:
        return join_path(expanduser('~'), '.local', 'voicecoder')

try:
    import readline
except ImportError:
    pass
else:
    readline.parse_and_bind('tab: complete')

DEFAULT_SAMPLERATE = 44100
DEFAULT_LEVEL = 'INFO'
APP_DATA_DIR = get_app_data_dir()
DEFAULT_LOGFILE = join_path(APP_DATA_DIR, 'session.log')

ARGS_CTRL = [1, 5]
DECODED_PAGE_UP   = (-1, 126, [5])
DECODED_PAGE_DOWN = (-1, 126, [6])

INPUT_PAGE_UP   = (b'\x1b[5~', DECODED_PAGE_UP)
INPUT_PAGE_DOWN = (b'\x1b[6~', DECODED_PAGE_DOWN)
INPUT_CURSOR_UP    = (b'\x1b[A', (-1, 65, []))
INPUT_CURSOR_DOWN  = (b'\x1b[B', (-1, 66, []))
INPUT_CURSOR_LEFT  = (b'\x1b[D', (-1, 68, []))
INPUT_CURSOR_RIGHT = (b'\x1b[C', (-1, 67, []))

LEVEL_INFO    = 0
LEVEL_WARNING = 1
LEVEL_ERROR   = 2

MAX_UNDO_STEPS = 1024

GptModelType = Literal[
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]

GPT_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]

DEFAULT_GPT_MODEL = 'gpt-4-1106-preview'

MAX_TOKENS = 4096

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "page_down",
            "description": "Scroll the view down by one page.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "page_up",
            "description": "Scroll the view up by one page.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cursor_down",
            "description": "Scroll the view down by one line.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cursor_up",
            "description": "Scroll the view up by one line.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cursor_left",
            "description": "Scroll the view left by one column.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cursor_right",
            "description": "Scroll the view right by one column.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo last change.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "redo",
            "description": "Redo previously reverted change.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save",
            "description": "Save changes of current file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "quit",
            "description": "Quit the code editor.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
]

REST_UNCHANGED_MARK = '[The rest of the file' + \
                      ' remains unchanged]'

SYSTEM_MESSAGE_CONTENT = f"""\
You're a programming accessibility tool. You will get natural language describing code and answer with valid {{lang}}. The natural language input comes from voice recognition and thus will omit a lot of special characters which you will have to fill in. Be sure to always echo back the whole edited file. Keep any messages about what you where doing very short.

If you reply with only the head of the file leaving the rest unchange please keep some unchanged context lines if possible, don't add an ellipsis, but insert this line as a comment at the end:
{REST_UNCHANGED_MARK}

The name of the current file is:
{{filename}}

The user is currently viewing line {{start_lineno}} to {{end_lineno}} of this file.

This is the content of the file:
{{code}}"""

HELP = """\
    Hotkeys
    -------

    h ......... show this help message
    w ......... write file (save)
    W ......... write to new file (save as)
    e ......... edit a new file (open)
    <SPACE> ... start voice recording
    : ......... enter command via typing
    u ......... undo
    r ......... redo
    <ENTER> ... clear message
    q ......... quit

    Use cursor keys, PAGE UP and PAGE DOWN to scroll.
    Use HOME/END to jump to the start/end of the line.
    Use Ctrl+HOME/Ctrl+END to jump to the start/end of the file.
"""

InputData = tuple[Union[bytes, bytearray], Optional[tuple[int, int, list[int]]]]

class ContentUpdate(NamedTuple):
    code: Optional[str] = None
    message: Optional[str] = None
    message_level: int = LEVEL_INFO

class EventType(Enum):
    REDRAW         = 0
    INPUT          = 1
    CONTENT_UPDATE = 2
    FUNC_CALL      = 3
    SET_MESSAGE    = 4
    CLEAR_MESSAGE  = 5

Event = Union[
    tuple[Literal[EventType.REDRAW]],
    tuple[Literal[EventType.INPUT], InputData],
    tuple[Literal[EventType.CONTENT_UPDATE], ContentUpdate],
    tuple[Literal[EventType.FUNC_CALL], str, list],
    tuple[Literal[EventType.SET_MESSAGE], str, int],
    tuple[Literal[EventType.CLEAR_MESSAGE]],
]

class MessageType(Enum):
    TEXT = 1
    VOICE = 2

Message = Union[
    tuple[Literal[MessageType.TEXT], str],
    tuple[Literal[MessageType.VOICE], Union[bytes, bytearray, memoryview]],
]

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

ELLIPSIS = r'(?:\.\.\.)?'
REST_UNCHANGED_MARK_PATTERN = rf'[ \t]*{ELLIPSIS}[ \t]*\[?(?:The )?(?:(?:rest of|remaining code in) the file|rest of the code) remains (?:unchanged|the same)\]?[ \t]*{ELLIPSIS}[ \t]*'
FALLBACK_REST_UNCHANGED_MARK = re.compile(rf'^[ \t]*(?:[#;]|//|--|/\*|\(\*\*|<!--)?{REST_UNCHANGED_MARK_PATTERN}(?:\*/|\*\)|-->)?[ \t]*$', re.M | re.I)

DASHDASH_COMMENT_REST_UNCHANGED_MARK    = re.compile(rf'^[ \t]*--{   REST_UNCHANGED_MARK_PATTERN}[ \t]*$', re.M | re.I)
HASH_COMMENT_REST_UNCHANGED_MARK        = re.compile(rf'^[ \t]*#{    REST_UNCHANGED_MARK_PATTERN}[ \t]*$', re.M | re.I)
INLINE_C_COMMENT_REST_UNCHANGED_MARK    = re.compile(rf'^[ \t]*//{   REST_UNCHANGED_MARK_PATTERN}[ \t]*$', re.M | re.I)
MULTILINE_C_COMMENT_REST_UNCHANGED_MARK = re.compile(rf'^[ \t]*/\*{  REST_UNCHANGED_MARK_PATTERN}\*/[ \t]*$', re.M | re.I)
OCAML_COMMENT_REST_UNCHANGED_MARK       = re.compile(rf'^[ \t]*(\*\*{REST_UNCHANGED_MARK_PATTERN}\*)[ \t]*$', re.M | re.I)
SEMICOLON_COMMENT_REST_UNCHANGED_MARK   = re.compile(rf'^[ \t]*;{    REST_UNCHANGED_MARK_PATTERN}[ \t]*$', re.M | re.I)
SGML_REST_UNCHANGED_MARK                = re.compile(rf'^[ \t]*<!--{ REST_UNCHANGED_MARK_PATTERN}-->[ \t]*$', re.M | re.I)

REST_UNCHANGED_MARKS: dict[Optional[str], Pattern] = {
    'Bash': HASH_COMMENT_REST_UNCHANGED_MARK,
    'C#': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'C': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'C++': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'CSS': MULTILINE_C_COMMENT_REST_UNCHANGED_MARK,
    'Dart': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'Go': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'Haskell': DASHDASH_COMMENT_REST_UNCHANGED_MARK,
    'HTML': SGML_REST_UNCHANGED_MARK,
    'Java': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'JavaScript': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'Kotlin': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'Lua': DASHDASH_COMMENT_REST_UNCHANGED_MARK,
    'Nginx configuration file': HASH_COMMENT_REST_UNCHANGED_MARK,
    'Nimrod': HASH_COMMENT_REST_UNCHANGED_MARK,
    'NSAM': SEMICOLON_COMMENT_REST_UNCHANGED_MARK,
    'OCaml': OCAML_COMMENT_REST_UNCHANGED_MARK,
    'Python 2.x': HASH_COMMENT_REST_UNCHANGED_MARK,
    'Python': HASH_COMMENT_REST_UNCHANGED_MARK,
    'Ruby': HASH_COMMENT_REST_UNCHANGED_MARK,
    'Rust': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'SGML': SGML_REST_UNCHANGED_MARK,
    'SQL': DASHDASH_COMMENT_REST_UNCHANGED_MARK,
    'TypeScript': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
    'XML': SGML_REST_UNCHANGED_MARK,
    'XSLT': SGML_REST_UNCHANGED_MARK,
    'YAML': HASH_COMMENT_REST_UNCHANGED_MARK,
    'Zig': INLINE_C_COMMENT_REST_UNCHANGED_MARK,
}

def parse_content_update(message: str, old_content: str, lang: Optional[str]) -> ContentUpdate:
    if message.startswith('```'):
        index = 0
    else:
        index = message.find('\n```')
        if index == -1:
            return ContentUpdate(message=message)
        index += 1

    start_index = message.find('\n', index)
    if start_index == -1:
        return ContentUpdate(message=message)

    start_index += 1

    end_index = message.find('\n```\n', start_index)

    if end_index == -1:
        if message.endswith('\n```'):
            end_index = len(message) - 3
        else:
            return ContentUpdate(message=message)
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

    # HACK: Try to handle when OpenAI is truncating the file marking with:
    #       # ... [The rest of the file remains unchanged] ...
    mark = REST_UNCHANGED_MARKS.get(lang, FALLBACK_REST_UNCHANGED_MARK)
    match = mark.search(code)

    if match and not code[match.end():].strip():
        new_code = code[:match.start()].rstrip()
        last_line_index = new_code.rfind('\n')
        if last_line_index == -1:
            last_line_index = 0
        else:
            last_line_index += 1
        last_line = new_code[last_line_index:]
        last_line_pattern = rf'^{re.escape(last_line)}[ \t]*$'
        logger.debug(f'reconstructing truncated file with last line: {last_line!r}')
        match = re.search(last_line_pattern, old_content, re.M)
        if match:
            logger.debug("MATCHED CONTEXT LINE")
            code = f'{new_code}\n{old_content[match.end():]}'
        else:
            logger.debug("CONTEXT LINE NOT FOUND!")
            code = f'{new_code}\n{old_content}'

        #line_count = code.count('\n') + 1
        #index = find_mark_line(last_line, old_content, line_count)
        #if index >= 0:
        #    code = new_code + '\n' + old_content[index:]
        #
        #    logger.info('reconstructing truncated code')

    return ContentUpdate(code=code, message=new_message or None)

def find_mark_line(mark_line: str, content: str, head_lines: int) -> int:
    line_count = 0
    prev = 0
    while line_count < head_lines:
        index = content.find('\n', prev)
        if index < 0:
            line = content[prev:]
            if line == mark_line:
                return prev
            return -1
        line = content[prev:index]
        if line == mark_line:
            return prev
        prev = index + 1
        line_count += 1
    return -1

def read_ansi(stdin: BinaryIO) -> InputData:
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
            if (tok_type is Token.Text or tok_type is Token.Whitespace) and tok_data == '\n':
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

        if lines and not lines[-1]:
            lines.pop()
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
                        current_offset += w
                        tok_data = tok_data[index:]
                        break
            if next_offset > max_offset:
                # TODO: don't cut in the middle of graphemes
                for index in range(len(tok_data), -1, -1):
                    new_tok_data = tok_data[:index]
                    w = wcswidth(new_tok_data)
                    next_offset = current_offset + w
                    if next_offset <= max_offset:
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
        'message', 'message_level', 'openai', 'undo_history', 'redo_history',
        'event_queue', 'input_thread', 'message_thread', 'recorder_thread',
        'message_queue', 'input_semaphore', 'waiting_for_openai', 'running',
        'recorder_semaphore', 'recording', 'silence', 'log_voice',
        'voicelog_dir', 'message_log', 'log_messages', 'voice_lang',
        'sound_device', 'samplerate', 'gpt_model', 'enable_tools',
    )

    scroll_xoffset: int
    scroll_yoffset: int
    filename: str
    stdin: BinaryIO
    message_log: Optional[TextIO]
    term_size: os.terminal_size
    lexer: Optional[Lexer]
    formatter: Formatter
    content: str
    lines: list[list[tuple[_TokenType, str]]]
    saved: bool
    max_line_width: int
    message: Optional[list[str]]
    message_level: int
    openai: OpenAI
    # XXX: xdelta3 is broken?
    #undo_history: list[tuple[Union[bytes, str], Union[bytes, str]]]
    #redo_history: list[tuple[Union[bytes, str], Union[bytes, str]]]
    undo_history: list[tuple[str, str]]
    redo_history: list[tuple[str, str]]
    event_queue: Queue[Event]
    message_queue: Queue[Optional[Message]]
    input_thread: Thread
    message_thread: Thread
    recorder_thread: Thread
    input_semaphore: Semaphore
    recorder_semaphore: Semaphore
    recording: bool
    waiting_for_openai: bool
    running: bool
    silence: bool
    log_voice: bool
    log_messages: bool
    voicelog_dir: str
    voice_lang: Optional[str]
    sound_device: Optional[str]
    samplerate: int
    gpt_model: GptModelType
    enable_tools: bool

    def __init__(self,
                 filename: str,
                 log_voice: bool = False,
                 log_messages: bool = False,
                 voice_lang: Optional[str] = None,
                 sound_device: Optional[str] = None,
                 samplerate: int = DEFAULT_SAMPLERATE,
                 gpt_model: GptModelType = DEFAULT_GPT_MODEL,
                 enable_tools: bool = False,
    ) -> None:
        self.scroll_xoffset = 0
        self.scroll_yoffset = 0
        self.filename = filename
        self.formatter = TerminalFormatter()
        self.saved = True
        self.message = None
        self.message_level = LEVEL_INFO
        self.openai = OpenAI()
        self.undo_history = []
        self.redo_history = []
        self.event_queue = Queue()
        self.message_queue = Queue()
        self.input_thread = Thread(target=self._input_thread_func, daemon=True)
        self.message_thread = Thread(target=self._message_thread_func, daemon=True)
        self.recorder_thread = Thread(target=self._recorder_thread_func, daemon=True)
        self.input_semaphore = Semaphore(0)
        self.recorder_semaphore = Semaphore(0)
        self.recording = False
        self.waiting_for_openai = False
        self.running = False
        self.silence = False
        self.log_voice = log_voice
        self.log_messages = log_messages
        self.voicelog_dir = join_path(APP_DATA_DIR, 'voicelog')
        self.voice_lang = voice_lang
        self.sound_device = sound_device
        self.samplerate = samplerate
        self.gpt_model = gpt_model
        self.enable_tools = enable_tools

        self.open_file(self.filename)

        self.stdin = io.open(sys.stdin.fileno(), 'rb', closefd=False, buffering=False)
        self.term_size = get_terminal_size()

    def open_file(self, filename: str) -> None:
        try:
            with open(filename, 'rt') as fp:
                self.content = fp.read()
        except FileNotFoundError:
            self.content = ''

        self.filename = filename

        lexer_class = find_lexer_class_for_filename(filename, self.content)
        if lexer_class is TextLexer or lexer_class is None:
            self.lexer = None
        else:
            self.lexer = lexer_class()

        self.lines = tokenized_lines(self.content, self.lexer)
        self.max_line_width = max(
            wcswidth(''.join(tok_data for _, tok_data in line))
            for line in self.lines
        )

    def _input_thread_func(self) -> None:
        while self.running:
            try:
                self.input_semaphore.acquire()
                if self.running:
                    item = read_ansi(self.stdin)
                    self.event_queue.put((EventType.INPUT, item))
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                logger.error('input thread error:', exc_info=exc)
                self.event_queue.put((EventType.SET_MESSAGE, f'input error: {exc}', LEVEL_ERROR))

    def _message_thread_func(self) -> None:
        while self.running:
            try:
                maybe_message = self.message_queue.get()
                if maybe_message is None or not self.running:
                    logger.debug("message thread: quit")
                    return
                else:
                    message_data = maybe_message

                self.waiting_for_openai = True
                self.event_queue.put((EventType.REDRAW, ))

                now = datetime.now()

                if message_data[0] == MessageType.VOICE:
                    voice_data = message_data[1]

                    encoder = lameenc.Encoder()
                    encoder.silence()
                    encoder.set_bit_rate(128)
                    encoder.set_in_sample_rate(self.samplerate)
                    encoder.set_channels(1)
                    encoder.set_quality(5)
                    mp3_data = encoder.encode(voice_data)
                    mp3_data += encoder.flush()

                    if self.log_voice:
                        try:
                            now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
                            mp3_filename = join_path(self.voicelog_dir, now_str + '.mp3')

                            try:
                                with open(mp3_filename, "wb") as fp:
                                    fp.write(mp3_data)
                            except FileNotFoundError:
                                os.makedirs(self.voicelog_dir, exist_ok=True)

                                with open(mp3_filename, "wb") as fp:
                                    fp.write(mp3_data)
                        except Exception as exc:
                            logger.error('saving voice log error:', exc_info=exc)

                    self.event_queue.put((EventType.SET_MESSAGE, 'Processing voice message...', LEVEL_INFO))
                    whisper_response = self.openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=("recording.mp3", io.BytesIO(mp3_data), 'audio/mpeg'),
                        language=self.voice_lang or NOT_GIVEN,
                    )

                    if not self.running:
                        self.event_queue.put((EventType.CLEAR_MESSAGE, ))
                        break

                    user_message = whisper_response.text
                    if user_message:
                        self.event_queue.put((EventType.SET_MESSAGE, ':' + user_message, LEVEL_INFO))
                else:
                    user_message = message_data[1]

                if not user_message:
                    continue

                if self.message_log:
                    now_str = now.strftime('%Y-%m-%d %H-%M-%S')
                    self.message_log.write(f'# {now_str}\n')
                    self.message_log.write(''.join(f'> {line}\n' for line in user_message.split('\n')))
                    self.message_log.write('\n')
                    self.message_log.flush()

                if self.lexer:
                    lang = self.lexer.name
                    lang_name = lang + ' code'
                else:
                    lang = None
                    lang_name = 'code'

                # TODO: put all context data in event, don't touch self here
                start_lineno = self.scroll_yoffset + 1
                end_lineno = max(start_lineno + self.term_size.lines + 1, 0)
                logger.info(f"sending message: {user_message!r}")
                content = self.content
                messages: List[ChatCompletionMessageParam] = []
                messages.append({
                    "role": "system",
                    "content": SYSTEM_MESSAGE_CONTENT.format(
                        lang=lang_name,
                        code=content,
                        start_lineno=start_lineno,
                        end_lineno=end_lineno,
                        filename=self.filename,
                    )
                })
                messages.append({
                    "role": "user",
                    "content": user_message,
                })

                # model="gpt-4-1106-preview"
                # model="gpt-3.5-turbo-16k"

                response = self.openai.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "page_down",
                                "description": "Scroll the view down by one page.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                }
                            }
                        }
                    ] if self.enable_tools else NOT_GIVEN,
                    tool_choice='auto' if self.enable_tools else NOT_GIVEN,
                )
                logger.debug(f"response message: {response}")

                choice = response.choices[0]
                tool_calls = choice.message.tool_calls

                if tool_calls and self.enable_tools:
                    # TODO: more functions?
                    if choice.message.content is None:
                        choice.message.content = ''
                    messages.append(choice.message) # type: ignore

                    illegal_tool = False
                    for tool_call in tool_calls:
                        func = tool_call.function
                        func_name = func.name
                        logger.info(f'message thread: called tool: {func_name} {func.arguments}')
                        args = json.loads(func.arguments)
                        if func_name == 'page_up':
                            self.event_queue.put((EventType.INPUT, INPUT_PAGE_UP))
                            retval = '{"success":true}'
                        elif func_name == 'page_down':
                            self.event_queue.put((EventType.INPUT, INPUT_PAGE_DOWN))
                            retval = '{"success":true}'
                        elif func_name == 'cursor_down':
                            self.event_queue.put((EventType.INPUT, INPUT_CURSOR_DOWN))
                            retval = '{"success":true}'
                        elif func_name == 'cursor_up':
                            self.event_queue.put((EventType.INPUT, INPUT_CURSOR_UP))
                            retval = '{"success":true}'
                        elif func_name == 'cursor_left':
                            self.event_queue.put((EventType.INPUT, INPUT_CURSOR_LEFT))
                            retval = '{"success":true}'
                        elif func_name == 'cursor_right':
                            self.event_queue.put((EventType.INPUT, INPUT_CURSOR_RIGHT))
                            retval = '{"success":true}'
                        elif func_name == 'undo':
                            self.event_queue.put((EventType.INPUT, (b'u', None)))
                            retval = '{"success":true}'
                        elif func_name == 'redo':
                            self.event_queue.put((EventType.INPUT, (b'r', None)))
                            retval = '{"success":true}'
                        elif func_name == 'save':
                            self.event_queue.put((EventType.INPUT, (b'w', None)))
                            retval = '{"success":true}'
                        elif func_name == 'quit':
                            self.event_queue.put((EventType.INPUT, (b'q', None)))
                            retval = '{"success":true}'
                        else:
                            logger.error(f'message thread: illegal tool call: {func_name} {func.arguments}')
                            illegal_tool = True
                            continue

                        messages.append({ # type: ignore
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": retval,
                        })

                    if illegal_tool:
                        continue

                    response = self.openai.chat.completions.create(
                        model=self.gpt_model,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                    )

                bot_message = choice.message.content or ''

                if self.message_log:
                    self.message_log.write(''.join(f'< {line}\n' for line in bot_message.split('\n')))
                    self.message_log.write('\n')
                    self.message_log.flush()

                update_data = parse_content_update(bot_message, content, lang)

                logger.info(f"message thread: parsed response:\n{update_data}")

                self.event_queue.put((EventType.CONTENT_UPDATE, update_data))
            except Exception as exc:
                logger.error('message thread error:', exc_info=exc)
                self.event_queue.put((EventType.SET_MESSAGE, f'OpenAI error: {exc}', LEVEL_ERROR))
            finally:
                self.waiting_for_openai = False
                self.event_queue.put((EventType.REDRAW, ))

    def _recorder_thread_func(self) -> None:
        #wavfile = io.BytesIO()

        while self.running:
            try:
                self.recorder_semaphore.acquire()
                self.recording = True
                self.event_queue.put((EventType.REDRAW,))

                chunks = []
                stream = sounddevice.InputStream(self.samplerate, channels=1, dtype=np.int16, device=self.sound_device)
                silence_treshold = int(~(-1 << (stream.samplesize * 8)) * 0.25 * 0.5) # type: ignore
                stream.start()
                try:
                    #no = 1
                    while self.running and self.recording:
                        # record for 2 seconds, i.e. 2 seconds silence is needed for input end
                        chunk, _was_discarded = stream.read(self.samplerate * 2)
                        # TODO: better silence detection
                        if np.all(abs(chunk) < silence_treshold):
                            self.silence = True
                            self.event_queue.put((EventType.REDRAW,))
                            if chunks:
                                data = np.concatenate(chunks)
                                if data.dtype.str != '<i2': # type: ignore
                                    data = data.astype('<i2', copy=False) # type: ignore
                                data = data.tobytes()
                                chunks.clear()

                                #wavfile.truncate(0)
                                #wavio.write(wavfile, data, self.samplerate, sampwidth=2)
                                #data = wavfile.getbuffer()

                                #with open(f"/tmp/recording_{no:04d}.wav", "wb") as fp:
                                #    fp.write(data)
                                #no += 1

                                self.message_queue.put((MessageType.VOICE, data))
                            #else:
                            #    self.event_queue.put((EventType.SET_MESSAGE, 'Are you still there?', LEVEL_WARNING))
                        else:
                            self.silence = False
                            self.event_queue.put((EventType.REDRAW,))
                            chunks.append(chunk)

                    self.silence = True
                    if self.running:
                        if chunks:
                            data = np.concatenate(chunks)
                            if data.dtype.str != '<i2': # type: ignore
                                data = data.astype('<i2', copy=False) # type: ignore
                            data = data.tobytes()
                            #wavfile.truncate(0)
                            #wavio.write(wavfile, data, self.samplerate)
                            #data = wavfile.getbuffer()

                            #with open(f"/tmp/recording_{no:04d}.wav", "wb") as fp:
                            #    fp.write(data)
                            #no += 1

                            self.message_queue.put((MessageType.VOICE, data))

                        self.event_queue.put((EventType.SET_MESSAGE, 'Stopped recording', LEVEL_INFO))
                finally:
                    stream.stop()
                    stream.close()

            except Exception as exc:
                logger.error('recorder thread error:', exc_info=exc)
                self.event_queue.put((EventType.SET_MESSAGE, f'recording error: {exc}', LEVEL_ERROR))

    @property
    def max_yoffset(self) -> int:
        return max(len(self.lines) - self.term_size.lines, 0)

    @property
    def max_xoffset(self) -> int:
        return max(self.max_line_width - (self.term_size.columns - len(str(len(self.lines))) - 2), 0)

    def clear_message(self) -> None:
        self.message = None
        self.message_level = LEVEL_INFO

    def set_message(self, message: str, level: int=LEVEL_INFO) -> None:
        self.message = message.split('\n')
        self.message_level = level

    def show_unknown_shortcut(self) -> None:
        self.set_message("Unknown shortcut", LEVEL_WARNING)

    def set_content(self, content: str) -> None:
        self.content = content

        self.lines = tokenized_lines(content, self.lexer)
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

    def edit(self, new_content: str) -> None:
        undo = self.content
        redo = new_content

        # XXX: xdelta is broken?
        #new_bytes = new_content.encode()
        #old_bytes = self.content.encode()
        #try:
        #    undo = xdelta3.encode(new_bytes, old_bytes)
        #except xdelta3.NoDeltaFound:
        #    undo = self.content

        #try:
        #    redo = xdelta3.encode(old_bytes, new_bytes)
        #except xdelta3.NoDeltaFound:
        #    redo = new_content

        self.undo_history.append((undo, redo))
        if len(self.undo_history) > MAX_UNDO_STEPS:
            del self.undo_history[0]
        self.set_content(new_content)
        self.redo_history.clear()
        self.saved = False

    def undo(self) -> None:
        if self.undo_history:
            item = self.undo_history.pop()
            undo, _redo = item
            #content = self.content.encode()
            new_content = undo
            #if isinstance(undo, str):
            #    new_content = undo
            #else:
            #    new_content = xdelta3.decode(content, undo).decode()
            self.set_content(new_content)
            self.redo_history.append(item)
            self.saved = False
        else:
            self.set_message('Already at oldest change', LEVEL_WARNING)

    def redo(self) -> None:
        if self.redo_history:
            item = self.redo_history.pop()
            _undo, redo = item
            #content = self.content.encode()
            new_content = redo
            #if isinstance(redo, str):
            #    new_content = redo
            #else:
            #    new_content = xdelta3.decode(content, redo).decode()
            self.set_content(new_content)
            self.undo_history.append(item)
            self.saved = False
        else:
            self.set_message('Already at newest change', LEVEL_WARNING)

    def prompt(self, prompt: str) -> str:
        sys.stdout.write('\x1b[%dE' % (self.term_size.lines))
        sys.stdout.write('\x1b[0K\r')
        sys.stdout.flush()
        with EchoedInput():
            return input(prompt)

    def save(self) -> None:
        try:
            with open(self.filename, 'w') as fp:
                fp.write(self.content)
        except Exception as exc:
            logger.error('Error saving file:', exc_info=exc)
            self.set_message(f'Error saving file: {exc}', LEVEL_ERROR)
        else:
            self.saved = True
            self.set_message('Written to file: ' + self.filename)

    def save_as(self) -> None:
        try:
            self.filename = self.prompt('Filename: ')
        except KeyboardInterrupt:
            self.set_message('Cancelled')
        else:
            self.save()

    def open(self) -> None:
        try:
            self.open_file(self.prompt('Filename: '))
        except KeyboardInterrupt:
            self.set_message('Cancelled')
        else:
            self.save()

    def _handle_input(self, input_data: InputData) -> bool:
        try:
            #raw, decoded = read_ansi(self.stdin)
            #print(raw, decoded)
            raw, decoded = input_data

            if raw == b'q':
                if not self.saved:
                    while True:
                        cmd = self.prompt('Save changes? y/n/C> ').strip().lower()

                        if cmd == 'y' or cmd == 'yes' or cmd == 'save':
                            self.save()
                            self.running = False
                            return False
                        elif cmd == 'n' or cmd == 'no':
                            self.running = False
                            return False
                        elif cmd == '' or cmd == 'c' or cmd == 'cancel':
                            break
                else:
                    self.running = False
                    return False
            elif raw == b'w':
                self.save()
            elif raw == b'W':
                self.save_as()
            elif raw == b'e':
                self.open()
            elif raw == b' ':
                if self.recording:
                    self.recording = False
                else:
                    self.recorder_semaphore.release()
            elif raw == b'\n':
                self.clear_message()
            elif raw == b'u':
                self.undo()
            elif raw == b'r':
                self.redo()
            elif raw == b'h':
                self.set_message(HELP)
            elif raw == b':':
                try:
                    if self.waiting_for_openai:
                        self.set_message("Already waiting for an OpenAI action...", LEVEL_WARNING)
                    else:
                        # edit via text input
                        self.clear_message()
                        try:
                            message = self.prompt(':')
                        except KeyboardInterrupt:
                            self.set_message('Cancelled')
                        else:
                            self.waiting_for_openai = True
                            self.message_queue.put((MessageType.TEXT, message))

                except Exception as exc:
                    logger.error('Error processing message:', exc_info=exc)
                    self.set_message(f'Error processing message: {exc}', LEVEL_ERROR)
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
                        if args == ARGS_CTRL: # CTRL+END
                            self.scroll_yoffset = self.max_yoffset
                        else:
                            self.scroll_xoffset = self.max_xoffset
                    elif suffix == 72: # HOME
                        if args == ARGS_CTRL: # CTRL+HOME
                            self.scroll_yoffset = 0
                        else:
                            self.scroll_xoffset = 0
                    elif decoded == DECODED_PAGE_UP:
                        self.scroll_yoffset = max(self.scroll_yoffset - self.term_size.lines, 0)
                    elif decoded == DECODED_PAGE_DOWN:
                        self.scroll_yoffset = min(self.scroll_yoffset + self.term_size.lines, self.max_yoffset)
                    else:
                        self.show_unknown_shortcut()
                else:
                    self.show_unknown_shortcut()
            else:
                self.show_unknown_shortcut()
        finally:
            self.input_semaphore.release()

        return True

    def start(self) -> None:
        signal(SIGWINCH, self._handle_sigwinch)

        if self.log_messages:
            msg_log_file = join_path(APP_DATA_DIR, 'messages.log')
            try:
                self.message_log = open(msg_log_file, 'at')
            except FileNotFoundError:
                os.makedirs(APP_DATA_DIR, exist_ok=True)
                self.message_log = open(msg_log_file, 'at')
        else:
            self.message_log = None

        self.running = True
        self.input_thread.start()
        self.message_thread.start()
        self.recorder_thread.start()
        self.input_semaphore.release()

        with UnbufferedInput():
            try:
                # hide cursor
                sys.stdout.write('\x1b[?25l')
                sys.stdout.flush()

                while self.running:
                    try:
                        self.redraw()
                        event = self.event_queue.get()

                        if event[0] == EventType.INPUT:
                            if not self._handle_input(event[1]):
                                break

                        elif event[0] == EventType.REDRAW:
                            self.redraw()

                        elif event[0] == EventType.CONTENT_UPDATE:
                            res = event[1]
                            if res.message:
                                self.set_message(res.message, res.message_level)
                            else:
                                self.clear_message()

                            new_content = res.code
                            if new_content is not None:
                                self.edit(new_content)

                        elif event[0] == EventType.FUNC_CALL:
                            pass # TODO

                        elif event[0] == EventType.SET_MESSAGE:
                            self.set_message(event[1], event[2])

                        elif event[0] == EventType.CLEAR_MESSAGE:
                            self.clear_message()

                        else:
                            logger.error(f'unhandled event: {event}')

                    except KeyboardInterrupt:
                        self.set_message("Keyboard Interrrupt, use q for quit", LEVEL_WARNING)
                    except Exception as exc:
                        logger.error("error in event loop:", exc_info=exc)
                        self.set_message(str(exc), LEVEL_ERROR)

                self.set_message("Exiting...")
                self.redraw()
            finally:
                try:
                    self.running = False
                    self.recording = False
                    self.message_queue.put(None)
                    self.input_semaphore.release()
                    self.recorder_semaphore.release()

                    self.input_thread.join()
                    self.message_thread.join()
                    self.recorder_thread.join()

                    if self.message_log is not None:
                        self.message_log.close()
                        self.message_log = None

                except Exception as exc:
                    logger.error("error while exiting:", exc_info=exc)
                finally:
                    # show cursor
                    sys.stdout.write('\x1b[?25h')
                    sys.stdout.write('\n')
                    sys.stdout.flush()

                    signal(SIGWINCH, SIG_DFL)

    def __enter__(self) -> 'VoiceCoder':
        return self

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

        self.event_queue.put((EventType.REDRAW, ))

    def redraw(self) -> None:
        columns = self.term_size.columns
        if columns == 0:
            # clear screen
            sys.stdout.write('\x1b[2J')
            return

        # move cursor to upper left corner
        sys.stdout.write('\x1b[H')

        scroll_xoffset = self.scroll_xoffset
        scroll_yoffset = self.scroll_yoffset
        lineno = scroll_yoffset + 1
        term_lines = self.term_size.lines
        max_yoffset = scroll_yoffset + term_lines
        message = self.message
        if not message:
            if self.waiting_for_openai:
                message = ['Waiting for OpenAI...']
            if self.recording:
                if self.silence:
                    message = ['Please speak now... (silence)']
                else:
                    message = ['Please speak now...']
        message_lines: list[str] = []
        if message and columns > 0:
            # wrapping message lines
            for line in message:
                w = wcswidth(line)
                if w > columns:
                    prev = 0
                    line_len = len(line)
                    while prev < line_len:
                        next_prev = line_len
                        start_index = prev
                        end_index = prev
                        for index in range(min(max(next_prev, prev + columns - 1), line_len), line_len + 1):
                            chunk = line[start_index:index]
                            w = wcswidth(chunk)
                            if w > columns:
                                end_index = index - 1
                                if end_index <= prev:
                                    next_prev = prev + 1
                                else:
                                    next_prev = end_index
                                break
                            elif w == columns:
                                next_prev = index
                                end_index = index
                                break

                        message_lines.append(line[start_index:end_index])
                        prev = next_prev
                else:
                    message_lines.append(line)

            max_yoffset -= len(message_lines)

            if max_yoffset < scroll_yoffset:
                max_yoffset = scroll_yoffset

        max_lineno_len = len(str(len(self.lines) + 1))
        avail_columns = max(columns - max_lineno_len - 1, 0)
        lines = self.lines[scroll_yoffset:max_yoffset]
        for line in lines:
            # line number color
            sys.stdout.write('\x1b[38;5;244m')
            str_lineno = str(lineno).rjust(max_lineno_len) + ' '
            if len(str_lineno) > columns:
                sys.stdout.write(str_lineno[:columns])
            else:
                sys.stdout.write(str_lineno)
            # normal color
            sys.stdout.write('\x1b[0m')
            if avail_columns > 0:
                tokens = slice_token_line(line, scroll_xoffset, avail_columns)
                colorize(tokens, self.formatter, sys.stdout)
            # clear to end of line
            sys.stdout.write('\x1b[0K')
            if lineno < max_yoffset:
                sys.stdout.write('\n')
            lineno += 1

        if message_lines:
            skip_count = term_lines - len(lines) - len(message_lines) - 1
            if skip_count > 0:
                sys.stdout.write('\x1b[0K\n' * skip_count)

            if self.message_level == LEVEL_WARNING:
                sys.stdout.write('\x1b[33m')
            elif self.message_level == LEVEL_ERROR:
                sys.stdout.write('\x1b[31m')

            for line in message_lines:
                # clear to end of line and newline
                sys.stdout.write('\x1b[0K\n')
                sys.stdout.write(line)

            if self.message_level != LEVEL_INFO:
                sys.stdout.write('\x1b[0m')

        # clear to end of screen
        sys.stdout.write('\x1b[0J')
        if self.recording:
            line = self.term_size.lines
            column = max(self.term_size.columns - 1, 1)
            sys.stdout.write(f'\x1b[{line};{column}H🎙️')

        sys.stdout.flush()

def main() -> None:
    log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

    optparser = optparse.OptionParser()
    optparser.add_option('--log-file', default=DEFAULT_LOGFILE, metavar='PATH', help=f"default: {DEFAULT_LOGFILE}")
    optparser.add_option('--log-level', choices=log_levels, default=DEFAULT_LEVEL, metavar='LEVEL', help=f"One of: {', '.join(log_levels)}. default: {DEFAULT_LEVEL}")
    optparser.add_option('--log-voice', action='store_true', default=False)
    optparser.add_option('--log-messages', action='store_true', default=False)
    optparser.add_option('--voice-lang', metavar='LANG', default=None, help='ISO-639-1 two-letter language code for voice input')
    optparser.add_option('--sound-device', metavar='DEVICE_ID', default=None)
    optparser.add_option('--samplerate', type=int, default=DEFAULT_SAMPLERATE, help=f'default: {DEFAULT_SAMPLERATE}')
    optparser.add_option('--gpt-model', metavar='MODEL', choices=GPT_MODELS, default=DEFAULT_GPT_MODEL, help=f"One of: {', '.join(GPT_MODELS)}. default: {DEFAULT_GPT_MODEL}")
    optparser.add_option('--enable-tools', action='store_true', default=False, help=f'Experiments for controlling other editor features via tools.')
    opts, args = optparser.parse_args()

    if len(args) != 1:
        raise ValueError("expected exactly one file argument")

    if opts.samplerate <= 0:
        print(f'illegal sample rate: {opts.samplerate}', file=sys.stderr)
        sys.exit(1)

    logfile: str = opts.log_file
    logdir = dirname(logfile)
    os.makedirs(logdir, exist_ok=True)

    logging.basicConfig(
        filename=logfile,
        level=opts.log_level,
    )

    filename = args[0]
    coder = VoiceCoder(
        filename     = filename,
        log_voice    = opts.log_voice,
        log_messages = opts.log_messages,
        voice_lang   = opts.voice_lang,
        sound_device = opts.sound_device,
        samplerate   = opts.samplerate,
        gpt_model    = opts.gpt_model,
        enable_tools = opts.enable_tools,
    )
    coder.start()

if __name__ == '__main__':
    main()
