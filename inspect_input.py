#!/usr/bin/env python3

# helper script to inspect keystrokes on the terminal

from typing import Union, Optional, BinaryIO

import io
import sys
import termios

InputData = tuple[Union[bytes, bytearray], Optional[tuple[int, int, list[int]]]]

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

def main() -> None:
    stdin = io.open(sys.stdin.fileno(), 'rb', closefd=False, buffering=False)
    with UnbufferedInput():
        try:
            while True:
                raw, decoded = read_ansi(stdin)
                if not raw:
                    break
                print((bytes(raw), decoded))
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
