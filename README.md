VoiceCoder
==========

**Work in progress!**

Experiments with OpenAI and Python to edit code via voice input.

**Note:** This sends your code and voice recording to OpenAI!

This is meant for people that can't use their hands. You can dictate code
without being precise about all the special characters and such, basically
dictate as you would to a human.

It kinda breaks down for large files, though. Then ChatGPT might only
return the changed part, but not in a way that can be automatically applied.

You need to set the environment variable `OPENAI_API_KEY` to a valid value.
It can be provided via a `.env` file in the current directory.

[Demo Video](https://www.youtube.com/watch?v=xj5YZlEFLpE)

```plain
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
```
