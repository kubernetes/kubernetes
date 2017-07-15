# go-ansiterm

This is a cross platform Ansi Terminal Emulation library.  It reads a stream of Ansi characters and produces the appropriate function calls.  The results of the function calls are platform dependent.

For example the parser might receive "ESC, [, A" as a stream of three characters.  This is the code for Cursor Up (http://www.vt100.net/docs/vt510-rm/CUU).  The parser then calls the cursor up function (CUU()) on an event handler.  The event handler determines what platform specific work must be done to cause the cursor to move up one position.

The parser (parser.go) is a partial implementation of this state machine (http://vt100.net/emu/vt500_parser.png).  There are also two event handler implementations, one for tests (test_event_handler.go) to validate that the expected events are being produced and called, the other is a Windows implementation (winterm/win_event_handler.go).

See parser_test.go for examples exercising the state machine and generating appropriate function calls.
