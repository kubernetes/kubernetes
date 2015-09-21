package winconsole

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// http://manpages.ubuntu.com/manpages/intrepid/man4/console_codes.4.html
const (
	ANSI_ESCAPE_PRIMARY   = 0x1B
	ANSI_ESCAPE_SECONDARY = 0x5B
	ANSI_COMMAND_FIRST    = 0x40
	ANSI_COMMAND_LAST     = 0x7E
	ANSI_PARAMETER_SEP    = ";"
	ANSI_CMD_G0           = '('
	ANSI_CMD_G1           = ')'
	ANSI_CMD_G2           = '*'
	ANSI_CMD_G3           = '+'
	ANSI_CMD_DECPNM       = '>'
	ANSI_CMD_DECPAM       = '='
	ANSI_CMD_OSC          = ']'
	ANSI_CMD_STR_TERM     = '\\'
	ANSI_BEL              = 0x07
	KEY_EVENT             = 1
)

// Interface that implements terminal handling
type terminalEmulator interface {
	HandleOutputCommand(fd uintptr, command []byte) (n int, err error)
	HandleInputSequence(fd uintptr, command []byte) (n int, err error)
	WriteChars(fd uintptr, w io.Writer, p []byte) (n int, err error)
	ReadChars(fd uintptr, w io.Reader, p []byte) (n int, err error)
}

type terminalWriter struct {
	wrappedWriter io.Writer
	emulator      terminalEmulator
	command       []byte
	inSequence    bool
	fd            uintptr
}

type terminalReader struct {
	wrappedReader io.ReadCloser
	emulator      terminalEmulator
	command       []byte
	inSequence    bool
	fd            uintptr
}

// http://manpages.ubuntu.com/manpages/intrepid/man4/console_codes.4.html
func isAnsiCommandChar(b byte) bool {
	switch {
	case ANSI_COMMAND_FIRST <= b && b <= ANSI_COMMAND_LAST && b != ANSI_ESCAPE_SECONDARY:
		return true
	case b == ANSI_CMD_G1 || b == ANSI_CMD_OSC || b == ANSI_CMD_DECPAM || b == ANSI_CMD_DECPNM:
		// non-CSI escape sequence terminator
		return true
	case b == ANSI_CMD_STR_TERM || b == ANSI_BEL:
		// String escape sequence terminator
		return true
	}
	return false
}

func isCharacterSelectionCmdChar(b byte) bool {
	return (b == ANSI_CMD_G0 || b == ANSI_CMD_G1 || b == ANSI_CMD_G2 || b == ANSI_CMD_G3)
}

func isXtermOscSequence(command []byte, current byte) bool {
	return (len(command) >= 2 && command[0] == ANSI_ESCAPE_PRIMARY && command[1] == ANSI_CMD_OSC && current != ANSI_BEL)
}

// Write writes len(p) bytes from p to the underlying data stream.
// http://golang.org/pkg/io/#Writer
func (tw *terminalWriter) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if tw.emulator == nil {
		return tw.wrappedWriter.Write(p)
	}
	// Emulate terminal by extracting commands and executing them
	totalWritten := 0
	start := 0 // indicates start of the next chunk
	end := len(p)
	for current := 0; current < end; current++ {
		if tw.inSequence {
			// inside escape sequence
			tw.command = append(tw.command, p[current])
			if isAnsiCommandChar(p[current]) {
				if !isXtermOscSequence(tw.command, p[current]) {
					// found the last command character.
					// Now we have a complete command.
					nchar, err := tw.emulator.HandleOutputCommand(tw.fd, tw.command)
					totalWritten += nchar
					if err != nil {
						return totalWritten, err
					}

					// clear the command
					// don't include current character again
					tw.command = tw.command[:0]
					start = current + 1
					tw.inSequence = false
				}
			}
		} else {
			if p[current] == ANSI_ESCAPE_PRIMARY {
				// entering escape sequnce
				tw.inSequence = true
				// indicates end of "normal sequence", write whatever you have so far
				if len(p[start:current]) > 0 {
					nw, err := tw.emulator.WriteChars(tw.fd, tw.wrappedWriter, p[start:current])
					totalWritten += nw
					if err != nil {
						return totalWritten, err
					}
				}
				// include the current character as part of the next sequence
				tw.command = append(tw.command, p[current])
			}
		}
	}
	// note that so far, start of the escape sequence triggers writing out of bytes to console.
	// For the part _after_ the end of last escape sequence, it is not written out yet. So write it out
	if !tw.inSequence {
		// assumption is that we can't be inside sequence and therefore command should be empty
		if len(p[start:]) > 0 {
			nw, err := tw.emulator.WriteChars(tw.fd, tw.wrappedWriter, p[start:])
			totalWritten += nw
			if err != nil {
				return totalWritten, err
			}
		}
	}
	return totalWritten, nil

}

// Read reads up to len(p) bytes into p.
// http://golang.org/pkg/io/#Reader
func (tr *terminalReader) Read(p []byte) (n int, err error) {
	//Implementations of Read are discouraged from returning a zero byte count
	// with a nil error, except when len(p) == 0.
	if len(p) == 0 {
		return 0, nil
	}
	if nil == tr.emulator {
		return tr.readFromWrappedReader(p)
	}
	return tr.emulator.ReadChars(tr.fd, tr.wrappedReader, p)
}

// Close the underlying stream
func (tr *terminalReader) Close() (err error) {
	return tr.wrappedReader.Close()
}

func (tr *terminalReader) readFromWrappedReader(p []byte) (n int, err error) {
	return tr.wrappedReader.Read(p)
}

type ansiCommand struct {
	CommandBytes []byte
	Command      string
	Parameters   []string
	IsSpecial    bool
}

func parseAnsiCommand(command []byte) *ansiCommand {
	if isCharacterSelectionCmdChar(command[1]) {
		// Is Character Set Selection commands
		return &ansiCommand{
			CommandBytes: command,
			Command:      string(command),
			IsSpecial:    true,
		}
	}
	// last char is command character
	lastCharIndex := len(command) - 1

	retValue := &ansiCommand{
		CommandBytes: command,
		Command:      string(command[lastCharIndex]),
		IsSpecial:    false,
	}
	// more than a single escape
	if lastCharIndex != 0 {
		start := 1
		// skip if double char escape sequence
		if command[0] == ANSI_ESCAPE_PRIMARY && command[1] == ANSI_ESCAPE_SECONDARY {
			start++
		}
		// convert this to GetNextParam method
		retValue.Parameters = strings.Split(string(command[start:lastCharIndex]), ANSI_PARAMETER_SEP)
	}
	return retValue
}

func (c *ansiCommand) getParam(index int) string {
	if len(c.Parameters) > index {
		return c.Parameters[index]
	}
	return ""
}

func (ac *ansiCommand) String() string {
	return fmt.Sprintf("0x%v \"%v\" (\"%v\")",
		bytesToHex(ac.CommandBytes),
		ac.Command,
		strings.Join(ac.Parameters, "\",\""))
}

func bytesToHex(b []byte) string {
	hex := make([]string, len(b))
	for i, ch := range b {
		hex[i] = fmt.Sprintf("%X", ch)
	}
	return strings.Join(hex, "")
}

func parseInt16OrDefault(s string, defaultValue int16) (n int16, err error) {
	if s == "" {
		return defaultValue, nil
	}
	parsedValue, err := strconv.ParseInt(s, 10, 16)
	if err != nil {
		return defaultValue, err
	}
	return int16(parsedValue), nil
}
