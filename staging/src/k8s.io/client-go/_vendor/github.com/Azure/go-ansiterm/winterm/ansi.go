// +build windows

package winterm

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"

	. "github.com/Azure/go-ansiterm"
)

// Windows keyboard constants
// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd375731(v=vs.85).aspx.
const (
	VK_PRIOR    = 0x21 // PAGE UP key
	VK_NEXT     = 0x22 // PAGE DOWN key
	VK_END      = 0x23 // END key
	VK_HOME     = 0x24 // HOME key
	VK_LEFT     = 0x25 // LEFT ARROW key
	VK_UP       = 0x26 // UP ARROW key
	VK_RIGHT    = 0x27 // RIGHT ARROW key
	VK_DOWN     = 0x28 // DOWN ARROW key
	VK_SELECT   = 0x29 // SELECT key
	VK_PRINT    = 0x2A // PRINT key
	VK_EXECUTE  = 0x2B // EXECUTE key
	VK_SNAPSHOT = 0x2C // PRINT SCREEN key
	VK_INSERT   = 0x2D // INS key
	VK_DELETE   = 0x2E // DEL key
	VK_HELP     = 0x2F // HELP key
	VK_F1       = 0x70 // F1 key
	VK_F2       = 0x71 // F2 key
	VK_F3       = 0x72 // F3 key
	VK_F4       = 0x73 // F4 key
	VK_F5       = 0x74 // F5 key
	VK_F6       = 0x75 // F6 key
	VK_F7       = 0x76 // F7 key
	VK_F8       = 0x77 // F8 key
	VK_F9       = 0x78 // F9 key
	VK_F10      = 0x79 // F10 key
	VK_F11      = 0x7A // F11 key
	VK_F12      = 0x7B // F12 key

	RIGHT_ALT_PRESSED  = 0x0001
	LEFT_ALT_PRESSED   = 0x0002
	RIGHT_CTRL_PRESSED = 0x0004
	LEFT_CTRL_PRESSED  = 0x0008
	SHIFT_PRESSED      = 0x0010
	NUMLOCK_ON         = 0x0020
	SCROLLLOCK_ON      = 0x0040
	CAPSLOCK_ON        = 0x0080
	ENHANCED_KEY       = 0x0100
)

type ansiCommand struct {
	CommandBytes []byte
	Command      string
	Parameters   []string
	IsSpecial    bool
}

func newAnsiCommand(command []byte) *ansiCommand {

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

	ac := &ansiCommand{
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
		ac.Parameters = strings.Split(string(command[start:lastCharIndex]), ANSI_PARAMETER_SEP)
	}

	return ac
}

func (ac *ansiCommand) paramAsSHORT(index int, defaultValue SHORT) SHORT {
	if index < 0 || index >= len(ac.Parameters) {
		return defaultValue
	}

	param, err := strconv.ParseInt(ac.Parameters[index], 10, 16)
	if err != nil {
		return defaultValue
	}

	return SHORT(param)
}

func (ac *ansiCommand) String() string {
	return fmt.Sprintf("0x%v \"%v\" (\"%v\")",
		bytesToHex(ac.CommandBytes),
		ac.Command,
		strings.Join(ac.Parameters, "\",\""))
}

// isAnsiCommandChar returns true if the passed byte falls within the range of ANSI commands.
// See http://manpages.ubuntu.com/manpages/intrepid/man4/console_codes.4.html.
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

func isXtermOscSequence(command []byte, current byte) bool {
	return (len(command) >= 2 && command[0] == ANSI_ESCAPE_PRIMARY && command[1] == ANSI_CMD_OSC && current != ANSI_BEL)
}

func isCharacterSelectionCmdChar(b byte) bool {
	return (b == ANSI_CMD_G0 || b == ANSI_CMD_G1 || b == ANSI_CMD_G2 || b == ANSI_CMD_G3)
}

// bytesToHex converts a slice of bytes to a human-readable string.
func bytesToHex(b []byte) string {
	hex := make([]string, len(b))
	for i, ch := range b {
		hex[i] = fmt.Sprintf("%X", ch)
	}
	return strings.Join(hex, "")
}

// ensureInRange adjusts the passed value, if necessary, to ensure it is within
// the passed min / max range.
func ensureInRange(n SHORT, min SHORT, max SHORT) SHORT {
	if n < min {
		return min
	} else if n > max {
		return max
	} else {
		return n
	}
}

func GetStdFile(nFile int) (*os.File, uintptr) {
	var file *os.File
	switch nFile {
	case syscall.STD_INPUT_HANDLE:
		file = os.Stdin
	case syscall.STD_OUTPUT_HANDLE:
		file = os.Stdout
	case syscall.STD_ERROR_HANDLE:
		file = os.Stderr
	default:
		panic(fmt.Errorf("Invalid standard handle identifier: %v", nFile))
	}

	fd, err := syscall.GetStdHandle(nFile)
	if err != nil {
		panic(fmt.Errorf("Invalid standard handle indentifier: %v -- %v", nFile, err))
	}

	return file, uintptr(fd)
}
