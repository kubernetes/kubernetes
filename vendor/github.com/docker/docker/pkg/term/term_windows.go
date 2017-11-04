// +build windows

package term

import (
	"io"
	"os"
	"os/signal"
	"syscall" // used for STD_INPUT_HANDLE, STD_OUTPUT_HANDLE and STD_ERROR_HANDLE

	"github.com/Azure/go-ansiterm/winterm"
	"github.com/docker/docker/pkg/term/windows"
)

// State holds the console mode for the terminal.
type State struct {
	mode uint32
}

// Winsize is used for window size.
type Winsize struct {
	Height uint16
	Width  uint16
}

const (
	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms683167(v=vs.85).aspx
	enableVirtualTerminalInput      = 0x0200
	enableVirtualTerminalProcessing = 0x0004
	disableNewlineAutoReturn        = 0x0008
)

// vtInputSupported is true if enableVirtualTerminalInput is supported by the console
var vtInputSupported bool

// StdStreams returns the standard streams (stdin, stdout, stderr).
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	// Turn on VT handling on all std handles, if possible. This might
	// fail, in which case we will fall back to terminal emulation.
	var emulateStdin, emulateStdout, emulateStderr bool
	fd := os.Stdin.Fd()
	if mode, err := winterm.GetConsoleMode(fd); err == nil {
		// Validate that enableVirtualTerminalInput is supported, but do not set it.
		if err = winterm.SetConsoleMode(fd, mode|enableVirtualTerminalInput); err != nil {
			emulateStdin = true
		} else {
			vtInputSupported = true
		}
		// Unconditionally set the console mode back even on failure because SetConsoleMode
		// remembers invalid bits on input handles.
		winterm.SetConsoleMode(fd, mode)
	}

	fd = os.Stdout.Fd()
	if mode, err := winterm.GetConsoleMode(fd); err == nil {
		// Validate disableNewlineAutoReturn is supported, but do not set it.
		if err = winterm.SetConsoleMode(fd, mode|enableVirtualTerminalProcessing|disableNewlineAutoReturn); err != nil {
			emulateStdout = true
		} else {
			winterm.SetConsoleMode(fd, mode|enableVirtualTerminalProcessing)
		}
	}

	fd = os.Stderr.Fd()
	if mode, err := winterm.GetConsoleMode(fd); err == nil {
		// Validate disableNewlineAutoReturn is supported, but do not set it.
		if err = winterm.SetConsoleMode(fd, mode|enableVirtualTerminalProcessing|disableNewlineAutoReturn); err != nil {
			emulateStderr = true
		} else {
			winterm.SetConsoleMode(fd, mode|enableVirtualTerminalProcessing)
		}
	}

	if os.Getenv("ConEmuANSI") == "ON" || os.Getenv("ConsoleZVersion") != "" {
		// The ConEmu and ConsoleZ terminals emulate ANSI on output streams well.
		emulateStdin = true
		emulateStdout = false
		emulateStderr = false
	}

	// Temporarily use STD_INPUT_HANDLE, STD_OUTPUT_HANDLE and
	// STD_ERROR_HANDLE from syscall rather than x/sys/windows as long as
	// go-ansiterm hasn't switch to x/sys/windows.
	// TODO: switch back to x/sys/windows once go-ansiterm has switched
	if emulateStdin {
		stdIn = windowsconsole.NewAnsiReader(syscall.STD_INPUT_HANDLE)
	} else {
		stdIn = os.Stdin
	}

	if emulateStdout {
		stdOut = windowsconsole.NewAnsiWriter(syscall.STD_OUTPUT_HANDLE)
	} else {
		stdOut = os.Stdout
	}

	if emulateStderr {
		stdErr = windowsconsole.NewAnsiWriter(syscall.STD_ERROR_HANDLE)
	} else {
		stdErr = os.Stderr
	}

	return
}

// GetFdInfo returns the file descriptor for an os.File and indicates whether the file represents a terminal.
func GetFdInfo(in interface{}) (uintptr, bool) {
	return windowsconsole.GetHandleInfo(in)
}

// GetWinsize returns the window size based on the specified file descriptor.
func GetWinsize(fd uintptr) (*Winsize, error) {
	info, err := winterm.GetConsoleScreenBufferInfo(fd)
	if err != nil {
		return nil, err
	}

	winsize := &Winsize{
		Width:  uint16(info.Window.Right - info.Window.Left + 1),
		Height: uint16(info.Window.Bottom - info.Window.Top + 1),
	}

	return winsize, nil
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	return windowsconsole.IsConsole(fd)
}

// RestoreTerminal restores the terminal connected to the given file descriptor
// to a previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	return winterm.SetConsoleMode(fd, state.mode)
}

// SaveState saves the state of the terminal connected to the given file descriptor.
func SaveState(fd uintptr) (*State, error) {
	mode, e := winterm.GetConsoleMode(fd)
	if e != nil {
		return nil, e
	}

	return &State{mode: mode}, nil
}

// DisableEcho disables echo for the terminal connected to the given file descriptor.
// -- See https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx
func DisableEcho(fd uintptr, state *State) error {
	mode := state.mode
	mode &^= winterm.ENABLE_ECHO_INPUT
	mode |= winterm.ENABLE_PROCESSED_INPUT | winterm.ENABLE_LINE_INPUT
	err := winterm.SetConsoleMode(fd, mode)
	if err != nil {
		return err
	}

	// Register an interrupt handler to catch and restore prior state
	restoreAtInterrupt(fd, state)
	return nil
}

// SetRawTerminal puts the terminal connected to the given file descriptor into
// raw mode and returns the previous state. On UNIX, this puts both the input
// and output into raw mode. On Windows, it only puts the input into raw mode.
func SetRawTerminal(fd uintptr) (*State, error) {
	state, err := MakeRaw(fd)
	if err != nil {
		return nil, err
	}

	// Register an interrupt handler to catch and restore prior state
	restoreAtInterrupt(fd, state)
	return state, err
}

// SetRawTerminalOutput puts the output of terminal connected to the given file
// descriptor into raw mode. On UNIX, this does nothing and returns nil for the
// state. On Windows, it disables LF -> CRLF translation.
func SetRawTerminalOutput(fd uintptr) (*State, error) {
	state, err := SaveState(fd)
	if err != nil {
		return nil, err
	}

	// Ignore failures, since disableNewlineAutoReturn might not be supported on this
	// version of Windows.
	winterm.SetConsoleMode(fd, state.mode|disableNewlineAutoReturn)
	return state, err
}

// MakeRaw puts the terminal (Windows Console) connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be restored.
func MakeRaw(fd uintptr) (*State, error) {
	state, err := SaveState(fd)
	if err != nil {
		return nil, err
	}

	mode := state.mode

	// See
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms686033(v=vs.85).aspx
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx

	// Disable these modes
	mode &^= winterm.ENABLE_ECHO_INPUT
	mode &^= winterm.ENABLE_LINE_INPUT
	mode &^= winterm.ENABLE_MOUSE_INPUT
	mode &^= winterm.ENABLE_WINDOW_INPUT
	mode &^= winterm.ENABLE_PROCESSED_INPUT

	// Enable these modes
	mode |= winterm.ENABLE_EXTENDED_FLAGS
	mode |= winterm.ENABLE_INSERT_MODE
	mode |= winterm.ENABLE_QUICK_EDIT_MODE
	if vtInputSupported {
		mode |= enableVirtualTerminalInput
	}

	err = winterm.SetConsoleMode(fd, mode)
	if err != nil {
		return nil, err
	}
	return state, nil
}

func restoreAtInterrupt(fd uintptr, state *State) {
	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan, os.Interrupt)

	go func() {
		_ = <-sigchan
		RestoreTerminal(fd, state)
		os.Exit(0)
	}()
}
