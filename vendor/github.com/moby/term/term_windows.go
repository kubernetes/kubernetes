package term

import (
	"io"
	"os"
	"os/signal"

	windowsconsole "github.com/moby/term/windows"
	"golang.org/x/sys/windows"
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

// vtInputSupported is true if winterm.ENABLE_VIRTUAL_TERMINAL_INPUT is supported by the console
var vtInputSupported bool

// StdStreams returns the standard streams (stdin, stdout, stderr).
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	// Turn on VT handling on all std handles, if possible. This might
	// fail, in which case we will fall back to terminal emulation.
	var (
		emulateStdin, emulateStdout, emulateStderr bool

		mode uint32
	)

	fd := windows.Handle(os.Stdin.Fd())
	if err := windows.GetConsoleMode(fd, &mode); err == nil {
		// Validate that winterm.ENABLE_VIRTUAL_TERMINAL_INPUT is supported, but do not set it.
		if err = windows.SetConsoleMode(fd, mode|windows.ENABLE_VIRTUAL_TERMINAL_INPUT); err != nil {
			emulateStdin = true
		} else {
			vtInputSupported = true
		}
		// Unconditionally set the console mode back even on failure because SetConsoleMode
		// remembers invalid bits on input handles.
		_ = windows.SetConsoleMode(fd, mode)
	}

	fd = windows.Handle(os.Stdout.Fd())
	if err := windows.GetConsoleMode(fd, &mode); err == nil {
		// Validate winterm.DISABLE_NEWLINE_AUTO_RETURN is supported, but do not set it.
		if err = windows.SetConsoleMode(fd, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING|windows.DISABLE_NEWLINE_AUTO_RETURN); err != nil {
			emulateStdout = true
		} else {
			_ = windows.SetConsoleMode(fd, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING)
		}
	}

	fd = windows.Handle(os.Stderr.Fd())
	if err := windows.GetConsoleMode(fd, &mode); err == nil {
		// Validate winterm.DISABLE_NEWLINE_AUTO_RETURN is supported, but do not set it.
		if err = windows.SetConsoleMode(fd, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING|windows.DISABLE_NEWLINE_AUTO_RETURN); err != nil {
			emulateStderr = true
		} else {
			_ = windows.SetConsoleMode(fd, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING)
		}
	}

	if emulateStdin {
		h := uint32(windows.STD_INPUT_HANDLE)
		stdIn = windowsconsole.NewAnsiReader(int(h))
	} else {
		stdIn = os.Stdin
	}

	if emulateStdout {
		h := uint32(windows.STD_OUTPUT_HANDLE)
		stdOut = windowsconsole.NewAnsiWriter(int(h))
	} else {
		stdOut = os.Stdout
	}

	if emulateStderr {
		h := uint32(windows.STD_ERROR_HANDLE)
		stdErr = windowsconsole.NewAnsiWriter(int(h))
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
	var info windows.ConsoleScreenBufferInfo
	if err := windows.GetConsoleScreenBufferInfo(windows.Handle(fd), &info); err != nil {
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
	var mode uint32
	err := windows.GetConsoleMode(windows.Handle(fd), &mode)
	return err == nil
}

// RestoreTerminal restores the terminal connected to the given file descriptor
// to a previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	return windows.SetConsoleMode(windows.Handle(fd), state.mode)
}

// SaveState saves the state of the terminal connected to the given file descriptor.
func SaveState(fd uintptr) (*State, error) {
	var mode uint32

	if err := windows.GetConsoleMode(windows.Handle(fd), &mode); err != nil {
		return nil, err
	}

	return &State{mode: mode}, nil
}

// DisableEcho disables echo for the terminal connected to the given file descriptor.
// -- See https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx
func DisableEcho(fd uintptr, state *State) error {
	mode := state.mode
	mode &^= windows.ENABLE_ECHO_INPUT
	mode |= windows.ENABLE_PROCESSED_INPUT | windows.ENABLE_LINE_INPUT
	err := windows.SetConsoleMode(windows.Handle(fd), mode)
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

	// Ignore failures, since winterm.DISABLE_NEWLINE_AUTO_RETURN might not be supported on this
	// version of Windows.
	_ = windows.SetConsoleMode(windows.Handle(fd), state.mode|windows.DISABLE_NEWLINE_AUTO_RETURN)
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
	mode &^= windows.ENABLE_ECHO_INPUT
	mode &^= windows.ENABLE_LINE_INPUT
	mode &^= windows.ENABLE_MOUSE_INPUT
	mode &^= windows.ENABLE_WINDOW_INPUT
	mode &^= windows.ENABLE_PROCESSED_INPUT

	// Enable these modes
	mode |= windows.ENABLE_EXTENDED_FLAGS
	mode |= windows.ENABLE_INSERT_MODE
	mode |= windows.ENABLE_QUICK_EDIT_MODE
	if vtInputSupported {
		mode |= windows.ENABLE_VIRTUAL_TERMINAL_INPUT
	}

	err = windows.SetConsoleMode(windows.Handle(fd), mode)
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
		_ = RestoreTerminal(fd, state)
		os.Exit(0)
	}()
}
