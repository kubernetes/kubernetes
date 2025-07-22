package term

import (
	"fmt"
	"io"
	"os"
	"os/signal"

	windowsconsole "github.com/moby/term/windows"
	"golang.org/x/sys/windows"
)

// terminalState holds the platform-specific state / console mode for the terminal.
type terminalState struct {
	mode uint32
}

// vtInputSupported is true if winterm.ENABLE_VIRTUAL_TERMINAL_INPUT is supported by the console
var vtInputSupported bool

func stdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
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

	return stdIn, stdOut, stdErr
}

func getFdInfo(in interface{}) (uintptr, bool) {
	return windowsconsole.GetHandleInfo(in)
}

func getWinsize(fd uintptr) (*Winsize, error) {
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

func setWinsize(fd uintptr, ws *Winsize) error {
	return fmt.Errorf("not implemented on Windows")
}

func isTerminal(fd uintptr) bool {
	var mode uint32
	err := windows.GetConsoleMode(windows.Handle(fd), &mode)
	return err == nil
}

func restoreTerminal(fd uintptr, state *State) error {
	return windows.SetConsoleMode(windows.Handle(fd), state.mode)
}

func saveState(fd uintptr) (*State, error) {
	var mode uint32

	if err := windows.GetConsoleMode(windows.Handle(fd), &mode); err != nil {
		return nil, err
	}

	return &State{mode: mode}, nil
}

func disableEcho(fd uintptr, state *State) error {
	// See https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx
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

func setRawTerminal(fd uintptr) (*State, error) {
	oldState, err := MakeRaw(fd)
	if err != nil {
		return nil, err
	}

	// Register an interrupt handler to catch and restore prior state
	restoreAtInterrupt(fd, oldState)
	return oldState, err
}

func setRawTerminalOutput(fd uintptr) (*State, error) {
	oldState, err := saveState(fd)
	if err != nil {
		return nil, err
	}

	// Ignore failures, since winterm.DISABLE_NEWLINE_AUTO_RETURN might not be supported on this
	// version of Windows.
	_ = windows.SetConsoleMode(windows.Handle(fd), oldState.mode|windows.DISABLE_NEWLINE_AUTO_RETURN)
	return oldState, err
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
