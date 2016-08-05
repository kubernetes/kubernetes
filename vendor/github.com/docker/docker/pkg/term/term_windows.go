// +build windows

package term

import (
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/Azure/go-ansiterm/winterm"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/term/windows"
)

// State holds the console mode for the terminal.
type State struct {
	inMode, outMode     uint32
	inHandle, outHandle syscall.Handle
}

// Winsize is used for window size.
type Winsize struct {
	Height uint16
	Width  uint16
	x      uint16
	y      uint16
}

const (
	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms683167(v=vs.85).aspx
	enableVirtualTerminalInput      = 0x0200
	enableVirtualTerminalProcessing = 0x0004
)

// usingNativeConsole is true if we are using the Windows native console
var usingNativeConsole bool

// StdStreams returns the standard streams (stdin, stdout, stedrr).
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	switch {
	case os.Getenv("ConEmuANSI") == "ON":
		// The ConEmu terminal emulates ANSI on output streams well.
		return windows.ConEmuStreams()
	case os.Getenv("MSYSTEM") != "":
		// MSYS (mingw) does not emulate ANSI well.
		return windows.ConsoleStreams()
	default:
		if useNativeConsole() {
			usingNativeConsole = true
			return os.Stdin, os.Stdout, os.Stderr
		}
		return windows.ConsoleStreams()
	}
}

// useNativeConsole determines if the docker client should use the built-in
// console which supports ANSI emulation, or fall-back to the golang emulator
// (github.com/azure/go-ansiterm).
func useNativeConsole() bool {
	osv, err := system.GetOSVersion()
	if err != nil {
		return false
	}

	// Native console is not available before major version 10
	if osv.MajorVersion < 10 {
		return false
	}

	// Must have a late pre-release TP4 build of Windows Server 2016/Windows 10 TH2 or later
	if osv.Build < 10578 {
		return false
	}

	// Get the console modes. If this fails, we can't use the native console
	state, err := getNativeConsole()
	if err != nil {
		return false
	}

	// Probe the console to see if it can be enabled.
	if nil != probeNativeConsole(state) {
		return false
	}

	// Environment variable override
	if e := os.Getenv("USE_NATIVE_CONSOLE"); e != "" {
		if e == "1" {
			return true
		}
		return false
	}

	// TODO Windows. The native emulator still has issues which
	// mean it shouldn't be enabled for everyone. Change this next line to true
	// to change the default to "enable if available". In the meantime, users
	// can still try it out by using USE_NATIVE_CONSOLE env variable.
	return false
}

// getNativeConsole returns the console modes ('state') for the native Windows console
func getNativeConsole() (State, error) {
	var (
		err   error
		state State
	)

	// Get the handle to stdout
	if state.outHandle, err = syscall.GetStdHandle(syscall.STD_OUTPUT_HANDLE); err != nil {
		return state, err
	}

	// Get the console mode from the consoles stdout handle
	if err = syscall.GetConsoleMode(state.outHandle, &state.outMode); err != nil {
		return state, err
	}

	// Get the handle to stdin
	if state.inHandle, err = syscall.GetStdHandle(syscall.STD_INPUT_HANDLE); err != nil {
		return state, err
	}

	// Get the console mode from the consoles stdin handle
	if err = syscall.GetConsoleMode(state.inHandle, &state.inMode); err != nil {
		return state, err
	}

	return state, nil
}

// probeNativeConsole probes the console to determine if native can be supported,
func probeNativeConsole(state State) error {
	if err := winterm.SetConsoleMode(uintptr(state.outHandle), state.outMode|enableVirtualTerminalProcessing); err != nil {
		return err
	}
	defer winterm.SetConsoleMode(uintptr(state.outHandle), state.outMode)

	if err := winterm.SetConsoleMode(uintptr(state.inHandle), state.inMode|enableVirtualTerminalInput); err != nil {
		return err
	}
	defer winterm.SetConsoleMode(uintptr(state.inHandle), state.inMode)

	return nil
}

// enableNativeConsole turns on native console mode
func enableNativeConsole(state State) error {
	if err := winterm.SetConsoleMode(uintptr(state.outHandle), state.outMode|enableVirtualTerminalProcessing); err != nil {
		return err
	}

	if err := winterm.SetConsoleMode(uintptr(state.inHandle), state.inMode|enableVirtualTerminalInput); err != nil {
		winterm.SetConsoleMode(uintptr(state.outHandle), state.outMode) // restore out if we can
		return err
	}

	return nil
}

// disableNativeConsole turns off native console mode
func disableNativeConsole(state *State) error {
	// Try and restore both in an out before error checking.
	errout := winterm.SetConsoleMode(uintptr(state.outHandle), state.outMode)
	errin := winterm.SetConsoleMode(uintptr(state.inHandle), state.inMode)
	if errout != nil {
		return errout
	}
	if errin != nil {
		return errin
	}
	return nil
}

// GetFdInfo returns the file descriptor for an os.File and indicates whether the file represents a terminal.
func GetFdInfo(in interface{}) (uintptr, bool) {
	return windows.GetHandleInfo(in)
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
		x:      0,
		y:      0}

	return winsize, nil
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	return windows.IsConsole(fd)
}

// RestoreTerminal restores the terminal connected to the given file descriptor
// to a previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	if usingNativeConsole {
		return disableNativeConsole(state)
	}
	return winterm.SetConsoleMode(fd, state.outMode)
}

// SaveState saves the state of the terminal connected to the given file descriptor.
func SaveState(fd uintptr) (*State, error) {
	if usingNativeConsole {
		state, err := getNativeConsole()
		if err != nil {
			return nil, err
		}
		return &state, nil
	}

	mode, e := winterm.GetConsoleMode(fd)
	if e != nil {
		return nil, e
	}

	return &State{outMode: mode}, nil
}

// DisableEcho disables echo for the terminal connected to the given file descriptor.
// -- See https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx
func DisableEcho(fd uintptr, state *State) error {
	mode := state.inMode
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

// SetRawTerminal puts the terminal connected to the given file descriptor into raw
// mode and returns the previous state.
func SetRawTerminal(fd uintptr) (*State, error) {
	state, err := MakeRaw(fd)
	if err != nil {
		return nil, err
	}

	// Register an interrupt handler to catch and restore prior state
	restoreAtInterrupt(fd, state)
	return state, err
}

// MakeRaw puts the terminal (Windows Console) connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be restored.
func MakeRaw(fd uintptr) (*State, error) {
	state, err := SaveState(fd)
	if err != nil {
		return nil, err
	}

	mode := state.inMode
	if usingNativeConsole {
		if err := enableNativeConsole(*state); err != nil {
			return nil, err
		}
		mode |= enableVirtualTerminalInput
	}

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
