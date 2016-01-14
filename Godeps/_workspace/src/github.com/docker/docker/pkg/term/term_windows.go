// +build windows

package term

import (
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/Azure/go-ansiterm/winterm"
	"github.com/Sirupsen/logrus"
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
	x      uint16
	y      uint16
}

// StdStreams returns the standard streams (stdin, stdout, stedrr).
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	switch {
	case os.Getenv("ConEmuANSI") == "ON":
		// The ConEmu shell emulates ANSI well by default.
		return os.Stdin, os.Stdout, os.Stderr
	case os.Getenv("MSYSTEM") != "":
		// MSYS (mingw) does not emulate ANSI well.
		return windows.ConsoleStreams()
	default:
		return windows.ConsoleStreams()
	}
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

	// Note: GetWinsize is called frequently -- uncomment only for excessive details
	// logrus.Debugf("[windows] GetWinsize: Console(%v)", info.String())
	// logrus.Debugf("[windows] GetWinsize: Width(%v), Height(%v), x(%v), y(%v)", winsize.Width, winsize.Height, winsize.x, winsize.y)
	return winsize, nil
}

// SetWinsize tries to set the specified window size for the specified file descriptor.
func SetWinsize(fd uintptr, ws *Winsize) error {

	// Ensure the requested dimensions are no larger than the maximum window size
	info, err := winterm.GetConsoleScreenBufferInfo(fd)
	if err != nil {
		return err
	}

	if ws.Width == 0 || ws.Height == 0 || ws.Width > uint16(info.MaximumWindowSize.X) || ws.Height > uint16(info.MaximumWindowSize.Y) {
		return fmt.Errorf("Illegal window size: (%v,%v) -- Maximum allow: (%v,%v)",
			ws.Width, ws.Height, info.MaximumWindowSize.X, info.MaximumWindowSize.Y)
	}

	// Narrow the sizes to that used by Windows
	width := winterm.SHORT(ws.Width)
	height := winterm.SHORT(ws.Height)

	// Set the dimensions while ensuring they remain within the bounds of the backing console buffer
	// -- Shrinking will always succeed. Growing may push the edges past the buffer boundary. When that occurs,
	//    shift the upper left just enough to keep the new window within the buffer.
	rect := info.Window
	if width < rect.Right-rect.Left+1 {
		rect.Right = rect.Left + width - 1
	} else if width > rect.Right-rect.Left+1 {
		rect.Right = rect.Left + width - 1
		if rect.Right >= info.Size.X {
			rect.Left = info.Size.X - width
			rect.Right = info.Size.X - 1
		}
	}

	if height < rect.Bottom-rect.Top+1 {
		rect.Bottom = rect.Top + height - 1
	} else if height > rect.Bottom-rect.Top+1 {
		rect.Bottom = rect.Top + height - 1
		if rect.Bottom >= info.Size.Y {
			rect.Top = info.Size.Y - height
			rect.Bottom = info.Size.Y - 1
		}
	}
	logrus.Debugf("[windows] SetWinsize: Requested((%v,%v)) Actual(%v)", ws.Width, ws.Height, rect)

	return winterm.SetConsoleWindowInfo(fd, true, rect)
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	return windows.IsConsole(fd)
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
	return &State{mode}, nil
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

	// See
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms686033(v=vs.85).aspx
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx
	mode := state.mode

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
