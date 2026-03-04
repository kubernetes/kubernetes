package term

import "io"

// State holds the platform-specific state / console mode for the terminal.
type State terminalState

// Winsize represents the size of the terminal window.
type Winsize struct {
	Height uint16
	Width  uint16

	// Only used on Unix
	x uint16
	y uint16
}

// StdStreams returns the standard streams (stdin, stdout, stderr).
//
// On Windows, it attempts to turn on VT handling on all std handles if
// supported, or falls back to terminal emulation. On Unix, this returns
// the standard [os.Stdin], [os.Stdout] and [os.Stderr].
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	return stdStreams()
}

// GetFdInfo returns the file descriptor for an os.File and indicates whether the file represents a terminal.
func GetFdInfo(in interface{}) (fd uintptr, isTerminal bool) {
	return getFdInfo(in)
}

// GetWinsize returns the window size based on the specified file descriptor.
func GetWinsize(fd uintptr) (*Winsize, error) {
	return getWinsize(fd)
}

// SetWinsize tries to set the specified window size for the specified file
// descriptor. It is only implemented on Unix, and returns an error on Windows.
func SetWinsize(fd uintptr, ws *Winsize) error {
	return setWinsize(fd, ws)
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	return isTerminal(fd)
}

// RestoreTerminal restores the terminal connected to the given file descriptor
// to a previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	return restoreTerminal(fd, state)
}

// SaveState saves the state of the terminal connected to the given file descriptor.
func SaveState(fd uintptr) (*State, error) {
	return saveState(fd)
}

// DisableEcho applies the specified state to the terminal connected to the file
// descriptor, with echo disabled.
func DisableEcho(fd uintptr, state *State) error {
	return disableEcho(fd, state)
}

// SetRawTerminal puts the terminal connected to the given file descriptor into
// raw mode and returns the previous state. On UNIX, this is the equivalent of
// [MakeRaw], and puts both the input and output into raw mode. On Windows, it
// only puts the input into raw mode.
func SetRawTerminal(fd uintptr) (previousState *State, err error) {
	return setRawTerminal(fd)
}

// SetRawTerminalOutput puts the output of terminal connected to the given file
// descriptor into raw mode. On UNIX, this does nothing and returns nil for the
// state. On Windows, it disables LF -> CRLF translation.
func SetRawTerminalOutput(fd uintptr) (previousState *State, err error) {
	return setRawTerminalOutput(fd)
}

// MakeRaw puts the terminal (Windows Console) connected to the
// given file descriptor into raw mode and returns the previous state of
// the terminal so that it can be restored.
func MakeRaw(fd uintptr) (previousState *State, err error) {
	return makeRaw(fd)
}
