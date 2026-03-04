//go:build !windows
// +build !windows

package term

import (
	"errors"
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// ErrInvalidState is returned if the state of the terminal is invalid.
//
// Deprecated: ErrInvalidState is no longer used.
var ErrInvalidState = errors.New("Invalid terminal state")

// terminalState holds the platform-specific state / console mode for the terminal.
type terminalState struct {
	termios unix.Termios
}

func stdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	return os.Stdin, os.Stdout, os.Stderr
}

func getFdInfo(in interface{}) (uintptr, bool) {
	var inFd uintptr
	var isTerminalIn bool
	if file, ok := in.(*os.File); ok {
		inFd = file.Fd()
		isTerminalIn = isTerminal(inFd)
	}
	return inFd, isTerminalIn
}

func getWinsize(fd uintptr) (*Winsize, error) {
	uws, err := unix.IoctlGetWinsize(int(fd), unix.TIOCGWINSZ)
	ws := &Winsize{Height: uws.Row, Width: uws.Col, x: uws.Xpixel, y: uws.Ypixel}
	return ws, err
}

func setWinsize(fd uintptr, ws *Winsize) error {
	return unix.IoctlSetWinsize(int(fd), unix.TIOCSWINSZ, &unix.Winsize{
		Row:    ws.Height,
		Col:    ws.Width,
		Xpixel: ws.x,
		Ypixel: ws.y,
	})
}

func isTerminal(fd uintptr) bool {
	_, err := tcget(fd)
	return err == nil
}

func restoreTerminal(fd uintptr, state *State) error {
	if state == nil {
		return errors.New("invalid terminal state")
	}
	return tcset(fd, &state.termios)
}

func saveState(fd uintptr) (*State, error) {
	termios, err := tcget(fd)
	if err != nil {
		return nil, err
	}
	return &State{termios: *termios}, nil
}

func disableEcho(fd uintptr, state *State) error {
	newState := state.termios
	newState.Lflag &^= unix.ECHO

	return tcset(fd, &newState)
}

func setRawTerminal(fd uintptr) (*State, error) {
	return makeRaw(fd)
}

func setRawTerminalOutput(fd uintptr) (*State, error) {
	return nil, nil
}

func tcget(fd uintptr) (*unix.Termios, error) {
	p, err := unix.IoctlGetTermios(int(fd), getTermios)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func tcset(fd uintptr, p *unix.Termios) error {
	return unix.IoctlSetTermios(int(fd), setTermios, p)
}
