//go:build js
// +build js
package term

import "syscall/js"
import "io"
import "os"
import "errors"

// terminalState holds the platform-specific state / console mode for the terminal.
type terminalState struct {}

// GetWinsize returns the window size based on the specified file descriptor.
func getWinsize(fd uintptr) (*Winsize, error) {
	window := js.Global().Get("window")
	width := uint16(window.Get("innerWidth").Int())
	height := uint16(window.Get("innerHeight").Int())
	return &Winsize{Width: width, Height: height}, nil
}

func isTerminal(fd uintptr) bool {
	return true
}

func saveState(fd uintptr) (*State, error) {
	return &State{}, nil
}

func stdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	return os.Stdin, os.Stdout, os.Stderr
}

func disableEcho(fd uintptr, state *State) error {
return nil
}

func setRawTerminal(fd uintptr) (*State, error) {
	return makeRaw(fd)
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

func setWinsize(fd uintptr, ws *Winsize) error {
	return nil
}

func makeRaw(fd uintptr) (*State, error) {
	oldState := State{}
	return &oldState, nil
}

func setRawTerminalOutput(fd uintptr) (*State, error) {
	return nil, nil
}

func restoreTerminal(fd uintptr, state *State) error {
	if state == nil {
		return errors.New("invalid terminal state")
	}
	return nil
}