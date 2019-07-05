// +build !windows

// Package term provides structures and helper functions to work with
// terminal (state, sizes).
package term // import "github.com/docker/docker/pkg/term"

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"

	"golang.org/x/sys/unix"
)

var (
	// ErrInvalidState is returned if the state of the terminal is invalid.
	ErrInvalidState = errors.New("Invalid terminal state")
)

// State represents the state of the terminal.
type State struct {
	termios Termios
}

// Winsize represents the size of the terminal window.
type Winsize struct {
	Height uint16
	Width  uint16
	x      uint16
	y      uint16
}

// StdStreams returns the standard streams (stdin, stdout, stderr).
func StdStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	return os.Stdin, os.Stdout, os.Stderr
}

// GetFdInfo returns the file descriptor for an os.File and indicates whether the file represents a terminal.
func GetFdInfo(in interface{}) (uintptr, bool) {
	var inFd uintptr
	var isTerminalIn bool
	if file, ok := in.(*os.File); ok {
		inFd = file.Fd()
		isTerminalIn = IsTerminal(inFd)
	}
	return inFd, isTerminalIn
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	var termios Termios
	return tcget(fd, &termios) == 0
}

// RestoreTerminal restores the terminal connected to the given file descriptor
// to a previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	if state == nil {
		return ErrInvalidState
	}
	if err := tcset(fd, &state.termios); err != 0 {
		return err
	}
	return nil
}

// SaveState saves the state of the terminal connected to the given file descriptor.
func SaveState(fd uintptr) (*State, error) {
	var oldState State
	if err := tcget(fd, &oldState.termios); err != 0 {
		return nil, err
	}

	return &oldState, nil
}

// DisableEcho applies the specified state to the terminal connected to the file
// descriptor, with echo disabled.
func DisableEcho(fd uintptr, state *State) error {
	newState := state.termios
	newState.Lflag &^= unix.ECHO

	if err := tcset(fd, &newState); err != 0 {
		return err
	}
	handleInterrupt(fd, state)
	return nil
}

// SetRawTerminal puts the terminal connected to the given file descriptor into
// raw mode and returns the previous state. On UNIX, this puts both the input
// and output into raw mode. On Windows, it only puts the input into raw mode.
func SetRawTerminal(fd uintptr) (*State, error) {
	oldState, err := MakeRaw(fd)
	if err != nil {
		return nil, err
	}
	handleInterrupt(fd, oldState)
	return oldState, err
}

// SetRawTerminalOutput puts the output of terminal connected to the given file
// descriptor into raw mode. On UNIX, this does nothing and returns nil for the
// state. On Windows, it disables LF -> CRLF translation.
func SetRawTerminalOutput(fd uintptr) (*State, error) {
	return nil, nil
}

func handleInterrupt(fd uintptr, state *State) {
	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan, os.Interrupt)
	go func() {
		for range sigchan {
			// quit cleanly and the new terminal item is on a new line
			fmt.Println()
			signal.Stop(sigchan)
			close(sigchan)
			RestoreTerminal(fd, state)
			os.Exit(1)
		}
	}()
}
