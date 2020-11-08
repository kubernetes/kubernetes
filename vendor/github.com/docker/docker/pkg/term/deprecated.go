// Package term provides structures and helper functions to work with
// terminal (state, sizes).
//
// Deprecated: use github.com/moby/term instead
package term // import "github.com/docker/docker/pkg/term"

import (
	"github.com/moby/term"
)

// EscapeError is special error which returned by a TTY proxy reader's Read()
// method in case its detach escape sequence is read.
// Deprecated: use github.com/moby/term.EscapeError
type EscapeError = term.EscapeError

// State represents the state of the terminal.
// Deprecated: use github.com/moby/term.State
type State = term.State

// Winsize represents the size of the terminal window.
// Deprecated: use github.com/moby/term.Winsize
type Winsize = term.Winsize

var (
	// ASCII list the possible supported ASCII key sequence
	ASCII = term.ASCII

	// ToBytes converts a string representing a suite of key-sequence to the corresponding ASCII code.
	// Deprecated: use github.com/moby/term.ToBytes
	ToBytes = term.ToBytes

	// StdStreams returns the standard streams (stdin, stdout, stderr).
	// Deprecated: use github.com/moby/term.StdStreams
	StdStreams = term.StdStreams

	// GetFdInfo returns the file descriptor for an os.File and indicates whether the file represents a terminal.
	// Deprecated: use github.com/moby/term.GetFdInfo
	GetFdInfo = term.GetFdInfo

	// GetWinsize returns the window size based on the specified file descriptor.
	// Deprecated: use github.com/moby/term.GetWinsize
	GetWinsize = term.GetWinsize

	// IsTerminal returns true if the given file descriptor is a terminal.
	// Deprecated: use github.com/moby/term.IsTerminal
	IsTerminal = term.IsTerminal

	// RestoreTerminal restores the terminal connected to the given file descriptor
	// to a previous state.
	// Deprecated: use github.com/moby/term.RestoreTerminal
	RestoreTerminal = term.RestoreTerminal

	// SaveState saves the state of the terminal connected to the given file descriptor.
	// Deprecated: use github.com/moby/term.SaveState
	SaveState = term.SaveState

	// DisableEcho applies the specified state to the terminal connected to the file
	// descriptor, with echo disabled.
	// Deprecated: use github.com/moby/term.DisableEcho
	DisableEcho = term.DisableEcho

	// SetRawTerminal puts the terminal connected to the given file descriptor into
	// raw mode and returns the previous state. On UNIX, this puts both the input
	// and output into raw mode. On Windows, it only puts the input into raw mode.
	// Deprecated: use github.com/moby/term.SetRawTerminal
	SetRawTerminal = term.SetRawTerminal

	// SetRawTerminalOutput puts the output of terminal connected to the given file
	// descriptor into raw mode. On UNIX, this does nothing and returns nil for the
	// state. On Windows, it disables LF -> CRLF translation.
	// Deprecated: use github.com/moby/term.SetRawTerminalOutput
	SetRawTerminalOutput = term.SetRawTerminalOutput

	// MakeRaw puts the terminal connected to the given file descriptor into raw
	// mode and returns the previous state of the terminal so that it can be restored.
	// Deprecated: use github.com/moby/term.MakeRaw
	MakeRaw = term.MakeRaw

	// NewEscapeProxy returns a new TTY proxy reader which wraps the given reader
	// and detects when the specified escape keys are read, in which case the Read
	// method will return an error of type EscapeError.
	// Deprecated: use github.com/moby/term.NewEscapeProxy
	NewEscapeProxy = term.NewEscapeProxy
)
