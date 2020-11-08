// +build !windows

package term // import "github.com/docker/docker/pkg/term"

import (
	"github.com/moby/term"
)

// Termios is the Unix API for terminal I/O.
// Deprecated: use github.com/moby/term.Termios
type Termios = term.Termios

var (
	// ErrInvalidState is returned if the state of the terminal is invalid.
	ErrInvalidState = term.ErrInvalidState

	// SetWinsize tries to set the specified window size for the specified file descriptor.
	// Deprecated: use github.com/moby/term.GetWinsize
	SetWinsize = term.SetWinsize
)
