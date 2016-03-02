package libcontainer

import "io"

// Console represents a pseudo TTY.
type Console interface {
	io.ReadWriter
	io.Closer

	// Path returns the filesystem path to the slave side of the pty.
	Path() string

	// Fd returns the fd for the master of the pty.
	Fd() uintptr
}
