//go:build !plan9
// +build !plan9

// Package fsnotify provides a cross-platform interface for file system
// notifications.
package fsnotify

import (
	"errors"
	"fmt"
	"strings"
)

// Event represents a file system notification.
type Event struct {
	// Path to the file or directory.
	//
	// Paths are relative to the input; for example with Add("dir") the Name
	// will be set to "dir/file" if you create that file, but if you use
	// Add("/path/to/dir") it will be "/path/to/dir/file".
	Name string

	// File operation that triggered the event.
	//
	// This is a bitmask and some systems may send multiple operations at once.
	// Use the Event.Has() method instead of comparing with ==.
	Op Op
}

// Op describes a set of file operations.
type Op uint32

// The operations fsnotify can trigger; see the documentation on [Watcher] for a
// full description, and check them with [Event.Has].
const (
	Create Op = 1 << iota
	Write
	Remove
	Rename
	Chmod
)

// Common errors that can be reported by a watcher
var (
	ErrNonExistentWatch = errors.New("can't remove non-existent watcher")
	ErrEventOverflow    = errors.New("fsnotify queue overflow")
)

func (op Op) String() string {
	var b strings.Builder
	if op.Has(Create) {
		b.WriteString("|CREATE")
	}
	if op.Has(Remove) {
		b.WriteString("|REMOVE")
	}
	if op.Has(Write) {
		b.WriteString("|WRITE")
	}
	if op.Has(Rename) {
		b.WriteString("|RENAME")
	}
	if op.Has(Chmod) {
		b.WriteString("|CHMOD")
	}
	if b.Len() == 0 {
		return "[no events]"
	}
	return b.String()[1:]
}

// Has reports if this operation has the given operation.
func (o Op) Has(h Op) bool { return o&h == h }

// Has reports if this event has the given operation.
func (e Event) Has(op Op) bool { return e.Op.Has(op) }

// String returns a string representation of the event with their path.
func (e Event) String() string {
	return fmt.Sprintf("%-13s %q", e.Op.String(), e.Name)
}
