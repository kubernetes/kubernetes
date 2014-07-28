// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!solaris

// Package fsnotify implements file system notification.
package fsnotify

import "fmt"

// Event represents a single file system event.
type Event struct {
	Name string // Relative path to the file/directory.
	Op   Op     // Platform-independent bitmask.
}

// Op describes a set of file operations.
type Op uint32

// These are the file operations that can trigger a notification.
const (
	Create Op = 1 << iota
	Write
	Remove
	Rename
	Chmod
)

// String formats the event e in the form
// "filename: REMOVE|WRITE|..."
func (e Event) String() string {
	events := ""

	if e.Op&Create == Create {
		events += "|" + "CREATE"
	}

	if e.Op&Remove == Remove {
		events += "|" + "REMOVE"
	}

	if e.Op&Write == Write {
		events += "|" + "WRITE"
	}

	if e.Op&Rename == Rename {
		events += "|" + "RENAME"
	}

	if e.Op&Chmod == Chmod {
		events += "|" + "CHMOD"
	}

	if len(events) > 0 {
		events = events[1:]
	}

	return fmt.Sprintf("%q: %s", e.Name, events)
}
