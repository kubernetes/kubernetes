// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !darwin && !dragonfly && !freebsd && !openbsd && !linux && !netbsd && !solaris && !windows
// +build !darwin,!dragonfly,!freebsd,!openbsd,!linux,!netbsd,!solaris,!windows

package fsnotify

import (
	"fmt"
	"runtime"
)

// Watcher watches a set of files, delivering events to a channel.
type Watcher struct{}

// NewWatcher establishes a new watcher with the underlying OS and begins waiting for events.
func NewWatcher() (*Watcher, error) {
	return nil, fmt.Errorf("fsnotify not supported on %s", runtime.GOOS)
}

// Close removes all watches and closes the events channel.
func (w *Watcher) Close() error {
	return nil
}

// Add starts watching the named file or directory (non-recursively).
func (w *Watcher) Add(name string) error {
	return nil
}

// Remove stops watching the the named file or directory (non-recursively).
func (w *Watcher) Remove(name string) error {
	return nil
}
