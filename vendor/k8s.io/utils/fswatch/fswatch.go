/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fswatch

import (
	"errors"
)

// Sentinel errors.
var (
	ErrClosed           = errors.New("fswatch: watcher is closed")
	ErrNonExistentWatch = errors.New("fswatch: path is not being watched")
	// ErrEventOverflow is delivered on the errors channel when the
	// kernel queue overflowed and events were dropped.
	ErrEventOverflow = errors.New("fswatch: event queue overflow")
)

// Op is the cross-platform event class. Bits match fsnotify v1's
// default set so direct callers can migrate by import-path swap.
type Op uint32

const (
	Create Op = 1 << iota
	Write
	Remove
	Rename
	Chmod
)

// String returns a pipe-separated list of bits set in op.
func (op Op) String() string {
	if op == 0 {
		return ""
	}
	var s string
	add := func(name string) {
		if s != "" {
			s += "|"
		}
		s += name
	}
	if op&Create != 0 {
		add("Create")
	}
	if op&Write != 0 {
		add("Write")
	}
	if op&Remove != 0 {
		add("Remove")
	}
	if op&Rename != 0 {
		add("Rename")
	}
	if op&Chmod != 0 {
		add("Chmod")
	}
	return s
}

// Event is a filesystem event.
type Event struct {
	Name string
	Op   Op
}

// Has reports whether op is set in e.Op.
func (e Event) Has(op Op) bool { return e.Op&op != 0 }

// Watcher delivers filesystem events. Add, Remove, and Close are safe
// to call from any goroutine; Close is idempotent. Reading Events and
// Errors is the caller's responsibility.
type Watcher struct {
	impl watcherImpl
}

// watcherImpl is the platform-specific backing for Watcher.
type watcherImpl interface {
	Add(path string) error
	Remove(path string) error
	Events() <-chan Event
	Errors() <-chan error
	Close() error
}

// NewWatcher constructs a Watcher.
func NewWatcher() (*Watcher, error) {
	impl, err := newWatcherImpl()
	if err != nil {
		return nil, err
	}
	return &Watcher{impl: impl}, nil
}

// Add starts watching path. Returns ErrClosed if the watcher has been
// closed.
func (w *Watcher) Add(path string) error { return w.impl.Add(path) }

// Remove stops watching path.
func (w *Watcher) Remove(path string) error { return w.impl.Remove(path) }

// Events returns the channel of filesystem events. Closed after Close.
func (w *Watcher) Events() <-chan Event { return w.impl.Events() }

// Errors returns the channel of backend errors. Closed after Close.
func (w *Watcher) Errors() <-chan error { return w.impl.Errors() }

// Close stops the watcher. In-flight events are dropped. Subsequent
// Add and Remove return ErrClosed. Close is idempotent.
func (w *Watcher) Close() error { return w.impl.Close() }
