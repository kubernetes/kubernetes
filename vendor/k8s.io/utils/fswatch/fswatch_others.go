//go:build !linux

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
	"sync"

	upstream "github.com/fsnotify/fsnotify"
)

// fsnotifyImpl wraps github.com/fsnotify/fsnotify on non-Linux
// platforms. The wrapper exists so the public API is identical across
// platforms; on Linux a raw inotify implementation is used instead, and
// fsnotify is therefore absent from the Linux build closure.
type fsnotifyImpl struct {
	w       *upstream.Watcher
	events  chan Event
	errors  chan error
	done    chan struct{}
	exited  chan struct{}
	closeMu sync.Mutex
	closed  bool
}

func newWatcherImpl() (watcherImpl, error) {
	w, err := upstream.NewWatcher()
	if err != nil {
		return nil, err
	}
	impl := &fsnotifyImpl{
		w:      w,
		events: make(chan Event, 64),
		errors: make(chan error, 64),
		done:   make(chan struct{}),
		exited: make(chan struct{}),
	}
	go impl.translate()
	return impl, nil
}

func (i *fsnotifyImpl) Add(path string) error {
	// Hold closeMu across the upstream syscall so a concurrent Close
	// cannot tear down fsnotify's internal FDs mid-call.
	i.closeMu.Lock()
	defer i.closeMu.Unlock()
	if i.closed {
		return ErrClosed
	}
	if err := i.w.Add(path); err != nil {
		return translateUpstreamErr(err)
	}
	return nil
}

func (i *fsnotifyImpl) Remove(path string) error {
	i.closeMu.Lock()
	defer i.closeMu.Unlock()
	if i.closed {
		return ErrClosed
	}
	if err := i.w.Remove(path); err != nil {
		if errors.Is(err, upstream.ErrClosed) {
			return ErrClosed
		}
		// fsnotify uses a generic error for non-existent watches.
		return ErrNonExistentWatch
	}
	return nil
}

func translateUpstreamErr(err error) error {
	switch {
	case errors.Is(err, upstream.ErrClosed):
		return ErrClosed
	case errors.Is(err, upstream.ErrEventOverflow):
		return ErrEventOverflow
	default:
		return err
	}
}

func (i *fsnotifyImpl) Events() <-chan Event { return i.events }
func (i *fsnotifyImpl) Errors() <-chan error { return i.errors }

func (i *fsnotifyImpl) Close() error {
	i.closeMu.Lock()
	if i.closed {
		i.closeMu.Unlock()
		return nil
	}
	i.closed = true
	// Close fsnotify under the lock so concurrent Add/Remove cannot
	// issue an upstream call on the FD we are tearing down.
	close(i.done)
	err := i.w.Close()
	i.closeMu.Unlock()
	<-i.exited
	close(i.events)
	close(i.errors)
	return err
}

func (i *fsnotifyImpl) translate() {
	defer close(i.exited)
	for {
		select {
		case <-i.done:
			return
		case ev, ok := <-i.w.Events:
			if !ok {
				return
			}
			select {
			case <-i.done:
				return
			case i.events <- Event{Name: ev.Name, Op: Op(ev.Op)}:
			}
		case err, ok := <-i.w.Errors:
			if !ok {
				return
			}
			translated := err
			if errors.Is(err, upstream.ErrEventOverflow) {
				translated = ErrEventOverflow
			}
			select {
			case <-i.done:
				return
			case i.errors <- translated:
			}
		}
	}
}
