/*
Copyright 2017 The Kubernetes Authors.

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

package filesystem

import (
	"context"
	"fmt"
	"time"

	"github.com/fsnotify/fsnotify"
)

// FSWatcher is a callback-based filesystem watcher abstraction for fsnotify.
type FSWatcher interface {
	// Initializes the watcher with the given watch handlers.
	// Called before all other methods.
	Init(FSEventHandler, FSErrorHandler) error

	// Starts listening for events and errors.
	// When an event or error occurs, the corresponding handler is called.
	Run()

	// Add a filesystem path to watch
	AddWatch(path string) error
}

// FSEventHandler is called when a fsnotify event occurs.
type FSEventHandler func(event fsnotify.Event)

// FSErrorHandler is called when a fsnotify error occurs.
type FSErrorHandler func(err error)

type fsnotifyWatcher struct {
	watcher      *fsnotify.Watcher
	eventHandler FSEventHandler
	errorHandler FSErrorHandler
}

var _ FSWatcher = &fsnotifyWatcher{}

// NewFsnotifyWatcher returns an implementation of FSWatcher that continuously listens for
// fsnotify events and calls the event handler as soon as an event is received.
func NewFsnotifyWatcher() FSWatcher {
	return &fsnotifyWatcher{}
}

func (w *fsnotifyWatcher) AddWatch(path string) error {
	return w.watcher.Add(path)
}

func (w *fsnotifyWatcher) Init(eventHandler FSEventHandler, errorHandler FSErrorHandler) error {
	var err error
	w.watcher, err = fsnotify.NewWatcher()
	if err != nil {
		return err
	}

	w.eventHandler = eventHandler
	w.errorHandler = errorHandler
	return nil
}

func (w *fsnotifyWatcher) Run() {
	go func() {
		defer w.watcher.Close()
		for {
			select {
			case event := <-w.watcher.Events:
				if w.eventHandler != nil {
					w.eventHandler(event)
				}
			case err := <-w.watcher.Errors:
				if w.errorHandler != nil {
					w.errorHandler(err)
				}
			}
		}
	}()
}

type watchAddRemover interface {
	Add(path string) error
	Remove(path string) error
}
type noopWatcher struct{}

func (noopWatcher) Add(path string) error    { return nil }
func (noopWatcher) Remove(path string) error { return nil }

// WatchUntil watches the specified path for changes and blocks until ctx is canceled.
// eventHandler() must be non-nil, and pollInterval must be greater than 0.
// eventHandler() is invoked whenever a change event is observed or pollInterval elapses.
// errorHandler() is invoked (if non-nil) whenever an error occurs initializing or watching the specified path.
//
// If path is a directory, only the directory and immediate children are watched.
//
// If path does not exist or cannot be watched, an error is passed to errorHandler() and eventHandler() is called at pollInterval.
//
// Multiple observed events may collapse to a single invocation of eventHandler().
//
// eventHandler() is invoked immediately after successful initialization of the filesystem watch,
// in case the path changed concurrent with calling WatchUntil().
func WatchUntil(ctx context.Context, pollInterval time.Duration, path string, eventHandler func(), errorHandler func(err error)) {
	if pollInterval <= 0 {
		panic(fmt.Errorf("pollInterval must be > 0"))
	}
	if eventHandler == nil {
		panic(fmt.Errorf("eventHandler must be non-nil"))
	}
	if errorHandler == nil {
		errorHandler = func(err error) {}
	}

	// Initialize watcher, fall back to no-op
	var (
		eventsCh chan fsnotify.Event
		errorCh  chan error
		watcher  watchAddRemover
	)
	if w, err := fsnotify.NewWatcher(); err != nil {
		errorHandler(fmt.Errorf("error creating file watcher, falling back to poll at interval %s: %w", pollInterval, err))
		watcher = noopWatcher{}
	} else {
		watcher = w
		eventsCh = w.Events
		errorCh = w.Errors
		defer func() {
			_ = w.Close()
		}()
	}

	// Initialize background poll
	t := time.NewTicker(pollInterval)
	defer t.Stop()

	attemptPeriodicRewatch := false

	// Start watching the path
	if err := watcher.Add(path); err != nil {
		errorHandler(err)
		attemptPeriodicRewatch = true
	} else {
		// Invoke handle() at least once after successfully registering the listener,
		// in case the file changed concurrent with calling WatchUntil.
		eventHandler()
	}

	for {
		select {
		case <-ctx.Done():
			return

		case <-t.C:
			// Prioritize exiting if context is canceled
			if ctx.Err() != nil {
				return
			}

			// Try to re-establish the watcher if we previously got a watch error
			if attemptPeriodicRewatch {
				_ = watcher.Remove(path)
				if err := watcher.Add(path); err != nil {
					errorHandler(err)
				} else {
					attemptPeriodicRewatch = false
				}
			}

			// Handle
			eventHandler()

		case e := <-eventsCh:
			// Prioritize exiting if context is canceled
			if ctx.Err() != nil {
				return
			}

			// Try to re-establish the watcher for events which dropped the existing watch
			if e.Name == path && (e.Has(fsnotify.Remove) || e.Has(fsnotify.Rename)) {
				_ = watcher.Remove(path)
				if err := watcher.Add(path); err != nil {
					errorHandler(err)
					attemptPeriodicRewatch = true
				}
			}

			// Handle
			eventHandler()

		case err := <-errorCh:
			// Prioritize exiting if context is canceled
			if ctx.Err() != nil {
				return
			}

			// If the error occurs in response to calling watcher.Add, re-adding here could hot-loop.
			// The periodic poll will attempt to re-establish the watch.
			errorHandler(err)
			attemptPeriodicRewatch = true
		}
	}
}
