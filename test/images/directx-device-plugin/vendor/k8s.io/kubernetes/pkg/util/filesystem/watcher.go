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
