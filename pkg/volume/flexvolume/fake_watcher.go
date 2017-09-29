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

package flexvolume

import (
	"github.com/fsnotify/fsnotify"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

// Mock filesystem watcher
type fakeWatcher struct {
	watches      []string // List of watches added by the prober, ordered from least recent to most recent.
	eventHandler utilfs.FSEventHandler
}

var _ utilfs.FSWatcher = &fakeWatcher{}

func NewFakeWatcher() *fakeWatcher {
	return &fakeWatcher{
		watches: nil,
	}
}

func (w *fakeWatcher) Init(eventHandler utilfs.FSEventHandler, _ utilfs.FSErrorHandler) error {
	w.eventHandler = eventHandler
	return nil
}

func (w *fakeWatcher) Run() { /* no-op */ }

func (w *fakeWatcher) AddWatch(path string) error {
	w.watches = append(w.watches, path)
	return nil
}

// Triggers a mock filesystem event.
func (w *fakeWatcher) TriggerEvent(op fsnotify.Op, filename string) {
	w.eventHandler(fsnotify.Event{Op: op, Name: filename})
}
