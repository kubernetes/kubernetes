// +build !linux

/*
Copyright 2020 The Kubernetes Authors.

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

package inotify // import "k8s.io/utils/inotify"

import (
	"fmt"
	"runtime"
)

var errNotSupported = fmt.Errorf("watch not supported on %s", runtime.GOOS)

// NewWatcher creates and returns a new inotify instance using inotify_init(2)
func NewWatcher() (*Watcher, error) {
	return nil, errNotSupported
}

// Close closes an inotify watcher instance
// It sends a message to the reader goroutine to quit and removes all watches
// associated with the inotify instance
func (w *Watcher) Close() error {
	return errNotSupported
}

// AddWatch adds path to the watched file set.
// The flags are interpreted as described in inotify_add_watch(2).
func (w *Watcher) AddWatch(path string, flags uint32) error {
	return errNotSupported
}

// Watch adds path to the watched file set, watching all events.
func (w *Watcher) Watch(path string) error {
	return errNotSupported
}

// RemoveWatch removes path from the watched file set.
func (w *Watcher) RemoveWatch(path string) error {
	return errNotSupported
}
