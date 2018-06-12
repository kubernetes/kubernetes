// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"sync"

	"github.com/fsnotify/fsnotify"
)

// Watcher for container-related fsnotify events in the cgroup hierarchy.
//
// Implementation is thread-safe.
type FsnotifyWatcher struct {
	// Underlying fsnotify watcher.
	watcher *fsnotify.Watcher

	// Map of containers being watched to cgroup paths watched for that container.
	containersWatched map[string]map[string]bool

	// Lock for all datastructure access.
	lock sync.Mutex
}

func NewFsnotifyWatcher() (*FsnotifyWatcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	return &FsnotifyWatcher{
		watcher:           w,
		containersWatched: make(map[string]map[string]bool),
	}, nil
}

// Add a watch to the specified directory. Returns if the container was already being watched.
func (iw *FsnotifyWatcher) AddWatch(containerName, dir string) (bool, error) {
	iw.lock.Lock()
	defer iw.lock.Unlock()

	cgroupsWatched, alreadyWatched := iw.containersWatched[containerName]

	// Register an fsnotify notification.
	if !cgroupsWatched[dir] {
		err := iw.watcher.Add(dir)
		if err != nil {
			return alreadyWatched, err
		}

		if cgroupsWatched == nil {
			cgroupsWatched = make(map[string]bool)
		}
		cgroupsWatched[dir] = true
	}

	// Record our watching of the container.
	if !alreadyWatched {
		iw.containersWatched[containerName] = cgroupsWatched
	}
	return alreadyWatched, nil
}

// Remove watch from the specified directory. Returns if this was the last watch on the specified container.
func (iw *FsnotifyWatcher) RemoveWatch(containerName, dir string) (bool, error) {
	iw.lock.Lock()
	defer iw.lock.Unlock()

	// If we don't have a watch registed for this, just return.
	cgroupsWatched, ok := iw.containersWatched[containerName]
	if !ok {
		return false, nil
	}

	// Remove the fsnotify watch if it exists.
	if cgroupsWatched[dir] {
		err := iw.watcher.Remove(dir)
		if err != nil {
			return false, nil
		}
		delete(cgroupsWatched, dir)
	}

	// Remove the record if this is the last watch.
	if len(cgroupsWatched) == 0 {
		delete(iw.containersWatched, containerName)
		return true, nil
	}

	return false, nil
}

// Errors are returned on this channel.
func (iw *FsnotifyWatcher) Error() chan error {
	return iw.watcher.Errors
}

// Events are returned on this channel.
func (iw *FsnotifyWatcher) Event() chan fsnotify.Event {
	return iw.watcher.Events
}

// Closes the fsnotify watcher.
func (iw *FsnotifyWatcher) Close() error {
	return iw.watcher.Close()
}

// Returns a map of containers to the cgroup paths being watched.
func (iw *FsnotifyWatcher) GetWatches() map[string][]string {
	out := make(map[string][]string, len(iw.containersWatched))
	for k, v := range iw.containersWatched {
		out[k] = mapToSlice(v)
	}
	return out
}

func mapToSlice(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
