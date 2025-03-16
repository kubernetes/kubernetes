/*
Copyright 2025 The Kubernetes Authors.

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

package logs

import (
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
)

const waitDuration = 150 * time.Millisecond

// dedupWriteEventsWatcher reduce write events from fsnotify.Watcher to reduce calling function isContainerRunning.
// When container output logs quickly, and user run command `kubectl logs CONTAINER_ID -f`,
// then there will be too much function call of `isContainerRunning`.
// This will consume CPU time of kubelet and containerd.
type dedupWriteEventsWatcher struct {
	*fsnotify.Watcher
	logFileName string

	// Events sends fsnotify events.
	Events chan fsnotify.Event
	done   chan struct{}
}

func newDedupWriteEventsWatcher(logfileName string, w *fsnotify.Watcher) *dedupWriteEventsWatcher {
	return &dedupWriteEventsWatcher{
		Watcher:     w,
		logFileName: logfileName,
		Events:      make(chan fsnotify.Event, 4),
		done:        make(chan struct{}),
	}
}

func (de *dedupWriteEventsWatcher) Close() error {
	err := de.Watcher.Close()
	// All channels of dedupWriteEventsWatcher will close after fsnotify.Watcher has been closed.
	<-de.done
	close(de.Events)
	// clean all events in channel
	for range de.Events {
	}
	return err
}

// dedupLoop deduplicate Write events.
func (de *dedupWriteEventsWatcher) dedupLoop() {
	defer close(de.done)

	var lastAdd time.Time
	for e := range de.Watcher.Events {
		if filepath.Base(e.Name) != de.logFileName {
			continue
		}

		switch e.Op {
		case fsnotify.Write:
			// If there is one Write event in the channel, we can discard this one.
			// If there is one Create event in the channel, we will reopen the log file, and read from the
			// start. It's OK to discard the new Write event.
			if len(de.Events) > 0 ||
				time.Since(lastAdd) < waitDuration {
				continue
			}
			lastAdd = time.Now()
		case fsnotify.Create:
			// Always add a write event before create event. In case lost some log lines.
			de.Events <- fsnotify.Event{
				Name: e.Name,
				Op:   fsnotify.Write,
			}
			lastAdd = time.Now()
		default:
		}
		de.Events <- e
	}
}
