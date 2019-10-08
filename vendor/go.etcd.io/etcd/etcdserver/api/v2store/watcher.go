// Copyright 2015 The etcd Authors
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

package v2store

type Watcher interface {
	EventChan() chan *Event
	StartIndex() uint64 // The EtcdIndex at which the Watcher was created
	Remove()
}

type watcher struct {
	eventChan  chan *Event
	stream     bool
	recursive  bool
	sinceIndex uint64
	startIndex uint64
	hub        *watcherHub
	removed    bool
	remove     func()
}

func (w *watcher) EventChan() chan *Event {
	return w.eventChan
}

func (w *watcher) StartIndex() uint64 {
	return w.startIndex
}

// notify function notifies the watcher. If the watcher interests in the given path,
// the function will return true.
func (w *watcher) notify(e *Event, originalPath bool, deleted bool) bool {
	// watcher is interested the path in three cases and under one condition
	// the condition is that the event happens after the watcher's sinceIndex

	// 1. the path at which the event happens is the path the watcher is watching at.
	// For example if the watcher is watching at "/foo" and the event happens at "/foo",
	// the watcher must be interested in that event.

	// 2. the watcher is a recursive watcher, it interests in the event happens after
	// its watching path. For example if watcher A watches at "/foo" and it is a recursive
	// one, it will interest in the event happens at "/foo/bar".

	// 3. when we delete a directory, we need to force notify all the watchers who watches
	// at the file we need to delete.
	// For example a watcher is watching at "/foo/bar". And we deletes "/foo". The watcher
	// should get notified even if "/foo" is not the path it is watching.
	if (w.recursive || originalPath || deleted) && e.Index() >= w.sinceIndex {
		// We cannot block here if the eventChan capacity is full, otherwise
		// etcd will hang. eventChan capacity is full when the rate of
		// notifications are higher than our send rate.
		// If this happens, we close the channel.
		select {
		case w.eventChan <- e:
		default:
			// We have missed a notification. Remove the watcher.
			// Removing the watcher also closes the eventChan.
			w.remove()
		}
		return true
	}
	return false
}

// Remove removes the watcher from watcherHub
// The actual remove function is guaranteed to only be executed once
func (w *watcher) Remove() {
	w.hub.mutex.Lock()
	defer w.hub.mutex.Unlock()

	close(w.eventChan)
	if w.remove != nil {
		w.remove()
	}
}

// nopWatcher is a watcher that receives nothing, always blocking.
type nopWatcher struct{}

func NewNopWatcher() Watcher                 { return &nopWatcher{} }
func (w *nopWatcher) EventChan() chan *Event { return nil }
func (w *nopWatcher) StartIndex() uint64     { return 0 }
func (w *nopWatcher) Remove()                {}
