/*
Copyright 2014 Google Inc. All rights reserved.

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

package watch

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Broadcaster distributes event notifications among any number of watchers. Every event
// is delivered to every watcher.
type Broadcaster struct {
	lock sync.Mutex

	watchers    map[int64]*broadcasterWatcher
	nextWatcher int64

	incoming chan Event
}

// NewBroadcaster creates a new Broadcaster. queueLength is the maximum number of events to queue.
// When queueLength is 0, Action will block until any prior event has been
// completely distributed. It is guaranteed that events will be distibuted in the
// order in which they ocurr, but the order in which a single event is distributed
// among all of the watchers is unspecified.
func NewBroadcaster(queueLength int) *Broadcaster {
	m := &Broadcaster{
		watchers: map[int64]*broadcasterWatcher{},
		incoming: make(chan Event, queueLength),
	}
	go m.loop()
	return m
}

// Watch adds a new watcher to the list and returns an Interface for it.
// Note: new watchers will only receive new events. They won't get an entire history
// of previous events.
func (m *Broadcaster) Watch() Interface {
	m.lock.Lock()
	defer m.lock.Unlock()
	id := m.nextWatcher
	m.nextWatcher++
	w := &broadcasterWatcher{
		result:  make(chan Event),
		stopped: make(chan struct{}),
		id:      id,
		m:       m,
	}
	m.watchers[id] = w
	return w
}

// stopWatching stops the given watcher and removes it from the list.
func (m *Broadcaster) stopWatching(id int64) {
	m.lock.Lock()
	defer m.lock.Unlock()
	w, ok := m.watchers[id]
	if !ok {
		// No need to do anything, it's already been removed from the list.
		return
	}
	delete(m.watchers, id)
	close(w.result)
}

// closeAll disconnects all watchers (presumably in response to a Shutdown call).
func (m *Broadcaster) closeAll() {
	m.lock.Lock()
	defer m.lock.Unlock()
	for _, w := range m.watchers {
		close(w.result)
	}
	// Delete everything from the map, since presence/absence in the map is used
	// by stopWatching to avoid double-closing the channel.
	m.watchers = map[int64]*broadcasterWatcher{}
}

// Action distributes the given event among all watchers.
func (m *Broadcaster) Action(action EventType, obj runtime.Object) {
	m.incoming <- Event{action, obj}
}

// Shutdown disconnects all watchers (but any queued events will still be distributed).
// You must not call Action after calling Shutdown.
func (m *Broadcaster) Shutdown() {
	close(m.incoming)
}

// loop recieves from m.incoming and distributes to all watchers.
func (m *Broadcaster) loop() {
	// Deliberately not catching crashes here. Yes, bring down the process if there's a
	// bug in watch.Broadcaster.
	for {
		event, ok := <-m.incoming
		if !ok {
			break
		}
		m.distribute(event)
	}
	m.closeAll()
}

// distribute sends event to all watchers. Blocking.
func (m *Broadcaster) distribute(event Event) {
	m.lock.Lock()
	defer m.lock.Unlock()
	for _, w := range m.watchers {
		select {
		case w.result <- event:
		case <-w.stopped:
		}
	}
}

// broadcasterWatcher handles a single watcher of a broadcaster
type broadcasterWatcher struct {
	result  chan Event
	stopped chan struct{}
	stop    sync.Once
	id      int64
	m       *Broadcaster
}

// ResultChan returns a channel to use for waiting on events.
func (mw *broadcasterWatcher) ResultChan() <-chan Event {
	return mw.result
}

// Stop stops watching and removes mw from its list.
func (mw *broadcasterWatcher) Stop() {
	mw.stop.Do(func() {
		close(mw.stopped)
		mw.m.stopWatching(mw.id)
	})
}
