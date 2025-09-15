/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// FullChannelBehavior controls how the Broadcaster reacts if a watcher's watch
// channel is full.
type FullChannelBehavior int

const (
	WaitIfChannelFull FullChannelBehavior = iota
	DropIfChannelFull
)

// Buffer the incoming queue a little bit even though it should rarely ever accumulate
// anything, just in case a few events are received in such a short window that
// Broadcaster can't move them onto the watchers' queues fast enough.
const incomingQueueLength = 25

// Broadcaster distributes event notifications among any number of watchers. Every event
// is delivered to every watcher.
type Broadcaster struct {
	watchers     map[int64]*broadcasterWatcher
	nextWatcher  int64
	distributing sync.WaitGroup

	// incomingBlock allows us to ensure we don't race and end up sending events
	// to a closed channel following a broadcaster shutdown.
	incomingBlock sync.Mutex
	incoming      chan Event
	stopped       chan struct{}

	// How large to make watcher's channel.
	watchQueueLength int
	// If one of the watch channels is full, don't wait for it to become empty.
	// Instead just deliver it to the watchers that do have space in their
	// channels and move on to the next event.
	// It's more fair to do this on a per-watcher basis than to do it on the
	// "incoming" channel, which would allow one slow watcher to prevent all
	// other watchers from getting new events.
	fullChannelBehavior FullChannelBehavior
}

// NewBroadcaster creates a new Broadcaster. queueLength is the maximum number of events to queue per watcher.
// It is guaranteed that events will be distributed in the order in which they occur,
// but the order in which a single event is distributed among all of the watchers is unspecified.
func NewBroadcaster(queueLength int, fullChannelBehavior FullChannelBehavior) *Broadcaster {
	m := &Broadcaster{
		watchers:            map[int64]*broadcasterWatcher{},
		incoming:            make(chan Event, incomingQueueLength),
		stopped:             make(chan struct{}),
		watchQueueLength:    queueLength,
		fullChannelBehavior: fullChannelBehavior,
	}
	m.distributing.Add(1)
	go m.loop()
	return m
}

// NewLongQueueBroadcaster functions nearly identically to NewBroadcaster,
// except that the incoming queue is the same size as the outgoing queues
// (specified by queueLength).
func NewLongQueueBroadcaster(queueLength int, fullChannelBehavior FullChannelBehavior) *Broadcaster {
	m := &Broadcaster{
		watchers:            map[int64]*broadcasterWatcher{},
		incoming:            make(chan Event, queueLength),
		stopped:             make(chan struct{}),
		watchQueueLength:    queueLength,
		fullChannelBehavior: fullChannelBehavior,
	}
	m.distributing.Add(1)
	go m.loop()
	return m
}

const internalRunFunctionMarker = "internal-do-function"

// a function type we can shoehorn into the queue.
type functionFakeRuntimeObject func()

func (obj functionFakeRuntimeObject) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj functionFakeRuntimeObject) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	// funcs are immutable. Hence, just return the original func.
	return obj
}

// Execute f, blocking the incoming queue (and waiting for it to drain first).
// The purpose of this terrible hack is so that watchers added after an event
// won't ever see that event, and will always see any event after they are
// added.
func (m *Broadcaster) blockQueue(f func()) {
	m.incomingBlock.Lock()
	defer m.incomingBlock.Unlock()
	select {
	case <-m.stopped:
		return
	default:
	}
	var wg sync.WaitGroup
	wg.Add(1)
	m.incoming <- Event{
		Type: internalRunFunctionMarker,
		Object: functionFakeRuntimeObject(func() {
			defer wg.Done()
			f()
		}),
	}
	wg.Wait()
}

// Watch adds a new watcher to the list and returns an Interface for it.
// Note: new watchers will only receive new events. They won't get an entire history
// of previous events. It will block until the watcher is actually added to the
// broadcaster.
func (m *Broadcaster) Watch() (Interface, error) {
	var w *broadcasterWatcher
	m.blockQueue(func() {
		id := m.nextWatcher
		m.nextWatcher++
		w = &broadcasterWatcher{
			result:  make(chan Event, m.watchQueueLength),
			stopped: make(chan struct{}),
			id:      id,
			m:       m,
		}
		m.watchers[id] = w
	})
	if w == nil {
		return nil, fmt.Errorf("broadcaster already stopped")
	}
	return w, nil
}

// WatchWithPrefix adds a new watcher to the list and returns an Interface for it. It sends
// queuedEvents down the new watch before beginning to send ordinary events from Broadcaster.
// The returned watch will have a queue length that is at least large enough to accommodate
// all of the items in queuedEvents. It will block until the watcher is actually added to
// the broadcaster.
func (m *Broadcaster) WatchWithPrefix(queuedEvents []Event) (Interface, error) {
	var w *broadcasterWatcher
	m.blockQueue(func() {
		id := m.nextWatcher
		m.nextWatcher++
		length := m.watchQueueLength
		if n := len(queuedEvents) + 1; n > length {
			length = n
		}
		w = &broadcasterWatcher{
			result:  make(chan Event, length),
			stopped: make(chan struct{}),
			id:      id,
			m:       m,
		}
		m.watchers[id] = w
		for _, e := range queuedEvents {
			w.result <- e
		}
	})
	if w == nil {
		return nil, fmt.Errorf("broadcaster already stopped")
	}
	return w, nil
}

// stopWatching stops the given watcher and removes it from the list.
func (m *Broadcaster) stopWatching(id int64) {
	m.blockQueue(func() {
		w, ok := m.watchers[id]
		if !ok {
			// No need to do anything, it's already been removed from the list.
			return
		}
		delete(m.watchers, id)
		close(w.result)
	})
}

// closeAll disconnects all watchers (presumably in response to a Shutdown call).
func (m *Broadcaster) closeAll() {
	for _, w := range m.watchers {
		close(w.result)
	}
	// Delete everything from the map, since presence/absence in the map is used
	// by stopWatching to avoid double-closing the channel.
	m.watchers = map[int64]*broadcasterWatcher{}
}

// Action distributes the given event among all watchers.
func (m *Broadcaster) Action(action EventType, obj runtime.Object) error {
	m.incomingBlock.Lock()
	defer m.incomingBlock.Unlock()
	select {
	case <-m.stopped:
		return fmt.Errorf("broadcaster already stopped")
	default:
	}

	m.incoming <- Event{action, obj}
	return nil
}

// Action distributes the given event among all watchers, or drops it on the floor
// if too many incoming actions are queued up.  Returns true if the action was sent,
// false if dropped.
func (m *Broadcaster) ActionOrDrop(action EventType, obj runtime.Object) (bool, error) {
	m.incomingBlock.Lock()
	defer m.incomingBlock.Unlock()

	// Ensure that if the broadcaster is stopped we do not send events to it.
	select {
	case <-m.stopped:
		return false, fmt.Errorf("broadcaster already stopped")
	default:
	}

	select {
	case m.incoming <- Event{action, obj}:
		return true, nil
	default:
		return false, nil
	}
}

// Shutdown disconnects all watchers (but any queued events will still be distributed).
// You must not call Action or Watch* after calling Shutdown. This call blocks
// until all events have been distributed through the outbound channels. Note
// that since they can be buffered, this means that the watchers might not
// have received the data yet as it can remain sitting in the buffered
// channel. It will block until the broadcaster stop request is actually executed
func (m *Broadcaster) Shutdown() {
	m.blockQueue(func() {
		close(m.stopped)
		close(m.incoming)
	})
	m.distributing.Wait()
}

// loop receives from m.incoming and distributes to all watchers.
func (m *Broadcaster) loop() {
	// Deliberately not catching crashes here. Yes, bring down the process if there's a
	// bug in watch.Broadcaster.
	for event := range m.incoming {
		if event.Type == internalRunFunctionMarker {
			event.Object.(functionFakeRuntimeObject)()
			continue
		}
		m.distribute(event)
	}
	m.closeAll()
	m.distributing.Done()
}

// distribute sends event to all watchers. Blocking.
func (m *Broadcaster) distribute(event Event) {
	if m.fullChannelBehavior == DropIfChannelFull {
		for _, w := range m.watchers {
			select {
			case w.result <- event:
			case <-w.stopped:
			default: // Don't block if the event can't be queued.
			}
		}
	} else {
		for _, w := range m.watchers {
			select {
			case w.result <- event:
			case <-w.stopped:
			}
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
// It will block until the watcher stop request is actually executed
func (mw *broadcasterWatcher) Stop() {
	mw.stop.Do(func() {
		close(mw.stopped)
		mw.m.stopWatching(mw.id)
	})
}
