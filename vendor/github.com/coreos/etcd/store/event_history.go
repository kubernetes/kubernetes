// Copyright 2015 CoreOS, Inc.
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

package store

import (
	"fmt"
	"path"
	"strings"
	"sync"

	etcdErr "github.com/coreos/etcd/error"
)

type EventHistory struct {
	Queue      eventQueue
	StartIndex uint64
	LastIndex  uint64
	rwl        sync.RWMutex
}

func newEventHistory(capacity int) *EventHistory {
	return &EventHistory{
		Queue: eventQueue{
			Capacity: capacity,
			Events:   make([]*Event, capacity),
		},
	}
}

// addEvent function adds event into the eventHistory
func (eh *EventHistory) addEvent(e *Event) *Event {
	eh.rwl.Lock()
	defer eh.rwl.Unlock()

	eh.Queue.insert(e)

	eh.LastIndex = e.Index()

	eh.StartIndex = eh.Queue.Events[eh.Queue.Front].Index()

	return e
}

// scan enumerates events from the index history and stops at the first point
// where the key matches.
func (eh *EventHistory) scan(key string, recursive bool, index uint64) (*Event, *etcdErr.Error) {
	eh.rwl.RLock()
	defer eh.rwl.RUnlock()

	// index should be after the event history's StartIndex
	if index < eh.StartIndex {
		return nil,
			etcdErr.NewError(etcdErr.EcodeEventIndexCleared,
				fmt.Sprintf("the requested history has been cleared [%v/%v]",
					eh.StartIndex, index), 0)
	}

	// the index should come before the size of the queue minus the duplicate count
	if index > eh.LastIndex { // future index
		return nil, nil
	}

	offset := index - eh.StartIndex
	i := (eh.Queue.Front + int(offset)) % eh.Queue.Capacity

	for {
		e := eh.Queue.Events[i]

		ok := (e.Node.Key == key)

		if recursive {
			// add tailing slash
			key = path.Clean(key)
			if key[len(key)-1] != '/' {
				key = key + "/"
			}

			ok = ok || strings.HasPrefix(e.Node.Key, key)
		}

		if (e.Action == Delete || e.Action == Expire) && e.PrevNode != nil && e.PrevNode.Dir {
			ok = ok || strings.HasPrefix(key, e.PrevNode.Key)
		}

		if ok {
			return e, nil
		}

		i = (i + 1) % eh.Queue.Capacity

		if i == eh.Queue.Back {
			return nil, nil
		}
	}
}

// clone will be protected by a stop-world lock
// do not need to obtain internal lock
func (eh *EventHistory) clone() *EventHistory {
	clonedQueue := eventQueue{
		Capacity: eh.Queue.Capacity,
		Events:   make([]*Event, eh.Queue.Capacity),
		Size:     eh.Queue.Size,
		Front:    eh.Queue.Front,
		Back:     eh.Queue.Back,
	}

	for i, e := range eh.Queue.Events {
		clonedQueue.Events[i] = e
	}

	return &EventHistory{
		StartIndex: eh.StartIndex,
		Queue:      clonedQueue,
		LastIndex:  eh.LastIndex,
	}

}
