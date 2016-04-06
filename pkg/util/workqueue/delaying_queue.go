/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package workqueue

import (
	"sort"
	"time"

	"k8s.io/kubernetes/pkg/util"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

// DelayingInterface is an Interface that can Add an item at a later time.  This makes it easier to
// requeue items after failures without ending up in a hot-loop.
type DelayingInterface interface {
	Interface
	// AddAfter adds an item to the workqueue after the indicated duration has passed
	AddAfter(item interface{}, duration time.Duration)
}

// NewDelayingQueue constructs a new workqueue with delayed queuing ability
func NewDelayingQueue() DelayingInterface {
	return newDelayingQueue(util.RealClock{})
}

func newDelayingQueue(clock util.Clock) DelayingInterface {
	ret := &delayingType{
		Interface:       New(),
		clock:           clock,
		heartbeat:       clock.Tick(maxWait),
		stopCh:          make(chan struct{}),
		waitingForAddCh: make(chan waitFor, 1000),
	}

	go ret.waitingLoop()

	return ret
}

// delayingType wraps an Interface and provides delayed re-enquing
type delayingType struct {
	Interface

	// clock tracks time for delayed firing
	clock util.Clock

	// stopCh lets us signal a shutdown to the waiting loop
	stopCh chan struct{}

	// heartbeat ensures we wait no more than maxWait before firing
	heartbeat <-chan time.Time

	// waitingForAdd is an ordered slice of items to be added to the contained work queue
	waitingForAdd []waitFor
	// waitingForAddCh is a buffered channel that feeds waitingForAdd
	waitingForAddCh chan waitFor
}

// waitFor holds the data to add and the time it should be added
type waitFor struct {
	data    t
	readyAt time.Time
}

// ShutDown gives a way to shut off this queue
func (q *delayingType) ShutDown() {
	q.Interface.ShutDown()
	close(q.stopCh)
}

// AddAfter adds the given item to the work queue after the given delay
func (q *delayingType) AddAfter(item interface{}, duration time.Duration) {
	// don't add if we're already shutting down
	if q.ShuttingDown() {
		return
	}

	// immediately add things with no delay
	if duration <= 0 {
		q.Add(item)
		return
	}

	select {
	case <-q.stopCh:
		// unblock if ShutDown() is called
	case q.waitingForAddCh <- waitFor{data: item, readyAt: q.clock.Now().Add(duration)}:
	}
}

// maxWait keeps a max bound on the wait time.  It's just insurance against weird things happening.
// Checking the queue every 10 seconds isn't expensive and we know that we'll never end up with an
// expired item sitting for more than 10 seconds.
const maxWait = 10 * time.Second

// waitingLoop runs until the workqueue is shutdown and keeps a check on the list of items to be added.
func (q *delayingType) waitingLoop() {
	defer utilruntime.HandleCrash()

	// Make a placeholder channel to use when there are no items in our list
	never := make(<-chan time.Time)

	for {
		if q.Interface.ShuttingDown() {
			// discard waiting entries
			q.waitingForAdd = nil
			return
		}

		now := q.clock.Now()

		// Add ready entries
		readyEntries := 0
		for _, entry := range q.waitingForAdd {
			if entry.readyAt.After(now) {
				break
			}
			q.Add(entry.data)
			readyEntries++
		}
		q.waitingForAdd = q.waitingForAdd[readyEntries:]

		// Set up a wait for the first item's readyAt (if one exists)
		nextReadyAt := never
		if len(q.waitingForAdd) > 0 {
			nextReadyAt = q.clock.After(q.waitingForAdd[0].readyAt.Sub(now))
		}

		select {
		case <-q.stopCh:
			return

		case <-q.heartbeat:
			// continue the loop, which will add ready items

		case <-nextReadyAt:
			// continue the loop, which will add ready items

		case waitEntry := <-q.waitingForAddCh:
			if waitEntry.readyAt.After(q.clock.Now()) {
				q.waitingForAdd = insert(q.waitingForAdd, waitEntry)
			} else {
				q.Add(waitEntry.data)
			}

			drained := false
			for !drained {
				select {
				case waitEntry := <-q.waitingForAddCh:
					if waitEntry.readyAt.After(q.clock.Now()) {
						q.waitingForAdd = insert(q.waitingForAdd, waitEntry)
					} else {
						q.Add(waitEntry.data)
					}
				default:
					drained = true
				}
			}
		}
	}
}

// inserts the given entry into the sorted entries list
// same semantics as append()... the given slice may be modified,
// and the returned value should be used
func insert(entries []waitFor, entry waitFor) []waitFor {
	insertionIndex := sort.Search(len(entries), func(i int) bool {
		return entry.readyAt.Before(entries[i].readyAt)
	})

	// grow by 1
	entries = append(entries, waitFor{})
	// shift items from the insertion point to the end
	copy(entries[insertionIndex+1:], entries[insertionIndex:])
	// insert the record
	entries[insertionIndex] = entry

	return entries
}
