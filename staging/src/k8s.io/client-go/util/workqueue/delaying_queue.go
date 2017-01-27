/*
Copyright 2016 The Kubernetes Authors.

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

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/util/clock"
)

// DelayingInterface is an Interface that can Add an item at a later time. This makes it easier to
// requeue items after failures without ending up in a hot-loop.
type DelayingInterface interface {
	Interface
	// AddAfter adds an item to the workqueue after the indicated duration has passed
	AddAfter(item interface{}, duration time.Duration)
}

// NewDelayingQueue constructs a new workqueue with delayed queuing ability
func NewDelayingQueue() DelayingInterface {
	return newDelayingQueue(clock.RealClock{}, "")
}

func NewNamedDelayingQueue(name string) DelayingInterface {
	return newDelayingQueue(clock.RealClock{}, name)
}

func newDelayingQueue(clock clock.Clock, name string) DelayingInterface {
	ret := &delayingType{
		Interface:          NewNamed(name),
		clock:              clock,
		heartbeat:          clock.Tick(maxWait),
		stopCh:             make(chan struct{}),
		waitingTimeByEntry: map[t]time.Time{},
		waitingForAddCh:    make(chan waitFor, 1000),
		metrics:            newRetryMetrics(name),
	}

	go ret.waitingLoop()

	return ret
}

// delayingType wraps an Interface and provides delayed re-enquing
type delayingType struct {
	Interface

	// clock tracks time for delayed firing
	clock clock.Clock

	// stopCh lets us signal a shutdown to the waiting loop
	stopCh chan struct{}

	// heartbeat ensures we wait no more than maxWait before firing
	//
	// TODO: replace with Ticker (and add to clock) so this can be cleaned up.
	// clock.Tick will leak.
	heartbeat <-chan time.Time

	// waitingForAdd is an ordered slice of items to be added to the contained work queue
	waitingForAdd []waitFor
	// waitingTimeByEntry holds wait time by entry, so we can lookup pre-existing indexes
	waitingTimeByEntry map[t]time.Time
	// waitingForAddCh is a buffered channel that feeds waitingForAdd
	waitingForAddCh chan waitFor

	// metrics counts the number of retries
	metrics retryMetrics
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

	q.metrics.retry()

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

// maxWait keeps a max bound on the wait time. It's just insurance against weird things happening.
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
			q.waitingTimeByEntry = nil
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
			delete(q.waitingTimeByEntry, entry.data)
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
				q.waitingForAdd = insert(q.waitingForAdd, q.waitingTimeByEntry, waitEntry)
			} else {
				q.Add(waitEntry.data)
			}

			drained := false
			for !drained {
				select {
				case waitEntry := <-q.waitingForAddCh:
					if waitEntry.readyAt.After(q.clock.Now()) {
						q.waitingForAdd = insert(q.waitingForAdd, q.waitingTimeByEntry, waitEntry)
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
//
// TODO: This should probably be converted to use container/heap to improve
// running time for a large number of items.
func insert(entries []waitFor, knownEntries map[t]time.Time, entry waitFor) []waitFor {
	// if the entry is already in our retry list and the existing time is before the new one, just skip it
	existingTime, exists := knownEntries[entry.data]
	if exists && existingTime.Before(entry.readyAt) {
		return entries
	}

	// if the entry exists and is scheduled for later, go ahead and remove the entry
	if exists {
		if existingIndex := findEntryIndex(entries, existingTime, entry.data); existingIndex >= 0 && existingIndex < len(entries) {
			entries = append(entries[:existingIndex], entries[existingIndex+1:]...)
		}
	}

	insertionIndex := sort.Search(len(entries), func(i int) bool {
		return entry.readyAt.Before(entries[i].readyAt)
	})

	// grow by 1
	entries = append(entries, waitFor{})
	// shift items from the insertion point to the end
	copy(entries[insertionIndex+1:], entries[insertionIndex:])
	// insert the record
	entries[insertionIndex] = entry

	knownEntries[entry.data] = entry.readyAt

	return entries
}

// findEntryIndex returns the index for an existing entry
func findEntryIndex(entries []waitFor, existingTime time.Time, data t) int {
	index := sort.Search(len(entries), func(i int) bool {
		return entries[i].readyAt.After(existingTime) || existingTime == entries[i].readyAt
	})

	// we know this is the earliest possible index, but there could be multiple with the same time
	// iterate from here to find the dupe
	for ; index < len(entries); index++ {
		if entries[index].data == data {
			break
		}
	}

	return index
}
