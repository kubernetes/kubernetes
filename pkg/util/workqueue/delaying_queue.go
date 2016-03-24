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
	"sync"
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
		Type:        New(),
		clock:       clock,
		waitingCond: sync.NewCond(&sync.Mutex{}),
	}

	go ret.waitingLoop()
	go ret.heartbeat()

	return ret
}

// delayingType wraps a Type and provides delayed re-enquing
type delayingType struct {
	*Type

	// clock tracks time for delayed firing
	clock util.Clock

	// waitingForAdd is an ordered slice of items to be added to the contained work queue
	waitingForAdd []waitFor
	// waitingLock synchronizes access to waitingForAdd
	waitingLock sync.Mutex
	// waitingCond is used to notify the adding go func that it needs to check for items to add
	waitingCond *sync.Cond

	// nextCheckTime is used to decide whether to add a notification timer.  If the requested time
	// is beyond the time we're already waiting for, we don't add a new timer thread
	nextCheckTime *time.Time
	// nextCheckLock serializes access to the notification time
	nextCheckLock sync.Mutex
	// nextCheckCancel is a channel to close to cancel the notification
	nextCheckCancel chan struct{}
}

// waitFor holds the data to add and the time it should be added
type waitFor struct {
	data    t
	readyAt time.Time
}

// ShutDown gives a way to shut off this queue
func (q *delayingType) ShutDown() {
	q.Type.ShutDown()
	q.waitingCond.Broadcast()
}

func (q *delayingType) AddAfter(item interface{}, duration time.Duration) {
	q.waitingLock.Lock()
	defer q.waitingLock.Unlock()
	waitEntry := waitFor{data: item, readyAt: q.clock.Now().Add(duration)}

	insertionIndex := sort.Search(len(q.waitingForAdd), func(i int) bool {
		return waitEntry.readyAt.Before(q.waitingForAdd[i].readyAt)
	})

	tail := q.waitingForAdd[insertionIndex:]
	q.waitingForAdd = append(make([]waitFor, 0, len(q.waitingForAdd)+1), q.waitingForAdd[:insertionIndex]...)
	q.waitingForAdd = append(q.waitingForAdd, waitEntry)
	q.waitingForAdd = append(q.waitingForAdd, tail...)

	q.notifyAt(waitEntry.readyAt)
}

// maxWait keeps a max bound on the wait time.  It's just insurance against weird things happening.
// Checking the queue every 10 seconds isn't expensive and we know that we'll never end up with an
// expired item sitting for more than 10 seconds.
const maxWait = 10 * time.Second

// waitingLoop runs until the workqueue is shutdown and keeps a check on the list of items to be added.
func (q *delayingType) waitingLoop() {
	defer utilruntime.HandleCrash()

	for {
		if q.shuttingDown {
			return
		}

		func() {
			q.waitingCond.L.Lock()
			defer q.waitingCond.L.Unlock()
			q.waitingCond.Wait()

			if q.shuttingDown {
				return
			}

			q.waitingLock.Lock()
			defer q.waitingLock.Unlock()

			nextReadyCheck := time.Time{}
			itemsAdded := 0

			for _, queuedItem := range q.waitingForAdd {
				nextReadyCheck = queuedItem.readyAt
				if queuedItem.readyAt.After(q.clock.Now()) {
					break
				}
				q.Type.Add(queuedItem.data)
				itemsAdded++
			}

			switch itemsAdded {
			case 0:
				// no change
			case len(q.waitingForAdd):
				// consumed everything
				q.waitingForAdd = make([]waitFor, 0, len(q.waitingForAdd))

			default:
				// consumed some
				q.waitingForAdd = q.waitingForAdd[itemsAdded:]

				if len(q.waitingForAdd) > 0 {
					q.notifyAt(nextReadyCheck)
				}
			}
		}()
	}
}

// heartbeat forces a check every maxWait seconds
func (q *delayingType) heartbeat() {
	defer utilruntime.HandleCrash()

	for {
		if q.shuttingDown {
			return
		}

		ch := q.clock.After(maxWait)
		<-ch
		q.waitingCond.Broadcast()
	}
}

// clearNextCheckTimeIf resets the nextCheckTime if it matches the expected value to ensure that the subsequent notification will take effect.
func (q *delayingType) clearNextCheckTimeIf(nextReadyCheck time.Time) {
	q.nextCheckLock.Lock()
	defer q.nextCheckLock.Unlock()

	if q.nextCheckTime != nil && *q.nextCheckTime == nextReadyCheck {
		q.nextCheckTime = nil
	}
}

// notifyAt: if the requested nextReadyCheck is sooner than the current check, then a new go func is
// spawned to notify the condition that the waitingLoop is waiting for after the time is up.  The previous go func
// is cancelled
func (q *delayingType) notifyAt(nextReadyCheck time.Time) {
	q.nextCheckLock.Lock()
	defer q.nextCheckLock.Unlock()

	now := q.clock.Now()
	if (q.nextCheckTime != nil && (nextReadyCheck.After(*q.nextCheckTime) || nextReadyCheck == *q.nextCheckTime)) || nextReadyCheck.Before(now) {
		return
	}

	duration := nextReadyCheck.Sub(now)
	q.nextCheckTime = &nextReadyCheck
	ch := q.clock.After(duration)

	newCancel := make(chan struct{})
	oldCancel := q.nextCheckCancel
	// always cancel the old notifier
	if oldCancel != nil {
		close(oldCancel)
	}
	q.nextCheckCancel = newCancel

	go func() {
		defer utilruntime.HandleCrash()

		select {
		case <-ch:
			// we only have one of these go funcs active at a time.  If we hit our timer, then clear
			// the check time so that the next add will win
			q.clearNextCheckTimeIf(nextReadyCheck)
			q.waitingCond.Broadcast()

		case <-newCancel:
			// do nothing, cancelled
		}
	}()
}
