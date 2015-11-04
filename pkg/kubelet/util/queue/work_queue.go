/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package queue

import (
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// WorkQueue allows queueing items with a timestamp. An item is
// considered ready to process if the timestamp has expired.
type WorkQueue interface {
	// GetWork dequeues and returns all ready items.
	GetWork() []types.UID
	// Enqueue inserts a new item or overwrites an existing item with the
	// new timestamp (time.Now() + delay) if it is greater.
	Enqueue(item types.UID, delay time.Duration)
}

type basicWorkQueue struct {
	clock util.Clock
	lock  sync.Mutex
	queue map[types.UID]time.Time
}

var _ WorkQueue = &basicWorkQueue{}

func NewBasicWorkQueue() WorkQueue {
	queue := make(map[types.UID]time.Time)
	return &basicWorkQueue{queue: queue, clock: util.RealClock{}}
}

func (q *basicWorkQueue) GetWork() []types.UID {
	q.lock.Lock()
	defer q.lock.Unlock()
	now := q.clock.Now()
	var items []types.UID
	for k, v := range q.queue {
		if v.Before(now) {
			items = append(items, k)
			delete(q.queue, k)
		}
	}
	return items
}

func (q *basicWorkQueue) Enqueue(item types.UID, delay time.Duration) {
	q.lock.Lock()
	defer q.lock.Unlock()
	now := q.clock.Now()
	timestamp := now.Add(delay)
	existing, ok := q.queue[item]
	if !ok || (ok && existing.Before(timestamp)) {
		q.queue[item] = timestamp
	}
}
