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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/util/clock"
)

type TimedWorkQueue struct {
	*Type
	clock clock.Clock
}

type timedWorkQueueItem struct {
	time time.Time
	obj  interface{}
}

func NewTimedWorkQueue(clock clock.Clock) *TimedWorkQueue {
	return &TimedWorkQueue{New(), clock}
}

// Add adds the obj along with the current timestamp to the queue.
func (q TimedWorkQueue) Add(obj interface{}) {
	start := q.clock.Now()
	item := &timedWorkQueueItem{start, obj}
	q.Type.Add(item)
}

// AddWithTimestamp is useful if the caller does not want to refresh the start
// time when requeuing an item. origin is the "origin" returned by Get().
func (q TimedWorkQueue) AddWithTimestamp(origin, obj interface{}, timestamp time.Time) error {
	originTimedItem, ok := origin.(*timedWorkQueueItem)
	if !ok {
		return fmt.Errorf("expect *timedWorkQueueItem, got %#v", origin)
	}
	originTimedItem.time = timestamp
	originTimedItem.obj = obj
	q.Type.Add(origin)
	return nil
}

// Get gets the obj along with its timestamp from the queue.
func (q TimedWorkQueue) Get() (origin, item interface{}, start time.Time, shutdown bool) {
	origin, shutdown = q.Type.Get()
	if item != nil {
		timed, _ := origin.(*timedWorkQueueItem)
		item = timed.obj
		start = timed.time
	}
	return origin, item, start, shutdown
}

func (q TimedWorkQueue) Done(origin interface{}) error {
	_, ok := origin.(*timedWorkQueueItem)
	if !ok {
		return fmt.Errorf("expect *timedWorkQueueItem, got %#v", origin)
	}
	q.Type.Done(origin)
	return nil
}
