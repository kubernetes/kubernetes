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
	item := timedWorkQueueItem{start, obj}
	q.Type.Add(item)
}

// AddWithTimestamp is useful if the caller does not want to refresh the start
// time when requeuing an item.
func (q TimedWorkQueue) AddWithTimestamp(obj interface{}, timestamp time.Time) {
	item := timedWorkQueueItem{timestamp, obj}
	q.Type.Add(item)
}

// Get gets the obj along with its timestamp from the queue.
func (q TimedWorkQueue) Get() (item interface{}, start time.Time, shutdown bool) {
	item, shutdown = q.Type.Get()
	if item != nil {
		timed, _ := item.(timedWorkQueueItem)
		item = timed.obj
		start = timed.time
	}
	return item, start, shutdown
}
