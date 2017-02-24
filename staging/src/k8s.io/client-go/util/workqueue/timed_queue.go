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

import "time"

type TimedWorkQueue struct {
	*Type
}

type TimedWorkQueueItem struct {
	StartTime time.Time
	Object    interface{}
}

func NewTimedWorkQueue() *TimedWorkQueue {
	return &TimedWorkQueue{New()}
}

// Add adds the obj along with the current timestamp to the queue.
func (q TimedWorkQueue) Add(timedItem *TimedWorkQueueItem) {
	q.Type.Add(timedItem)
}

// Get gets the obj along with its timestamp from the queue.
func (q TimedWorkQueue) Get() (timedItem *TimedWorkQueueItem, shutdown bool) {
	origin, shutdown := q.Type.Get()
	if origin == nil {
		return nil, shutdown
	}
	timedItem, _ = origin.(*TimedWorkQueueItem)
	return timedItem, shutdown
}

func (q TimedWorkQueue) Done(timedItem *TimedWorkQueueItem) error {
	q.Type.Done(timedItem)
	return nil
}
