/*
Copyright 2022 The Kubernetes Authors.

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

package metrics

import (
	"sync"
	"time"

	"k8s.io/utils/clock"
)

type QueueMetrics interface {
	// AddRateLimited tracks adding an item to the workqueue rate limiter
	AddRateLimited(item interface{})

	// Forget tracks removing an item to the workqueue rate limiter
	Forget(item interface{})

	// GetRetrySinceDurations gets time durations for all items that are pending a retry since they were first recognized by rate limiter
	GetRetrySinceDurations() map[interface{}]time.Duration
}

var _ QueueMetrics = &queueMetrics{}

func NewQueueMetrics() QueueMetrics {
	return &queueMetrics{
		clock:           clock.RealClock{},
		startRetryTimes: map[interface{}]time.Time{},
	}
}

// queueMetrics start retry times for metrics. Its methods are thread-safe.
type queueMetrics struct {
	clock               clock.Clock
	startRetryTimesLock sync.RWMutex
	startRetryTimes     map[interface{}]time.Time
}

func (q *queueMetrics) AddRateLimited(item interface{}) {
	q.startRetryTimesLock.Lock()
	defer q.startRetryTimesLock.Unlock()
	if _, ok := q.startRetryTimes[item]; !ok {
		q.startRetryTimes[item] = q.clock.Now()
	}
}

func (q *queueMetrics) Forget(item interface{}) {
	q.startRetryTimesLock.Lock()
	defer q.startRetryTimesLock.Unlock()
	delete(q.startRetryTimes, item)
}

func (q *queueMetrics) GetRetrySinceDurations() map[interface{}]time.Duration {
	q.startRetryTimesLock.RLock()
	defer q.startRetryTimesLock.RUnlock()
	now := q.clock.Now()
	result := make(map[interface{}]time.Duration, len(q.startRetryTimes))
	for k, startRetry := range q.startRetryTimes {
		result[k] = now.Sub(startRetry)
	}
	return result
}
