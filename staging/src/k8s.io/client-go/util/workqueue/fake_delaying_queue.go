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
	"k8s.io/utils/clock"
	"time"
)

type fake_delayingType struct {
	*delayingType
}

func newFakeDelayingQueue(clock clock.WithTicker, q Interface, name string) *fake_delayingType {
	return &fake_delayingType{
		delayingType: newDelayingQueue(clock, q, name),
	}
}

func NewFakeDelayingQueue() DelayingInterface {
	return NewFakeDelayingQueueWithCustomClock(clock.RealClock{}, "")
}

func NewNamedFakeDelayingQueue(name string) DelayingInterface {
	return NewFakeDelayingQueueWithCustomClock(clock.RealClock{}, name)
}

// NewDelayingQueueWithCustomClock constructs a new named workqueue
// with ability to inject real or fake clock for testing purposes
func NewFakeDelayingQueueWithCustomClock(clock clock.WithTicker, name string) DelayingInterface {
	return newFakeDelayingQueue(clock, NewNamedFake(name), name)
}

func (q *fake_delayingType) AddAfter(item interface{}, _ time.Duration) {
	q.Add(item)
}
