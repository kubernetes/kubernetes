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
	"sync"
)

type FakeType struct {
	*Type
	mutex sync.Mutex
}

func NewNamedFake(name string) *FakeType {
	rc := clock.RealClock{}
	return &FakeType{
		Type: newQueue(
			rc,
			globalMetricsFactory.newQueueMetrics(name, rc),
			defaultUnfinishedWorkUpdatePeriod,
		),
	}
}

func NewFakeType() *FakeType {
	return NewNamedFake("")
}

func (q *FakeType) Add(item interface{}) {
	q.mutex.Lock()
	q.shuttingDown = false
	q.Type.Add(item)
	q.mutex.Unlock()
}

func (q *FakeType) Get() (item interface{}, shutdown bool) {
	q.mutex.Lock()
	defer q.mutex.Unlock()

	q.Type.ShutDown()
	return q.Type.Get()
}

func (q *FakeType) ShutDown() {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	q.Type.ShutDown()
}

func (q *FakeType) ShuttingDown() bool {
	q.mutex.Lock()
	defer q.mutex.Unlock()

	return q.Type.ShuttingDown()
}
