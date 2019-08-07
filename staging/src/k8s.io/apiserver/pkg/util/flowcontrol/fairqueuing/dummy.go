/*
Copyright 2019 The Kubernetes Authors.

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

package fairqueuing

import (
	"time"
)

type dummyQueueSet struct{}

var _ QueueSet = dummyQueueSet{}

// NewDummyQueueSet makes a QueueSet that exerts no control --- it
// always says to execute each request immediately
func NewDummyQueueSet() QueueSet {
	return dummyQueueSet{}
}

func (dummyQueueSet) SetConfiguration(concurrencyLimit, desiredNumQueues, queueLengthLimit int, requestWaitLimit time.Duration) {
}

func (dummyQueueSet) Quiesce(EmptyHandler) {
}

func (dummyQueueSet) Wait(hashValue uint64, handSize int32) (quiescent, execute bool, afterExecution func()) {
	return false, true, func() {}
}
