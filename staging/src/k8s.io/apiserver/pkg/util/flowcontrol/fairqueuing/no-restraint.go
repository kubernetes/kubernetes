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

// NewNoRestraintFactory makes a QueueSetFactory that produces
// QueueSets that exert no restraint --- every request is dispatched
// for execution as soon as it arrives.
func NewNoRestraintFactory() QueueSetFactory {
	return noRestraintFactory{}
}

type noRestraintFactory struct{}

func (noRestraintFactory) NewQueueSet(concurrencyLimit, numQueues, queueLengthLimit int, requestWaitLimit time.Duration) QueueSet {
	return noRestraint{}
}

type noRestraint struct{}

func (noRestraint) SetConfiguration(concurrencyLimit, desiredNumQueues, queueLengthLimit int, requestWaitLimit time.Duration) {
}

func (noRestraint) Quiesce(EmptyHandler) {
}

func (noRestraint) Wait(hashValue uint64, handSize int32) (quiescent, execute bool, afterExecution func()) {
	return false, true, func() {}
}
