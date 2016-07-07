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

package kubelet

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util"

	"github.com/stretchr/testify/assert"
)

var gcCallBackCount = 0

func TestContainerGarbageCollectionInvokedPeriodically(t *testing.T) {
	oldValue := gcCallBackCount
	mockClock := util.NewFakeClock(time.Now())
	mockGCController := getMockGarbageCollectionController(mockClock)
	timer := mockGCController.clock.After(containerGCPeriod)
	earliestTimeToGC := mockClock.Now()
	mockClock.Step(containerGCPeriod)
	_, earliestTimeToGC = mockGCController.doContainerGarbageCollection(mockGCCallback, timer, earliestTimeToGC)
	assert := assert.New(t)
	assert.Equal(oldValue+1, gcCallBackCount)
	assert.Equal(mockClock.Now().Add(containerGCPerioMinInterval), earliestTimeToGC)
}

func TestContainerGarbageCollectionInvocationBeingCapped(t *testing.T) {
	oldValue := gcCallBackCount
	mockClock := util.NewFakeClock(time.Now())
	mockGCController := getMockGarbageCollectionController(mockClock)
	earliestTimeToGC := mockClock.Now().Add(containerGCPerioMinInterval * 2)
	timer := mockGCController.clock.After(containerGCPerioMinInterval)

	mockClock.Step(containerGCPerioMinInterval)
	// The GC cannot run because it's not yet the next eligible time to run
	timer, earliestTimeToGC = mockGCController.doContainerGarbageCollection(mockGCCallback, timer, earliestTimeToGC)
	assert := assert.New(t)
	assert.Equal(oldValue, gcCallBackCount)

	mockClock.Step(containerGCPerioMinInterval)
	mockGCController.doContainerGarbageCollection(mockGCCallback, timer, earliestTimeToGC)
	assert.Equal(oldValue+1, gcCallBackCount)
}

func getMockGarbageCollectionController(mockClock util.Clock) garbageCollectionController {
	gcController := newGarbageCollectionController()
	gcController.clock = mockClock
	return gcController
}

func mockGCCallback() {
	gcCallBackCount++
}
