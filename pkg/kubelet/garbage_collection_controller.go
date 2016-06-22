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
	"time"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// Minimum period between consecutive container garbage collection.
	containerGCPerioMinInterval = time.Second
	// Period for performing container garbage collection.
	containerGCPeriod = 60 * containerGCPerioMinInterval
)

type garbageCollectionController struct {
	clock              util.Clock
	containerGCTrigger chan struct{}
}

func newGarbageCollectionController() garbageCollectionController {
	gcController := garbageCollectionController{clock: util.RealClock{}}
	gcController.containerGCTrigger = make(chan struct{}, 1)
	return gcController
}

func (gcController *garbageCollectionController) startContainerGarbageCollection(gcCallBack func()) {
	go wait.Until(func() {
		// Garbage collection may be triggered by a timer or by an explicit invocation of triggerContainerGarbageCollection().
		timer := gcController.clock.After(containerGCPeriod)
		earliestTimeToGC := gcController.clock.Now()
		for {
			timer, earliestTimeToGC = gcController.doContainerGarbageCollection(gcCallBack, timer, earliestTimeToGC)
		}
	}, 0, wait.NeverStop)
}

func (gcController *garbageCollectionController) doContainerGarbageCollection(gcCallBack func(), timer <-chan time.Time, earliestTimeToGC time.Time) (<-chan time.Time, time.Time) {
	for {
		select {
		case <-gcController.containerGCTrigger:
			now := gcController.clock.Now()
			if now.Before(earliestTimeToGC) {
				// It's too soon to run GC, reschedule the timer
				return gcController.clock.After(earliestTimeToGC.Sub(now)), earliestTimeToGC
			}
			gcCallBack()
			return gcController.clock.After(containerGCPeriod), gcController.clock.Now().Add(containerGCPerioMinInterval)
		case <-timer:
			gcController.triggerContainerGarbageCollection()
		}
	}
}

func (gcController *garbageCollectionController) triggerContainerGarbageCollection() {
	select {
	case gcController.containerGCTrigger <- struct{}{}:
	default:
	}
}
