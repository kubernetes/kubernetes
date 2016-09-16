/*
Copyright 2015 The Kubernetes Authors.

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

package benchmark

import (
	"fmt"
	"testing"
	"time"
)

// TestSchedule100Node3KPods schedules 3k pods on 100 nodes.
func TestSchedule100Node3KPods(t *testing.T) {
	schedulePods(100, 3000)
}

// TestSchedule1000Node30KPods schedules 30k pods on 1000 nodes.
func TestSchedule1000Node30KPods(t *testing.T) {
	schedulePods(1000, 30000)
}

// schedulePods schedules specific number of pods on specific number of nodes.
// This is used to learn the scheduling throughput on various
// sizes of cluster and changes as more and more pods are scheduled.
// It won't stop until all pods are scheduled.
func schedulePods(numNodes, numPods int) {
	schedulerConfigFactory, destroyFunc := mustSetupScheduler()
	defer destroyFunc()
	c := schedulerConfigFactory.Client

	makeNodes(c, numNodes)
	makePodsFromRC(c, "rc1", numPods)

	prev := 0
	start := time.Now()
	for {
		// This can potentially affect performance of scheduler, since List() is done under mutex.
		// Listing 10000 pods is an expensive operation, so running it frequently may impact scheduler.
		// TODO: Setup watch on apiserver and wait until all pods scheduled.
		scheduled := schedulerConfigFactory.ScheduledPodLister.Indexer.List()
		fmt.Printf("%ds\trate: %d\ttotal: %d\n", time.Since(start)/time.Second, len(scheduled)-prev, len(scheduled))
		if len(scheduled) >= numPods {
			return
		}
		prev = len(scheduled)
		time.Sleep(1 * time.Second)
	}
}
