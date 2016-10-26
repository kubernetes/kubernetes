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
	"math"
	"testing"
	"time"

	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

const (
	threshold3K  = 100
	threshold30K = 30
	threshold60K = 30
)

// TestSchedule100Node3KPods schedules 3k pods on 100 nodes.
func TestSchedule100Node3KPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}
	if min := schedulePods(100, 3000); min < threshold3K {
		t.Errorf("To small pod scheduling throughput for 3k pods. Expected %v got %v", threshold3K, min)
	} else {
		fmt.Printf("Minimal observed throughput for 3k pod test: %v\n", min)
	}
}

// TestSchedule1000Node30KPods schedules 30k pods on 1000 nodes.
func TestSchedule1000Node30KPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}
	if min := schedulePods(1000, 30000); min < threshold30K {
		t.Errorf("To small pod scheduling throughput for 30k pods. Expected %v got %v", threshold30K, min)
	} else {
		fmt.Printf("Minimal observed throughput for 30k pod test: %v\n", min)
	}
}

// TestSchedule2000Node60KPods schedules 60k pods on 2000 nodes.
// This test won't fit in normal 10 minutes time window.
// func TestSchedule2000Node60KPods(t *testing.T) {
// 	if testing.Short() {
// 		t.Skip("Skipping because we want to run short tests")
// 	}
// 	if min := schedulePods(2000, 60000); min < threshold60K {
// 		t.Errorf("To small pod scheduling throughput for 60k pods. Expected %v got %v", threshold60K, min)
// 	} else {
// 		fmt.Printf("Minimal observed throughput for 60k pod test: %v\n", min)
// 	}
// }

// schedulePods schedules specific number of pods on specific number of nodes.
// This is used to learn the scheduling throughput on various
// sizes of cluster and changes as more and more pods are scheduled.
// It won't stop until all pods are scheduled.
// It retruns the minimum of throughput over whole run.
func schedulePods(numNodes, numPods int) int32 {
	schedulerConfigFactory, destroyFunc := mustSetupScheduler()
	defer destroyFunc()
	c := schedulerConfigFactory.Client

	nodePreparer := framework.NewIntegrationTestNodePreparer(
		c,
		[]testutils.CountToStrategy{{Count: numNodes, Strategy: &testutils.TrivialNodePrepareStrategy{}}},
		"scheduler-perf-",
	)
	if err := nodePreparer.PrepareNodes(); err != nil {
		glog.Fatalf("%v", err)
	}
	defer nodePreparer.CleanupNodes()

	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy("sched-test", numPods, testutils.NewSimpleWithControllerCreatePodStrategy("rc1"))
	podCreator := testutils.NewTestPodCreator(c, config)
	podCreator.CreatePods()

	prev := 0
	minQps := int32(math.MaxInt32)
	start := time.Now()
	for {
		// This can potentially affect performance of scheduler, since List() is done under mutex.
		// Listing 10000 pods is an expensive operation, so running it frequently may impact scheduler.
		// TODO: Setup watch on apiserver and wait until all pods scheduled.
		scheduled := schedulerConfigFactory.ScheduledPodLister.Indexer.List()
		if len(scheduled) >= numPods {
			fmt.Printf("Scheduled %v Pods in %v seconds (%v per second on average).\n",
				numPods, int(time.Since(start)/time.Second), numPods/int(time.Since(start)/time.Second))
			return minQps
		}
		// There's no point in printing it for the last iteration, as the value is random
		qps := len(scheduled) - prev
		if int32(qps) < minQps {
			minQps = int32(qps)
		}

		fmt.Printf("%ds\trate: %d\ttotal: %d\n", time.Since(start)/time.Second, qps, len(scheduled))
		prev = len(scheduled)
		time.Sleep(1 * time.Second)
	}
}
