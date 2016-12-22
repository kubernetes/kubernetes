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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"
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

	config := defaultSchedulerBenchmarkConfig(100, 3000)
	if min := schedulePods(config); min < threshold3K {
		t.Errorf("Too small pod scheduling throughput for 3k pods. Expected %v got %v", threshold3K, min)
	} else {
		fmt.Printf("Minimal observed throughput for 3k pod test: %v\n", min)
	}
}

// TestSchedule100Node3KNodeAffinityPods schedules 3k pods using Node affinity on 100 nodes.
func TestSchedule100Node3KNodeAffinityPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}

	config := baseConfig()
	config.numNodes = 100
	config.numPods = 3000

	// number of Node-Pod sets with Pods NodeAffinity matching given Nodes.
	numGroups := 10
	nodeAffinityKey := "kubernetes.io/sched-perf-node-affinity"

	nodeStrategies := make([]testutils.CountToStrategy, 0, 10)
	for i := 0; i < numGroups; i++ {
		nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{
			Count:    config.numNodes / numGroups,
			Strategy: testutils.NewLabelNodePrepareStrategy(nodeAffinityKey, fmt.Sprintf("%v", i)),
		})
	}
	config.nodePreparer = framework.NewIntegrationTestNodePreparer(
		config.schedulerConfigFactory.Client,
		nodeStrategies,
		"scheduler-perf-",
	)

	affinityTemplate := dedent.Dedent(`
		{
			"nodeAffinity": {
				"requiredDuringSchedulingIgnoredDuringExecution": {
					"nodeSelectorTerms": [{
						"matchExpressions": [{
							"key": "` + nodeAffinityKey + `",
							"operator": "In",
							"values": ["%v"]
						}]
					}]
				}
			}
		}`)

	podCreatorConfig := testutils.NewTestPodCreatorConfig()
	for i := 0; i < numGroups; i++ {
		podCreatorConfig.AddStrategy("sched-perf-node-affinity", config.numPods/numGroups,
			testutils.NewCustomCreatePodStrategy(&v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					GenerateName: "sched-perf-node-affinity-pod-",
					Annotations:  map[string]string{v1.AffinityAnnotationKey: fmt.Sprintf(affinityTemplate, i)},
				},
				Spec: testutils.MakePodSpec(),
			}),
		)
	}
	config.podCreator = testutils.NewTestPodCreator(config.schedulerConfigFactory.Client, podCreatorConfig)

	if min := schedulePods(config); min < threshold30K {
		t.Errorf("Too small pod scheduling throughput for 30k pods. Expected %v got %v", threshold30K, min)
	} else {
		fmt.Printf("Minimal observed throughput for 30k pod test: %v\n", min)
	}
}

// TestSchedule1000Node30KPods schedules 30k pods on 1000 nodes.
func TestSchedule1000Node30KPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}

	config := defaultSchedulerBenchmarkConfig(1000, 30000)
	if min := schedulePods(config); min < threshold30K {
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
// 	config := defaultSchedulerBenchmarkConfig(2000, 60000)
// 	if min := schedulePods(config); min < threshold60K {
// 		t.Errorf("To small pod scheduling throughput for 60k pods. Expected %v got %v", threshold60K, min)
// 	} else {
// 		fmt.Printf("Minimal observed throughput for 60k pod test: %v\n", min)
// 	}
// }

type testConfig struct {
	numPods                int
	numNodes               int
	nodePreparer           testutils.TestNodePreparer
	podCreator             *testutils.TestPodCreator
	schedulerConfigFactory *factory.ConfigFactory
	destroyFunc            func()
}

func baseConfig() *testConfig {
	schedulerConfigFactory, destroyFunc := mustSetupScheduler()
	return &testConfig{
		schedulerConfigFactory: schedulerConfigFactory,
		destroyFunc:            destroyFunc,
	}
}

func defaultSchedulerBenchmarkConfig(numNodes, numPods int) *testConfig {
	baseConfig := baseConfig()

	nodePreparer := framework.NewIntegrationTestNodePreparer(
		baseConfig.schedulerConfigFactory.Client,
		[]testutils.CountToStrategy{{Count: numNodes, Strategy: &testutils.TrivialNodePrepareStrategy{}}},
		"scheduler-perf-",
	)

	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy("sched-test", numPods, testutils.NewSimpleWithControllerCreatePodStrategy("rc1"))
	podCreator := testutils.NewTestPodCreator(baseConfig.schedulerConfigFactory.Client, config)

	baseConfig.nodePreparer = nodePreparer
	baseConfig.podCreator = podCreator
	baseConfig.numPods = numPods
	baseConfig.numNodes = numNodes

	return baseConfig
}

// schedulePods schedules specific number of pods on specific number of nodes.
// This is used to learn the scheduling throughput on various
// sizes of cluster and changes as more and more pods are scheduled.
// It won't stop until all pods are scheduled.
// It returns the minimum of throughput over whole run.
func schedulePods(config *testConfig) int32 {
	defer config.destroyFunc()
	if err := config.nodePreparer.PrepareNodes(); err != nil {
		glog.Fatalf("%v", err)
	}
	defer config.nodePreparer.CleanupNodes()
	config.podCreator.CreatePods()

	prev := 0
	// On startup there may be a latent period where NO scheduling occurs (qps = 0).
	// We are interested in low scheduling rates (i.e. qps=2),
	minQps := int32(math.MaxInt32)
	start := time.Now()

	// Bake in time for the first pod scheduling event.
	for {
		time.Sleep(50 * time.Millisecond)
		scheduled := config.schedulerConfigFactory.ScheduledPodLister.Indexer.List()
		// 30,000 pods -> wait till @ least 300 are scheduled to start measuring.
		// TODO Find out why sometimes there may be scheduling blips in the beggining.
		if len(scheduled) > config.numPods/100 {
			break
		}
	}
	// map minimum QPS entries in a counter, useful for debugging tests.
	qpsStats := map[int]int{}

	// Now that scheduling has started, lets start taking the pulse on how many pods are happening per second.
	for {
		// This can potentially affect performance of scheduler, since List() is done under mutex.
		// Listing 10000 pods is an expensive operation, so running it frequently may impact scheduler.
		// TODO: Setup watch on apiserver and wait until all pods scheduled.
		scheduled := config.schedulerConfigFactory.ScheduledPodLister.Indexer.List()

		// We will be completed when all pods are done being scheduled.
		// return the worst-case-scenario interval that was seen during this time.
		// Note this should never be low due to cold-start, so allow bake in sched time if necessary.
		if len(scheduled) >= config.numPods {
			fmt.Printf("Scheduled %v Pods in %v seconds (%v per second on average). min QPS was %v\n",
				config.numPods, int(time.Since(start)/time.Second), config.numPods/int(time.Since(start)/time.Second), minQps)
			return minQps
		}

		// There's no point in printing it for the last iteration, as the value is random
		qps := len(scheduled) - prev
		qpsStats[qps] += 1
		if int32(qps) < minQps {
			minQps = int32(qps)
		}
		fmt.Printf("%ds\trate: %d\ttotal: %d (qps frequency: %v)\n", time.Since(start)/time.Second, qps, len(scheduled), qpsStats)
		prev = len(scheduled)
		time.Sleep(1 * time.Second)
	}
}
