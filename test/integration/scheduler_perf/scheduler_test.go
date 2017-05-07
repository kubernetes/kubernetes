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
	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"math"
	"testing"
	"time"
)

const (
	warning3K    = 100
	threshold3K  = 30
	threshold30K = 30
	threshold60K = 30
)

// TestSchedule100Node3KPods schedules 3k pods on 100 nodes.
func TestSchedule100Node3KPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}
	config := getBaseConfig(1000, 30000)
	writePodAndNodeTopologyToConfig(config)
	min := schedulePods(config)
	if min < threshold3K {
		t.Errorf("Failing: Scheduling rate was too low for an interval, we saw rate of %v, which is the allowed minimum of %v ! ", min, threshold3K)
	} else if min < warning3K {
		fmt.Printf("Warning: pod scheduling throughput for 3k pods was slow for an interval... Saw a interval with very low (%v) scheduling rate!", min)
	} else {
		fmt.Printf("Minimal observed throughput for 3k pod test: %v\n", min)
	}
}

// TestSchedule100Node3KNodeAffinityPods schedules 3k pods using Node affinity on 100 nodes.
func TestSchedule100Node3KNodeAffinityPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}

	config := getBaseConfig(100, 3000)
	// number of Node-Pod sets with Pods NodeAffinity matching given Nodes.
	numGroups := 10
	nodeAffinityKey := "kubernetes.io/sched-perf-node-affinity"
	nodeStrategies := make([]testutils.CountToStrategy, 0, numGroups)
	for i := 0; i < numGroups; i++ {
		nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{
			Count:    config.numNodes / numGroups,
			Strategy: testutils.NewLabelNodePrepareStrategy(nodeAffinityKey, fmt.Sprintf("%v", i)),
		})
	}
	config.nodePreparer = framework.NewIntegrationTestNodePreparer(
		config.schedulerSupportFunctions.GetClient(),
		nodeStrategies,
		"scheduler-perf-",
	)

	podCreatorConfig := testutils.NewTestPodCreatorConfig()
	for i := 0; i < numGroups; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "sched-perf-node-affinity-pod-",
			},
			Spec: testutils.MakePodSpec(),
		}
		pod.Spec.Affinity = &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      nodeAffinityKey,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{fmt.Sprintf("%v", i)},
								},
							},
						},
					},
				},
			},
		}

		podCreatorConfig.AddStrategy("sched-perf-node-affinity", config.numPods/numGroups,
			testutils.NewCustomCreatePodStrategy(pod),
		)
	}
	config.podCreator = testutils.NewTestPodCreator(config.schedulerSupportFunctions.GetClient(), podCreatorConfig)

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
	config := getBaseConfig(1000, 30000)
	writePodAndNodeTopologyToConfig(config)
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

// testConfig contains the some input parameters needed for running test-suite
type testConfig struct {
	// Note: We don't need numPods, numNodes anymore in this struct but keeping them for backward compatibility
	numPods                   int
	numNodes                  int
	nodePreparer              testutils.TestNodePreparer
	podCreator                *testutils.TestPodCreator
	schedulerSupportFunctions scheduler.Configurator
	destroyFunc               func()
}

//  baseConfig returns a minimal testConfig to be customized for different tests.
func baseConfig() *testConfig {
	schedulerConfigFactory, destroyFunc := mustSetupScheduler()
	return &testConfig{
		schedulerSupportFunctions: schedulerConfigFactory,
		destroyFunc:               destroyFunc,
	}
}

// getBaseConfig returns baseConfig after initializing number of nodes and pods.
// We have to function for backward compatibility. We can combine this into baseConfig.
// TODO: Remove this function once the backward compatibility is not needed.
func getBaseConfig(nodes int, pods int) *testConfig {
	config := baseConfig()
	config.numNodes = nodes
	config.numPods = pods
	return config
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
		scheduled, err := config.schedulerSupportFunctions.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}
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
		scheduled, err := config.schedulerSupportFunctions.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}

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

// mutateNodeSpec returns the strategy needed for creation of nodes.
// TODO: It should take the nodespec and return the modified version of it. As of now, returning the strategies for backward compatibilty.
func (na nodeAffinity) mutateNodeSpec(numNodes int) []testutils.CountToStrategy {
	numGroups := na.numGroups
	nodeAffinityKey := na.nodeAffinityKey
	nodeStrategies := make([]testutils.CountToStrategy, 0, numGroups)
	for i := 0; i < numGroups; i++ {
		nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{
			Count:    numNodes / numGroups,
			Strategy: testutils.NewLabelNodePrepareStrategy(nodeAffinityKey, fmt.Sprintf("%v", i)),
		})
	}
	return nodeStrategies
}

// mutatePodSpec returns the list of pods after mutating the pod spec based on predicates and priorities.
// TODO: It should take the podspec and return the modified version of it. As of now, returning the podlist for backward compatibilty.
func (na nodeAffinity) mutatePodSpec(numPods int, pod *v1.Pod) []*v1.Pod {
	numGroups := na.numGroups
	nodeAffinityKey := na.nodeAffinityKey
	podList := make([]*v1.Pod, 0, numGroups)
	for i := 0; i < numGroups; i++ {
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "sched-perf-node-affinity-pod-",
			},
			Spec: testutils.MakePodSpec(),
		}
		pod.Spec.Affinity = &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      nodeAffinityKey,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{fmt.Sprintf("%v", i)},
								},
							},
						},
					},
				},
			},
		}
		podList = append(podList, pod)
	}
	return podList
}

// generatePodAndNodeTopology is the wrapper function for modifying both pods and node objects.
func (inputConfig *schedulerPerfConfig) generatePodAndNodeTopology(config *testConfig) {
	nodeAffinity := inputConfig.NodeAffinity
	podCreatorConfig := testutils.NewTestPodCreatorConfig()
	var nodeStrategies []testutils.CountToStrategy
	var pod *v1.Pod
	var podList []*v1.Pod
	if nodeAffinity != nil {
		// Mutate Node
		nodeStrategies = nodeAffinity.mutateNodeSpec(config.numNodes)
		// Mutate Pod TODO: Make this to return to podSpec.
		podList = nodeAffinity.mutatePodSpec(config.numPods, pod)
		numGroups := nodeAffinity.numGroups
		for _, pod := range podList {
			podCreatorConfig.AddStrategy("sched-perf-node-affinity", config.numPods/numGroups,
				testutils.NewCustomCreatePodStrategy(pod),
			)
		}
		config.nodePreparer = framework.NewIntegrationTestNodePreparer(
			config.schedulerSupportFunctions.GetClient(),
			nodeStrategies, "scheduler-perf-")
		config.podCreator = testutils.NewTestPodCreator(config.schedulerSupportFunctions.GetClient(), podCreatorConfig)
		// TODO: other predicates/priorities will be processed in subsequent if statements.
	} else {
		// Default configuration.
		nodePreparer := framework.NewIntegrationTestNodePreparer(
			config.schedulerSupportFunctions.GetClient(),
			[]testutils.CountToStrategy{{Count: config.numNodes, Strategy: &testutils.TrivialNodePrepareStrategy{}}},
			"scheduler-perf-",
		)

		podConfig := testutils.NewTestPodCreatorConfig()
		podConfig.AddStrategy("sched-test", config.numPods, testutils.NewSimpleWithControllerCreatePodStrategy("rc1"))
		podCreator := testutils.NewTestPodCreator(config.schedulerSupportFunctions.GetClient(), podConfig)
		config.nodePreparer = nodePreparer
		config.podCreator = podCreator
	}
	return
}

// writePodAndNodeTopologyToConfig reads a configuration and then applies it to a test configuration.
//TODO: As of now, this function is not doing anything expect for reading input values to priority structs.
func writePodAndNodeTopologyToConfig(config *testConfig) {
	// High Level structure that should be filled for every predicate or priority.
	inputConfig := &schedulerPerfConfig{
		NodeAffinity: &nodeAffinity{
			//number of Node-Pod sets with Pods NodeAffinity matching given Nodes.
			numGroups:       10,
			nodeAffinityKey: "kubernetes.io/sched-perf-node-affinity",
		},
	}
	inputConfig.generatePodAndNodeTopology(config)
	return
}
