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
	"context"
	"fmt"
	"math"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	testutils "k8s.io/kubernetes/test/utils"

	"k8s.io/klog/v2"
)

const (
	warning3K   = 100
	threshold3K = 30
)

var (
	basePodTemplate = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "sched-perf-pod-",
		},
		// TODO: this needs to be configurable.
		Spec: testutils.MakePodSpec(),
	}
	baseNodeTemplate = &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "sample-node-",
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}
)

// TestSchedule100Node3KPods schedules 3k pods on 100 nodes.
func TestSchedule100Node3KPods(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping because we want to run short tests")
	}

	config := getBaseConfig(100, 3000)
	err := writePodAndNodeTopologyToConfig(config)
	if err != nil {
		t.Errorf("Misconfiguration happened for nodes/pods chosen to have predicates and priorities")
	}
	min := schedulePods(config)
	if min < threshold3K {
		t.Errorf("Failing: Scheduling rate was too low for an interval, we saw rate of %v, which is the allowed minimum of %v ! ", min, threshold3K)
	} else if min < warning3K {
		fmt.Printf("Warning: pod scheduling throughput for 3k pods was slow for an interval... Saw an interval with very low (%v) scheduling rate!", min)
	} else {
		fmt.Printf("Minimal observed throughput for 3k pod test: %v\n", min)
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
// 		t.Errorf("Too small pod scheduling throughput for 60k pods. Expected %v got %v", threshold60K, min)
// 	} else {
// 		fmt.Printf("Minimal observed throughput for 60k pod test: %v\n", min)
// 	}
// }

// testConfig contains the some input parameters needed for running test-suite
type testConfig struct {
	numPods             int
	numNodes            int
	mutatedNodeTemplate *v1.Node
	mutatedPodTemplate  *v1.Pod
	clientset           clientset.Interface
	podInformer         coreinformers.PodInformer
	destroyFunc         func()
}

// getBaseConfig returns baseConfig after initializing number of nodes and pods.
func getBaseConfig(nodes int, pods int) *testConfig {
	destroyFunc, podInformer, clientset := mustSetupScheduler()
	return &testConfig{
		clientset:   clientset,
		destroyFunc: destroyFunc,
		numNodes:    nodes,
		numPods:     pods,
		podInformer: podInformer,
	}
}

// schedulePods schedules specific number of pods on specific number of nodes.
// This is used to learn the scheduling throughput on various
// sizes of cluster and changes as more and more pods are scheduled.
// It won't stop until all pods are scheduled.
// It returns the minimum of throughput over whole run.
func schedulePods(config *testConfig) int32 {
	defer config.destroyFunc()
	prev := int32(0)
	// On startup there may be a latent period where NO scheduling occurs (qps = 0).
	// We are interested in low scheduling rates (i.e. qps=2),
	minQPS := int32(math.MaxInt32)
	start := time.Now()

	// Bake in time for the first pod scheduling event.
	for {
		time.Sleep(50 * time.Millisecond)
		scheduled, err := getScheduledPods(config.podInformer)
		if err != nil {
			klog.Fatalf("%v", err)
		}
		// 30,000 pods -> wait till @ least 300 are scheduled to start measuring.
		// TODO Find out why sometimes there may be scheduling blips in the beginning.
		if len(scheduled) > config.numPods/100 {
			break
		}
	}

	scheduled := int32(0)
	ctx, cancel := context.WithCancel(context.Background())
	config.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, cur interface{}) {
			curPod := cur.(*v1.Pod)
			oldPod := old.(*v1.Pod)

			if len(oldPod.Spec.NodeName) == 0 && len(curPod.Spec.NodeName) > 0 {
				if atomic.AddInt32(&scheduled, 1) >= int32(config.numPods) {
					cancel()
				}
			}
		},
	})

	// map minimum QPS entries in a counter, useful for debugging tests.
	qpsStats := map[int32]int{}

	ticker := time.NewTicker(1 * time.Second)
	go func() {
		for {
			select {
			case <-ticker.C:
				scheduled := atomic.LoadInt32(&scheduled)
				qps := scheduled - prev
				qpsStats[qps]++
				if qps < minQPS {
					minQPS = qps
				}
				fmt.Printf("%ds\trate: %d\ttotal: %d (qps frequency: %v)\n", time.Since(start)/time.Second, qps, scheduled, qpsStats)
				prev = scheduled

			case <-ctx.Done():
				return
			}
		}
	}()

	<-ctx.Done()

	ticker.Stop()

	// We will be completed when all pods are done being scheduled.
	// return the worst-case-scenario interval that was seen during this time.
	// Note this should never be low due to cold-start, so allow bake in sched time if necessary.
	consumed := int(time.Since(start) / time.Second)
	if consumed <= 0 {
		consumed = 1
	}
	fmt.Printf("Scheduled %v Pods in %v seconds (%v per second on average). min QPS was %v\n",
		config.numPods, consumed, config.numPods/consumed, minQPS)
	return minQPS
}

// mutateNodeTemplate returns the modified node needed for creation of nodes.
func (na nodeAffinity) mutateNodeTemplate(node *v1.Node) {
	labels := make(map[string]string)
	for i := 0; i < na.LabelCount; i++ {
		value := strconv.Itoa(i)
		key := na.nodeAffinityKey + value
		labels[key] = value
	}
	node.ObjectMeta.Labels = labels
	return
}

// mutatePodTemplate returns the modified pod template after applying mutations.
func (na nodeAffinity) mutatePodTemplate(pod *v1.Pod) {
	var nodeSelectorRequirements []v1.NodeSelectorRequirement
	for i := 0; i < na.LabelCount; i++ {
		value := strconv.Itoa(i)
		key := na.nodeAffinityKey + value
		nodeSelector := v1.NodeSelectorRequirement{Key: key, Values: []string{value}, Operator: v1.NodeSelectorOpIn}
		nodeSelectorRequirements = append(nodeSelectorRequirements, nodeSelector)
	}
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: nodeSelectorRequirements,
					},
				},
			},
		},
	}
}

// generateNodes generates nodes to be used for scheduling.
func (inputConfig *schedulerPerfConfig) generateNodes(config *testConfig) {
	for i := 0; i < inputConfig.NodeCount; i++ {
		config.clientset.CoreV1().Nodes().Create(context.TODO(), config.mutatedNodeTemplate, metav1.CreateOptions{})

	}
	for i := 0; i < config.numNodes-inputConfig.NodeCount; i++ {
		config.clientset.CoreV1().Nodes().Create(context.TODO(), baseNodeTemplate, metav1.CreateOptions{})
	}
}

// generatePods generates pods to be used for scheduling.
func (inputConfig *schedulerPerfConfig) generatePods(config *testConfig) {
	testutils.CreatePod(config.clientset, "sample", inputConfig.PodCount, config.mutatedPodTemplate)
	testutils.CreatePod(config.clientset, "sample", config.numPods-inputConfig.PodCount, basePodTemplate)
}

// generatePodAndNodeTopology is the wrapper function for modifying both pods and node objects.
func (inputConfig *schedulerPerfConfig) generatePodAndNodeTopology(config *testConfig) error {
	if config.numNodes < inputConfig.NodeCount || config.numPods < inputConfig.PodCount {
		return fmt.Errorf("NodeCount cannot be greater than numNodes")
	}
	nodeAffinity := inputConfig.NodeAffinity
	// Node template that needs to be mutated.
	mutatedNodeTemplate := baseNodeTemplate
	// Pod template that needs to be mutated.
	mutatedPodTemplate := basePodTemplate
	if nodeAffinity != nil {
		nodeAffinity.mutateNodeTemplate(mutatedNodeTemplate)
		nodeAffinity.mutatePodTemplate(mutatedPodTemplate)

	} // TODO: other predicates/priorities will be processed in subsequent if statements or a switch:).
	config.mutatedPodTemplate = mutatedPodTemplate
	config.mutatedNodeTemplate = mutatedNodeTemplate
	inputConfig.generateNodes(config)
	inputConfig.generatePods(config)
	return nil
}

// writePodAndNodeTopologyToConfig reads a configuration and then applies it to a test configuration.
//TODO: As of now, this function is not doing anything except for reading input values to priority structs.
func writePodAndNodeTopologyToConfig(config *testConfig) error {
	// High Level structure that should be filled for every predicate or priority.
	inputConfig := &schedulerPerfConfig{
		NodeCount: 100,
		PodCount:  3000,
		NodeAffinity: &nodeAffinity{
			nodeAffinityKey: "kubernetes.io/sched-perf-node-affinity-",
			LabelCount:      10,
		},
	}
	err := inputConfig.generatePodAndNodeTopology(config)
	if err != nil {
		return err
	}
	return nil
}
