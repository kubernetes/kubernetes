/*
Copyright 2014 The Kubernetes Authors.

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

package core

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	algorithmpredicates "k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	algorithmpriorities "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/scheduler/core/equivalence"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	errPrioritize = fmt.Errorf("priority map encounters an error")
	order         = []string{"false", "true", "matches", "nopods", algorithmpredicates.MatchInterPodAffinityPred}
)

func falsePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func truePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return true, nil, nil
}

func matchesPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if pod.Name == node.Name {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func hasNoPodsPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if len(nodeInfo.Pods()) == 0 {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func numericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	result := []schedulerapi.HostPriority{}
	for _, node := range nodes {
		score, err := strconv.Atoi(node.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, schedulerapi.HostPriority{
			Host:  node.Name,
			Score: score,
		})
	}
	return result, nil
}

func reverseNumericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []schedulerapi.HostPriority{}
	result, err := numericPriority(pod, nodeNameToInfo, nodes)
	if err != nil {
		return nil, err
	}

	for _, hostPriority := range result {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for _, hostPriority := range result {
		reverseResult = append(reverseResult, schedulerapi.HostPriority{
			Host:  hostPriority.Host,
			Score: int(maxScore + minScore - float64(hostPriority.Score)),
		})
	}

	return reverseResult, nil
}

func trueMapPriority(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	return schedulerapi.HostPriority{
		Host:  nodeInfo.Node().Name,
		Score: 1,
	}, nil
}

func falseMapPriority(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	return schedulerapi.HostPriority{}, errPrioritize
}

func getNodeReducePriority(pod *v1.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {
	for _, host := range result {
		if host.Host == "" {
			return fmt.Errorf("unexpected empty host name")
		}
	}
	return nil
}

func makeNodeList(nodeNames []string) []*v1.Node {
	result := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		result = append(result, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return result
}

func TestSelectHost(t *testing.T) {
	scheduler := genericScheduler{}
	tests := []struct {
		name          string
		list          schedulerapi.HostPriorityList
		possibleHosts sets.String
		expectsErr    bool
	}{
		{
			name: "unique properly ordered scores",
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine2.1"),
			expectsErr:    false,
		},
		{
			name: "equal scores",
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine1.2", Score: 2},
				{Host: "machine1.3", Score: 2},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine1.2", "machine1.3", "machine2.1"),
			expectsErr:    false,
		},
		{
			name: "out of order scores",
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 3},
				{Host: "machine1.2", Score: 3},
				{Host: "machine2.1", Score: 2},
				{Host: "machine3.1", Score: 1},
				{Host: "machine1.3", Score: 3},
			},
			possibleHosts: sets.NewString("machine1.1", "machine1.2", "machine1.3"),
			expectsErr:    false,
		},
		{
			name:          "empty priority list",
			list:          []schedulerapi.HostPriority{},
			possibleHosts: sets.NewString(),
			expectsErr:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// increase the randomness
			for i := 0; i < 10; i++ {
				got, err := scheduler.selectHost(test.list)
				if test.expectsErr {
					if err == nil {
						t.Error("Unexpected non-error")
					}
				} else {
					if err != nil {
						t.Errorf("Unexpected error: %v", err)
					}
					if !test.possibleHosts.Has(got) {
						t.Errorf("got %s is not in the possible map %v", got, test.possibleHosts)
					}
				}
			}
		})
	}
}

func TestGenericScheduler(t *testing.T) {
	algorithmpredicates.SetPredicatesOrdering(order)
	tests := []struct {
		name                     string
		predicates               map[string]algorithm.FitPredicate
		prioritizers             []algorithm.PriorityConfig
		alwaysCheckAllPredicates bool
		nodes                    []string
		pvcs                     []*v1.PersistentVolumeClaim
		pod                      *v1.Pod
		pods                     []*v1.Pod
		expectedHosts            sets.String
		expectsErr               bool
		wErr                     error
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:         "test 1",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FailedPredicates: FailedPredicateMap{
					"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				}},
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine2", UID: types.UID("machine2")}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectsErr:   true,
			name:         "test 7",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 3,
				FailedPredicates: FailedPredicateMap{
					"3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			predicates: map[string]algorithm.FitPredicate{
				"nopods":  hasNoPodsPredicate,
				"matches": matchesPredicate,
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")},
					Spec: v1.PodSpec{
						NodeName: "2",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"1", "2"},
			expectsErr:   true,
			name:         "test 8",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FailedPredicates: FailedPredicateMap{
					"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			// Pod with existing PVC
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			pvcs:         []*v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC"}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "existingPVC",
								},
							},
						},
					},
				},
			},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "existing PVC",
			wErr:          nil,
		},
		{
			// Pod with non existing PVC
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "unknownPVC",
								},
							},
						},
					},
				},
			},
			name:       "unknown PVC",
			expectsErr: true,
			wErr:       fmt.Errorf("persistentvolumeclaim \"unknownPVC\" not found"),
		},
		{
			// Pod with deleting PVC
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			pvcs:         []*v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", DeletionTimestamp: &metav1.Time{}}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "existingPVC",
								},
							},
						},
					},
				},
			},
			name:       "deleted PVC",
			expectsErr: true,
			wErr:       fmt.Errorf("persistentvolumeclaim \"existingPVC\" is being deleted"),
		},
		{
			// alwaysCheckAllPredicates is true
			predicates:               map[string]algorithm.FitPredicate{"true": truePredicate, "matches": matchesPredicate, "false": falsePredicate},
			prioritizers:             []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			alwaysCheckAllPredicates: true,
			nodes: []string{"1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test alwaysCheckAllPredicates is true",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 1,
				FailedPredicates: FailedPredicateMap{
					"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate, algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: falseMapPriority, Weight: 1}, {Map: trueMapPriority, Reduce: getNodeReducePriority, Weight: 2}},
			nodes:        []string{"2", "1"},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			name:         "test error with priority map",
			wErr:         errors.NewAggregate([]error{errPrioritize, errPrioritize}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := schedulercache.New(time.Duration(0), wait.NeverStop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			for _, name := range test.nodes {
				cache.AddNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
			}
			pvcs := []*v1.PersistentVolumeClaim{}
			pvcs = append(pvcs, test.pvcs...)

			pvcLister := schedulertesting.FakePersistentVolumeClaimLister(pvcs)

			scheduler := NewGenericScheduler(
				cache,
				nil,
				internalqueue.NewSchedulingQueue(),
				test.predicates,
				algorithm.EmptyPredicateMetadataProducer,
				test.prioritizers,
				algorithm.EmptyPriorityMetadataProducer,
				[]algorithm.SchedulerExtender{},
				nil,
				pvcLister,
				schedulertesting.FakePDBLister{},
				test.alwaysCheckAllPredicates,
				false,
				schedulerapi.DefaultPercentageOfNodesToScore)
			machine, err := scheduler.Schedule(test.pod, schedulertesting.FakeNodeLister(makeNodeList(test.nodes)))

			if !reflect.DeepEqual(err, test.wErr) {
				t.Errorf("Unexpected error: %v, expected: %v", err, test.wErr)
			}
			if test.expectedHosts != nil && !test.expectedHosts.Has(machine) {
				t.Errorf("Expected: %s, got: %s", test.expectedHosts, machine)
			}
		})
	}
}

// makeScheduler makes a simple genericScheduler for testing.
func makeScheduler(predicates map[string]algorithm.FitPredicate, nodes []*v1.Node) *genericScheduler {
	algorithmpredicates.SetPredicatesOrdering(order)
	cache := schedulercache.New(time.Duration(0), wait.NeverStop)
	for _, n := range nodes {
		cache.AddNode(n)
	}
	prioritizers := []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}}

	s := NewGenericScheduler(
		cache,
		nil,
		internalqueue.NewSchedulingQueue(),
		predicates,
		algorithm.EmptyPredicateMetadataProducer,
		prioritizers,
		algorithm.EmptyPriorityMetadataProducer,
		nil, nil, nil, nil, false, false,
		schedulerapi.DefaultPercentageOfNodesToScore)
	cache.UpdateNodeNameToInfoMap(s.(*genericScheduler).cachedNodeInfoMap)
	return s.(*genericScheduler)

}

func TestFindFitAllError(t *testing.T) {
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "matches": matchesPredicate}
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(predicates, nodes)

	_, predicateMap, err := scheduler.findNodesThatFit(&v1.Pod{}, nodes)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != len(nodes) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		t.Run(node.Name, func(t *testing.T) {
			failures, found := predicateMap[node.Name]
			if !found {
				t.Errorf("failed to find node in %v", predicateMap)
			}
			if len(failures) != 1 || failures[0] != algorithmpredicates.ErrFakePredicate {
				t.Errorf("unexpected failures: %v", failures)
			}
		})
	}
}

func TestFindFitSomeError(t *testing.T) {
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "matches": matchesPredicate}
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(predicates, nodes)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1", UID: types.UID("1")}}
	_, predicateMap, err := scheduler.findNodesThatFit(pod, nodes)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != (len(nodes) - 1) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		if node.Name == pod.Name {
			continue
		}
		t.Run(node.Name, func(t *testing.T) {
			failures, found := predicateMap[node.Name]
			if !found {
				t.Errorf("failed to find node in %v", predicateMap)
			}
			if len(failures) != 1 || failures[0] != algorithmpredicates.ErrFakePredicate {
				t.Errorf("unexpected failures: %v", failures)
			}
		})
	}
}

func makeNode(node string, milliCPU, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
			Allocatable: v1.ResourceList{

				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
		},
	}
}

func TestHumanReadableFitError(t *testing.T) {
	err := &FitError{
		Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
		NumAllNodes: 3,
		FailedPredicates: FailedPredicateMap{
			"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderMemoryPressure},
			"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
			"3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
		},
	}
	if strings.Contains(err.Error(), "0/3 nodes are available") {
		if strings.Contains(err.Error(), "2 node(s) had disk pressure") && strings.Contains(err.Error(), "1 node(s) had memory pressure") {
			return
		}
	}
	t.Errorf("Error message doesn't have all the information content: [" + err.Error() + "]")
}

// The point of this test is to show that you:
// - get the same priority for a zero-request pod as for a pod with the defaults requests,
//   both when the zero-request pod is already on the machine and when the zero-request pod
//   is the one being scheduled.
// - don't get the same score no matter what we schedule.
func TestZeroRequest(t *testing.T) {
	// A pod with no resources. We expect spreading to count it as having the default resources.
	noResources := v1.PodSpec{
		Containers: []v1.Container{
			{},
		},
	}
	noResources1 := noResources
	noResources1.NodeName = "machine1"
	// A pod with the same resources as a 0-request pod gets by default as its resources (for spreading).
	small := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCPURequest, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
					},
				},
			},
		},
	}
	small2 := small
	small2.NodeName = "machine2"
	// A larger pod.
	large := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCPURequest*3, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
					},
				},
			},
		},
	}
	large1 := large
	large1.NodeName = "machine1"
	large2 := large
	large2.NodeName = "machine2"
	tests := []struct {
		pod           *v1.Pod
		pods          []*v1.Pod
		nodes         []*v1.Node
		name          string
		expectedScore int
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request pod
		// as for a pod with the defaults requests, both when the zero-request pod is already on the machine
		// and when the zero-request pod is the one being scheduled.
		{
			pod:   &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			name:  "test priority of zero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 25,
		},
		{
			pod:   &v1.Pod{Spec: small},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			name:  "test priority of nonzero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 25,
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			pod:   &v1.Pod{Spec: large},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			name:  "test priority of larger pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 23,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// This should match the configuration in defaultPriorities() in
			// pkg/scheduler/algorithmprovider/defaults/defaults.go if you want
			// to test what's actually in production.
			priorityConfigs := []algorithm.PriorityConfig{
				{Map: algorithmpriorities.LeastRequestedPriorityMap, Weight: 1},
				{Map: algorithmpriorities.BalancedResourceAllocationMap, Weight: 1},
			}
			selectorSpreadPriorityMap, selectorSpreadPriorityReduce := algorithmpriorities.NewSelectorSpreadPriority(
				schedulertesting.FakeServiceLister([]*v1.Service{}),
				schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
				schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
				schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))
			pc := algorithm.PriorityConfig{Map: selectorSpreadPriorityMap, Reduce: selectorSpreadPriorityReduce, Weight: 1}
			priorityConfigs = append(priorityConfigs, pc)

			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)

			mataDataProducer := algorithmpriorities.NewPriorityMetadataFactory(
				schedulertesting.FakeServiceLister([]*v1.Service{}),
				schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
				schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
				schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))
			mataData := mataDataProducer(test.pod, nodeNameToInfo)

			list, err := PrioritizeNodes(
				test.pod, nodeNameToInfo, mataData, priorityConfigs,
				schedulertesting.FakeNodeLister(test.nodes), []algorithm.SchedulerExtender{})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, hp := range list {
				if hp.Score != test.expectedScore {
					t.Errorf("expected %d for all priorities, got list %#v", test.expectedScore, list)
				}
			}
		})
	}
}

func printNodeToVictims(nodeToVictims map[*v1.Node]*schedulerapi.Victims) string {
	var output string
	for node, victims := range nodeToVictims {
		output += node.Name + ": ["
		for _, pod := range victims.Pods {
			output += pod.Name + ", "
		}
		output += "]"
	}
	return output
}

func checkPreemptionVictims(expected map[string]map[string]bool, nodeToPods map[*v1.Node]*schedulerapi.Victims) error {
	if len(expected) == len(nodeToPods) {
		for k, victims := range nodeToPods {
			if expPods, ok := expected[k.Name]; ok {
				if len(victims.Pods) != len(expPods) {
					return fmt.Errorf("unexpected number of pods. expected: %v, got: %v", expected, printNodeToVictims(nodeToPods))
				}
				prevPriority := int32(math.MaxInt32)
				for _, p := range victims.Pods {
					// Check that pods are sorted by their priority.
					if *p.Spec.Priority > prevPriority {
						return fmt.Errorf("pod %v of node %v was not sorted by priority", p.Name, k)
					}
					prevPriority = *p.Spec.Priority
					if _, ok := expPods[p.Name]; !ok {
						return fmt.Errorf("pod %v was not expected. Expected: %v", p.Name, expPods)
					}
				}
			} else {
				return fmt.Errorf("unexpected machines. expected: %v, got: %v", expected, printNodeToVictims(nodeToPods))
			}
		}
	} else {
		return fmt.Errorf("unexpected number of machines. expected: %v, got: %v", expected, printNodeToVictims(nodeToPods))
	}
	return nil
}

type FakeNodeInfo v1.Node

func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	node := v1.Node(n)
	return &node, nil
}

func PredicateMetadata(p *v1.Pod, nodeInfo map[string]*schedulercache.NodeInfo) algorithm.PredicateMetadata {
	return algorithmpredicates.NewPredicateMetadataFactory(schedulertesting.FakePodLister{p})(p, nodeInfo)
}

var smallContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCPURequest, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
			},
		},
	},
}
var mediumContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCPURequest*2, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*2, 10)),
			},
		},
	},
}
var largeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCPURequest*3, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
			},
		},
	},
}
var veryLargeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCPURequest*5, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*5, 10)),
			},
		},
	},
}
var negPriority, lowPriority, midPriority, highPriority, veryHighPriority = int32(-100), int32(0), int32(100), int32(1000), int32(10000)

// TestSelectNodesForPreemption tests selectNodesForPreemption. This test assumes
// that podsFitsOnNode works correctly and is tested separately.
func TestSelectNodesForPreemption(t *testing.T) {
	algorithmpredicates.SetPredicatesOrdering(order)
	tests := []struct {
		name                 string
		predicates           map[string]algorithm.FitPredicate
		nodes                []string
		pod                  *v1.Pod
		pods                 []*v1.Pod
		expected             map[string]map[string]bool // Map from node name to a list of pods names which should be preempted.
		addAffinityPredicate bool
	}{
		{
			name:       "a pod that does not fit on any machine",
			predicates: map[string]algorithm.FitPredicate{"matches": falsePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "a pod that fits with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": truePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}, "machine2": {}},
		},
		{
			name:       "a pod that fits on one machine with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"a": true}, "machine2": {"b": true}},
		},
		{
			name:       "a pod that would fit on the machines, but other pods running are higher priority",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &lowPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "medium priority pod is preempted, but lower priority one stays as it is small",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"b": true}, "machine2": {"c": true}},
		},
		{
			name:       "mixed priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d", UID: types.UID("d")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e", UID: types.UID("e")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"b": true, "c": true}},
		},
		{
			name:       "pod with anti-affinity is preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name:   "machine1",
				Labels: map[string]string{"pod": "preemptor"}}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a"), Labels: map[string]string{"service": "securityscan"}}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1", Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"preemptor", "value2"},
										},
									},
								},
								TopologyKey: "hostname",
							},
						},
					}}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d", UID: types.UID("d")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e", UID: types.UID("e")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}}},
			expected:             map[string]map[string]bool{"machine1": {"a": true}, "machine2": {}},
			addAffinityPredicate: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := []*v1.Node{}
			for _, n := range test.nodes {
				node := makeNode(n, 1000*5, priorityutil.DefaultMemoryRequest*5)
				node.ObjectMeta.Labels = map[string]string{"hostname": node.Name}
				nodes = append(nodes, node)
			}
			if test.addAffinityPredicate {
				test.predicates[algorithmpredicates.MatchInterPodAffinityPred] = algorithmpredicates.NewPodAffinityPredicate(FakeNodeInfo(*nodes[0]), schedulertesting.FakePodLister(test.pods))
			}
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nodes)
			nodeToPods, err := selectNodesForPreemption(test.pod, nodeNameToInfo, nodes, test.predicates, PredicateMetadata, nil, nil)
			if err != nil {
				t.Error(err)
			}
			if err := checkPreemptionVictims(test.expected, nodeToPods); err != nil {
				t.Error(err)
			}
		})
	}
}

// TestPickOneNodeForPreemption tests pickOneNodeForPreemption.
func TestPickOneNodeForPreemption(t *testing.T) {
	algorithmpredicates.SetPredicatesOrdering(order)
	tests := []struct {
		name       string
		predicates map[string]algorithm.FitPredicate
		nodes      []string
		pod        *v1.Pod
		pods       []*v1.Pod
		expected   []string // any of the items is valid
	}{
		{
			name:       "No node needs preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}}},
			expected: []string{"machine1"},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: []string{"machine1", "machine2"},
		},
		{
			name:       "a pod that fits on a machine with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: []string{"machine3"},
		},
		{
			name:       "machine with min highest priority pod is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine3"},
		},
		{
			name:       "when highest priorities are the same, minimum sum of priorities is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "when highest priority and sum are the same, minimum number of pods is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.4", UID: types.UID("m1.4")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			// pickOneNodeForPreemption adjusts pod priorities when finding the sum of the victims. This
			// test ensures that the logic works correctly.
			name:       "sum of adjusted priorities is considered",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "non-overlapping lowest high priority, sum priorities, and number of pods",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3", "machine4"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &veryHighPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.4", UID: types.UID("m3.4")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m4.1", UID: types.UID("m4.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.2", UID: types.UID("m4.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.3", UID: types.UID("m4.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.4", UID: types.UID("m4.4")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine4"}},
			},
			expected: []string{"machine1"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := []*v1.Node{}
			for _, n := range test.nodes {
				nodes = append(nodes, makeNode(n, priorityutil.DefaultMilliCPURequest*5, priorityutil.DefaultMemoryRequest*5))
			}
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nodes)
			candidateNodes, _ := selectNodesForPreemption(test.pod, nodeNameToInfo, nodes, test.predicates, PredicateMetadata, nil, nil)
			node := pickOneNodeForPreemption(candidateNodes)
			found := false
			for _, nodeName := range test.expected {
				if node.Name == nodeName {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("unexpected node: %v", node)
			}
		})
	}
}

func TestNodesWherePreemptionMightHelp(t *testing.T) {
	// Prepare 4 node names.
	nodeNames := []string{}
	for i := 1; i < 5; i++ {
		nodeNames = append(nodeNames, fmt.Sprintf("machine%d", i))
	}

	tests := []struct {
		name          string
		failedPredMap FailedPredicateMap
		expected      map[string]bool // set of expected node names. Value is ignored.
	}{
		{
			name: "No node should be attempted",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeSelectorNotMatch},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrTaintsTolerationsNotMatch},
				"machine4": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeLabelPresenceViolated},
			},
			expected: map[string]bool{},
		},
		{
			name: "ErrPodAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnschedulable},
			},
			expected: map[string]bool{"machine1": true, "machine4": true},
		},
		{
			name: "pod with both pod affinity and anti-affinity should be tried",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
			},
			expected: map[string]bool{"machine1": true, "machine3": true, "machine4": true},
		},
		{
			name: "ErrPodAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrPodAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodAffinityRulesNotMatch},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
			},
			expected: map[string]bool{"machine2": true, "machine3": true, "machine4": true},
		},
		{
			name: "Mix of failed predicates works fine",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeSelectorNotMatch, algorithmpredicates.ErrNodeOutOfDisk, algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName, algorithmpredicates.ErrDiskConflict},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
				"machine4": []algorithm.PredicateFailureReason{},
			},
			expected: map[string]bool{"machine3": true, "machine4": true},
		},
		{
			name: "Node condition errors should be considered unresolvable",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderPIDPressure},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderMemoryPressure},
				"machine4": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeOutOfDisk},
			},
			expected: map[string]bool{},
		},
		{
			name: "Node condition errors and ErrNodeUnknownCondition should be considered unresolvable",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeNotReady},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeNetworkUnavailable},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnknownCondition},
			},
			expected: map[string]bool{"machine4": true},
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrVolumeZoneConflict},
				"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrVolumeNodeConflict},
				"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrVolumeBindConflict},
			},
			expected: map[string]bool{"machine4": true},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := nodesWherePreemptionMightHelp(makeNodeList(nodeNames), test.failedPredMap)
			if len(test.expected) != len(nodes) {
				t.Errorf("number of nodes is not the same as expected. exptectd: %d, got: %d. Nodes: %v", len(test.expected), len(nodes), nodes)
			}
			for _, node := range nodes {
				if _, found := test.expected[node.Name]; !found {
					t.Errorf("node %v is not expected.", node.Name)
				}
			}
		})
	}
}

func TestPreempt(t *testing.T) {
	failedPredMap := FailedPredicateMap{
		"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
		"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrDiskConflict},
		"machine3": []algorithm.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
	}
	// Prepare 3 node names.
	nodeNames := []string{}
	for i := 1; i < 4; i++ {
		nodeNames = append(nodeNames, fmt.Sprintf("machine%d", i))
	}
	tests := []struct {
		name         string
		pod          *v1.Pod
		pods         []*v1.Pod
		extenders    []*FakeExtender
		expectedNode string
		expectedPods []string // list of preempted pods
	}{
		{
			name: "basic preemption logic",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "One node doesn't need any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			expectedNode: "machine3",
			expectedPods: []string{},
		},
		{
			name: "Scheduler extenders allow only machine1, otherwise machine3 would have been chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "Scheduler extenders do not allow any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{falsePredicateExtender},
				},
			},
			expectedNode: "",
			expectedPods: []string{},
		},
		{
			name: "One scheduler extender allows only machine1, the other returns error but ignorable. Only machine1 would be chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{errorPredicateExtender},
					ignorable:  true,
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "One scheduler extender allows only machine1, but it is not interested in given pod, otherwise machine1 would have been chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates:   []fitPredicate{machine1PredicateExtender},
					unInterested: true,
				},
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
			},
			expectedNode: "machine3",
			expectedPods: []string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stop := make(chan struct{})
			cache := schedulercache.New(time.Duration(0), stop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			cachedNodeInfoMap := map[string]*schedulercache.NodeInfo{}
			for _, name := range nodeNames {
				node := makeNode(name, 1000*5, priorityutil.DefaultMemoryRequest*5)
				cache.AddNode(node)

				// Set nodeInfo to extenders to mock extenders' cache for preemption.
				cachedNodeInfo := schedulercache.NewNodeInfo()
				cachedNodeInfo.SetNode(node)
				cachedNodeInfoMap[name] = cachedNodeInfo
			}
			extenders := []algorithm.SchedulerExtender{}
			for _, extender := range test.extenders {
				// Set nodeInfoMap as extenders cached node information.
				extender.cachedNodeNameToInfo = cachedNodeInfoMap
				extenders = append(extenders, extender)
			}
			scheduler := NewGenericScheduler(
				cache,
				nil,
				internalqueue.NewSchedulingQueue(),
				map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
				algorithm.EmptyPredicateMetadataProducer,
				[]algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
				algorithm.EmptyPriorityMetadataProducer,
				extenders,
				nil,
				schedulertesting.FakePersistentVolumeClaimLister{},
				schedulertesting.FakePDBLister{},
				false,
				false,
				schedulerapi.DefaultPercentageOfNodesToScore)
			// Call Preempt and check the expected results.
			node, victims, _, err := scheduler.Preempt(test.pod, schedulertesting.FakeNodeLister(makeNodeList(nodeNames)), error(&FitError{Pod: test.pod, FailedPredicates: failedPredMap}))
			if err != nil {
				t.Errorf("unexpected error in preemption: %v", err)
			}
			if (node != nil && node.Name != test.expectedNode) || (node == nil && len(test.expectedNode) != 0) {
				t.Errorf("expected node: %v, got: %v", test.expectedNode, node.GetName())
			}
			if len(victims) != len(test.expectedPods) {
				t.Errorf("expected %v pods, got %v.", len(test.expectedPods), len(victims))
			}
			for _, victim := range victims {
				found := false
				for _, expPod := range test.expectedPods {
					if expPod == victim.Name {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("pod %v is not expected to be a victim.", victim.Name)
				}
				// Mark the victims for deletion and record the preemptor's nominated node name.
				now := metav1.Now()
				victim.DeletionTimestamp = &now
				test.pod.Status.NominatedNodeName = node.Name
			}
			// Call preempt again and make sure it doesn't preempt any more pods.
			node, victims, _, err = scheduler.Preempt(test.pod, schedulertesting.FakeNodeLister(makeNodeList(nodeNames)), error(&FitError{Pod: test.pod, FailedPredicates: failedPredMap}))
			if err != nil {
				t.Errorf("unexpected error in preemption: %v", err)
			}
			if node != nil && len(victims) > 0 {
				t.Errorf("didn't expect any more preemption. Node %v is selected for preemption.", node)
			}
			close(stop)
		})
	}
}

// syncingMockCache delegates method calls to an actual Cache,
// but calls to UpdateNodeNameToInfoMap synchronize with the test.
type syncingMockCache struct {
	schedulercache.Cache
	cycleStart, cacheInvalidated chan struct{}
	once                         sync.Once
}

// UpdateNodeNameToInfoMap delegates to the real implementation, but on the first call, it
// synchronizes with the test.
//
// Since UpdateNodeNameToInfoMap is one of the first steps of (*genericScheduler).Schedule, we use
// this point to signal to the test that a scheduling cycle has started.
func (c *syncingMockCache) UpdateNodeNameToInfoMap(infoMap map[string]*schedulercache.NodeInfo) error {
	err := c.Cache.UpdateNodeNameToInfoMap(infoMap)
	c.once.Do(func() {
		c.cycleStart <- struct{}{}
		<-c.cacheInvalidated
	})
	return err
}

// TestCacheInvalidationRace tests that equivalence cache invalidation is correctly
// handled when an invalidation event happens early in a scheduling cycle. Specifically, the event
// occurs after schedulercache is snapshotted and before equivalence cache lock is acquired.
func TestCacheInvalidationRace(t *testing.T) {
	// Create a predicate that returns false the first time and true on subsequent calls.
	podWillFit := false
	var callCount int
	testPredicate := func(pod *v1.Pod,
		meta algorithm.PredicateMetadata,
		nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
		callCount++
		if !podWillFit {
			podWillFit = true
			return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
		}
		return true, nil, nil
	}

	// Set up the mock cache.
	cache := schedulercache.New(time.Duration(0), wait.NeverStop)
	testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
	cache.AddNode(testNode)
	mockCache := &syncingMockCache{
		Cache:            cache,
		cycleStart:       make(chan struct{}),
		cacheInvalidated: make(chan struct{}),
	}

	ps := map[string]algorithm.FitPredicate{"testPredicate": testPredicate}
	algorithmpredicates.SetPredicatesOrdering([]string{"testPredicate"})
	eCache := equivalence.NewCache(algorithmpredicates.Ordering())
	eCache.GetNodeCache(testNode.Name)
	// Ensure that equivalence cache invalidation happens after the scheduling cycle starts, but before
	// the equivalence cache would be updated.
	go func() {
		<-mockCache.cycleStart
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "new-pod", UID: "new-pod"},
			Spec:       v1.PodSpec{NodeName: "machine1"}}
		if err := cache.AddPod(pod); err != nil {
			t.Errorf("Could not add pod to cache: %v", err)
		}
		eCache.InvalidateAllPredicatesOnNode("machine1")
		mockCache.cacheInvalidated <- struct{}{}
	}()

	// Set up the scheduler.
	prioritizers := []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}}
	pvcLister := schedulertesting.FakePersistentVolumeClaimLister([]*v1.PersistentVolumeClaim{})
	pdbLister := schedulertesting.FakePDBLister{}
	scheduler := NewGenericScheduler(
		mockCache,
		eCache,
		internalqueue.NewSchedulingQueue(),
		ps,
		algorithm.EmptyPredicateMetadataProducer,
		prioritizers,
		algorithm.EmptyPriorityMetadataProducer,
		nil, nil, pvcLister, pdbLister,
		true, false,
		schedulerapi.DefaultPercentageOfNodesToScore)

	// First scheduling attempt should fail.
	nodeLister := schedulertesting.FakeNodeLister(makeNodeList([]string{"machine1"}))
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}}
	machine, err := scheduler.Schedule(pod, nodeLister)
	if machine != "" || err == nil {
		t.Error("First scheduling attempt did not fail")
	}

	// Second scheduling attempt should succeed because cache was invalidated.
	_, err = scheduler.Schedule(pod, nodeLister)
	if err != nil {
		t.Errorf("Second scheduling attempt failed: %v", err)
	}
	if callCount != 2 {
		t.Errorf("Predicate should have been called twice. Was called %d times.", callCount)
	}
}

// TestCacheInvalidationRace2 tests that cache invalidation is correctly handled
// when an invalidation event happens while a predicate is running.
func TestCacheInvalidationRace2(t *testing.T) {
	// Create a predicate that returns false the first time and true on subsequent calls.
	var (
		podWillFit       = false
		callCount        int
		cycleStart       = make(chan struct{})
		cacheInvalidated = make(chan struct{})
		once             sync.Once
	)
	testPredicate := func(pod *v1.Pod,
		meta algorithm.PredicateMetadata,
		nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
		callCount++
		once.Do(func() {
			cycleStart <- struct{}{}
			<-cacheInvalidated
		})
		if !podWillFit {
			podWillFit = true
			return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
		}
		return true, nil, nil
	}

	// Set up the mock cache.
	cache := schedulercache.New(time.Duration(0), wait.NeverStop)
	testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}}
	cache.AddNode(testNode)

	ps := map[string]algorithm.FitPredicate{"testPredicate": testPredicate}
	algorithmpredicates.SetPredicatesOrdering([]string{"testPredicate"})
	eCache := equivalence.NewCache(algorithmpredicates.Ordering())
	eCache.GetNodeCache(testNode.Name)
	// Ensure that equivalence cache invalidation happens after the scheduling cycle starts, but before
	// the equivalence cache would be updated.
	go func() {
		<-cycleStart
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "new-pod", UID: "new-pod"},
			Spec:       v1.PodSpec{NodeName: "machine1"}}
		if err := cache.AddPod(pod); err != nil {
			t.Errorf("Could not add pod to cache: %v", err)
		}
		eCache.InvalidateAllPredicatesOnNode("machine1")
		cacheInvalidated <- struct{}{}
	}()

	// Set up the scheduler.
	prioritizers := []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}}
	pvcLister := schedulertesting.FakePersistentVolumeClaimLister([]*v1.PersistentVolumeClaim{})
	pdbLister := schedulertesting.FakePDBLister{}
	scheduler := NewGenericScheduler(
		cache,
		eCache,
		internalqueue.NewSchedulingQueue(),
		ps,
		algorithm.EmptyPredicateMetadataProducer,
		prioritizers,
		algorithm.EmptyPriorityMetadataProducer,
		nil, nil, pvcLister, pdbLister, true, false,
		schedulerapi.DefaultPercentageOfNodesToScore)

	// First scheduling attempt should fail.
	nodeLister := schedulertesting.FakeNodeLister(makeNodeList([]string{"machine1"}))
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}}
	machine, err := scheduler.Schedule(pod, nodeLister)
	if machine != "" || err == nil {
		t.Error("First scheduling attempt did not fail")
	}

	// Second scheduling attempt should succeed because cache was invalidated.
	_, err = scheduler.Schedule(pod, nodeLister)
	if err != nil {
		t.Errorf("Second scheduling attempt failed: %v", err)
	}
	if callCount != 2 {
		t.Errorf("Predicate should have been called twice. Was called %d times.", callCount)
	}
}
