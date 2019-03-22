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
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulerinternalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	errPrioritize = fmt.Errorf("priority map encounters an error")
	order         = []string{"false", "true", "matches", "nopods", algorithmpredicates.MatchInterPodAffinityPred}
)

func falsePredicate(pod *v1.Pod, meta algorithmpredicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []algorithmpredicates.PredicateFailureReason, error) {
	return false, []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func truePredicate(pod *v1.Pod, meta algorithmpredicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []algorithmpredicates.PredicateFailureReason, error) {
	return true, nil, nil
}

func matchesPredicate(pod *v1.Pod, meta algorithmpredicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []algorithmpredicates.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if pod.Name == node.Name {
		return true, nil, nil
	}
	return false, []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func hasNoPodsPredicate(pod *v1.Pod, meta algorithmpredicates.PredicateMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []algorithmpredicates.PredicateFailureReason, error) {
	if len(nodeInfo.Pods()) == 0 {
		return true, nil, nil
	}
	return false, []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func numericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
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

func reverseNumericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
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

func trueMapPriority(pod *v1.Pod, meta interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (schedulerapi.HostPriority, error) {
	return schedulerapi.HostPriority{
		Host:  nodeInfo.Node().Name,
		Score: 1,
	}, nil
}

func falseMapPriority(pod *v1.Pod, meta interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (schedulerapi.HostPriority, error) {
	return schedulerapi.HostPriority{}, errPrioritize
}

func getNodeReducePriority(pod *v1.Pod, meta interface{}, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, result schedulerapi.HostPriorityList) error {
	for _, host := range result {
		if host.Host == "" {
			return fmt.Errorf("unexpected empty host name")
		}
	}
	return nil
}

// EmptyPluginSet is a test plugin set used by the default scheduler.
type EmptyPluginSet struct{}

var _ plugins.PluginSet = EmptyPluginSet{}

// ReservePlugins returns a slice of default reserve plugins.
func (r EmptyPluginSet) ReservePlugins() []plugins.ReservePlugin {
	return []plugins.ReservePlugin{}
}

// PrebindPlugins returns a slice of default prebind plugins.
func (r EmptyPluginSet) PrebindPlugins() []plugins.PrebindPlugin {
	return []plugins.PrebindPlugin{}
}

// Data returns a pointer to PluginData.
func (r EmptyPluginSet) Data() *plugins.PluginData {
	return &plugins.PluginData{}
}

var emptyPluginSet = &EmptyPluginSet{}

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
		predicates               map[string]algorithmpredicates.FitPredicate
		prioritizers             []priorities.PriorityConfig
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
			predicates:   map[string]algorithmpredicates.FitPredicate{"false": falsePredicate},
			prioritizers: []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:         "test 1",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FailedPredicates: FailedPredicateMap{
					"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				}},
		},
		{
			predicates:    map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers:  []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:    map[string]algorithmpredicates.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine2", UID: types.UID("machine2")}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers:  []priorities.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithmpredicates.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []priorities.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers:  []priorities.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			predicates:   map[string]algorithmpredicates.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []priorities.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectsErr:   true,
			name:         "test 7",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 3,
				FailedPredicates: FailedPredicateMap{
					"3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			predicates: map[string]algorithmpredicates.FitPredicate{
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
			prioritizers: []priorities.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"1", "2"},
			expectsErr:   true,
			name:         "test 8",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FailedPredicates: FailedPredicateMap{
					"1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			// Pod with existing PVC
			predicates:   map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers: []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:   map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers: []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:   map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers: []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:               map[string]algorithmpredicates.FitPredicate{"true": truePredicate, "matches": matchesPredicate, "false": falsePredicate},
			prioritizers:             []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			alwaysCheckAllPredicates: true,
			nodes:                    []string{"1"},
			pod:                      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:                     "test alwaysCheckAllPredicates is true",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 1,
				FailedPredicates: FailedPredicateMap{
					"1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrFakePredicate, algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			predicates:   map[string]algorithmpredicates.FitPredicate{"true": truePredicate},
			prioritizers: []priorities.PriorityConfig{{Map: falseMapPriority, Weight: 1}, {Map: trueMapPriority, Reduce: getNodeReducePriority, Weight: 2}},
			nodes:        []string{"2", "1"},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			name:         "test error with priority map",
			wErr:         errors.NewAggregate([]error{errPrioritize, errPrioritize}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := schedulerinternalcache.New(time.Duration(0), wait.NeverStop)
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
				internalqueue.NewSchedulingQueue(nil),
				test.predicates,
				algorithmpredicates.EmptyPredicateMetadataProducer,
				test.prioritizers,
				priorities.EmptyPriorityMetadataProducer,
				emptyPluginSet,
				[]algorithm.SchedulerExtender{},
				nil,
				pvcLister,
				schedulertesting.FakePDBLister{},
				test.alwaysCheckAllPredicates,
				false,
				schedulerapi.DefaultPercentageOfNodesToScore)
			result, err := scheduler.Schedule(test.pod, schedulertesting.FakeNodeLister(makeNodeList(test.nodes)))

			if !reflect.DeepEqual(err, test.wErr) {
				t.Errorf("Unexpected error: %v, expected: %v", err, test.wErr)
			}
			if test.expectedHosts != nil && !test.expectedHosts.Has(result.SuggestedHost) {
				t.Errorf("Expected: %s, got: %s", test.expectedHosts, result.SuggestedHost)
			}
		})
	}
}

// makeScheduler makes a simple genericScheduler for testing.
func makeScheduler(predicates map[string]algorithmpredicates.FitPredicate, nodes []*v1.Node) *genericScheduler {
	algorithmpredicates.SetPredicatesOrdering(order)
	cache := schedulerinternalcache.New(time.Duration(0), wait.NeverStop)
	for _, n := range nodes {
		cache.AddNode(n)
	}
	prioritizers := []priorities.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}}

	s := NewGenericScheduler(
		cache,
		internalqueue.NewSchedulingQueue(nil),
		predicates,
		algorithmpredicates.EmptyPredicateMetadataProducer,
		prioritizers,
		priorities.EmptyPriorityMetadataProducer,
		emptyPluginSet,
		nil, nil, nil, nil, false, false,
		schedulerapi.DefaultPercentageOfNodesToScore)
	cache.UpdateNodeInfoSnapshot(&s.(*genericScheduler).nodeInfoSnapshot)
	return s.(*genericScheduler)

}

func TestFindFitAllError(t *testing.T) {
	predicates := map[string]algorithmpredicates.FitPredicate{"true": truePredicate, "matches": matchesPredicate}
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
	predicates := map[string]algorithmpredicates.FitPredicate{"true": truePredicate, "matches": matchesPredicate}
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
			"1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderMemoryPressure},
			"2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
			"3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
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
			priorityConfigs := []priorities.PriorityConfig{
				{Map: priorities.LeastRequestedPriorityMap, Weight: 1},
				{Map: priorities.BalancedResourceAllocationMap, Weight: 1},
			}
			selectorSpreadPriorityMap, selectorSpreadPriorityReduce := priorities.NewSelectorSpreadPriority(
				schedulertesting.FakeServiceLister([]*v1.Service{}),
				schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
				schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
				schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))
			pc := priorities.PriorityConfig{Map: selectorSpreadPriorityMap, Reduce: selectorSpreadPriorityReduce, Weight: 1}
			priorityConfigs = append(priorityConfigs, pc)

			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, test.nodes)

			metaDataProducer := priorities.NewPriorityMetadataFactory(
				schedulertesting.FakeServiceLister([]*v1.Service{}),
				schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
				schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
				schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))
			metaData := metaDataProducer(test.pod, nodeNameToInfo)

			list, err := PrioritizeNodes(
				test.pod, nodeNameToInfo, metaData, priorityConfigs,
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

func PredicateMetadata(p *v1.Pod, nodeInfo map[string]*schedulernodeinfo.NodeInfo) algorithmpredicates.PredicateMetadata {
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

var startTime = metav1.Date(2019, 1, 1, 1, 1, 1, 0, time.UTC)

var startTime20190102 = metav1.Date(2019, 1, 2, 1, 1, 1, 0, time.UTC)
var startTime20190103 = metav1.Date(2019, 1, 3, 1, 1, 1, 0, time.UTC)
var startTime20190104 = metav1.Date(2019, 1, 4, 1, 1, 1, 0, time.UTC)
var startTime20190105 = metav1.Date(2019, 1, 5, 1, 1, 1, 0, time.UTC)
var startTime20190106 = metav1.Date(2019, 1, 6, 1, 1, 1, 0, time.UTC)
var startTime20190107 = metav1.Date(2019, 1, 7, 1, 1, 1, 0, time.UTC)

// TestSelectNodesForPreemption tests selectNodesForPreemption. This test assumes
// that podsFitsOnNode works correctly and is tested separately.
func TestSelectNodesForPreemption(t *testing.T) {
	algorithmpredicates.SetPredicatesOrdering(order)
	tests := []struct {
		name                 string
		predicates           map[string]algorithmpredicates.FitPredicate
		nodes                []string
		pod                  *v1.Pod
		pods                 []*v1.Pod
		expected             map[string]map[string]bool // Map from node name to a list of pods names which should be preempted.
		addAffinityPredicate bool
	}{
		{
			name:       "a pod that does not fit on any machine",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": falsePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "a pod that fits with no preemption",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": truePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}, "machine2": {}},
		},
		{
			name:       "a pod that fits on one machine with no preemption",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": matchesPredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"a": true}, "machine2": {"b": true}},
		},
		{
			name:       "a pod that would fit on the machines, but other pods running are higher priority",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &lowPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "medium priority pod is preempted, but lower priority one stays as it is small",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
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
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
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
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
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
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, nodes)
			// newnode simulate a case that a new node is added to the cluster, but nodeNameToInfo
			// doesn't have it yet.
			newnode := makeNode("newnode", 1000*5, priorityutil.DefaultMemoryRequest*5)
			newnode.ObjectMeta.Labels = map[string]string{"hostname": "newnode"}
			nodes = append(nodes, newnode)
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
		predicates map[string]algorithmpredicates.FitPredicate
		nodes      []string
		pod        *v1.Pod
		pods       []*v1.Pod
		expected   []string // any of the items is valid
	}{
		{
			name:       "No node needs preemption",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine1"},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine1", "machine2"},
		},
		{
			name:       "a pod that fits on a machine with no preemption",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine3"},
		},
		{
			name:       "machine with min highest priority pod is picked",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
			},
			expected: []string{"machine3"},
		},
		{
			name:       "when highest priorities are the same, minimum sum of priorities is picked",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "when highest priority and sum are the same, minimum number of pods is picked",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.4", UID: types.UID("m1.4")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
			},
			expected: []string{"machine2"},
		},
		{
			// pickOneNodeForPreemption adjusts pod priorities when finding the sum of the victims. This
			// test ensures that the logic works correctly.
			name:       "sum of adjusted priorities is considered",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "non-overlapping lowest high priority, sum priorities, and number of pods",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3", "machine4"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &veryHighPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3", UID: types.UID("m1.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3", UID: types.UID("m3.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.4", UID: types.UID("m3.4")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m4.1", UID: types.UID("m4.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine4"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.2", UID: types.UID("m4.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.3", UID: types.UID("m4.3")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}, Status: v1.PodStatus{StartTime: &startTime}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.4", UID: types.UID("m4.4")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine4"}, Status: v1.PodStatus{StartTime: &startTime}},
			},
			expected: []string{"machine1"},
		},
		{
			name:       "same priority, same number of victims, different start time for each machine's pod",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190103}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190103}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190104}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190104}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190102}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190102}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "same priority, same number of victims, different start time for all pods",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190105}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190103}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190106}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190102}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190104}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190107}},
			},
			expected: []string{"machine3"},
		},
		{
			name:       "different priority, same number of victims, different start time for all pods",
			predicates: map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190105}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190103}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190107}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2", UID: types.UID("m2.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190102}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190104}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2", UID: types.UID("m3.2")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{StartTime: &startTime20190106}},
			},
			expected: []string{"machine2"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := []*v1.Node{}
			for _, n := range test.nodes {
				nodes = append(nodes, makeNode(n, priorityutil.DefaultMilliCPURequest*5, priorityutil.DefaultMemoryRequest*5))
			}
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, nodes)
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
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeSelectorNotMatch},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrTaintsTolerationsNotMatch},
				"machine4": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeLabelPresenceViolated},
			},
			expected: map[string]bool{},
		},
		{
			name: "ErrPodAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnschedulable},
			},
			expected: map[string]bool{"machine1": true, "machine4": true},
		},
		{
			name: "pod with both pod affinity and anti-affinity should be tried",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName},
			},
			expected: map[string]bool{"machine1": true, "machine3": true, "machine4": true},
		},
		{
			name: "ErrPodAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrPodAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodAffinityRulesNotMatch},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodAffinityNotMatch},
			},
			expected: map[string]bool{"machine2": true, "machine3": true, "machine4": true},
		},
		{
			name: "Mix of failed predicates works fine",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeSelectorNotMatch, algorithmpredicates.ErrNodeUnderDiskPressure, algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrPodNotMatchHostName, algorithmpredicates.ErrDiskConflict},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
				"machine4": []algorithmpredicates.PredicateFailureReason{},
			},
			expected: map[string]bool{"machine3": true, "machine4": true},
		},
		{
			name: "Node condition errors should be considered unresolvable",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderPIDPressure},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnderMemoryPressure},
			},
			expected: map[string]bool{"machine4": true},
		},
		{
			name: "Node condition errors and ErrNodeUnknownCondition should be considered unresolvable",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeNotReady},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeNetworkUnavailable},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrNodeUnknownCondition},
			},
			expected: map[string]bool{"machine4": true},
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrVolumeZoneConflict},
				"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrVolumeNodeConflict},
				"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrVolumeBindConflict},
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
		"machine1": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
		"machine2": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.ErrDiskConflict},
		"machine3": []algorithmpredicates.PredicateFailureReason{algorithmpredicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
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
			t.Logf("===== Running test %v", t.Name())
			stop := make(chan struct{})
			cache := schedulerinternalcache.New(time.Duration(0), stop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			cachedNodeInfoMap := map[string]*schedulernodeinfo.NodeInfo{}
			for _, name := range nodeNames {
				node := makeNode(name, 1000*5, priorityutil.DefaultMemoryRequest*5)
				cache.AddNode(node)

				// Set nodeInfo to extenders to mock extenders' cache for preemption.
				cachedNodeInfo := schedulernodeinfo.NewNodeInfo()
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
				internalqueue.NewSchedulingQueue(nil),
				map[string]algorithmpredicates.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
				algorithmpredicates.EmptyPredicateMetadataProducer,
				[]priorities.PriorityConfig{{Function: numericPriority, Weight: 1}},
				priorities.EmptyPriorityMetadataProducer,
				emptyPluginSet,
				extenders,
				nil,
				schedulertesting.FakePersistentVolumeClaimLister{},
				schedulertesting.FakePDBLister{},
				false,
				false,
				schedulerapi.DefaultPercentageOfNodesToScore)
			scheduler.(*genericScheduler).snapshot()
			// Call Preempt and check the expected results.
			node, victims, _, err := scheduler.Preempt(test.pod, schedulertesting.FakeNodeLister(makeNodeList(nodeNames)), error(&FitError{Pod: test.pod, FailedPredicates: failedPredMap}))
			if err != nil {
				t.Errorf("unexpected error in preemption: %v", err)
			}
			if node != nil && node.Name != test.expectedNode {
				t.Errorf("expected node: %v, got: %v", test.expectedNode, node.GetName())
			}
			if node == nil && len(test.expectedNode) != 0 {
				t.Errorf("expected node: %v, got: nothing", test.expectedNode)
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

func TestNumFeasibleNodesToFind(t *testing.T) {
	tests := []struct {
		name                     string
		percentageOfNodesToScore int32
		numAllNodes              int32
		wantNumNodes             int32
	}{
		{
			name:         "not set percentageOfNodesToScore and nodes number not more than 50",
			numAllNodes:  10,
			wantNumNodes: 10,
		},
		{
			name:                     "set percentageOfNodesToScore and nodes number not more than 50",
			percentageOfNodesToScore: 40,
			numAllNodes:              10,
			wantNumNodes:             10,
		},
		{
			name:         "not set percentageOfNodesToScore and nodes number more than 50",
			numAllNodes:  1000,
			wantNumNodes: 420,
		},
		{
			name:                     "set percentageOfNodesToScore and nodes number more than 50",
			percentageOfNodesToScore: 40,
			numAllNodes:              1000,
			wantNumNodes:             400,
		},
		{
			name:         "not set percentageOfNodesToScore and nodes number more than 50*125",
			numAllNodes:  6000,
			wantNumNodes: 300,
		},
		{
			name:                     "set percentageOfNodesToScore and nodes number more than 50*125",
			percentageOfNodesToScore: 40,
			numAllNodes:              6000,
			wantNumNodes:             2400,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &genericScheduler{
				percentageOfNodesToScore: tt.percentageOfNodesToScore,
			}
			if gotNumNodes := g.numFeasibleNodesToFind(tt.numAllNodes); gotNumNodes != tt.wantNumNodes {
				t.Errorf("genericScheduler.numFeasibleNodesToFind() = %v, want %v", gotNumNodes, tt.wantNumNodes)
			}
		})
	}
}
