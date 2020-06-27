/*
Copyright 2020 The Kubernetes Authors.

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

package defaultpreemption

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	volumescheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var smallContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMilliCPURequest, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMemoryRequest, 10)),
			},
		},
	},
}
var mediumContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMilliCPURequest*2, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMemoryRequest*2, 10)),
			},
		},
	},
}
var largeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMilliCPURequest*3, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMemoryRequest*3, 10)),
			},
		},
	},
}
var veryLargeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMilliCPURequest*5, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(schedutil.DefaultMemoryRequest*5, 10)),
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

type victims struct {
	pods             sets.String
	numPDBViolations int64
}

func checkPreemptionVictims(expected map[string]victims, nodeToPods map[string]*extenderv1.Victims) error {
	if len(expected) == len(nodeToPods) {
		for k, victims := range nodeToPods {
			if expVictims, ok := expected[k]; ok {
				if len(victims.Pods) != len(expVictims.pods) {
					return fmt.Errorf("unexpected number of pods. expected: %v, got: %v", expected, printNodeNameToVictims(nodeToPods))
				}
				prevPriority := int32(math.MaxInt32)
				for _, p := range victims.Pods {
					// Check that pods are sorted by their priority.
					if *p.Spec.Priority > prevPriority {
						return fmt.Errorf("pod %v of node %v was not sorted by priority", p.Name, k)
					}
					prevPriority = *p.Spec.Priority
					if !expVictims.pods.Has(p.Name) {
						return fmt.Errorf("pod %v was not expected. Expected: %v", p.Name, expVictims.pods)
					}
				}
				if expVictims.numPDBViolations != victims.NumPDBViolations {
					return fmt.Errorf("unexpected numPDBViolations. expected: %d, got: %d", expVictims.numPDBViolations, victims.NumPDBViolations)
				}
			} else {
				return fmt.Errorf("unexpected machines. expected: %v, got: %v", expected, printNodeNameToVictims(nodeToPods))
			}
		}
	} else {
		return fmt.Errorf("unexpected number of machines. expected: %v, got: %v", expected, printNodeNameToVictims(nodeToPods))
	}
	return nil
}

func printNodeNameToVictims(nodeNameToVictims map[string]*extenderv1.Victims) string {
	var output string
	for nodeName, victims := range nodeNameToVictims {
		output += nodeName + ": ["
		for _, pod := range victims.Pods {
			output += pod.Name + ", "
		}
		output += "]"
	}
	return output
}

func assignDefaultStartTime(pods []*v1.Pod) {
	now := metav1.Now()
	for i := range pods {
		pod := pods[i]
		if pod.Status.StartTime == nil {
			pod.Status.StartTime = &now
		}
	}
}

func nodesToNodeInfos(nodes []*v1.Node, snapshot *internalcache.Snapshot) ([]*framework.NodeInfo, error) {
	var nodeInfos []*framework.NodeInfo
	for _, n := range nodes {
		nodeInfo, err := snapshot.NodeInfos().Get(n.Name)
		if err != nil {
			return nil, err
		}
		nodeInfos = append(nodeInfos, nodeInfo)
	}
	return nodeInfos, nil
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

func makeNodeList(nodeNames []string) []*v1.Node {
	result := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		result = append(result, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return result
}

func mergeObjs(pod *v1.Pod, pods []*v1.Pod) []runtime.Object {
	var objs []runtime.Object
	if pod != nil {
		objs = append(objs, pod)
	}
	for i := range pods {
		objs = append(objs, pods[i])
	}
	return objs
}

// TestSelectNodesForPreemption tests selectNodesForPreemption. This test assumes
// that podsFitsOnNode works correctly and is tested separately.
func TestSelectNodesForPreemption(t *testing.T) {
	tests := []struct {
		name                    string
		registerPlugins         []st.RegisterPluginFunc
		nodes                   []string
		pod                     *v1.Pod
		pods                    []*v1.Pod
		pdbs                    []*policy.PodDisruptionBudget
		filterReturnCode        framework.Code
		expected                map[string]victims
		expectedNumFilterCalled int32
	}{
		{
			name: "a pod that does not fit on any machine",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{},
			expectedNumFilterCalled: 2,
		},
		{
			name: "a pod that fits with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{"machine1": {}, "machine2": {}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that fits on one machine with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{"machine1": {}},
			expectedNumFilterCalled: 3,
		},
		{
			name: "a pod that fits on both machines when lower priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a")}, "machine2": {pods: sets.NewString("b")}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that would fit on the machines, but other pods running are higher priority",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &lowPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{},
			expectedNumFilterCalled: 2,
		},
		{
			name: "medium priority pod is preempted, but lower priority one stays as it is small",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("b")}, "machine2": {pods: sets.NewString("c")}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d", UID: types.UID("d")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e", UID: types.UID("e")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("b", "c")}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted, pick later StartTime one when priorities are equal",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190107}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190106}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190105}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d", UID: types.UID("d")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime20190104}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e", UID: types.UID("e")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime20190103}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a", "c")}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "pod with anti-affinity is preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterPluginAsExtensions(interpodaffinity.Name, interpodaffinity.New, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
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
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a")}, "machine2": {}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "preemption to resolve even pods spread FitError",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"node-a/zone1", "node-b/zone1", "node-x/zone2"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "p",
					Labels: map[string]string{"foo": ""},
				},
				Spec: v1.PodSpec{
					Priority: &highPriority,
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           1,
							TopologyKey:       "zone",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
						{
							MaxSkew:           1,
							TopologyKey:       "hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-a1", UID: types.UID("pod-a1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-a", Priority: &midPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-a2", UID: types.UID("pod-a2"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-a", Priority: &lowPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-b1", UID: types.UID("pod-b1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-b", Priority: &lowPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-x1", UID: types.UID("pod-x1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-x", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-x2", UID: types.UID("pod-x2"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-x", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
			},
			expected: map[string]victims{
				"node-a": {pods: sets.NewString("pod-a2")},
				"node-b": {pods: sets.NewString("pod-b1")},
			},
			expectedNumFilterCalled: 6,
		},
		{
			name: "get Unschedulable in the preemption phase when the filter plugins filtering the nodes",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			filterReturnCode:        framework.Unschedulable,
			expected:                map[string]victims{},
			expectedNumFilterCalled: 2,
		},
		{
			name: "preemption with violation of same pdb",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}}},
			pdbs: []*policy.PodDisruptionBudget{
				{Spec: policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}}, Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a", "b"), numPDBViolations: 1}},
			expectedNumFilterCalled: 3,
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim doesn't belong to DisruptedPods",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}}},
			pdbs: []*policy.PodDisruptionBudget{
				{Spec: policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}}, Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"c": {Time: time.Now()}}}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a", "b"), numPDBViolations: 1}},
			expectedNumFilterCalled: 3,
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim belongs to DisruptedPods",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}}},
			pdbs: []*policy.PodDisruptionBudget{
				{Spec: policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}}, Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"b": {Time: time.Now()}}}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a", "b"), numPDBViolations: 0}},
			expectedNumFilterCalled: 3,
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim which belongs to DisruptedPods is treated as 'nonViolating'",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c"), Labels: map[string]string{"app": "foo"}}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}}},
			pdbs: []*policy.PodDisruptionBudget{
				{Spec: policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}}, Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"c": {Time: time.Now()}}}}},
			expected:                map[string]victims{"machine1": {pods: sets.NewString("a", "b", "c"), numPDBViolations: 1}},
			expectedNumFilterCalled: 4,
		},
	}
	labelKeys := []string{"hostname", "zone", "region"}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filterFailedNodeReturnCodeMap := map[string]framework.Code{}
			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			for _, name := range test.nodes {
				filterFailedNodeReturnCodeMap[name] = test.filterReturnCode
				cache.AddNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"hostname": name}}})
			}

			var nodes []*v1.Node
			for _, n := range test.nodes {
				node := makeNode(n, 1000*5, schedutil.DefaultMemoryRequest*5)
				// if possible, split node name by '/' to form labels in a format of
				// {"hostname": node.Name[0], "zone": node.Name[1], "region": node.Name[2]}
				node.ObjectMeta.Labels = make(map[string]string)
				for i, label := range strings.Split(node.Name, "/") {
					node.ObjectMeta.Labels[labelKeys[i]] = label
				}
				node.Name = node.ObjectMeta.Labels["hostname"]
				nodes = append(nodes, node)
			}

			// For each test, prepend a FakeFilterPlugin.
			fakePlugin := st.FakeFilterPlugin{}
			fakePlugin.FailedNodeReturnCodeMap = filterFailedNodeReturnCodeMap
			registerFakeFilterFunc := st.RegisterFilterPlugin(
				"FakeFilter",
				func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
					return &fakePlugin, nil
				},
			)
			registerPlugins := append([]st.RegisterPluginFunc{registerFakeFilterFunc}, test.registerPlugins...)
			// Use a real snapshot since it's needed in some Filter Plugin (e.g., PodAffinity)
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(
				registerPlugins,
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatal(err)
			}

			assignDefaultStartTime(test.pods)

			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			preFilterStatus := fwk.RunPreFilterPlugins(context.Background(), state, test.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("Unexpected preFilterStatus: %v", preFilterStatus)
			}
			nodeInfos, err := nodesToNodeInfos(nodes, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			nodeToPods, err := selectNodesForPreemption(context.Background(), fwk.PreemptHandle(), state, test.pod, nodeInfos, test.pdbs)
			if err != nil {
				t.Error(err)
			}

			if test.expectedNumFilterCalled != fakePlugin.NumFilterCalled {
				t.Errorf("expected fakePlugin.numFilterCalled is %d, but got %d", test.expectedNumFilterCalled, fakePlugin.NumFilterCalled)
			}

			if err := checkPreemptionVictims(test.expected, nodeToPods); err != nil {
				t.Error(err)
			}
		})
	}
}

// TestPickOneNodeForPreemption tests pickOneNodeForPreemption.
func TestPickOneNodeForPreemption(t *testing.T) {
	tests := []struct {
		name            string
		registerPlugins []st.RegisterPluginFunc
		nodes           []string
		pod             *v1.Pod
		pods            []*v1.Pod
		expected        []string // any of the items is valid
	}{
		{
			name: "No node needs preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine1"},
		},
		{
			name: "a pod that fits on both machines when lower priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine1", "machine2"},
		},
		{
			name: "a pod that fits on a machine with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{StartTime: &startTime}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{StartTime: &startTime}}},
			expected: []string{"machine3"},
		},
		{
			name: "machine with min highest priority pod is picked",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "when highest priorities are the same, minimum sum of priorities is picked",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "when highest priority and sum are the same, minimum number of pods is picked",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "sum of adjusted priorities is considered",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "non-overlapping lowest high priority, sum priorities, and number of pods",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3", "machine4"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &veryHighPriority}},
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
			name: "same priority, same number of victims, different start time for each machine's pod",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "same priority, same number of victims, different start time for all pods",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			name: "different priority, same number of victims, different start time for all pods",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
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
			var nodes []*v1.Node
			for _, n := range test.nodes {
				nodes = append(nodes, makeNode(n, schedutil.DefaultMilliCPURequest*5, schedutil.DefaultMemoryRequest*5))
			}
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(
				test.registerPlugins,
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatal(err)
			}

			assignDefaultStartTime(test.pods)

			nodeInfos, err := nodesToNodeInfos(nodes, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			preFilterStatus := fwk.RunPreFilterPlugins(context.Background(), state, test.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("Unexpected preFilterStatus: %v", preFilterStatus)
			}
			candidateNodes, _ := selectNodesForPreemption(context.Background(), fwk.PreemptHandle(), state, test.pod, nodeInfos, nil)
			node := pickOneNodeForPreemption(candidateNodes)
			found := false
			for _, nodeName := range test.expected {
				if node == nodeName {
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
	nodeNames := make([]string, 0, 4)
	for i := 1; i < 5; i++ {
		nodeNames = append(nodeNames, fmt.Sprintf("machine%d", i))
	}

	tests := []struct {
		name          string
		nodesStatuses framework.NodeToStatusMap
		expected      map[string]bool // set of expected node names. Value is ignored.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeaffinity.ErrReason),
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"machine3": framework.NewStatus(framework.UnschedulableAndUnresolvable, tainttoleration.ErrReasonNotMatch),
				"machine4": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodelabel.ErrReasonPresenceViolated),
			},
			expected: map[string]bool{},
		},
		{
			name: "ErrReasonAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"machine3": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnschedulable),
			},
			expected: map[string]bool{"machine1": true, "machine4": true},
		},
		{
			name: "pod with both pod affinity and anti-affinity should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
			},
			expected: map[string]bool{"machine1": true, "machine3": true, "machine4": true},
		},
		{
			name: "ErrReasonAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrReasonAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, interpodaffinity.ErrReasonAffinityRulesNotMatch),
				"machine2": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
			},
			expected: map[string]bool{"machine2": true, "machine3": true, "machine4": true},
		},
		{
			name: "Mix of failed predicates works fine",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumerestrictions.ErrReasonDiskConflict),
				"machine2": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
			},
			expected: map[string]bool{"machine2": true, "machine3": true, "machine4": true},
		},
		{
			name: "Node condition errors should be considered unresolvable",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnknownCondition),
			},
			expected: map[string]bool{"machine2": true, "machine3": true, "machine4": true},
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumezone.ErrReasonConflict),
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumescheduling.ErrReasonNodeConflict)),
				"machine3": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumescheduling.ErrReasonBindConflict)),
			},
			expected: map[string]bool{"machine4": true},
		},
		{
			name: "ErrTopologySpreadConstraintsNotMatch should be tried as it indicates that the pod is unschedulable due to topology spread constraints",
			nodesStatuses: framework.NodeToStatusMap{
				"machine1": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"machine3": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
			},
			expected: map[string]bool{"machine1": true, "machine3": true, "machine4": true},
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
				"machine3": framework.NewStatus(framework.Unschedulable, ""),
				"machine4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: map[string]bool{"machine1": true, "machine3": true},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var nodeInfos []*framework.NodeInfo
			for _, n := range makeNodeList(nodeNames) {
				ni := framework.NewNodeInfo()
				ni.SetNode(n)
				nodeInfos = append(nodeInfos, ni)
			}
			nodes := nodesWherePreemptionMightHelp(nodeInfos, test.nodesStatuses)
			if len(test.expected) != len(nodes) {
				t.Errorf("number of nodes is not the same as expected. exptectd: %d, got: %d. Nodes: %v", len(test.expected), len(nodes), nodes)
			}
			for _, node := range nodes {
				name := node.Node().Name
				if _, found := test.expected[name]; !found {
					t.Errorf("node %v is not expected.", name)
				}
			}
		})
	}
}

func TestPreempt(t *testing.T) {
	defaultFailedNodeToStatusMap := framework.NodeToStatusMap{
		"machine1": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
		"machine2": framework.NewStatus(framework.Unschedulable, volumerestrictions.ErrReasonDiskConflict),
		"machine3": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
	}
	// Prepare 3 node names.
	var defaultNodeNames []string
	for i := 1; i < 4; i++ {
		defaultNodeNames = append(defaultNodeNames, fmt.Sprintf("machine%d", i))
	}
	var (
		preemptLowerPriority = v1.PreemptLowerPriority
		preemptNever         = v1.PreemptNever
	)
	tests := []struct {
		name                  string
		pod                   *v1.Pod
		pods                  []*v1.Pod
		extenders             []*st.FakeExtender
		failedNodeToStatusMap framework.NodeToStatusMap
		nodeNames             []string
		registerPlugins       []st.RegisterPluginFunc
		expectedNode          string
		expectedPods          []string // list of preempted pods
	}{
		{
			name: "basic preemption logic",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "One node doesn't need any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine3",
			expectedPods: []string{},
		},
		{
			name: "preemption for topology spread constraints",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "p",
					Labels: map[string]string{"foo": ""},
				},
				Spec: v1.PodSpec{
					Priority: &highPriority,
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           1,
							TopologyKey:       "zone",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
						{
							MaxSkew:           1,
							TopologyKey:       "hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-a1", UID: types.UID("pod-a1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-a", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-a2", UID: types.UID("pod-a2"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-a", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-b1", UID: types.UID("pod-b1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-b", Priority: &lowPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-x1", UID: types.UID("pod-x1"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-x", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-x2", UID: types.UID("pod-x2"), Labels: map[string]string{"foo": ""}},
					Spec:       v1.PodSpec{NodeName: "node-x", Priority: &highPriority},
					Status:     v1.PodStatus{Phase: v1.PodRunning},
				},
			},
			failedNodeToStatusMap: framework.NodeToStatusMap{
				"node-a": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"node-b": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"node-x": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
			},
			nodeNames: []string{"node-a/zone1", "node-b/zone1", "node-x/zone2"},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "node-b",
			expectedPods: []string{"pod-b1"},
		},
		{
			name: "Scheduler extenders allow only machine1, otherwise machine3 would have been chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.TruePredicateExtender},
				},
				{
					Predicates: []st.FitPredicate{st.Machine1PredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "Scheduler extenders do not allow any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.FalsePredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "",
			expectedPods: []string{},
		},
		{
			name: "One scheduler extender allows only machine1, the other returns error but ignorable. Only machine1 would be chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.ErrorPredicateExtender},
					Ignorable:  true,
				},
				{
					Predicates: []st.FitPredicate{st.Machine1PredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "One scheduler extender allows only machine1, but it is not interested in given pod, otherwise machine1 would have been chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptLowerPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*st.FakeExtender{
				{
					Predicates:   []st.FitPredicate{st.Machine1PredicateExtender},
					UnInterested: true,
				},
				{
					Predicates: []st.FitPredicate{st.TruePredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine3",
			expectedPods: []string{},
		},
		{
			name: "no preempting in pod",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: &preemptNever},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "",
			expectedPods: nil,
		},
		{
			name: "PreemptionPolicy is nil",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1")}, Spec: v1.PodSpec{
				Containers:       veryLargeContainers,
				Priority:         &highPriority,
				PreemptionPolicy: nil},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1", UID: types.UID("m1.1")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2", UID: types.UID("m1.2")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1", UID: types.UID("m2.1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1", UID: types.UID("m3.1")}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			apiObjs := mergeObjs(test.pod, test.pods)
			client := clientsetfake.NewSimpleClientset(apiObjs...)
			deletedPodNames := make(sets.String)
			client.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				deletedPodNames.Insert(action.(clienttesting.DeleteAction).GetName())
				return true, nil, nil
			})

			stop := make(chan struct{})
			cache := internalcache.New(time.Duration(0), stop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			cachedNodeInfoMap := map[string]*framework.NodeInfo{}
			nodeNames := defaultNodeNames
			if len(test.nodeNames) != 0 {
				nodeNames = test.nodeNames
			}
			var nodes []*v1.Node
			for i, name := range nodeNames {
				node := makeNode(name, 1000*5, schedutil.DefaultMemoryRequest*5)
				// if possible, split node name by '/' to form labels in a format of
				// {"hostname": node.Name[0], "zone": node.Name[1], "region": node.Name[2]}
				node.ObjectMeta.Labels = make(map[string]string)
				for i, label := range strings.Split(node.Name, "/") {
					node.ObjectMeta.Labels[labelKeys[i]] = label
				}
				node.Name = node.ObjectMeta.Labels["hostname"]
				cache.AddNode(node)
				nodes = append(nodes, node)
				nodeNames[i] = node.Name

				// Set nodeInfo to extenders to mock extenders' cache for preemption.
				cachedNodeInfo := framework.NewNodeInfo()
				cachedNodeInfo.SetNode(node)
				cachedNodeInfoMap[node.Name] = cachedNodeInfo
			}
			var extenders []framework.Extender
			for _, extender := range test.extenders {
				// Set nodeInfoMap as extenders cached node information.
				extender.CachedNodeNameToInfo = cachedNodeInfoMap
				extenders = append(extenders, extender)
			}

			podNominator := internalqueue.NewPodNominator()
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(
				test.registerPlugins,
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
				frameworkruntime.WithExtenders(extenders),
				frameworkruntime.WithPodNominator(podNominator),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informers.NewSharedInformerFactory(client, 0)),
			)
			if err != nil {
				t.Fatal(err)
			}

			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			preFilterStatus := fwk.RunPreFilterPlugins(context.Background(), state, test.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("Unexpected preFilterStatus: %v", preFilterStatus)
			}
			// Call preempt and check the expected results.
			failedNodeToStatusMap := defaultFailedNodeToStatusMap
			if test.failedNodeToStatusMap != nil {
				failedNodeToStatusMap = test.failedNodeToStatusMap
			}
			node, err := preempt(context.Background(), fwk, state, test.pod, failedNodeToStatusMap)
			if err != nil {
				t.Errorf("unexpected error in preemption: %v", err)
			}
			if len(node) != 0 && node != test.expectedNode {
				t.Errorf("expected node: %v, got: %v", test.expectedNode, node)
			}
			if len(node) == 0 && len(test.expectedNode) != 0 {
				t.Errorf("expected node: %v, got: nothing", test.expectedNode)
			}
			if len(deletedPodNames) != len(test.expectedPods) {
				t.Errorf("expected %v pods, got %v.", len(test.expectedPods), len(deletedPodNames))
			}
			for victimName := range deletedPodNames {
				found := false
				for _, expPod := range test.expectedPods {
					if expPod == victimName {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("pod %v is not expected to be a victim.", victimName)
				}
			}
			test.pod.Status.NominatedNodeName = node
			client.CoreV1().Pods(test.pod.Namespace).Update(context.TODO(), test.pod, metav1.UpdateOptions{})

			// Manually set the deleted Pods' deletionTimestamp to non-nil.
			for _, pod := range test.pods {
				if deletedPodNames.Has(pod.Name) {
					now := metav1.Now()
					pod.DeletionTimestamp = &now
					deletedPodNames.Delete(pod.Name)
				}
			}

			// Call preempt again and make sure it doesn't preempt any more pods.
			node, err = preempt(context.Background(), fwk, state, test.pod, failedNodeToStatusMap)
			if err != nil {
				t.Errorf("unexpected error in preemption: %v", err)
			}
			if len(node) != 0 && len(deletedPodNames) > 0 {
				t.Errorf("didn't expect any more preemption. Node %v is selected for preemption.", node)
			}
			close(stop)
		})
	}
}
