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
	"context"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	extenderv1 "k8s.io/kubernetes/pkg/scheduler/apis/extender/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpodtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	fakelisters "k8s.io/kubernetes/pkg/scheduler/listers/fake"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	errPrioritize = fmt.Errorf("priority map encounters an error")
)

const ErrReasonFake = "Nodes failed the fake predicate"

type trueFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *trueFilterPlugin) Name() string {
	return "TrueFilter"
}

// Filter invoked at the filter extension point.
func (pl *trueFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	return nil
}

// NewTrueFilterPlugin initializes a trueFilterPlugin and returns it.
func NewTrueFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &trueFilterPlugin{}, nil
}

type falseFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *falseFilterPlugin) Name() string {
	return "FalseFilter"
}

// Filter invoked at the filter extension point.
func (pl *falseFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	return framework.NewStatus(framework.Unschedulable, ErrReasonFake)
}

// NewFalseFilterPlugin initializes a falseFilterPlugin and returns it.
func NewFalseFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &falseFilterPlugin{}, nil
}

type matchFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *matchFilterPlugin) Name() string {
	return "MatchFilter"
}

// Filter invoked at the filter extension point.
func (pl *matchFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	if pod.Name == node.Name {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, ErrReasonFake)
}

// NewMatchFilterPlugin initializes a matchFilterPlugin and returns it.
func NewMatchFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &matchFilterPlugin{}, nil
}

type noPodsFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *noPodsFilterPlugin) Name() string {
	return "NoPodsFilter"
}

// Filter invoked at the filter extension point.
func (pl *noPodsFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	if len(nodeInfo.Pods()) == 0 {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, ErrReasonFake)
}

// NewNoPodsFilterPlugin initializes a noPodsFilterPlugin and returns it.
func NewNoPodsFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &noPodsFilterPlugin{}, nil
}

// fakeFilterPlugin is a test filter plugin to record how many times its Filter() function have
// been called, and it returns different 'Code' depending on its internal 'failedNodeReturnCodeMap'.
type fakeFilterPlugin struct {
	numFilterCalled         int32
	failedNodeReturnCodeMap map[string]framework.Code
}

// Name returns name of the plugin.
func (pl *fakeFilterPlugin) Name() string {
	return "FakeFilter"
}

// Filter invoked at the filter extension point.
func (pl *fakeFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	atomic.AddInt32(&pl.numFilterCalled, 1)

	if returnCode, ok := pl.failedNodeReturnCodeMap[nodeInfo.Node().Name]; ok {
		return framework.NewStatus(returnCode, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// NewFakeFilterPlugin initializes a fakeFilterPlugin and returns it.
func NewFakeFilterPlugin(failedNodeReturnCodeMap map[string]framework.Code) framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &fakeFilterPlugin{
			failedNodeReturnCodeMap: failedNodeReturnCodeMap,
		}, nil
	}
}

type numericMapPlugin struct{}

func newNumericMapPlugin() framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &numericMapPlugin{}, nil
	}
}

func (pl *numericMapPlugin) Name() string {
	return "NumericMap"
}

func (pl *numericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *numericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

type reverseNumericMapPlugin struct{}

func newReverseNumericMapPlugin() framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &reverseNumericMapPlugin{}, nil
	}
}

func (pl *reverseNumericMapPlugin) Name() string {
	return "ReverseNumericMap"
}

func (pl *reverseNumericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *reverseNumericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *reverseNumericMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	var maxScore float64
	minScore := math.MaxFloat64

	for _, hostPriority := range nodeScores {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for i, hostPriority := range nodeScores {
		nodeScores[i] = framework.NodeScore{
			Name:  hostPriority.Name,
			Score: int64(maxScore + minScore - float64(hostPriority.Score)),
		}
	}
	return nil
}

type trueMapPlugin struct{}

func newTrueMapPlugin() framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &trueMapPlugin{}, nil
	}
}

func (pl *trueMapPlugin) Name() string {
	return "TrueMap"
}

func (pl *trueMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 1, nil
}

func (pl *trueMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *trueMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	for _, host := range nodeScores {
		if host.Name == "" {
			return framework.NewStatus(framework.Error, "unexpected empty host name")
		}
	}
	return nil
}

type falseMapPlugin struct{}

func newFalseMapPlugin() framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &falseMapPlugin{}, nil
	}
}

func (pl *falseMapPlugin) Name() string {
	return "FalseMap"
}

func (pl *falseMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 0, framework.NewStatus(framework.Error, errPrioritize.Error())
}

func (pl *falseMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

var emptySnapshot = internalcache.NewEmptySnapshot()

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
		list          framework.NodeScoreList
		possibleHosts sets.String
		expectsErr    bool
	}{
		{
			name: "unique properly ordered scores",
			list: []framework.NodeScore{
				{Name: "machine1.1", Score: 1},
				{Name: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine2.1"),
			expectsErr:    false,
		},
		{
			name: "equal scores",
			list: []framework.NodeScore{
				{Name: "machine1.1", Score: 1},
				{Name: "machine1.2", Score: 2},
				{Name: "machine1.3", Score: 2},
				{Name: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine1.2", "machine1.3", "machine2.1"),
			expectsErr:    false,
		},
		{
			name: "out of order scores",
			list: []framework.NodeScore{
				{Name: "machine1.1", Score: 3},
				{Name: "machine1.2", Score: 3},
				{Name: "machine2.1", Score: 2},
				{Name: "machine3.1", Score: 1},
				{Name: "machine1.3", Score: 3},
			},
			possibleHosts: sets.NewString("machine1.1", "machine1.2", "machine1.3"),
			expectsErr:    false,
		},
		{
			name:          "empty priority list",
			list:          []framework.NodeScore{},
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
	tests := []struct {
		name            string
		registerPlugins []st.RegisterPluginFunc
		nodes           []string
		pvcs            []v1.PersistentVolumeClaim
		pod             *v1.Pod
		pods            []*v1.Pod
		expectedHosts   sets.String
		wErr            error
	}{
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("FalseFilter", NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test 1",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"machine1": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
					"machine2": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine2", UID: types.UID("machine2")}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterScorePlugin("ReverseNumericMap", newReverseNumericMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterFilterPlugin("FalseFilter", NewFalseFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"3", "2", "1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test 7",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 3,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"3": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
					"2": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
					"1": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("NoPodsFilter", NewNoPodsFilterPlugin),
				st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
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
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			nodes: []string{"1", "2"},
			name:  "test 8",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"1": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
					"2": framework.NewStatus(framework.Unschedulable, ErrReasonFake),
				},
			},
		},
		{
			// Pod with existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pvcs:  []v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC"}}},
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
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
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
			name: "unknown PVC",
			wErr: fmt.Errorf("persistentvolumeclaim \"unknownPVC\" not found"),
		},
		{
			// Pod with deleting PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pvcs:  []v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", DeletionTimestamp: &metav1.Time{}}}},
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
			name: "deleted PVC",
			wErr: fmt.Errorf("persistentvolumeclaim \"existingPVC\" is being deleted"),
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterScorePlugin("FalseMap", newFalseMapPlugin(), 1),
				st.RegisterScorePlugin("TrueMap", newTrueMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"2", "1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			name:  "test error with priority map",
			wErr:  fmt.Errorf("error while running score plugin for pod \"2\": %+v", errPrioritize),
		},
		{
			name: "test even pods spread predicate - 2 nodes with maxskew=1",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					1,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p", UID: types.UID("p"), Labels: map[string]string{"foo": ""}},
				Spec: v1.PodSpec{
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
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
					ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			expectedHosts: sets.NewString("machine2"),
			wErr:          nil,
		},
		{
			name: "test even pods spread predicate - 3 nodes with maxskew=2",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					1,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p", UID: types.UID("p"), Labels: map[string]string{"foo": ""}},
				Spec: v1.PodSpec{
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
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
					ObjectMeta: metav1.ObjectMeta{Name: "pod1a", UID: types.UID("pod1a"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod1b", UID: types.UID("pod1b"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: types.UID("pod2"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine2",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			expectedHosts: sets.NewString("machine2", "machine3"),
			wErr:          nil,
		},
		{
			name: "test with filter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					NewFakeFilterPlugin(map[string]framework.Code{"3": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
				NumAllNodes: 1,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"3": framework.NewStatus(framework.Unschedulable, "injecting failure for pod test-filter"),
				},
			},
		},
		{
			name: "test with filter plugin returning UnschedulableAndUnresolvable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					NewFakeFilterPlugin(map[string]framework.Code{"3": framework.UnschedulableAndUnresolvable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
				NumAllNodes: 1,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"3": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injecting failure for pod test-filter"),
				},
			},
		},
		{
			name: "test with partial failed filter plugin",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					NewFakeFilterPlugin(map[string]framework.Code{"1": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr:          nil,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			var nodes []*v1.Node
			for _, name := range test.nodes {
				node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"hostname": name}}}
				nodes = append(nodes, node)
				cache.AddNode(node)
			}

			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(test.registerPlugins, framework.WithSnapshotSharedLister(snapshot))
			if err != nil {
				t.Fatal(err)
			}

			var pvcs []v1.PersistentVolumeClaim
			pvcs = append(pvcs, test.pvcs...)
			pvcLister := fakelisters.PersistentVolumeClaimLister(pvcs)

			scheduler := NewGenericScheduler(
				cache,
				internalqueue.NewSchedulingQueue(nil),
				snapshot,
				fwk,
				[]SchedulerExtender{},
				nil,
				pvcLister,
				informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
				false,
				schedulerapi.DefaultPercentageOfNodesToScore,
				false)
			result, err := scheduler.Schedule(context.Background(), framework.NewCycleState(), test.pod)
			if !reflect.DeepEqual(err, test.wErr) {
				t.Errorf("Unexpected error: %v, expected: %v", err.Error(), test.wErr)
			}
			if test.expectedHosts != nil && !test.expectedHosts.Has(result.SuggestedHost) {
				t.Errorf("Expected: %s, got: %s", test.expectedHosts, result.SuggestedHost)
			}
			if test.wErr == nil && len(test.nodes) != result.EvaluatedNodes {
				t.Errorf("Expected EvaluatedNodes: %d, got: %d", len(test.nodes), result.EvaluatedNodes)
			}
		})
	}
}

// makeScheduler makes a simple genericScheduler for testing.
func makeScheduler(nodes []*v1.Node, fns ...st.RegisterPluginFunc) *genericScheduler {
	cache := internalcache.New(time.Duration(0), wait.NeverStop)
	for _, n := range nodes {
		cache.AddNode(n)
	}

	fwk, _ := st.NewFramework(fns)
	s := NewGenericScheduler(
		cache,
		internalqueue.NewSchedulingQueue(nil),
		emptySnapshot,
		fwk,
		nil, nil, nil, nil, false,
		schedulerapi.DefaultPercentageOfNodesToScore, false)
	cache.UpdateSnapshot(s.(*genericScheduler).nodeInfoSnapshot)
	return s.(*genericScheduler)

}

func TestFindFitAllError(t *testing.T) {
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(
		nodes,
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
		st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	)

	_, nodeToStatusMap, err := scheduler.findNodesThatFitPod(context.Background(), framework.NewCycleState(), &v1.Pod{})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(nodeToStatusMap) != len(nodes) {
		t.Errorf("unexpected failed status map: %v", nodeToStatusMap)
	}

	for _, node := range nodes {
		t.Run(node.Name, func(t *testing.T) {
			status, found := nodeToStatusMap[node.Name]
			if !found {
				t.Errorf("failed to find node %v in %v", node.Name, nodeToStatusMap)
			}
			reasons := status.Reasons()
			if len(reasons) != 1 || reasons[0] != ErrReasonFake {
				t.Errorf("unexpected failure reasons: %v", reasons)
			}
		})
	}
}

func TestFindFitSomeError(t *testing.T) {
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(
		nodes,
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
		st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1", UID: types.UID("1")}}
	_, nodeToStatusMap, err := scheduler.findNodesThatFitPod(context.Background(), framework.NewCycleState(), pod)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(nodeToStatusMap) != len(nodes)-1 {
		t.Errorf("unexpected failed status map: %v", nodeToStatusMap)
	}

	for _, node := range nodes {
		if node.Name == pod.Name {
			continue
		}
		t.Run(node.Name, func(t *testing.T) {
			status, found := nodeToStatusMap[node.Name]
			if !found {
				t.Errorf("failed to find node %v in %v", node.Name, nodeToStatusMap)
			}
			reasons := status.Reasons()
			if len(reasons) != 1 || reasons[0] != ErrReasonFake {
				t.Errorf("unexpected failures: %v", reasons)
			}
		})
	}
}

func TestFindFitPredicateCallCounts(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		expectedCount int32
	}{
		{
			name:          "nominated pods have lower priority, predicate is called once",
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1", UID: types.UID("1")}, Spec: v1.PodSpec{Priority: &highPriority}},
			expectedCount: 1,
		},
		{
			name:          "nominated pods have higher priority, predicate is called twice",
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1", UID: types.UID("1")}, Spec: v1.PodSpec{Priority: &lowPriority}},
			expectedCount: 2,
		},
	}

	for _, test := range tests {
		nodes := makeNodeList([]string{"1"})

		cache := internalcache.New(time.Duration(0), wait.NeverStop)
		for _, n := range nodes {
			cache.AddNode(n)
		}

		plugin := fakeFilterPlugin{}
		registerFakeFilterFunc := st.RegisterFilterPlugin(
			"FakeFilter",
			func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
				return &plugin, nil
			},
		)
		registerPlugins := []st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			registerFakeFilterFunc,
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		}
		fwk, err := st.NewFramework(registerPlugins)
		if err != nil {
			t.Fatal(err)
		}

		queue := internalqueue.NewSchedulingQueue(nil)
		scheduler := NewGenericScheduler(
			cache,
			queue,
			emptySnapshot,
			fwk,
			nil, nil, nil, nil, false,
			schedulerapi.DefaultPercentageOfNodesToScore, false).(*genericScheduler)
		cache.UpdateSnapshot(scheduler.nodeInfoSnapshot)
		queue.UpdateNominatedPodForNode(&v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("nominated")}, Spec: v1.PodSpec{Priority: &midPriority}}, "1")

		_, _, err = scheduler.findNodesThatFitPod(context.Background(), framework.NewCycleState(), test.pod)

		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectedCount != plugin.numFilterCalled {
			t.Errorf("predicate was called %d times, expected is %d", plugin.numFilterCalled, test.expectedCount)
		}
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
							strconv.FormatInt(schedutil.DefaultMilliCPURequest, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMemoryRequest, 10)),
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
							strconv.FormatInt(schedutil.DefaultMilliCPURequest*3, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(schedutil.DefaultMemoryRequest*3, 10)),
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
		expectedScore int64
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request pod
		// as for a pod with the defaults requests, both when the zero-request pod is already on the machine
		// and when the zero-request pod is the one being scheduled.
		{
			pod:   &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{makeNode("machine1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of zero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 250,
		},
		{
			pod:   &v1.Pod{Spec: small},
			nodes: []*v1.Node{makeNode("machine1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of nonzero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 250,
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			pod:   &v1.Pod{Spec: large},
			nodes: []*v1.Node{makeNode("machine1", 1000, schedutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, schedutil.DefaultMemoryRequest*10)},
			name:  "test priority of larger pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
			expectedScore: 230,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			snapshot := internalcache.NewSnapshot(test.pods, test.nodes)

			pluginRegistrations := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterScorePlugin(noderesources.LeastAllocatedName, noderesources.NewLeastAllocated, 1),
				st.RegisterScorePlugin(noderesources.BalancedAllocationName, noderesources.NewBalancedAllocation, 1),
				st.RegisterScorePlugin(defaultpodtopologyspread.Name, defaultpodtopologyspread.New, 1),
				st.RegisterPostFilterPlugin(defaultpodtopologyspread.Name, defaultpodtopologyspread.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			fwk, err := st.NewFramework(
				pluginRegistrations,
				framework.WithInformerFactory(informerFactory),
				framework.WithSnapshotSharedLister(snapshot),
				framework.WithClientSet(client),
			)
			if err != nil {
				t.Fatalf("error creating framework: %+v", err)
			}

			scheduler := NewGenericScheduler(
				nil,
				nil,
				emptySnapshot,
				fwk,
				[]SchedulerExtender{},
				nil,
				nil,
				nil,
				false,
				schedulerapi.DefaultPercentageOfNodesToScore,
				false).(*genericScheduler)
			scheduler.nodeInfoSnapshot = snapshot

			ctx := context.Background()
			state := framework.NewCycleState()
			_, filteredNodesStatuses, err := scheduler.findNodesThatFitPod(ctx, state, test.pod)
			if err != nil {
				t.Fatalf("error filtering nodes: %+v", err)
			}
			scheduler.framework.RunPostFilterPlugins(ctx, state, test.pod, test.nodes, filteredNodesStatuses)
			list, err := scheduler.prioritizeNodes(
				ctx,
				state,
				test.pod,
				test.nodes,
			)
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

func printNodeToVictims(nodeToVictims map[*v1.Node]*extenderv1.Victims) string {
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

func checkPreemptionVictims(expected map[string]map[string]bool, nodeToPods map[*v1.Node]*extenderv1.Victims) error {
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

// TestSelectNodesForPreemption tests selectNodesForPreemption. This test assumes
// that podsFitsOnNode works correctly and is tested separately.
func TestSelectNodesForPreemption(t *testing.T) {
	tests := []struct {
		name                    string
		registerPlugins         []st.RegisterPluginFunc
		nodes                   []string
		pod                     *v1.Pod
		pods                    []*v1.Pod
		filterReturnCode        framework.Code
		expected                map[string]map[string]bool // Map from node name to a list of pods names which should be preempted.
		expectedNumFilterCalled int32
	}{
		{
			name: "a pod that does not fit on any machine",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("FalseFilter", NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{},
			expectedNumFilterCalled: 2,
		},
		{
			name: "a pod that fits with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", UID: types.UID("new")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{"machine1": {}, "machine2": {}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that fits on one machine with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{"machine1": {}},
			expectedNumFilterCalled: 3,
		},
		{
			name: "a pod that fits on both machines when lower priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{"machine1": {"a": true}, "machine2": {"b": true}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that would fit on the machines, but other pods running are higher priority",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &lowPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{},
			expectedNumFilterCalled: 2,
		},
		{
			name: "medium priority pod is preempted, but lower priority one stays as it is small",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c", UID: types.UID("c")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected:                map[string]map[string]bool{"machine1": {"b": true}, "machine2": {"c": true}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			expected:                map[string]map[string]bool{"machine1": {"b": true, "c": true}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted, pick later StartTime one when priorities are equal",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			expected:                map[string]map[string]bool{"machine1": {"a": true, "c": true}},
			expectedNumFilterCalled: 5,
		},
		{
			name: "pod with anti-affinity is preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterFilterPlugin(interpodaffinity.Name, interpodaffinity.New),
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
			expected:                map[string]map[string]bool{"machine1": {"a": true}, "machine2": {}},
			expectedNumFilterCalled: 4,
		},
		{
			name: "preemption to resolve even pods spread FitError",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					1,
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
			expected: map[string]map[string]bool{
				"node-a": {"pod-a2": true},
				"node-b": {"pod-b1": true},
			},
			expectedNumFilterCalled: 6,
		},
		{
			name: "get Unschedulable in the preemption phase when the filter plugins filtering the nodes",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", UID: types.UID("a")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b", UID: types.UID("b")}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			filterReturnCode:        framework.Unschedulable,
			expected:                map[string]map[string]bool{},
			expectedNumFilterCalled: 2,
		},
	}
	labelKeys := []string{"hostname", "zone", "region"}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

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
			fakePlugin := fakeFilterPlugin{}
			fakePlugin.failedNodeReturnCodeMap = filterFailedNodeReturnCodeMap
			registerFakeFilterFunc := st.RegisterFilterPlugin(
				"FakeFilter",
				func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
					return &fakePlugin, nil
				},
			)
			registerPlugins := append([]st.RegisterPluginFunc{registerFakeFilterFunc}, test.registerPlugins...)
			// Use a real snapshot since it's needed in some Filter Plugin (e.g., PodAffinity)
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(registerPlugins, framework.WithSnapshotSharedLister(snapshot))
			if err != nil {
				t.Fatal(err)
			}

			scheduler := NewGenericScheduler(
				nil,
				internalqueue.NewSchedulingQueue(nil),
				snapshot,
				fwk,
				[]SchedulerExtender{},
				nil,
				nil,
				informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
				false,
				schedulerapi.DefaultPercentageOfNodesToScore,
				false)
			g := scheduler.(*genericScheduler)

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
			nodeToPods, err := g.selectNodesForPreemption(context.Background(), state, test.pod, nodeInfos, nil)
			if err != nil {
				t.Error(err)
			}

			if test.expectedNumFilterCalled != fakePlugin.numFilterCalled {
				t.Errorf("expected fakePlugin.numFilterCalled is %d, but got %d", test.expectedNumFilterCalled, fakePlugin.numFilterCalled)
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			fwk, err := st.NewFramework(test.registerPlugins, framework.WithSnapshotSharedLister(snapshot))
			if err != nil {
				t.Fatal(err)
			}

			g := &genericScheduler{
				framework:        fwk,
				nodeInfoSnapshot: snapshot,
			}
			assignDefaultStartTime(test.pods)

			nodeInfos, err := nodesToNodeInfos(nodes, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			state := framework.NewCycleState()
			candidateNodes, _ := g.selectNodesForPreemption(context.Background(), state, test.pod, nodeInfos, nil)
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
				"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumebinding.ErrReasonNodeConflict),
				"machine3": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumebinding.ErrReasonBindConflict),
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
			fitErr := FitError{
				FilteredNodesStatuses: test.nodesStatuses,
			}
			var nodeInfos []*schedulernodeinfo.NodeInfo
			for _, n := range makeNodeList(nodeNames) {
				ni := schedulernodeinfo.NewNodeInfo()
				ni.SetNode(n)
				nodeInfos = append(nodeInfos, ni)
			}
			nodes := nodesWherePreemptionMightHelp(nodeInfos, &fitErr)
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
		extenders             []*FakeExtender
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
					1,
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
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{falsePredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{errorPredicateExtender},
					ignorable:  true,
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
			extenders: []*FakeExtender{
				{
					predicates:   []fitPredicate{machine1PredicateExtender},
					unInterested: true,
				},
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
			},
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
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
				st.RegisterFilterPlugin(noderesources.FitName, noderesources.NewFit),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			stop := make(chan struct{})
			cache := internalcache.New(time.Duration(0), stop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			cachedNodeInfoMap := map[string]*schedulernodeinfo.NodeInfo{}
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
				cachedNodeInfo := schedulernodeinfo.NewNodeInfo()
				cachedNodeInfo.SetNode(node)
				cachedNodeInfoMap[node.Name] = cachedNodeInfo
			}
			var extenders []SchedulerExtender
			for _, extender := range test.extenders {
				// Set nodeInfoMap as extenders cached node information.
				extender.cachedNodeNameToInfo = cachedNodeInfoMap
				extenders = append(extenders, extender)
			}

			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(test.registerPlugins, framework.WithSnapshotSharedLister(snapshot))
			if err != nil {
				t.Fatal(err)
			}

			scheduler := NewGenericScheduler(
				cache,
				internalqueue.NewSchedulingQueue(nil),
				snapshot,
				fwk,
				extenders,
				nil,
				informerFactory.Core().V1().PersistentVolumeClaims().Lister(),
				informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
				false,
				schedulerapi.DefaultPercentageOfNodesToScore,
				true)
			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			preFilterStatus := fwk.RunPreFilterPlugins(context.Background(), state, test.pod)
			if !preFilterStatus.IsSuccess() {
				t.Errorf("Unexpected preFilterStatus: %v", preFilterStatus)
			}
			// Call Preempt and check the expected results.
			failedNodeToStatusMap := defaultFailedNodeToStatusMap
			if test.failedNodeToStatusMap != nil {
				failedNodeToStatusMap = test.failedNodeToStatusMap
			}
			node, victims, _, err := scheduler.Preempt(context.Background(), state, test.pod, error(&FitError{Pod: test.pod, FilteredNodesStatuses: failedNodeToStatusMap}))
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
			node, victims, _, err = scheduler.Preempt(context.Background(), state, test.pod, error(&FitError{Pod: test.pod, FilteredNodesStatuses: failedNodeToStatusMap}))
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

func assignDefaultStartTime(pods []*v1.Pod) {
	now := metav1.Now()
	for i := range pods {
		pod := pods[i]
		if pod.Status.StartTime == nil {
			pod.Status.StartTime = &now
		}
	}
}

func TestFairEvaluationForNodes(t *testing.T) {
	numAllNodes := 500
	nodeNames := make([]string, 0, numAllNodes)
	for i := 0; i < numAllNodes; i++ {
		nodeNames = append(nodeNames, strconv.Itoa(i))
	}
	nodes := makeNodeList(nodeNames)
	g := makeScheduler(
		nodes,
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	)
	// To make numAllNodes % nodesToFind != 0
	g.percentageOfNodesToScore = 30
	nodesToFind := int(g.numFeasibleNodesToFind(int32(numAllNodes)))

	// Iterating over all nodes more than twice
	for i := 0; i < 2*(numAllNodes/nodesToFind+1); i++ {
		nodesThatFit, _, err := g.findNodesThatFitPod(context.Background(), framework.NewCycleState(), &v1.Pod{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(nodesThatFit) != nodesToFind {
			t.Errorf("got %d nodes filtered, want %d", len(nodesThatFit), nodesToFind)
		}
		if g.nextStartNodeIndex != (i+1)*nodesToFind%numAllNodes {
			t.Errorf("got %d lastProcessedNodeIndex, want %d", g.nextStartNodeIndex, (i+1)*nodesToFind%numAllNodes)
		}
	}
}

func nodesToNodeInfos(nodes []*v1.Node, snapshot *internalcache.Snapshot) ([]*schedulernodeinfo.NodeInfo, error) {
	var nodeInfos []*schedulernodeinfo.NodeInfo
	for _, n := range nodes {
		nodeInfo, err := snapshot.NodeInfos().Get(n.Name)
		if err != nil {
			return nil, err
		}
		nodeInfos = append(nodeInfos, nodeInfo)
	}
	return nodeInfos, nil
}
