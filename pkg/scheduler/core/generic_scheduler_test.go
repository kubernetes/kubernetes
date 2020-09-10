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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	fakeframework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1/fake"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	errPrioritize = fmt.Errorf("priority map encounters an error")
)

type noPodsFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *noPodsFilterPlugin) Name() string {
	return "NoPodsFilter"
}

// Filter invoked at the filter extension point.
func (pl *noPodsFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if len(nodeInfo.Pods) == 0 {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, st.ErrReasonFake)
}

// NewNoPodsFilterPlugin initializes a noPodsFilterPlugin and returns it.
func NewNoPodsFilterPlugin(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &noPodsFilterPlugin{}, nil
}

type numericMapPlugin struct{}

func newNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
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

func newReverseNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
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

func newTrueMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
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

func newFalseMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
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
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test 1",
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"machine1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
					"machine2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
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
					"3": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
					"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
					"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("NoPodsFilter", NewNoPodsFilterPlugin),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
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
					"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
					"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake),
				},
			},
		},
		{
			// Pod with existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
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
			name: "test podtopologyspread plugin - 2 nodes with maxskew=1",
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
			name: "test podtopologyspread plugin - 3 nodes with maxskew=2",
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
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.Unschedulable}),
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
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.UnschedulableAndUnresolvable}),
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
					st.NewFakeFilterPlugin(map[string]framework.Code{"1": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr:          nil,
		},
		{
			name: "test prefilter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin(framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
			expectedHosts: nil,
			wErr: &FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
				NumAllNodes: 2,
				FilteredNodesStatuses: framework.NodeToStatusMap{
					"1": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status"),
					"2": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status"),
				},
			},
		},
		{
			name: "test prefilter plugin returning error status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin(framework.NewStatus(framework.Error, "injected error status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
			expectedHosts: nil,
			wErr:          fmt.Errorf(`prefilter plugin "FakePreFilter" failed for pod "test-prefilter": injected error status`),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
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
			fwk, err := st.NewFramework(
				test.registerPlugins,
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
			)
			if err != nil {
				t.Fatal(err)
			}
			prof := &profile.Profile{
				Framework: fwk,
			}

			var pvcs []v1.PersistentVolumeClaim
			pvcs = append(pvcs, test.pvcs...)
			pvcLister := fakeframework.PersistentVolumeClaimLister(pvcs)

			scheduler := NewGenericScheduler(
				cache,
				snapshot,
				[]framework.Extender{},
				pvcLister,
				schedulerapi.DefaultPercentageOfNodesToScore)
			result, err := scheduler.Schedule(context.Background(), prof, framework.NewCycleState(), test.pod)
			if !reflect.DeepEqual(err, test.wErr) {
				t.Errorf("want: %v, got: %v", test.wErr, err)
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
func makeScheduler(nodes []*v1.Node) *genericScheduler {
	cache := internalcache.New(time.Duration(0), wait.NeverStop)
	for _, n := range nodes {
		cache.AddNode(n)
	}

	s := NewGenericScheduler(
		cache,
		emptySnapshot,
		nil, nil,
		schedulerapi.DefaultPercentageOfNodesToScore)
	cache.UpdateSnapshot(s.(*genericScheduler).nodeInfoSnapshot)
	return s.(*genericScheduler)
}

func TestFindFitAllError(t *testing.T) {
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(nodes)
	fwk, err := st.NewFramework(
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
	)
	if err != nil {
		t.Fatal(err)
	}
	prof := &profile.Profile{Framework: fwk}

	_, nodeToStatusMap, err := scheduler.findNodesThatFitPod(context.Background(), prof, framework.NewCycleState(), &v1.Pod{})

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
			if len(reasons) != 1 || reasons[0] != st.ErrReasonFake {
				t.Errorf("unexpected failure reasons: %v", reasons)
			}
		})
	}
}

func TestFindFitSomeError(t *testing.T) {
	nodes := makeNodeList([]string{"3", "2", "1"})
	scheduler := makeScheduler(nodes)
	fwk, err := st.NewFramework(
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
	)
	if err != nil {
		t.Fatal(err)
	}
	prof := &profile.Profile{Framework: fwk}

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1", UID: types.UID("1")}}
	_, nodeToStatusMap, err := scheduler.findNodesThatFitPod(context.Background(), prof, framework.NewCycleState(), pod)

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
			if len(reasons) != 1 || reasons[0] != st.ErrReasonFake {
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

		plugin := st.FakeFilterPlugin{}
		registerFakeFilterFunc := st.RegisterFilterPlugin(
			"FakeFilter",
			func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
				return &plugin, nil
			},
		)
		registerPlugins := []st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			registerFakeFilterFunc,
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		}
		fwk, err := st.NewFramework(
			registerPlugins,
			frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
		)
		if err != nil {
			t.Fatal(err)
		}
		prof := &profile.Profile{Framework: fwk}

		scheduler := makeScheduler(nodes)
		if err := scheduler.cache.UpdateSnapshot(scheduler.nodeInfoSnapshot); err != nil {
			t.Fatal(err)
		}
		fwk.PreemptHandle().AddNominatedPod(&v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "nominated"}, Spec: v1.PodSpec{Priority: &midPriority}}, "1")

		_, _, err = scheduler.findNodesThatFitPod(context.Background(), prof, framework.NewCycleState(), test.pod)

		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectedCount != plugin.NumFilterCalled {
			t.Errorf("predicate was called %d times, expected is %d", plugin.NumFilterCalled, test.expectedCount)
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
				st.RegisterScorePlugin(selectorspread.Name, selectorspread.New, 1),
				st.RegisterPreScorePlugin(selectorspread.Name, selectorspread.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			fwk, err := st.NewFramework(
				pluginRegistrations,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
			)
			if err != nil {
				t.Fatalf("error creating framework: %+v", err)
			}
			prof := &profile.Profile{Framework: fwk}

			scheduler := NewGenericScheduler(
				nil,
				emptySnapshot,
				[]framework.Extender{},
				nil,
				schedulerapi.DefaultPercentageOfNodesToScore).(*genericScheduler)
			scheduler.nodeInfoSnapshot = snapshot

			ctx := context.Background()
			state := framework.NewCycleState()
			_, _, err = scheduler.findNodesThatFitPod(ctx, prof, state, test.pod)
			if err != nil {
				t.Fatalf("error filtering nodes: %+v", err)
			}
			prof.RunPreScorePlugins(ctx, state, test.pod, test.nodes)
			list, err := scheduler.prioritizeNodes(ctx, prof, state, test.pod, test.nodes)
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

var lowPriority, midPriority, highPriority = int32(0), int32(100), int32(1000)

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

func TestFairEvaluationForNodes(t *testing.T) {
	numAllNodes := 500
	nodeNames := make([]string, 0, numAllNodes)
	for i := 0; i < numAllNodes; i++ {
		nodeNames = append(nodeNames, strconv.Itoa(i))
	}
	nodes := makeNodeList(nodeNames)
	g := makeScheduler(nodes)
	fwk, err := st.NewFramework(
		[]st.RegisterPluginFunc{
			st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
			st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
	)
	if err != nil {
		t.Fatal(err)
	}
	prof := &profile.Profile{Framework: fwk}
	// To make numAllNodes % nodesToFind != 0
	g.percentageOfNodesToScore = 30
	nodesToFind := int(g.numFeasibleNodesToFind(int32(numAllNodes)))

	// Iterating over all nodes more than twice
	for i := 0; i < 2*(numAllNodes/nodesToFind+1); i++ {
		nodesThatFit, _, err := g.findNodesThatFitPod(context.Background(), prof, framework.NewCycleState(), &v1.Pod{})
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
