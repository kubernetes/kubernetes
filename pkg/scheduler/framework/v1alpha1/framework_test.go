/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

const (
	queueSortPlugin                   = "no-op-queue-sort-plugin"
	scoreWithNormalizePlugin1         = "score-with-normalize-plugin-1"
	scoreWithNormalizePlugin2         = "score-with-normalize-plugin-2"
	scorePlugin1                      = "score-plugin-1"
	pluginNotImplementingScore        = "plugin-not-implementing-score"
	preFilterPluginName               = "prefilter-plugin"
	preFilterWithExtensionsPluginName = "prefilter-with-extensions-plugin"
	duplicatePluginName               = "duplicate-plugin"
	testPlugin                        = "test-plugin"
	permitPlugin                      = "permit-plugin"
	bindPlugin                        = "bind-plugin"
)

// TestScoreWithNormalizePlugin implements ScoreWithNormalizePlugin interface.
// TestScorePlugin only implements ScorePlugin interface.
var _ ScorePlugin = &TestScoreWithNormalizePlugin{}
var _ ScorePlugin = &TestScorePlugin{}

func newScoreWithNormalizePlugin1(injArgs *runtime.Unknown, f FrameworkHandle) (Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin1, inj}, nil
}

func newScoreWithNormalizePlugin2(injArgs *runtime.Unknown, f FrameworkHandle) (Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin2, inj}, nil
}

func newScorePlugin1(injArgs *runtime.Unknown, f FrameworkHandle) (Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScorePlugin{scorePlugin1, inj}, nil
}

func newPluginNotImplementingScore(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
	return &PluginNotImplementingScore{}, nil
}

type TestScoreWithNormalizePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestScoreWithNormalizePlugin) Name() string {
	return pl.name
}

func (pl *TestScoreWithNormalizePlugin) NormalizeScore(ctx context.Context, state *CycleState, pod *v1.Pod, scores NodeScoreList) *Status {
	return injectNormalizeRes(pl.inj, scores)
}

func (pl *TestScoreWithNormalizePlugin) Score(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
	return setScoreRes(pl.inj)
}

func (pl *TestScoreWithNormalizePlugin) ScoreExtensions() ScoreExtensions {
	return pl
}

// TestScorePlugin only implements ScorePlugin interface.
type TestScorePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestScorePlugin) Name() string {
	return pl.name
}

func (pl *TestScorePlugin) Score(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
	return setScoreRes(pl.inj)
}

func (pl *TestScorePlugin) ScoreExtensions() ScoreExtensions {
	return nil
}

// PluginNotImplementingScore doesn't implement the ScorePlugin interface.
type PluginNotImplementingScore struct{}

func (pl *PluginNotImplementingScore) Name() string {
	return pluginNotImplementingScore
}

// TestPlugin implements all Plugin interfaces.
type TestPlugin struct {
	name string
	inj  injectedResult
}

type TestPluginPreFilterExtension struct {
	inj injectedResult
}

func (e *TestPluginPreFilterExtension) AddPod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	return NewStatus(Code(e.inj.PreFilterAddPodStatus), "injected status")
}
func (e *TestPluginPreFilterExtension) RemovePod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	return NewStatus(Code(e.inj.PreFilterRemovePodStatus), "injected status")
}

func (pl *TestPlugin) Name() string {
	return pl.name
}

func (pl *TestPlugin) Score(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
	return 0, NewStatus(Code(pl.inj.ScoreStatus), "injected status")
}

func (pl *TestPlugin) ScoreExtensions() ScoreExtensions {
	return nil
}

func (pl *TestPlugin) PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status {
	return NewStatus(Code(pl.inj.PreFilterStatus), "injected status")
}

func (pl *TestPlugin) PreFilterExtensions() PreFilterExtensions {
	return &TestPluginPreFilterExtension{inj: pl.inj}
}

func (pl *TestPlugin) Filter(ctx context.Context, state *CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	return NewStatus(Code(pl.inj.FilterStatus), "injected filter status")
}

func (pl *TestPlugin) PostFilter(ctx context.Context, state *CycleState, pod *v1.Pod, nodes []*v1.Node, filteredNodesStatuses NodeToStatusMap) *Status {
	return NewStatus(Code(pl.inj.PostFilterStatus), "injected status")
}

func (pl *TestPlugin) Reserve(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.ReserveStatus), "injected status")
}

func (pl *TestPlugin) PreBind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.PreBindStatus), "injected status")
}

func (pl *TestPlugin) PostBind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) {}

func (pl *TestPlugin) Unreserve(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) {}

func (pl *TestPlugin) Permit(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (*Status, time.Duration) {
	return NewStatus(Code(pl.inj.PermitStatus), "injected status"), time.Duration(0)
}

func (pl *TestPlugin) Bind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.BindStatus), "injected status")
}

// TestPreFilterPlugin only implements PreFilterPlugin interface.
type TestPreFilterPlugin struct {
	PreFilterCalled int
}

func (pl *TestPreFilterPlugin) Name() string {
	return preFilterPluginName
}

func (pl *TestPreFilterPlugin) PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status {
	pl.PreFilterCalled++
	return nil
}

func (pl *TestPreFilterPlugin) PreFilterExtensions() PreFilterExtensions {
	return nil
}

// TestPreFilterWithExtensionsPlugin implements Add/Remove interfaces.
type TestPreFilterWithExtensionsPlugin struct {
	PreFilterCalled int
	AddCalled       int
	RemoveCalled    int
}

func (pl *TestPreFilterWithExtensionsPlugin) Name() string {
	return preFilterWithExtensionsPluginName
}

func (pl *TestPreFilterWithExtensionsPlugin) PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status {
	pl.PreFilterCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) AddPod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod,
	podToAdd *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	pl.AddCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) RemovePod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod,
	podToRemove *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	pl.RemoveCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) PreFilterExtensions() PreFilterExtensions {
	return pl
}

type TestDuplicatePlugin struct {
}

func (dp *TestDuplicatePlugin) Name() string {
	return duplicatePluginName
}

func (dp *TestDuplicatePlugin) PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status {
	return nil
}

func (dp *TestDuplicatePlugin) PreFilterExtensions() PreFilterExtensions {
	return nil
}

var _ PreFilterPlugin = &TestDuplicatePlugin{}

func newDuplicatePlugin(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
	return &TestDuplicatePlugin{}, nil
}

// TestPermitPlugin only implements PermitPlugin interface.
type TestPermitPlugin struct {
	PreFilterCalled int
}

func (pp *TestPermitPlugin) Name() string {
	return permitPlugin
}
func (pp *TestPermitPlugin) Permit(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (*Status, time.Duration) {
	return NewStatus(Wait, ""), time.Duration(10 * time.Second)
}

var _ QueueSortPlugin = &TestQueueSortPlugin{}

func newQueueSortPlugin(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
	return &TestQueueSortPlugin{}, nil
}

// TestQueueSortPlugin is a no-op implementation for QueueSort extension point.
type TestQueueSortPlugin struct{}

func (pl *TestQueueSortPlugin) Name() string {
	return queueSortPlugin
}

func (pl *TestQueueSortPlugin) Less(_, _ *PodInfo) bool {
	return false
}

var _ BindPlugin = &TestBindPlugin{}

func newBindPlugin(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
	return &TestBindPlugin{}, nil
}

// TestBindPlugin is a no-op implementation for Bind extension point.
type TestBindPlugin struct{}

func (t TestBindPlugin) Name() string {
	return bindPlugin
}

func (t TestBindPlugin) Bind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status {
	return nil
}

var registry = func() Registry {
	r := make(Registry)
	r.Register(scoreWithNormalizePlugin1, newScoreWithNormalizePlugin1)
	r.Register(scoreWithNormalizePlugin2, newScoreWithNormalizePlugin2)
	r.Register(scorePlugin1, newScorePlugin1)
	r.Register(pluginNotImplementingScore, newPluginNotImplementingScore)
	r.Register(duplicatePluginName, newDuplicatePlugin)
	return r
}()

var defaultWeights = map[string]int32{
	scoreWithNormalizePlugin1: 1,
	scoreWithNormalizePlugin2: 2,
	scorePlugin1:              1,
}

var emptyArgs = make([]config.PluginConfig, 0)
var state = &CycleState{}

// Pod is only used for logging errors.
var pod = &v1.Pod{}
var nodes = []*v1.Node{
	{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
	{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
}

func newFrameworkWithQueueSortAndBind(r Registry, pl *config.Plugins, plc []config.PluginConfig, opts ...Option) (Framework, error) {
	if _, ok := r[queueSortPlugin]; !ok {
		r[queueSortPlugin] = newQueueSortPlugin
	}
	if _, ok := r[bindPlugin]; !ok {
		r[bindPlugin] = newBindPlugin
	}
	plugins := &config.Plugins{}
	plugins.Append(pl)
	if plugins.QueueSort == nil || len(plugins.QueueSort.Enabled) == 0 {
		plugins.Append(&config.Plugins{
			QueueSort: &config.PluginSet{
				Enabled: []config.Plugin{{Name: queueSortPlugin}},
			},
		})
	}
	if plugins.Bind == nil || len(plugins.Bind.Enabled) == 0 {
		plugins.Append(&config.Plugins{
			Bind: &config.PluginSet{
				Enabled: []config.Plugin{{Name: bindPlugin}},
			},
		})
	}
	return NewFramework(r, plugins, plc, opts...)
}

func TestInitFrameworkWithScorePlugins(t *testing.T) {
	tests := []struct {
		name    string
		plugins *config.Plugins
		// If initErr is true, we expect framework initialization to fail.
		initErr bool
	}{
		{
			name:    "enabled Score plugin doesn't exist in registry",
			plugins: buildScoreConfigDefaultWeights("notExist"),
			initErr: true,
		},
		{
			name:    "enabled Score plugin doesn't extend the ScorePlugin interface",
			plugins: buildScoreConfigDefaultWeights(pluginNotImplementingScore),
			initErr: true,
		},
		{
			name:    "Score plugins are nil",
			plugins: &config.Plugins{Score: nil},
		},
		{
			name:    "enabled Score plugin list is empty",
			plugins: buildScoreConfigDefaultWeights(),
		},
		{
			name:    "enabled plugin only implements ScorePlugin interface",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
		},
		{
			name:    "enabled plugin implements ScoreWithNormalizePlugin interface",
			plugins: buildScoreConfigDefaultWeights(scoreWithNormalizePlugin1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newFrameworkWithQueueSortAndBind(registry, tt.plugins, emptyArgs)
			if tt.initErr && err == nil {
				t.Fatal("Framework initialization should fail")
			}
			if !tt.initErr && err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
		})
	}
}

func TestRegisterDuplicatePluginWouldFail(t *testing.T) {
	plugin := config.Plugin{Name: duplicatePluginName, Weight: 1}

	pluginSet := config.PluginSet{
		Enabled: []config.Plugin{
			plugin,
			plugin,
		},
	}
	plugins := config.Plugins{}
	plugins.PreFilter = &pluginSet

	_, err := NewFramework(registry, &plugins, emptyArgs)
	if err == nil {
		t.Fatal("Framework initialization should fail")
	}

	if err != nil && !strings.Contains(err.Error(), "already registered") {
		t.Fatalf("Unexpected error, got %s, expect: plugin already registered", err.Error())
	}
}

func TestRunScorePlugins(t *testing.T) {
	tests := []struct {
		name          string
		registry      Registry
		plugins       *config.Plugins
		pluginConfigs []config.PluginConfig
		want          PluginToNodeScores
		// If err is true, we expect RunScorePlugin to fail.
		err bool
	}{
		{
			name:    "no Score plugins",
			plugins: buildScoreConfigDefaultWeights(),
			want:    PluginToNodeScores{},
		},
		{
			name:    "single Score plugin",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
			},
			// scorePlugin1 Score returns 1, weight=1, so want=1.
			want: PluginToNodeScores{
				scorePlugin1: {{Name: "node1", Score: 1}, {Name: "node2", Score: 1}},
			},
		},
		{
			name: "single ScoreWithNormalize plugin",
			//registry: registry,
			plugins: buildScoreConfigDefaultWeights(scoreWithNormalizePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 10, "normalizeRes": 5 }`),
					},
				},
			},
			// scoreWithNormalizePlugin1 Score returns 10, but NormalizeScore overrides to 5, weight=1, so want=5
			want: PluginToNodeScores{
				scoreWithNormalizePlugin1: {{Name: "node1", Score: 5}, {Name: "node2", Score: 5}},
			},
		},
		{
			name:    "2 Score plugins, 2 NormalizeScore plugins",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1, scoreWithNormalizePlugin2),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 3, "normalizeRes": 4}`),
					},
				},
				{
					Name: scoreWithNormalizePlugin2,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 4, "normalizeRes": 5}`),
					},
				},
			},
			// scorePlugin1 Score returns 1, weight =1, so want=1.
			// scoreWithNormalizePlugin1 Score returns 3, but NormalizeScore overrides to 4, weight=1, so want=4.
			// scoreWithNormalizePlugin2 Score returns 4, but NormalizeScore overrides to 5, weight=2, so want=10.
			want: PluginToNodeScores{
				scorePlugin1:              {{Name: "node1", Score: 1}, {Name: "node2", Score: 1}},
				scoreWithNormalizePlugin1: {{Name: "node1", Score: 4}, {Name: "node2", Score: 4}},
				scoreWithNormalizePlugin2: {{Name: "node1", Score: 10}, {Name: "node2", Score: 10}},
			},
		},
		{
			name: "score fails",
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "scoreStatus": 1 }`),
					},
				},
			},
			plugins: buildScoreConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1),
			err:     true,
		},
		{
			name: "normalize fails",
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(`{ "normalizeStatus": 1 }`),
					},
				},
			},
			plugins: buildScoreConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1),
			err:     true,
		},
		{
			name:    "Score plugin return score greater than MaxNodeScore",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "scoreRes": %d }`, MaxNodeScore+1)),
					},
				},
			},
			err: true,
		},
		{
			name:    "Score plugin return score less than MinNodeScore",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "scoreRes": %d }`, MinNodeScore-1)),
					},
				},
			},
			err: true,
		},
		{
			name:    "ScoreWithNormalize plugin return score greater than MaxNodeScore",
			plugins: buildScoreConfigDefaultWeights(scoreWithNormalizePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "normalizeRes": %d }`, MaxNodeScore+1)),
					},
				},
			},
			err: true,
		},
		{
			name:    "ScoreWithNormalize plugin return score less than MinNodeScore",
			plugins: buildScoreConfigDefaultWeights(scoreWithNormalizePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "normalizeRes": %d }`, MinNodeScore-1)),
					},
				},
			},
			err: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Inject the results via Args in PluginConfig.
			f, err := newFrameworkWithQueueSortAndBind(registry, tt.plugins, tt.pluginConfigs)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			res, status := f.RunScorePlugins(context.Background(), state, pod, nodes)

			if tt.err {
				if status.IsSuccess() {
					t.Errorf("Expected status to be non-success. got: %v", status.Code().String())
				}
				return
			}

			if !status.IsSuccess() {
				t.Errorf("Expected status to be success.")
			}
			if !reflect.DeepEqual(res, tt.want) {
				t.Errorf("Score map after RunScorePlugin: %+v, want: %+v.", res, tt.want)
			}
		})
	}
}

func TestPreFilterPlugins(t *testing.T) {
	preFilter1 := &TestPreFilterPlugin{}
	preFilter2 := &TestPreFilterWithExtensionsPlugin{}
	r := make(Registry)
	r.Register(preFilterPluginName,
		func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
			return preFilter1, nil
		})
	r.Register(preFilterWithExtensionsPluginName,
		func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
			return preFilter2, nil
		})
	plugins := &config.Plugins{PreFilter: &config.PluginSet{Enabled: []config.Plugin{{Name: preFilterWithExtensionsPluginName}, {Name: preFilterPluginName}}}}
	t.Run("TestPreFilterPlugin", func(t *testing.T) {
		f, err := newFrameworkWithQueueSortAndBind(r, plugins, emptyArgs)
		if err != nil {
			t.Fatalf("Failed to create framework for testing: %v", err)
		}
		f.RunPreFilterPlugins(context.Background(), nil, nil)
		f.RunPreFilterExtensionAddPod(context.Background(), nil, nil, nil, nil)
		f.RunPreFilterExtensionRemovePod(context.Background(), nil, nil, nil, nil)

		if preFilter1.PreFilterCalled != 1 {
			t.Errorf("preFilter1 called %v, expected: 1", preFilter1.PreFilterCalled)
		}
		if preFilter2.PreFilterCalled != 1 {
			t.Errorf("preFilter2 called %v, expected: 1", preFilter2.PreFilterCalled)
		}
		if preFilter2.AddCalled != 1 {
			t.Errorf("AddPod called %v, expected: 1", preFilter2.AddCalled)
		}
		if preFilter2.RemoveCalled != 1 {
			t.Errorf("AddPod called %v, expected: 1", preFilter2.RemoveCalled)
		}
	})
}

func TestFilterPlugins(t *testing.T) {
	tests := []struct {
		name          string
		plugins       []*TestPlugin
		wantStatus    *Status
		wantStatusMap PluginToStatus
		runAllFilters bool
	}{
		{
			name: "SuccessFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(Success)},
				},
			},
			wantStatus:    nil,
			wantStatusMap: PluginToStatus{},
		},
		{
			name: "ErrorFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(Error)},
				},
			},
			wantStatus:    NewStatus(Error, `running "TestPlugin" filter plugin for pod "": injected filter status`),
			wantStatusMap: PluginToStatus{"TestPlugin": NewStatus(Error, `running "TestPlugin" filter plugin for pod "": injected filter status`)},
		},
		{
			name: "UnschedulableFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(Unschedulable)},
				},
			},
			wantStatus:    NewStatus(Unschedulable, "injected filter status"),
			wantStatusMap: PluginToStatus{"TestPlugin": NewStatus(Unschedulable, "injected filter status")},
		},
		{
			name: "UnschedulableAndUnresolvableFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj: injectedResult{
						FilterStatus: int(UnschedulableAndUnresolvable)},
				},
			},
			wantStatus:    NewStatus(UnschedulableAndUnresolvable, "injected filter status"),
			wantStatusMap: PluginToStatus{"TestPlugin": NewStatus(UnschedulableAndUnresolvable, "injected filter status")},
		},
		// followings tests cover multiple-plugins scenarios
		{
			name: "ErrorAndErrorFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Error)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Error)},
				},
			},
			wantStatus:    NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`),
			wantStatusMap: PluginToStatus{"TestPlugin1": NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`)},
		},
		{
			name: "SuccessAndSuccessFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Success)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Success)},
				},
			},
			wantStatus:    nil,
			wantStatusMap: PluginToStatus{},
		},
		{
			name: "ErrorAndSuccessFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Error)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Success)},
				},
			},
			wantStatus:    NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`),
			wantStatusMap: PluginToStatus{"TestPlugin1": NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`)},
		},
		{
			name: "SuccessAndErrorFilters",
			plugins: []*TestPlugin{
				{

					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Success)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Error)},
				},
			},
			wantStatus:    NewStatus(Error, `running "TestPlugin2" filter plugin for pod "": injected filter status`),
			wantStatusMap: PluginToStatus{"TestPlugin2": NewStatus(Error, `running "TestPlugin2" filter plugin for pod "": injected filter status`)},
		},
		{
			name: "SuccessAndUnschedulableFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Success)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Unschedulable)},
				},
			},
			wantStatus:    NewStatus(Unschedulable, "injected filter status"),
			wantStatusMap: PluginToStatus{"TestPlugin2": NewStatus(Unschedulable, "injected filter status")},
		},
		{
			name: "SuccessFilterWithRunAllFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(Success)},
				},
			},
			runAllFilters: true,
			wantStatus:    nil,
			wantStatusMap: PluginToStatus{},
		},
		{
			name: "ErrorAndErrorFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(Error)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Error)},
				},
			},
			runAllFilters: true,
			wantStatus:    NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`),
			wantStatusMap: PluginToStatus{"TestPlugin1": NewStatus(Error, `running "TestPlugin1" filter plugin for pod "": injected filter status`)},
		},
		{
			name: "ErrorAndErrorFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(UnschedulableAndUnresolvable)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(Unschedulable)},
				},
			},
			runAllFilters: true,
			wantStatus:    NewStatus(UnschedulableAndUnresolvable, "injected filter status", "injected filter status"),
			wantStatusMap: PluginToStatus{
				"TestPlugin1": NewStatus(UnschedulableAndUnresolvable, "injected filter status"),
				"TestPlugin2": NewStatus(Unschedulable, "injected filter status"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := Registry{}
			cfgPls := &config.Plugins{Filter: &config.PluginSet{}}
			for _, pl := range tt.plugins {
				// register all plugins
				tmpPl := pl
				if err := registry.Register(pl.name,
					func(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
						return tmpPl, nil
					}); err != nil {
					t.Fatalf("fail to register filter plugin (%s)", pl.name)
				}
				// append plugins to filter pluginset
				cfgPls.Filter.Enabled = append(
					cfgPls.Filter.Enabled,
					config.Plugin{Name: pl.name})
			}

			f, err := newFrameworkWithQueueSortAndBind(registry, cfgPls, emptyArgs, WithRunAllFilters(tt.runAllFilters))
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			gotStatusMap := f.RunFilterPlugins(context.TODO(), nil, pod, nil)
			gotStatus := gotStatusMap.Merge()
			if !reflect.DeepEqual(gotStatus, tt.wantStatus) {
				t.Errorf("wrong status code. got: %v, want:%v", gotStatus, tt.wantStatus)
			}
			if !reflect.DeepEqual(gotStatusMap, tt.wantStatusMap) {
				t.Errorf("wrong status map. got: %+v, want: %+v", gotStatusMap, tt.wantStatusMap)
			}

		})
	}
}

func TestRecordingMetrics(t *testing.T) {
	state := &CycleState{
		recordPluginMetrics: true,
	}
	tests := []struct {
		name               string
		action             func(f Framework)
		inject             injectedResult
		wantExtensionPoint string
		wantStatus         Code
	}{
		{
			name:               "PreFilter - Success",
			action:             func(f Framework) { f.RunPreFilterPlugins(context.Background(), state, pod) },
			wantExtensionPoint: "PreFilter",
			wantStatus:         Success,
		},
		{
			name:               "PostFilter - Success",
			action:             func(f Framework) { f.RunPostFilterPlugins(context.Background(), state, pod, nil, nil) },
			wantExtensionPoint: "PostFilter",
			wantStatus:         Success,
		},
		{
			name:               "Score - Success",
			action:             func(f Framework) { f.RunScorePlugins(context.Background(), state, pod, nodes) },
			wantExtensionPoint: "Score",
			wantStatus:         Success,
		},
		{
			name:               "Reserve - Success",
			action:             func(f Framework) { f.RunReservePlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "Reserve",
			wantStatus:         Success,
		},
		{
			name:               "Unreserve - Success",
			action:             func(f Framework) { f.RunUnreservePlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "Unreserve",
			wantStatus:         Success,
		},
		{
			name:               "PreBind - Success",
			action:             func(f Framework) { f.RunPreBindPlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "PreBind",
			wantStatus:         Success,
		},
		{
			name:               "Bind - Success",
			action:             func(f Framework) { f.RunBindPlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "Bind",
			wantStatus:         Success,
		},
		{
			name:               "PostBind - Success",
			action:             func(f Framework) { f.RunPostBindPlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "PostBind",
			wantStatus:         Success,
		},
		{
			name:               "Permit - Success",
			action:             func(f Framework) { f.RunPermitPlugins(context.Background(), state, pod, "") },
			wantExtensionPoint: "Permit",
			wantStatus:         Success,
		},

		{
			name:               "PreFilter - Error",
			action:             func(f Framework) { f.RunPreFilterPlugins(context.Background(), state, pod) },
			inject:             injectedResult{PreFilterStatus: int(Error)},
			wantExtensionPoint: "PreFilter",
			wantStatus:         Error,
		},
		{
			name:               "PostFilter - Error",
			action:             func(f Framework) { f.RunPostFilterPlugins(context.Background(), state, pod, nil, nil) },
			inject:             injectedResult{PostFilterStatus: int(Error)},
			wantExtensionPoint: "PostFilter",
			wantStatus:         Error,
		},
		{
			name:               "Score - Error",
			action:             func(f Framework) { f.RunScorePlugins(context.Background(), state, pod, nodes) },
			inject:             injectedResult{ScoreStatus: int(Error)},
			wantExtensionPoint: "Score",
			wantStatus:         Error,
		},
		{
			name:               "Reserve - Error",
			action:             func(f Framework) { f.RunReservePlugins(context.Background(), state, pod, "") },
			inject:             injectedResult{ReserveStatus: int(Error)},
			wantExtensionPoint: "Reserve",
			wantStatus:         Error,
		},
		{
			name:               "PreBind - Error",
			action:             func(f Framework) { f.RunPreBindPlugins(context.Background(), state, pod, "") },
			inject:             injectedResult{PreBindStatus: int(Error)},
			wantExtensionPoint: "PreBind",
			wantStatus:         Error,
		},
		{
			name:               "Bind - Error",
			action:             func(f Framework) { f.RunBindPlugins(context.Background(), state, pod, "") },
			inject:             injectedResult{BindStatus: int(Error)},
			wantExtensionPoint: "Bind",
			wantStatus:         Error,
		},
		{
			name:               "Permit - Error",
			action:             func(f Framework) { f.RunPermitPlugins(context.Background(), state, pod, "") },
			inject:             injectedResult{PermitStatus: int(Error)},
			wantExtensionPoint: "Permit",
			wantStatus:         Error,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.FrameworkExtensionPointDuration.Reset()
			metrics.PluginExecutionDuration.Reset()

			plugin := &TestPlugin{name: testPlugin, inj: tt.inject}
			r := make(Registry)
			r.Register(testPlugin,
				func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
					return plugin, nil
				})
			pluginSet := &config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}}
			plugins := &config.Plugins{
				Score:      pluginSet,
				PreFilter:  pluginSet,
				Filter:     pluginSet,
				PostFilter: pluginSet,
				Reserve:    pluginSet,
				Permit:     pluginSet,
				PreBind:    pluginSet,
				Bind:       pluginSet,
				PostBind:   pluginSet,
				Unreserve:  pluginSet,
			}
			recorder := newMetricsRecorder(100, time.Nanosecond)
			f, err := newFrameworkWithQueueSortAndBind(r, plugins, emptyArgs, withMetricsRecorder(recorder))
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			tt.action(f)

			// Stop the goroutine which records metrics and ensure it's stopped.
			close(recorder.stopCh)
			<-recorder.isStoppedCh
			// Try to clean up the metrics buffer again in case it's not empty.
			recorder.flushMetrics()

			collectAndCompareFrameworkMetrics(t, tt.wantExtensionPoint, tt.wantStatus)
			collectAndComparePluginMetrics(t, tt.wantExtensionPoint, testPlugin, tt.wantStatus)
		})
	}
}

func TestRunBindPlugins(t *testing.T) {
	tests := []struct {
		name       string
		injects    []Code
		wantStatus Code
	}{
		{
			name:       "simple success",
			injects:    []Code{Success},
			wantStatus: Success,
		},
		{
			name:       "error on second",
			injects:    []Code{Skip, Error, Success},
			wantStatus: Error,
		},
		{
			name:       "all skip",
			injects:    []Code{Skip, Skip, Skip},
			wantStatus: Skip,
		},
		{
			name:       "error on third, but not reached",
			injects:    []Code{Skip, Success, Error},
			wantStatus: Success,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.FrameworkExtensionPointDuration.Reset()
			metrics.PluginExecutionDuration.Reset()

			pluginSet := &config.PluginSet{}
			r := make(Registry)
			for i, inj := range tt.injects {
				name := fmt.Sprintf("bind-%d", i)
				plugin := &TestPlugin{name: name, inj: injectedResult{BindStatus: int(inj)}}
				r.Register(name,
					func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
						return plugin, nil
					})
				pluginSet.Enabled = append(pluginSet.Enabled, config.Plugin{Name: name})
			}
			plugins := &config.Plugins{Bind: pluginSet}
			recorder := newMetricsRecorder(100, time.Nanosecond)
			fwk, err := newFrameworkWithQueueSortAndBind(r, plugins, emptyArgs, withMetricsRecorder(recorder))
			if err != nil {
				t.Fatal(err)
			}

			st := fwk.RunBindPlugins(context.Background(), state, pod, "")
			if st.Code() != tt.wantStatus {
				t.Errorf("got status code %s, want %s", st.Code(), tt.wantStatus)
			}

			// Stop the goroutine which records metrics and ensure it's stopped.
			close(recorder.stopCh)
			<-recorder.isStoppedCh
			// Try to clean up the metrics buffer again in case it's not empty.
			recorder.flushMetrics()
			collectAndCompareFrameworkMetrics(t, "Bind", tt.wantStatus)
		})
	}
}

func TestPermitWaitingMetric(t *testing.T) {
	tests := []struct {
		name    string
		inject  injectedResult
		wantRes string
	}{
		{
			name: "Permit - Success",
		},
		{
			name:    "Permit - Wait Timeout",
			inject:  injectedResult{PermitStatus: int(Wait)},
			wantRes: "Unschedulable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.PermitWaitDuration.Reset()

			plugin := &TestPlugin{name: testPlugin, inj: tt.inject}
			r := make(Registry)
			err := r.Register(testPlugin,
				func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
					return plugin, nil
				})
			if err != nil {
				t.Fatal(err)
			}
			plugins := &config.Plugins{
				Permit: &config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}},
			}
			f, err := newFrameworkWithQueueSortAndBind(r, plugins, emptyArgs)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			f.RunPermitPlugins(context.TODO(), nil, pod, "")

			collectAndComparePermitWaitDuration(t, tt.wantRes)
		})
	}
}

func TestRejectWaitingPod(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod",
			UID:  types.UID("pod"),
		},
	}

	testPermitPlugin := &TestPermitPlugin{}
	r := make(Registry)
	r.Register(permitPlugin,
		func(_ *runtime.Unknown, fh FrameworkHandle) (Plugin, error) {
			return testPermitPlugin, nil
		})
	plugins := &config.Plugins{
		Permit: &config.PluginSet{Enabled: []config.Plugin{{Name: permitPlugin, Weight: 1}}},
	}

	f, err := newFrameworkWithQueueSortAndBind(r, plugins, emptyArgs)
	if err != nil {
		t.Fatalf("Failed to create framework for testing: %v", err)
	}

	go func() {
		for {
			waitingPod := f.GetWaitingPod(pod.UID)
			if waitingPod != nil {
				break
			}
		}
		f.RejectWaitingPod(pod.UID)
	}()
	permitStatus := f.RunPermitPlugins(context.Background(), nil, pod, "")
	if permitStatus.Message() != "pod \"pod\" rejected while waiting at permit: removed" {
		t.Fatalf("RejectWaitingPod failed, permitStatus: %v", permitStatus)
	}
}

func buildScoreConfigDefaultWeights(ps ...string) *config.Plugins {
	return buildScoreConfigWithWeights(defaultWeights, ps...)
}

func buildScoreConfigWithWeights(weights map[string]int32, ps ...string) *config.Plugins {
	var plugins []config.Plugin
	for _, p := range ps {
		plugins = append(plugins, config.Plugin{Name: p, Weight: weights[p]})
	}
	return &config.Plugins{Score: &config.PluginSet{Enabled: plugins}}
}

type injectedResult struct {
	ScoreRes                 int64 `json:"scoreRes,omitempty"`
	NormalizeRes             int64 `json:"normalizeRes,omitempty"`
	ScoreStatus              int   `json:"scoreStatus,omitempty"`
	NormalizeStatus          int   `json:"normalizeStatus,omitempty"`
	PreFilterStatus          int   `json:"preFilterStatus,omitempty"`
	PreFilterAddPodStatus    int   `json:"preFilterAddPodStatus,omitempty"`
	PreFilterRemovePodStatus int   `json:"preFilterRemovePodStatus,omitempty"`
	FilterStatus             int   `json:"filterStatus,omitempty"`
	PostFilterStatus         int   `json:"postFilterStatus,omitempty"`
	ReserveStatus            int   `json:"reserveStatus,omitempty"`
	PreBindStatus            int   `json:"preBindStatus,omitempty"`
	BindStatus               int   `json:"bindStatus,omitempty"`
	PermitStatus             int   `json:"permitStatus,omitempty"`
}

func setScoreRes(inj injectedResult) (int64, *Status) {
	if Code(inj.ScoreStatus) != Success {
		return 0, NewStatus(Code(inj.ScoreStatus), "injecting failure.")
	}
	return inj.ScoreRes, nil
}

func injectNormalizeRes(inj injectedResult, scores NodeScoreList) *Status {
	if Code(inj.NormalizeStatus) != Success {
		return NewStatus(Code(inj.NormalizeStatus), "injecting failure.")
	}
	for i := range scores {
		scores[i].Score = inj.NormalizeRes
	}
	return nil
}

func collectAndComparePluginMetrics(t *testing.T, wantExtensionPoint, wantPlugin string, wantStatus Code) {
	t.Helper()
	m := collectHistogramMetric(metrics.PluginExecutionDuration)
	if len(m.Label) != 3 {
		t.Fatalf("Unexpected number of label pairs, got: %v, want: 2", len(m.Label))
	}

	if *m.Label[0].Value != wantExtensionPoint {
		t.Errorf("Unexpected extension point label, got: %q, want %q", *m.Label[0].Value, wantExtensionPoint)
	}

	if *m.Label[1].Value != wantPlugin {
		t.Errorf("Unexpected plugin label, got: %q, want %q", *m.Label[1].Value, wantPlugin)
	}

	if *m.Label[2].Value != wantStatus.String() {
		t.Errorf("Unexpected status code label, got: %q, want %q", *m.Label[2].Value, wantStatus)
	}

	if *m.Histogram.SampleCount == 0 {
		t.Error("Expect at least 1 sample")
	}

	if *m.Histogram.SampleSum <= 0 {
		t.Errorf("Expect latency to be greater than 0, got: %v", *m.Histogram.SampleSum)
	}
}

func collectAndCompareFrameworkMetrics(t *testing.T, wantExtensionPoint string, wantStatus Code) {
	t.Helper()
	m := collectHistogramMetric(metrics.FrameworkExtensionPointDuration)

	if len(m.Label) != 2 {
		t.Fatalf("Unexpected number of label pairs, got: %v, want: 2", len(m.Label))
	}

	if *m.Label[0].Value != wantExtensionPoint {
		t.Errorf("Unexpected extension point label, got: %q, want %q", *m.Label[0].Value, wantExtensionPoint)
	}

	if *m.Label[1].Value != wantStatus.String() {
		t.Errorf("Unexpected status code label, got: %q, want %q", *m.Label[1].Value, wantStatus)
	}

	if *m.Histogram.SampleCount != 1 {
		t.Errorf("Expect 1 sample, got: %v", *m.Histogram.SampleCount)
	}

	if *m.Histogram.SampleSum <= 0 {
		t.Errorf("Expect latency to be greater than 0, got: %v", *m.Histogram.SampleSum)
	}
}

func collectAndComparePermitWaitDuration(t *testing.T, wantRes string) {
	m := collectHistogramMetric(metrics.PermitWaitDuration)
	if wantRes == "" {
		if m != nil {
			t.Errorf("PermitWaitDuration shouldn't be recorded but got %+v", m)
		}
		return
	}
	if wantRes != "" {
		if len(m.Label) != 1 {
			t.Fatalf("Unexpected number of label pairs, got: %v, want: 1", len(m.Label))
		}

		if *m.Label[0].Value != wantRes {
			t.Errorf("Unexpected result label, got: %q, want %q", *m.Label[0].Value, wantRes)
		}

		if *m.Histogram.SampleCount != 1 {
			t.Errorf("Expect 1 sample, got: %v", *m.Histogram.SampleCount)
		}

		if *m.Histogram.SampleSum <= 0 {
			t.Errorf("Expect latency to be greater than 0, got: %v", *m.Histogram.SampleSum)
		}
	}
}

func collectHistogramMetric(metric prometheus.Collector) *dto.Metric {
	ch := make(chan prometheus.Metric, 100)
	metric.Collect(ch)
	select {
	case got := <-ch:
		m := &dto.Metric{}
		got.Write(m)
		return m
	default:
		return nil
	}
}
