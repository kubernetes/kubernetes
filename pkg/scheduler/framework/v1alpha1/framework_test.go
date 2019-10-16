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
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"reflect"
	"strings"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

const (
	scoreWithNormalizePlugin1         = "score-with-normalize-plugin-1"
	scoreWithNormalizePlugin2         = "score-with-normalize-plugin-2"
	scorePlugin1                      = "score-plugin-1"
	pluginNotImplementingScore        = "plugin-not-implementing-score"
	preFilterPluginName               = "prefilter-plugin"
	preFilterWithExtensionsPluginName = "prefilter-with-extensions-plugin"
	duplicatePluginName               = "duplicate-plugin"
	testPlugin                        = "test-plugin"
)

// TestScoreWithNormalizePlugin implements ScoreWithNormalizePlugin interface.
// TestScorePlugin only implements ScorePlugin interface.
var _ = ScorePlugin(&TestScoreWithNormalizePlugin{})
var _ = ScorePlugin(&TestScorePlugin{})

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

func (pl *TestScoreWithNormalizePlugin) NormalizeScore(state *CycleState, pod *v1.Pod, scores NodeScoreList) *Status {
	return injectNormalizeRes(pl.inj, scores)
}

func (pl *TestScoreWithNormalizePlugin) Score(state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
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

func (pl *TestScorePlugin) Score(state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
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

func (pl *TestPlugin) Name() string {
	return pl.name
}

func (pl *TestPlugin) Score(state *CycleState, p *v1.Pod, nodeName string) (int64, *Status) {
	return 0, NewStatus(Code(pl.inj.ScoreStatus), "injected status")
}

func (pl *TestPlugin) ScoreExtensions() ScoreExtensions {
	return nil
}

func (pl *TestPlugin) PreFilter(state *CycleState, p *v1.Pod) *Status {
	return NewStatus(Code(pl.inj.PreFilterStatus), "injected status")
}
func (pl *TestPlugin) PreFilterExtensions() PreFilterExtensions {
	return nil
}
func (pl *TestPlugin) Filter(state *CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	return NewStatus(Code(pl.inj.FilterStatus), "injected status")
}
func (pl *TestPlugin) PostFilter(state *CycleState, pod *v1.Pod, nodes []*v1.Node, filteredNodesStatuses NodeToStatusMap) *Status {
	return NewStatus(Code(pl.inj.PostFilterStatus), "injected status")
}
func (pl *TestPlugin) Reserve(state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.ReserveStatus), "injected status")
}
func (pl *TestPlugin) PreBind(state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.PreBindStatus), "injected status")
}
func (pl *TestPlugin) PostBind(state *CycleState, p *v1.Pod, nodeName string)  {}
func (pl *TestPlugin) Unreserve(state *CycleState, p *v1.Pod, nodeName string) {}
func (pl *TestPlugin) Permit(state *CycleState, p *v1.Pod, nodeName string) (*Status, time.Duration) {
	return NewStatus(Code(pl.inj.PermitStatus), "injected status"), time.Duration(0)
}
func (pl *TestPlugin) Bind(state *CycleState, p *v1.Pod, nodeName string) *Status {
	return NewStatus(Code(pl.inj.BindStatus), "injected status")
}

// TestPreFilterPlugin only implements PreFilterPlugin interface.
type TestPreFilterPlugin struct {
	PreFilterCalled int
}

func (pl *TestPreFilterPlugin) Name() string {
	return preFilterPluginName
}

func (pl *TestPreFilterPlugin) PreFilter(state *CycleState, p *v1.Pod) *Status {
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

func (pl *TestPreFilterWithExtensionsPlugin) PreFilter(state *CycleState, p *v1.Pod) *Status {
	pl.PreFilterCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) AddPod(state *CycleState, podToSchedule *v1.Pod,
	podToAdd *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *Status {
	pl.AddCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) RemovePod(state *CycleState, podToSchedule *v1.Pod,
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

func (dp *TestDuplicatePlugin) PreFilter(state *CycleState, p *v1.Pod) *Status {
	return nil
}

func (dp *TestDuplicatePlugin) PreFilterExtensions() PreFilterExtensions {
	return nil
}

var _ PreFilterPlugin = &TestDuplicatePlugin{}

func newDuplicatePlugin(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
	return &TestDuplicatePlugin{}, nil
}

var registry Registry = func() Registry {
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

var emptyArgs []config.PluginConfig = make([]config.PluginConfig, 0)
var state = &CycleState{}

// Pod is only used for logging errors.
var pod = &v1.Pod{}
var nodes = []*v1.Node{
	{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
	{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
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
			_, err := NewFramework(registry, tt.plugins, emptyArgs)
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
			f, err := NewFramework(registry, tt.plugins, tt.pluginConfigs)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			res, status := f.RunScorePlugins(state, pod, nodes)

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
		f, err := NewFramework(r, plugins, emptyArgs)
		if err != nil {
			t.Fatalf("Failed to create framework for testing: %v", err)
		}
		f.RunPreFilterPlugins(nil, nil)
		f.RunPreFilterExtensionAddPod(nil, nil, nil, nil)
		f.RunPreFilterExtensionRemovePod(nil, nil, nil, nil)

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

func TestRecordingMetrics(t *testing.T) {
	tests := []struct {
		name               string
		action             func(f Framework)
		inject             injectedResult
		wantExtensionPoint string
		wantStatus         Code
	}{
		{
			name:               "PreFilter - Success",
			action:             func(f Framework) { f.RunPreFilterPlugins(nil, pod) },
			wantExtensionPoint: "PreFilter",
			wantStatus:         Success,
		},
		{
			name:               "Filter - Success",
			action:             func(f Framework) { f.RunFilterPlugins(nil, pod, nil) },
			wantExtensionPoint: "Filter",
			wantStatus:         Success,
		},
		{
			name:               "PostFilter - Success",
			action:             func(f Framework) { f.RunPostFilterPlugins(nil, pod, nil, nil) },
			wantExtensionPoint: "PostFilter",
			wantStatus:         Success,
		},
		{
			name:               "Score - Success",
			action:             func(f Framework) { f.RunScorePlugins(nil, pod, nodes) },
			wantExtensionPoint: "Score",
			wantStatus:         Success,
		},
		{
			name:               "Reserve - Success",
			action:             func(f Framework) { f.RunReservePlugins(nil, pod, "") },
			wantExtensionPoint: "Reserve",
			wantStatus:         Success,
		},
		{
			name:               "Unreserve - Success",
			action:             func(f Framework) { f.RunUnreservePlugins(nil, pod, "") },
			wantExtensionPoint: "Unreserve",
			wantStatus:         Success,
		},
		{
			name:               "PreBind - Success",
			action:             func(f Framework) { f.RunPreBindPlugins(nil, pod, "") },
			wantExtensionPoint: "PreBind",
			wantStatus:         Success,
		},
		{
			name:               "Bind - Success",
			action:             func(f Framework) { f.RunBindPlugins(nil, pod, "") },
			wantExtensionPoint: "Bind",
			wantStatus:         Success,
		},
		{
			name:               "PostBind - Success",
			action:             func(f Framework) { f.RunPostBindPlugins(nil, pod, "") },
			wantExtensionPoint: "PostBind",
			wantStatus:         Success,
		},
		{
			name:               "Permit - Success",
			action:             func(f Framework) { f.RunPermitPlugins(nil, pod, "") },
			wantExtensionPoint: "Permit",
			wantStatus:         Success,
		},

		{
			name:               "PreFilter - Error",
			action:             func(f Framework) { f.RunPreFilterPlugins(nil, pod) },
			inject:             injectedResult{PreFilterStatus: int(Error)},
			wantExtensionPoint: "PreFilter",
			wantStatus:         Error,
		},
		{
			name:               "Filter - Error",
			action:             func(f Framework) { f.RunFilterPlugins(nil, pod, nil) },
			inject:             injectedResult{FilterStatus: int(Error)},
			wantExtensionPoint: "Filter",
			wantStatus:         Error,
		},
		{
			name:               "PostFilter - Error",
			action:             func(f Framework) { f.RunPostFilterPlugins(nil, pod, nil, nil) },
			inject:             injectedResult{PostFilterStatus: int(Error)},
			wantExtensionPoint: "PostFilter",
			wantStatus:         Error,
		},
		{
			name:               "Score - Error",
			action:             func(f Framework) { f.RunScorePlugins(nil, pod, nodes) },
			inject:             injectedResult{ScoreStatus: int(Error)},
			wantExtensionPoint: "Score",
			wantStatus:         Error,
		},
		{
			name:               "Reserve - Error",
			action:             func(f Framework) { f.RunReservePlugins(nil, pod, "") },
			inject:             injectedResult{ReserveStatus: int(Error)},
			wantExtensionPoint: "Reserve",
			wantStatus:         Error,
		},
		{
			name:               "PreBind - Error",
			action:             func(f Framework) { f.RunPreBindPlugins(nil, pod, "") },
			inject:             injectedResult{PreBindStatus: int(Error)},
			wantExtensionPoint: "PreBind",
			wantStatus:         Error,
		},
		{
			name:               "Bind - Error",
			action:             func(f Framework) { f.RunBindPlugins(nil, pod, "") },
			inject:             injectedResult{BindStatus: int(Error)},
			wantExtensionPoint: "Bind",
			wantStatus:         Error,
		},
		{
			name:               "Permit - Error",
			action:             func(f Framework) { f.RunPermitPlugins(nil, pod, "") },
			inject:             injectedResult{PermitStatus: int(Error)},
			wantExtensionPoint: "Permit",
			wantStatus:         Error,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
			f, err := NewFramework(r, plugins, emptyArgs)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			metrics.Register()
			metrics.FrameworkExtensionPointDuration.Reset()

			tt.action(f)

			collectAndCompare(t, tt.wantExtensionPoint, tt.wantStatus)
		})
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
	ScoreRes         int64 `json:"scoreRes,omitempty"`
	NormalizeRes     int64 `json:"normalizeRes,omitempty"`
	ScoreStatus      int   `json:"scoreStatus,omitempty"`
	NormalizeStatus  int   `json:"normalizeStatus,omitempty"`
	PreFilterStatus  int   `json:"preFilterStatus,omitempty"`
	FilterStatus     int   `json:"filterStatus,omitempty"`
	PostFilterStatus int   `json:"postFilterStatus,omitempty"`
	ReserveStatus    int   `json:"reserveStatus,omitempty"`
	PreBindStatus    int   `json:"preBindStatus,omitempty"`
	BindStatus       int   `json:"bindStatus,omitempty"`
	PermitStatus     int   `json:"permitStatus,omitempty"`
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

func collectAndCompare(t *testing.T, wantExtensionPoint string, wantStatus Code) {
	ch := make(chan prometheus.Metric, 1)
	m := &dto.Metric{}
	metrics.FrameworkExtensionPointDuration.Collect(ch)
	got := <-ch
	got.Write(m)

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
		t.Errorf("Expect 1 sample, got: %v", m.Histogram.SampleCount)
	}

	if *m.Histogram.SampleSum <= 0 {
		t.Errorf("Expect latency to be greater than 0, got: %v", m.Histogram.SampleSum)
	}
}
