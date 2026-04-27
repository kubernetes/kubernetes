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

package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

const (
	preEnqueuePlugin                  = "preEnqueue-plugin"
	queueSortPlugin                   = "no-op-queue-sort-plugin"
	scoreWithNormalizePlugin1         = "score-with-normalize-plugin-1"
	scoreWithNormalizePlugin2         = "score-with-normalize-plugin-2"
	scorePlugin1                      = "score-plugin-1"
	scorePlugin2                      = "score-plugin-2"
	placementScorePlugin1             = "placement-score-plugin-1"
	pluginNotImplementingScore        = "plugin-not-implementing-score"
	preFilterPluginName               = "prefilter-plugin"
	preFilterWithExtensionsPluginName = "prefilter-with-extensions-plugin"
	duplicatePluginName               = "duplicate-plugin"
	testPlugin                        = "test-plugin"
	permitPlugin                      = "permit-plugin"
	bindPlugin                        = "bind-plugin"
	testCloseErrorPlugin              = "test-close-error-plugin"
	placementGeneratePlugin           = "placement-generate-plugin"
	defaultPreemptionPlugin           = names.DefaultPreemption

	testProfileName              = "test-profile"
	testPercentageOfNodesToScore = 35
	nodeName                     = "testNode"

	injectReason       = "injected status"
	injectFilterReason = "injected filter status"
)

func init() {
	metrics.Register()
}

// TestScoreWithNormalizePlugin implements ScoreWithNormalizePlugin interface.
// TestScorePlugin only implements ScorePlugin interface.
var _ fwk.ScorePlugin = &TestScoreWithNormalizePlugin{}
var _ fwk.ScorePlugin = &TestScorePlugin{}

var statusCmpOpts = []cmp.Option{
	cmp.Comparer(func(s1 *fwk.Status, s2 *fwk.Status) bool {
		if s1 == nil || s2 == nil {
			return s1.IsSuccess() && s2.IsSuccess()
		}
		if s1.Code() == fwk.Error {
			return s1.AsError().Error() == s2.AsError().Error()
		}
		return s1.Code() == s2.Code() && s1.Plugin() == s2.Plugin() && s1.Message() == s2.Message()
	}),
}

func newScoreWithNormalizePlugin1(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin1, inj}, nil
}

func newScoreWithNormalizePlugin2(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin2, inj}, nil
}

func newScorePlugin1(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScorePlugin{scorePlugin1, inj}, nil
}

func newScorePlugin2(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestScorePlugin{scorePlugin2, inj}, nil
}

func newPlacementScorePluginFactory(name string) func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) {
	return func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) {
		return &testPlacementScorePlugin{name: name}, nil
	}
}

func newPluginNotImplementingScore(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &PluginNotImplementingScore{}, nil
}

type TestScoreWithNormalizePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestScoreWithNormalizePlugin) Name() string {
	return pl.name
}

func (pl *TestScoreWithNormalizePlugin) NormalizeScore(ctx context.Context, state fwk.CycleState, pod *v1.Pod, scores fwk.NodeScoreList) *fwk.Status {
	return injectNormalizeRes(pl.inj, scores)
}

func (pl *TestScoreWithNormalizePlugin) Score(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	return setScoreRes(pl.inj)
}

func (pl *TestScoreWithNormalizePlugin) ScoreExtensions() fwk.ScoreExtensions {
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

func (pl *TestScorePlugin) PreScore(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.PreScoreStatus), injectReason)
}

func (pl *TestScorePlugin) Score(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	return setScoreRes(pl.inj)
}

func (pl *TestScorePlugin) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}

// PluginNotImplementingScore doesn't implement the ScorePlugin interface.
type PluginNotImplementingScore struct{}

func (pl *PluginNotImplementingScore) Name() string {
	return pluginNotImplementingScore
}

func newTestPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &TestPlugin{name: testPlugin}, nil
}

// TestPlugin implements all Plugin interfaces.
type TestPlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestPlugin) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.PreFilterAddPodStatus), injectReason)
}
func (pl *TestPlugin) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.PreFilterRemovePodStatus), injectReason)
}

func (pl *TestPlugin) Name() string {
	return pl.name
}

func (pl *TestPlugin) Less(fwk.QueuedPodInfo, fwk.QueuedPodInfo) bool {
	return false
}

func (pl *TestPlugin) Score(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	return 0, fwk.NewStatus(fwk.Code(pl.inj.ScoreStatus), injectReason)
}

func (pl *TestPlugin) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}

func (pl *TestPlugin) PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return pl.inj.PreFilterResult, fwk.NewStatus(fwk.Code(pl.inj.PreFilterStatus), injectReason)
}

func (pl *TestPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return pl
}

func (pl *TestPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.FilterStatus), injectFilterReason)
}

func (pl *TestPlugin) PostFilter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	return nil, fwk.NewStatus(fwk.Code(pl.inj.PostFilterStatus), injectReason)
}

func (pl *TestPlugin) PreScore(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.PreScoreStatus), injectReason)
}

func (pl *TestPlugin) Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.ReserveStatus), injectReason)
}

func (pl *TestPlugin) Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
}

func (pl *TestPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return &fwk.PreBindPreFlightResult{AllowParallel: false}, fwk.NewStatus(fwk.Code(pl.inj.PreBindPreFlightStatus), injectReason)
}

func (pl *TestPlugin) PreBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.PreBindStatus), injectReason)
}

func (pl *TestPlugin) PostBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
}

func (pl *TestPlugin) Permit(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return fwk.NewStatus(fwk.Code(pl.inj.PermitStatus), injectReason), pl.inj.PermitTimeout
}

func (pl *TestPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	return fwk.NewStatus(fwk.Code(pl.inj.BindStatus), injectReason)
}

func (pl *TestPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	return &fwk.GeneratePlacementsResult{Placements: pl.inj.GeneratePlacementsResult}, fwk.NewStatus(fwk.Code(pl.inj.GeneratePlacementsStatus), injectReason)
}

func (pl *TestPlugin) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	return 0, fwk.NewStatus(fwk.Code(pl.inj.PlacementScoreStatus), injectReason)
}

func (pl *TestPlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}

func (pl *TestPlugin) PodGroupPostFilter(ctx context.Context, pg *v1alpha2.PodGroup, pods []*v1.Pod, pgSchedulingFunc func(ctx context.Context) *fwk.Status) *fwk.Status {
	return nil
}

func newTestCloseErrorPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &TestCloseErrorPlugin{name: testCloseErrorPlugin}, nil
}

// TestCloseErrorPlugin implements for Close test.
type TestCloseErrorPlugin struct {
	name string
}

func (pl *TestCloseErrorPlugin) Name() string {
	return pl.name
}

var errClose = errors.New("close err")

func (pl *TestCloseErrorPlugin) Close() error {
	return errClose
}

// TestPreFilterPlugin only implements PreFilterPlugin interface.
type TestPreFilterPlugin struct {
	PreFilterCalled int
}

func (pl *TestPreFilterPlugin) Name() string {
	return preFilterPluginName
}

func (pl *TestPreFilterPlugin) PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	pl.PreFilterCalled++
	return nil, nil
}

func (pl *TestPreFilterPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
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

func (pl *TestPreFilterWithExtensionsPlugin) PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	pl.PreFilterCalled++
	return nil, nil
}

func (pl *TestPreFilterWithExtensionsPlugin) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod,
	podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	pl.AddCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod,
	podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	pl.RemoveCalled++
	return nil
}

func (pl *TestPreFilterWithExtensionsPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return pl
}

type TestDuplicatePlugin struct {
}

func (dp *TestDuplicatePlugin) Name() string {
	return duplicatePluginName
}

func (dp *TestDuplicatePlugin) PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, nil
}

func (dp *TestDuplicatePlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

var _ fwk.PreFilterPlugin = &TestDuplicatePlugin{}

func newDuplicatePlugin(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &TestDuplicatePlugin{}, nil
}

// TestPermitPlugin only implements PermitPlugin interface.
type TestPermitPlugin struct {
	PreFilterCalled int
}

func (pp *TestPermitPlugin) Name() string {
	return permitPlugin
}
func (pp *TestPermitPlugin) Permit(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	return fwk.NewStatus(fwk.Wait), 10 * time.Second
}

var _ fwk.PreEnqueuePlugin = &TestPreEnqueuePlugin{}

type TestPreEnqueuePlugin struct{}

func (pl *TestPreEnqueuePlugin) Name() string {
	return preEnqueuePlugin
}

func (pl *TestPreEnqueuePlugin) PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status {
	return nil
}

var _ fwk.QueueSortPlugin = &TestQueueSortPlugin{}

func newQueueSortPlugin(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &TestQueueSortPlugin{}, nil
}

// TestQueueSortPlugin is a no-op implementation for QueueSort extension point.
type TestQueueSortPlugin struct{}

func (pl *TestQueueSortPlugin) Name() string {
	return queueSortPlugin
}

func (pl *TestQueueSortPlugin) Less(_, _ fwk.QueuedPodInfo) bool {
	return false
}

var _ fwk.BindPlugin = &TestBindPlugin{}

func newBindPlugin(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &TestBindPlugin{}, nil
}

// TestBindPlugin is a no-op implementation for Bind extension point.
type TestBindPlugin struct{}

func (t TestBindPlugin) Name() string {
	return bindPlugin
}

func (t TestBindPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	return nil
}

// TestPlacementGeneratePlugin only implements GeneratePlacements extension point.
type TestPlacementGeneratePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestPlacementGeneratePlugin) Name() string {
	return pl.name
}

func (pl *TestPlacementGeneratePlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	return &fwk.GeneratePlacementsResult{Placements: pl.inj.GeneratePlacementsResult}, fwk.NewStatus(fwk.Code(pl.inj.GeneratePlacementsStatus), injectReason)
}

func newTestPlacementGeneratePlugin(_ context.Context, injArgs runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	var inj injectedResult
	if err := DecodeInto(injArgs, &inj); err != nil {
		return nil, err
	}
	return &TestPlacementGeneratePlugin{placementGeneratePlugin, inj}, nil
}

// nolint:errcheck   // Ignore the error returned by Register as before
var registry = func() Registry {
	r := make(Registry)
	r.Register(scoreWithNormalizePlugin1, newScoreWithNormalizePlugin1)
	r.Register(scoreWithNormalizePlugin2, newScoreWithNormalizePlugin2)
	r.Register(scorePlugin1, newScorePlugin1)
	r.Register(scorePlugin2, newScorePlugin2)
	r.Register(pluginNotImplementingScore, newPluginNotImplementingScore)
	r.Register(duplicatePluginName, newDuplicatePlugin)
	r.Register(testPlugin, newTestPlugin)
	r.Register(queueSortPlugin, newQueueSortPlugin)
	r.Register(bindPlugin, newBindPlugin)
	r.Register(testCloseErrorPlugin, newTestCloseErrorPlugin)
	r.Register(placementGeneratePlugin, newTestPlacementGeneratePlugin)
	r.Register(placementScorePlugin1, newPlacementScorePluginFactory(placementScorePlugin1))
	r.Register(defaultPreemptionPlugin, newTestPlugin)
	return r
}()

var defaultWeights = map[string]int32{
	scoreWithNormalizePlugin1: 1,
	scoreWithNormalizePlugin2: 2,
	scorePlugin1:              1,
}

var state = framework.NewCycleState()

// Pod is only used for logging errors.
var pod = &v1.Pod{}
var node = &v1.Node{
	ObjectMeta: metav1.ObjectMeta{
		Name: nodeName,
	},
}
var lowPriority, highPriority = int32(0), int32(1000)
var lowPriorityPod = &v1.Pod{
	ObjectMeta: metav1.ObjectMeta{UID: "low"},
	Spec:       v1.PodSpec{Priority: &lowPriority},
}
var highPriorityPod = &v1.Pod{
	ObjectMeta: metav1.ObjectMeta{UID: "high"},
	Spec:       v1.PodSpec{Priority: &highPriority},
}
var nodes = []*v1.Node{
	{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
	{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
}

var (
	errInjectedStatus       = errors.New(injectReason)
	errInjectedFilterStatus = errors.New(injectFilterReason)
)

func newFrameworkWithQueueSortAndBind(ctx context.Context, r Registry, profile config.KubeSchedulerProfile, opts ...Option) (framework.Framework, error) {
	if _, ok := r[queueSortPlugin]; !ok {
		r[queueSortPlugin] = newQueueSortPlugin
	}
	if _, ok := r[bindPlugin]; !ok {
		r[bindPlugin] = newBindPlugin
	}

	if len(profile.Plugins.QueueSort.Enabled) == 0 {
		profile.Plugins.QueueSort.Enabled = append(profile.Plugins.QueueSort.Enabled, config.Plugin{Name: queueSortPlugin})
	}
	if len(profile.Plugins.Bind.Enabled) == 0 {
		profile.Plugins.Bind.Enabled = append(profile.Plugins.Bind.Enabled, config.Plugin{Name: bindPlugin})
	}
	return NewFramework(ctx, r, &profile, opts...)
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
			plugins: &config.Plugins{},
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
			profile := config.KubeSchedulerProfile{Plugins: tt.plugins}
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			_, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if tt.initErr && err == nil {
				t.Fatal("Framework initialization should fail")
			}
			if !tt.initErr && err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
		})
	}
}

func TestNewFrameworkErrors(t *testing.T) {
	tests := []struct {
		name      string
		plugins   *config.Plugins
		pluginCfg []config.PluginConfig
		wantErr   string
	}{
		{
			name: "duplicate plugin name",
			plugins: &config.Plugins{
				PreFilter: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: duplicatePluginName, Weight: 1},
						{Name: duplicatePluginName, Weight: 1},
					},
				},
			},
			pluginCfg: []config.PluginConfig{
				{Name: duplicatePluginName},
			},
			wantErr: "already registered",
		},
		{
			name: "duplicate plugin config",
			plugins: &config.Plugins{
				PreFilter: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: duplicatePluginName, Weight: 1},
					},
				},
			},
			pluginCfg: []config.PluginConfig{
				{Name: duplicatePluginName},
				{Name: duplicatePluginName},
			},
			wantErr: "repeated config for plugin",
		},
		{
			name: "more than one PlacementGeneratePlugin",
			plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: queueSortPlugin},
					},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: bindPlugin},
					},
				},
				PlacementGenerate: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: placementGeneratePlugin},
					},
				},
			},
			pluginCfg: []config.PluginConfig{
				{Name: queueSortPlugin},
				{Name: bindPlugin},
				{Name: testPlugin},
				{Name: placementGeneratePlugin},
			},
			wantErr: "at most one placement generate plugin is allowed",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			profile := &config.KubeSchedulerProfile{
				Plugins:      tc.plugins,
				PluginConfig: tc.pluginCfg,
			}
			_, err := NewFramework(ctx, registry, profile)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Unexpected error, got %v, expect: %s", err, tc.wantErr)
			}
		})
	}
}

func TestPodGroupPostFilterPlugins(t *testing.T) {
	tests := []struct {
		name                   string
		plugins                *config.Plugins
		featureGate            bool
		wantPodGroupPostFilter bool
	}{
		{
			name: "should fill pod group post filter with feature gate and default preemption",
			plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: queueSortPlugin},
					},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: bindPlugin},
					},
				},
				PostFilter: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: defaultPreemptionPlugin},
					},
				},
			},
			featureGate:            true,
			wantPodGroupPostFilter: true,
		},
		{
			name: "should not fill pod group post filter when feature gate is disabled",
			plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: queueSortPlugin},
					},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: bindPlugin},
					},
				},
				PostFilter: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: defaultPreemptionPlugin},
					},
				},
			},
			featureGate:            false,
			wantPodGroupPostFilter: false,
		},
		{
			name: "should not fill pod group post filter when post filter plugin is not default preemption",
			plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: queueSortPlugin},
					},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: bindPlugin},
					},
				},
				PostFilter: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
					},
				},
			},
			featureGate:            false,
			wantPodGroupPostFilter: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.featureGate {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:         true,
					features.GangScheduling:          true,
					features.WorkloadAwarePreemption: true,
				})
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			profile := &config.KubeSchedulerProfile{
				Plugins: &config.Plugins{
					QueueSort: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: queueSortPlugin},
						},
					},
					Bind: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: bindPlugin},
						},
					},
					PostFilter: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: defaultPreemptionPlugin},
						},
					},
				},
			}
			f, _ := NewFramework(ctx, registry, profile)

			if tc.wantPodGroupPostFilter && len(f.PodGroupPostFilterPlugins()) != 1 {
				t.Errorf("Expected 1 pod group post filter plugin, got %d", len(f.PodGroupPostFilterPlugins()))
			}
			if !tc.wantPodGroupPostFilter && len(f.PodGroupPostFilterPlugins()) != 0 {
				t.Errorf("Expected 0 pod group post filter plugin, got %d", len(f.PodGroupPostFilterPlugins()))
			}
		})
	}

}

func TestNewFrameworkMultiPointExpansion(t *testing.T) {
	tests := []struct {
		name        string
		plugins     *config.Plugins
		wantPlugins *config.Plugins
		wantErr     string
	}{
		{
			name: "plugin expansion",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin, Weight: 5},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:         config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:         config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter:        config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Score:             config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 5}}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore:    config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 5}}},
			},
		},
		{
			name: "disable MultiPoint plugin at some extension points",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
					},
				},
				PreScore: config.PluginSet{
					Disabled: []config.Plugin{
						{Name: testPlugin},
					},
				},
				Score: config.PluginSet{
					Disabled: []config.Plugin{
						{Name: testPlugin},
					},
				},
				PlacementScore: config.PluginSet{
					Disabled: []config.Plugin{
						{Name: testPlugin},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:         config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:         config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter:        config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
			},
		},
		{
			name: "Multiple MultiPoint plugins",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: scorePlugin1},
						{Name: placementScorePlugin1},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin},
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin, Weight: 1},
					{Name: scorePlugin1, Weight: 1},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin, Weight: 1},
					{Name: placementScorePlugin1, Weight: 1},
				}},
			},
		},
		{
			name: "disable MultiPoint extension",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: scorePlugin1},
					},
				},
				PreScore: config.PluginSet{
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin, Weight: 1},
					{Name: scorePlugin1, Weight: 1},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore:    config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}},
			},
		},
		{
			name: "Reorder MultiPoint plugins (specified extension takes precedence)",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scoreWithNormalizePlugin1},
						{Name: testPlugin},
						{Name: scorePlugin1},
					},
				},
				Score: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scorePlugin1},
						{Name: testPlugin},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin},
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 1},
					{Name: testPlugin, Weight: 1},
					{Name: scoreWithNormalizePlugin1, Weight: 1},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore:    config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}},
			},
		},
		{
			name: "Reorder MultiPoint plugins (specified extension only takes precedence when it exists in MultiPoint)",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: scorePlugin1},
					},
				},
				Score: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scoreWithNormalizePlugin1},
						{Name: scorePlugin1},
						{Name: testPlugin},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin},
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 1},
					{Name: testPlugin, Weight: 1},
					{Name: scoreWithNormalizePlugin1, Weight: 1},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore:    config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}},
			},
		},
		{
			name: "Override MultiPoint plugins weights",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: scorePlugin1},
					},
				},
				Score: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scorePlugin1, Weight: 5},
						{Name: testPlugin, Weight: 3},
					},
				},
				PlacementScore: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin, Weight: 2},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin},
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 5},
					{Name: testPlugin, Weight: 3},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore:    config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 2}}},
			},
		},
		{
			name: "disable and enable MultiPoint plugins with '*'",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: queueSortPlugin},
						{Name: bindPlugin},
						{Name: scorePlugin1},
						{Name: placementScorePlugin1},
					},
					Disabled: []config.Plugin{
						{Name: "*"},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 1},
				}},
				Bind: config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
				PlacementScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: placementScorePlugin1, Weight: 1},
				}},
			},
		},
		{
			name: "disable and enable MultiPoint plugin by name",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: bindPlugin},
						{Name: queueSortPlugin},
						{Name: scorePlugin1},
					},
					Disabled: []config.Plugin{
						{Name: scorePlugin1},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 1},
				}},
				Bind: config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
			},
		},
		{
			name: "Expect 'already registered' error",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: testPlugin},
					},
				},
			},
			wantErr: "already registered",
		},
		{
			name: "Override MultiPoint plugins weights and avoid a plugin being loaded multiple times",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin},
						{Name: scorePlugin1},
						{Name: placementScorePlugin1},
					},
				},
				Score: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scorePlugin1, Weight: 5},
						{Name: scorePlugin2, Weight: 5},
						{Name: testPlugin, Weight: 3},
					},
				},
				PlacementScore: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin, Weight: 2},
						{Name: placementScorePlugin1, Weight: 6},
					},
				},
			},
			wantPlugins: &config.Plugins{
				QueueSort:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreFilter:  config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Filter:     config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin},
					{Name: scorePlugin1},
				}},
				Score: config.PluginSet{Enabled: []config.Plugin{
					{Name: scorePlugin1, Weight: 5},
					{Name: testPlugin, Weight: 3},
					{Name: scorePlugin2, Weight: 5},
				}},
				Reserve:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Permit:            config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PreBind:           config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				Bind:              config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PostBind:          config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementGenerate: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin}}},
				PlacementScore: config.PluginSet{Enabled: []config.Plugin{
					{Name: testPlugin, Weight: 2},
					{Name: placementScorePlugin1, Weight: 6},
				}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fw, err := NewFramework(ctx, registry, &config.KubeSchedulerProfile{Plugins: tc.plugins})
			defer func() {
				if fw != nil {
					_ = fw.Close()
				}
			}()
			if err != nil {
				if tc.wantErr == "" || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Unexpected error, got %v, expect: %s", err, tc.wantErr)
				}
			} else {
				if tc.wantErr != "" {
					t.Fatalf("Unexpected error, got %v, expect: %s", err, tc.wantErr)
				}
			}
			if tc.wantErr == "" {
				if diff := cmp.Diff(tc.wantPlugins, fw.ListPlugins()); diff != "" {
					t.Fatalf("Unexpected eventToPlugin map (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestPreEnqueuePlugins(t *testing.T) {
	tests := []struct {
		name    string
		plugins []fwk.Plugin
		want    []fwk.PreEnqueuePlugin
	}{
		{
			name: "no PreEnqueuePlugin registered",
		},
		{
			name: "one PreEnqueuePlugin registered",
			plugins: []fwk.Plugin{
				&TestPreEnqueuePlugin{},
			},
			want: []fwk.PreEnqueuePlugin{
				&TestPreEnqueuePlugin{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			cfgPls := &config.Plugins{}
			for _, pl := range tt.plugins {
				// register all plugins
				tmpPl := pl
				if err := registry.Register(pl.Name(),
					func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tmpPl, nil
					}); err != nil {
					t.Fatalf("fail to register preEnqueue plugin (%s)", pl.Name())
				}
				// append plugins to filter pluginset
				cfgPls.PreEnqueue.Enabled = append(
					cfgPls.PreEnqueue.Enabled,
					config.Plugin{Name: pl.Name()},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: cfgPls}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()
			got := f.PreEnqueuePlugins()
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Unexpected PreEnqueuePlugins(): (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestRunPreScorePlugins(t *testing.T) {
	tests := []struct {
		name               string
		plugins            []*TestPlugin
		wantSkippedPlugins sets.Set[string]
		wantStatusCode     fwk.Code
	}{
		{
			name: "all PreScorePlugins returned success",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "success2",
				},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "one PreScore plugin returned success, but another PreScore plugin returned non-success",
			plugins: []*TestPlugin{
				{
					name: "success",
				},
				{
					name: "error",
					inj:  injectedResult{PreScoreStatus: int(fwk.Error)},
				},
			},
			wantStatusCode: fwk.Error,
		},
		{
			name: "one PreScore plugin returned skip, but another PreScore plugin returned non-success",
			plugins: []*TestPlugin{
				{
					name: "skip",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
				{
					name: "error",
					inj:  injectedResult{PreScoreStatus: int(fwk.Error)},
				},
			},
			wantSkippedPlugins: sets.New("skip"),
			wantStatusCode:     fwk.Error,
		},
		{
			name: "all PreScore plugins returned skip",
			plugins: []*TestPlugin{
				{
					name: "skip1",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
				{
					name: "skip2",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
				{
					name: "skip3",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
			},
			wantSkippedPlugins: sets.New("skip1", "skip2", "skip3"),
			wantStatusCode:     fwk.Success,
		},
		{
			name: "some PreScore plugins returned skip",
			plugins: []*TestPlugin{
				{
					name: "skip1",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
				{
					name: "success1",
				},
				{
					name: "skip2",
					inj:  injectedResult{PreScoreStatus: int(fwk.Skip)},
				},
				{
					name: "success2",
				},
			},
			wantSkippedPlugins: sets.New("skip1", "skip2"),
			wantStatusCode:     fwk.Success,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := make(Registry)
			enabled := make([]config.Plugin, len(tt.plugins))
			for i, p := range tt.plugins {
				enabled[i].Name = p.name
				if err := r.Register(p.name, func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return p, nil
				}); err != nil {
					t.Fatalf("fail to register PreScorePlugins plugin (%s)", p.Name())
				}
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			f, err := newFrameworkWithQueueSortAndBind(
				ctx,
				r,
				config.KubeSchedulerProfile{Plugins: &config.Plugins{PreScore: config.PluginSet{Enabled: enabled}}},
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()
			state := framework.NewCycleState()
			status := f.RunPreScorePlugins(ctx, state, nil, nil)
			if status.Code() != tt.wantStatusCode {
				t.Errorf("wrong status code. got: %v, want: %v", status, tt.wantStatusCode)
			}
			skipped := state.GetSkipScorePlugins()
			if diff := cmp.Diff(tt.wantSkippedPlugins, skipped); diff != "" {
				t.Errorf("wrong skip score plugins (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestRunScorePlugins(t *testing.T) {
	tests := []struct {
		name           string
		registry       Registry
		plugins        *config.Plugins
		pluginConfigs  []config.PluginConfig
		want           []fwk.NodePluginScores
		skippedPlugins sets.Set[string]
		// If err is true, we expect RunScorePlugin to fail.
		err bool
	}{
		{
			name:    "no Score plugins",
			plugins: buildScoreConfigDefaultWeights(),
			want: []fwk.NodePluginScores{
				{
					Name:   "node1",
					Scores: []fwk.PluginScore{},
				},
				{
					Name:   "node2",
					Scores: []fwk.PluginScore{},
				},
			},
		},
		{
			name:    "single Score plugin",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
			},
			// scorePlugin1 Score returns 1, weight=1, so want=1.
			want: []fwk.NodePluginScores{
				{
					Name: "node1",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
					},
					TotalScore: 1,
				},
				{
					Name: "node2",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
					},
					TotalScore: 1,
				},
			},
		},
		{
			name: "single ScoreWithNormalize plugin",
			// registry: registry,
			plugins: buildScoreConfigDefaultWeights(scoreWithNormalizePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 10, "normalizeRes": 5 }`),
					},
				},
			},
			// scoreWithNormalizePlugin1 Score returns 10, but NormalizeScore overrides to 5, weight=1, so want=5
			want: []fwk.NodePluginScores{
				{
					Name: "node1",
					Scores: []fwk.PluginScore{
						{
							Name:  scoreWithNormalizePlugin1,
							Score: 5,
						},
					},
					TotalScore: 5,
				},
				{
					Name: "node2",
					Scores: []fwk.PluginScore{
						{
							Name:  scoreWithNormalizePlugin1,
							Score: 5,
						},
					},
					TotalScore: 5,
				},
			},
		},
		{
			name:    "3 Score plugins, 2 NormalizeScore plugins",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1, scoreWithNormalizePlugin2),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
				{
					Name: scoreWithNormalizePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 3, "normalizeRes": 4}`),
					},
				},
				{
					Name: scoreWithNormalizePlugin2,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 4, "normalizeRes": 5}`),
					},
				},
			},
			// scorePlugin1 Score returns 1, weight =1, so want=1.
			// scoreWithNormalizePlugin1 Score returns 3, but NormalizeScore overrides to 4, weight=1, so want=4.
			// scoreWithNormalizePlugin2 Score returns 4, but NormalizeScore overrides to 5, weight=2, so want=10.
			want: []fwk.NodePluginScores{
				{
					Name: "node1",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
						{
							Name:  scoreWithNormalizePlugin1,
							Score: 4,
						},
						{
							Name:  scoreWithNormalizePlugin2,
							Score: 10,
						},
					},
					TotalScore: 15,
				},
				{
					Name: "node2",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
						{
							Name:  scoreWithNormalizePlugin1,
							Score: 4,
						},
						{
							Name:  scoreWithNormalizePlugin2,
							Score: 10,
						},
					},
					TotalScore: 15,
				},
			},
		},
		{
			name: "score fails",
			pluginConfigs: []config.PluginConfig{
				{
					Name: scoreWithNormalizePlugin1,
					Args: &runtime.Unknown{
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
					Args: &runtime.Unknown{
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
					Args: &runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "scoreRes": %d }`, fwk.MaxNodeScore+1)),
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
					Args: &runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "scoreRes": %d }`, fwk.MinNodeScore-1)),
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
					Args: &runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "normalizeRes": %d }`, fwk.MaxNodeScore+1)),
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
					Args: &runtime.Unknown{
						Raw: []byte(fmt.Sprintf(`{ "normalizeRes": %d }`, fwk.MinNodeScore-1)),
					},
				},
			},
			err: true,
		},
		{
			name: "single Score plugin with MultiPointExpansion",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scorePlugin1},
					},
				},
				Score: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: scorePlugin1, Weight: 3},
					},
				},
			},
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
			},
			// scorePlugin1 Score returns 1, weight=3, so want=3.
			want: []fwk.NodePluginScores{
				{
					Name: "node1",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 3,
						},
					},
					TotalScore: 3,
				},
				{
					Name: "node2",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 3,
						},
					},
					TotalScore: 3,
				},
			},
		},
		{
			name:    "one success plugin, one skip plugin",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreRes": 1 }`),
					},
				},
				{
					Name: scoreWithNormalizePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreStatus": 1 }`), // To make sure this plugin isn't called, set error as an injected result.
					},
				},
			},
			skippedPlugins: sets.New(scoreWithNormalizePlugin1),
			want: []fwk.NodePluginScores{
				{
					Name: "node1",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
					},
					TotalScore: 1,
				},
				{
					Name: "node2",
					Scores: []fwk.PluginScore{
						{
							Name:  scorePlugin1,
							Score: 1,
						},
					},
					TotalScore: 1,
				},
			},
		},
		{
			name:    "all plugins are skipped in prescore",
			plugins: buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs: []config.PluginConfig{
				{
					Name: scorePlugin1,
					Args: &runtime.Unknown{
						Raw: []byte(`{ "scoreStatus": 1 }`), // To make sure this plugin isn't called, set error as an injected result.
					},
				},
			},
			skippedPlugins: sets.New(scorePlugin1),
			want: []fwk.NodePluginScores{
				{
					Name:   "node1",
					Scores: []fwk.PluginScore{},
				},
				{
					Name:   "node2",
					Scores: []fwk.PluginScore{},
				},
			},
		},
		{
			name:           "skipped prescore plugin number greater than the number of score plugins",
			plugins:        buildScoreConfigDefaultWeights(scorePlugin1),
			pluginConfigs:  nil,
			skippedPlugins: sets.New(scorePlugin1, "score-plugin-unknown"),
			want: []fwk.NodePluginScores{
				{
					Name:   "node1",
					Scores: []fwk.PluginScore{},
				},
				{
					Name:   "node2",
					Scores: []fwk.PluginScore{},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Inject the results via Args in PluginConfig.
			profile := config.KubeSchedulerProfile{
				Plugins:      tt.plugins,
				PluginConfig: tt.pluginConfigs,
			}
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			state := framework.NewCycleState()
			state.SetSkipScorePlugins(tt.skippedPlugins)
			res, status := f.RunScorePlugins(ctx, state, pod, BuildNodeInfos(nodes))

			if tt.err {
				if status.IsSuccess() {
					t.Errorf("Expected status to be non-success. got: %v", status.Code().String())
				}
				return
			}

			if !status.IsSuccess() {
				t.Errorf("Expected status to be success.")
			}
			if diff := cmp.Diff(tt.want, res); diff != "" {
				t.Errorf("Score map after RunScorePlugin (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPreFilterPlugins(t *testing.T) {
	preFilter1 := &TestPreFilterPlugin{}
	preFilter2 := &TestPreFilterWithExtensionsPlugin{}
	r := make(Registry)
	r.Register(preFilterPluginName,
		func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
			return preFilter1, nil
		})
	r.Register(preFilterWithExtensionsPluginName,
		func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
			return preFilter2, nil
		})
	plugins := &config.Plugins{PreFilter: config.PluginSet{Enabled: []config.Plugin{{Name: preFilterWithExtensionsPluginName}, {Name: preFilterPluginName}}}}
	t.Run("TestPreFilterPlugin", func(t *testing.T) {
		profile := config.KubeSchedulerProfile{Plugins: plugins}
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		f, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
		if err != nil {
			t.Fatalf("Failed to create framework for testing: %v", err)
		}
		defer func() {
			_ = f.Close()
		}()
		state := framework.NewCycleState()
		f.RunPreFilterPlugins(ctx, state, nil)
		f.RunPreFilterExtensionAddPod(ctx, state, nil, nil, nil)
		f.RunPreFilterExtensionRemovePod(ctx, state, nil, nil, nil)

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

func TestRunPreFilterPlugins(t *testing.T) {
	tests := []struct {
		name                string
		plugins             []*TestPlugin
		wantPreFilterResult *fwk.PreFilterResult
		wantSkippedPlugins  sets.Set[string]
		wantStatusCode      fwk.Code
	}{
		{
			name: "all PreFilter returned success",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "success2",
				},
			},
			wantPreFilterResult: nil,
			wantStatusCode:      fwk.Success,
		},
		{
			name: "one PreFilter plugin returned success, but another PreFilter plugin returned non-success",
			plugins: []*TestPlugin{
				{
					name: "success",
				},
				{
					name: "error",
					inj:  injectedResult{PreFilterStatus: int(fwk.Error)},
				},
			},
			wantPreFilterResult: nil,
			wantStatusCode:      fwk.Error,
		},
		{
			name: "one PreFilter plugin returned skip, but another PreFilter plugin returned non-success",
			plugins: []*TestPlugin{
				{
					name: "skip",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
				{
					name: "error",
					inj:  injectedResult{PreFilterStatus: int(fwk.Error)},
				},
			},
			wantSkippedPlugins: sets.New("skip"),
			wantStatusCode:     fwk.Error,
		},
		{
			name: "all PreFilter plugins returned skip",
			plugins: []*TestPlugin{
				{
					name: "skip1",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
				{
					name: "skip2",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
				{
					name: "skip3",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
			},
			wantPreFilterResult: nil,
			wantSkippedPlugins:  sets.New("skip1", "skip2", "skip3"),
			wantStatusCode:      fwk.Success,
		},
		{
			name: "some PreFilter plugins returned skip",
			plugins: []*TestPlugin{
				{
					name: "skip1",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
				{
					name: "success1",
				},
				{
					name: "skip2",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
				{
					name: "success2",
				},
			},
			wantPreFilterResult: nil,
			wantSkippedPlugins:  sets.New("skip1", "skip2"),
			wantStatusCode:      fwk.Success,
		},
		{
			name: "one PreFilter plugin returned Unschedulable, but another PreFilter plugin should be executed",
			plugins: []*TestPlugin{
				{
					name: "unschedulable",
					inj:  injectedResult{PreFilterStatus: int(fwk.Unschedulable)},
				},
				{
					// to make sure this plugin is executed, this plugin return Skip and we confirm it via wantSkippedPlugins.
					name: "skip",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
			},
			wantPreFilterResult: nil,
			wantSkippedPlugins:  sets.New("skip"),
			wantStatusCode:      fwk.Unschedulable,
		},
		{
			name: "one PreFilter plugin returned UnschedulableAndUnresolvable, and all other plugins aren't executed",
			plugins: []*TestPlugin{
				{
					name: "unresolvable",
					inj:  injectedResult{PreFilterStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
				{
					// to make sure this plugin is not executed, this plugin return Skip and we confirm it via wantSkippedPlugins.
					name: "skip",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
			},
			wantPreFilterResult: nil,
			wantStatusCode:      fwk.UnschedulableAndUnresolvable,
		},
		{
			name: "all nodes are filtered out by prefilter result, but other plugins aren't executed because we consider all nodes are filtered out by UnschedulableAndUnresolvable",
			plugins: []*TestPlugin{
				{
					name: "reject-all-nodes",
					inj:  injectedResult{PreFilterResult: &fwk.PreFilterResult{NodeNames: sets.New[string]()}},
				},
				{
					// to make sure this plugin is not executed, this plugin return Skip and we confirm it via wantSkippedPlugins.
					name: "skip",
					inj:  injectedResult{PreFilterStatus: int(fwk.Skip)},
				},
			},
			wantPreFilterResult: &fwk.PreFilterResult{NodeNames: sets.New[string]()},
			wantSkippedPlugins:  sets.New[string](), // "skip" plugin isn't executed.
			wantStatusCode:      fwk.UnschedulableAndUnresolvable,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := make(Registry)
			enabled := make([]config.Plugin, len(tt.plugins))
			for i, p := range tt.plugins {
				enabled[i].Name = p.name
				if err := r.Register(p.name, func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return p, nil
				}); err != nil {
					t.Fatalf("fail to register PreFilter plugin (%s)", p.Name())
				}
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			f, err := newFrameworkWithQueueSortAndBind(
				ctx,
				r,
				config.KubeSchedulerProfile{Plugins: &config.Plugins{PreFilter: config.PluginSet{Enabled: enabled}}},
				WithSnapshotSharedLister(cache.NewEmptySnapshot()),
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			state := framework.NewCycleState()
			result, status, _ := f.RunPreFilterPlugins(ctx, state, nil)
			if diff := cmp.Diff(tt.wantPreFilterResult, result); diff != "" {
				t.Errorf("wrong status (-want,+got):\n%s", diff)
			}
			if status.Code() != tt.wantStatusCode {
				t.Errorf("wrong status code. got: %v, want: %v", status, tt.wantStatusCode)
			}
			skipped := state.GetSkipFilterPlugins()
			if diff := cmp.Diff(tt.wantSkippedPlugins, skipped); diff != "" {
				t.Errorf("wrong skip filter plugins (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestRunPreFilterExtensionRemovePod(t *testing.T) {
	tests := []struct {
		name               string
		plugins            []*TestPlugin
		skippedPluginNames sets.Set[string]
		wantStatusCode     fwk.Code
	}{
		{
			name: "no plugins are skipped and all RemovePod() returned success",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "success2",
				},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "one RemovePod() returned error",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "error1",
					inj:  injectedResult{PreFilterRemovePodStatus: int(fwk.Error)},
				},
			},
			wantStatusCode: fwk.Error,
		},
		{
			name: "one RemovePod() is skipped",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "skipped",
					// To confirm it's skipped, return error so that this test case will fail when it isn't skipped.
					inj: injectedResult{PreFilterRemovePodStatus: int(fwk.Error)},
				},
			},
			skippedPluginNames: sets.New("skipped"),
			wantStatusCode:     fwk.Success,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := make(Registry)
			enabled := make([]config.Plugin, len(tt.plugins))
			for i, p := range tt.plugins {
				enabled[i].Name = p.name
				if err := r.Register(p.name, func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return p, nil
				}); err != nil {
					t.Fatalf("fail to register PreFilterExtension plugin (%s)", p.Name())
				}
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			f, err := newFrameworkWithQueueSortAndBind(
				ctx,
				r,
				config.KubeSchedulerProfile{Plugins: &config.Plugins{PreFilter: config.PluginSet{Enabled: enabled}}},
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			state := framework.NewCycleState()
			state.SetSkipFilterPlugins(tt.skippedPluginNames)
			status := f.RunPreFilterExtensionRemovePod(ctx, state, nil, nil, nil)
			if status.Code() != tt.wantStatusCode {
				t.Errorf("wrong status code. got: %v, want: %v", status, tt.wantStatusCode)
			}
		})
	}
}

func TestRunPreFilterExtensionAddPod(t *testing.T) {
	tests := []struct {
		name               string
		plugins            []*TestPlugin
		skippedPluginNames sets.Set[string]
		wantStatusCode     fwk.Code
	}{
		{
			name: "no plugins are skipped and all AddPod() returned success",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "success2",
				},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "one AddPod() returned error",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "error1",
					inj:  injectedResult{PreFilterAddPodStatus: int(fwk.Error)},
				},
			},
			wantStatusCode: fwk.Error,
		},
		{
			name: "one AddPod() is skipped",
			plugins: []*TestPlugin{
				{
					name: "success1",
				},
				{
					name: "skipped",
					// To confirm it's skipped, return error so that this test case will fail when it isn't skipped.
					inj: injectedResult{PreFilterAddPodStatus: int(fwk.Error)},
				},
			},
			skippedPluginNames: sets.New("skipped"),
			wantStatusCode:     fwk.Success,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := make(Registry)
			enabled := make([]config.Plugin, len(tt.plugins))
			for i, p := range tt.plugins {
				enabled[i].Name = p.name
				if err := r.Register(p.name, func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return p, nil
				}); err != nil {
					t.Fatalf("fail to register PreFilterExtension plugin (%s)", p.Name())
				}
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			f, err := newFrameworkWithQueueSortAndBind(
				ctx,
				r,
				config.KubeSchedulerProfile{Plugins: &config.Plugins{PreFilter: config.PluginSet{Enabled: enabled}}},
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			state := framework.NewCycleState()
			state.SetSkipFilterPlugins(tt.skippedPluginNames)
			ni := framework.NewNodeInfo()
			status := f.RunPreFilterExtensionAddPod(ctx, state, nil, nil, ni)
			if status.Code() != tt.wantStatusCode {
				t.Errorf("wrong status code. got: %v, want: %v", status, tt.wantStatusCode)
			}
		})
	}
}

func TestFilterPlugins(t *testing.T) {
	tests := []struct {
		name           string
		plugins        []*TestPlugin
		skippedPlugins sets.Set[string]
		wantStatus     *fwk.Status
	}{
		{
			name: "SuccessFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "ErrorFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running "TestPlugin" filter plugin: %w`, errInjectedFilterStatus)).WithPlugin("TestPlugin"),
		},
		{
			name: "UnschedulableFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{FilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectFilterReason).WithPlugin("TestPlugin"),
		},
		{
			name: "UnschedulableAndUnresolvableFilter",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj: injectedResult{
						FilterStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectFilterReason).WithPlugin("TestPlugin"),
		},
		// following tests cover multiple-plugins scenarios
		{
			name: "ErrorAndErrorFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running "TestPlugin1" filter plugin: %w`, errInjectedFilterStatus)).WithPlugin("TestPlugin1"),
		},
		{
			name: "UnschedulableAndUnschedulableFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectFilterReason).WithPlugin("TestPlugin1"),
		},
		{
			name: "UnschedulableAndUnschedulableAndUnresolvableFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectFilterReason).WithPlugin("TestPlugin1"),
		},
		{
			name: "SuccessAndSuccessFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "SuccessAndSkipFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},

				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Error)}, // To make sure this plugins isn't called, set error as an injected result.
				},
			},
			wantStatus:     nil,
			skippedPlugins: sets.New("TestPlugin2"),
		},
		{
			name: "ErrorAndSuccessFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running "TestPlugin1" filter plugin: %w`, errInjectedFilterStatus)).WithPlugin("TestPlugin1"),
		},
		{
			name: "SuccessAndErrorFilters",
			plugins: []*TestPlugin{
				{

					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running "TestPlugin2" filter plugin: %w`, errInjectedFilterStatus)).WithPlugin("TestPlugin2"),
		},
		{
			name: "SuccessAndUnschedulableFilters",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{FilterStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{FilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectFilterReason).WithPlugin("TestPlugin2"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			cfgPls := &config.Plugins{}
			for _, pl := range tt.plugins {
				// register all plugins
				tmpPl := pl
				if err := registry.Register(pl.name,
					func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tmpPl, nil
					}); err != nil {
					t.Fatalf("fail to register filter plugin (%s)", pl.name)
				}
				// append plugins to filter pluginset
				cfgPls.Filter.Enabled = append(
					cfgPls.Filter.Enabled,
					config.Plugin{Name: pl.name})
			}
			profile := config.KubeSchedulerProfile{Plugins: cfgPls}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()
			state := framework.NewCycleState()
			state.SetSkipFilterPlugins(tt.skippedPlugins)
			gotStatus := f.RunFilterPlugins(ctx, state, pod, nil)
			if diff := cmp.Diff(tt.wantStatus, gotStatus, statusCmpOpts...); diff != "" {
				t.Errorf("Unexpected status: (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPostFilterPlugins(t *testing.T) {
	tests := []struct {
		name        string
		skipPlugins bool
		plugins     []*TestPlugin
		wantStatus  *fwk.Status
	}{
		{
			name: "a single plugin makes a Pod schedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PostFilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Success, injectReason),
		},
		{
			name:        "skips all post filters if state has SkipPostFilterPlugins",
			skipPlugins: true,
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "All PostFilter plugins are skipped"),
		},
		{
			name: "plugin1 failed to make a Pod schedulable, followed by plugin2 which makes the Pod schedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Success, injectReason),
		},
		{
			name: "plugin1 makes a Pod schedulable, followed by plugin2 which cannot make the Pod schedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Success, injectReason),
		},
		{
			name: "plugin1 failed to make a Pod schedulable, followed by plugin2 which makes the Pod schedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.AsStatus(errors.New(injectReason)).WithPlugin("TestPlugin1"),
		},
		{
			name: "plugin1 failed to make a Pod schedulable, followed by plugin2 which makes the Pod unresolvable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectReason).WithPlugin("TestPlugin2"),
		},
		{
			name: "both plugins failed to make a Pod schedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PostFilterStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, []string{injectReason, injectReason}...).WithPlugin("TestPlugin1"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			cfgPls := &config.Plugins{}
			for _, pl := range tt.plugins {
				// register all plugins
				tmpPl := pl
				if err := registry.Register(pl.name,
					func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tmpPl, nil
					}); err != nil {
					t.Fatalf("fail to register postFilter plugin (%s)", pl.name)
				}
				// append plugins to filter pluginset
				cfgPls.PostFilter.Enabled = append(
					cfgPls.PostFilter.Enabled,
					config.Plugin{Name: pl.name},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: cfgPls}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()
			state := framework.NewCycleState()
			state.SetSkipAllPostFilterPlugins(tt.skipPlugins)
			_, gotStatus := f.RunPostFilterPlugins(ctx, state, pod, nil)

			if diff := cmp.Diff(tt.wantStatus, gotStatus, statusCmpOpts...); diff != "" {
				t.Errorf("Unexpected status (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestFilterPluginsWithNominatedPods(t *testing.T) {
	tests := []struct {
		name            string
		preFilterPlugin *TestPlugin
		filterPlugin    *TestPlugin
		pod             *v1.Pod
		nominatedPod    *v1.Pod
		node            *v1.Node
		nodeInfo        *framework.NodeInfo
		wantStatus      *fwk.Status
	}{
		{
			name:            "node has no nominated pod",
			preFilterPlugin: nil,
			filterPlugin:    nil,
			pod:             lowPriorityPod,
			nominatedPod:    nil,
			node:            node,
			nodeInfo:        framework.NewNodeInfo(pod),
			wantStatus:      nil,
		},
		{
			name: "node has a high-priority nominated pod and all filters succeed",
			preFilterPlugin: &TestPlugin{
				name: "TestPlugin1",
				inj: injectedResult{
					PreFilterAddPodStatus: int(fwk.Success),
				},
			},
			filterPlugin: &TestPlugin{
				name: "TestPlugin2",
				inj: injectedResult{
					FilterStatus: int(fwk.Success),
				},
			},
			pod:          lowPriorityPod,
			nominatedPod: highPriorityPod,
			node:         node,
			nodeInfo:     framework.NewNodeInfo(pod),
			wantStatus:   nil,
		},
		{
			name: "node has a high-priority nominated pod and pre filters fail",
			preFilterPlugin: &TestPlugin{
				name: "TestPlugin1",
				inj: injectedResult{
					PreFilterAddPodStatus: int(fwk.Error),
				},
			},
			filterPlugin: nil,
			pod:          lowPriorityPod,
			nominatedPod: highPriorityPod,
			node:         node,
			nodeInfo:     framework.NewNodeInfo(pod),
			wantStatus:   fwk.AsStatus(fmt.Errorf(`running AddPod on PreFilter plugin "TestPlugin1": %w`, errInjectedStatus)),
		},
		{
			name: "node has a high-priority nominated pod and filters fail",
			preFilterPlugin: &TestPlugin{
				name: "TestPlugin1",
				inj: injectedResult{
					PreFilterAddPodStatus: int(fwk.Success),
				},
			},
			filterPlugin: &TestPlugin{
				name: "TestPlugin2",
				inj: injectedResult{
					FilterStatus: int(fwk.Error),
				},
			},
			pod:          lowPriorityPod,
			nominatedPod: highPriorityPod,
			node:         node,
			nodeInfo:     framework.NewNodeInfo(pod),
			wantStatus:   fwk.AsStatus(fmt.Errorf(`running "TestPlugin2" filter plugin: %w`, errInjectedFilterStatus)).WithPlugin("TestPlugin2"),
		},
		{
			name: "node has a low-priority nominated pod and pre filters return unschedulable",
			preFilterPlugin: &TestPlugin{
				name: "TestPlugin1",
				inj: injectedResult{
					PreFilterAddPodStatus: int(fwk.Unschedulable),
				},
			},
			filterPlugin: &TestPlugin{
				name: "TestPlugin2",
				inj: injectedResult{
					FilterStatus: int(fwk.Success),
				},
			},
			pod:          highPriorityPod,
			nominatedPod: lowPriorityPod,
			node:         node,
			nodeInfo:     framework.NewNodeInfo(pod),
			wantStatus:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			cfgPls := &config.Plugins{}

			if tt.preFilterPlugin != nil {
				if err := registry.Register(tt.preFilterPlugin.name,
					func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.preFilterPlugin, nil
					}); err != nil {
					t.Fatalf("fail to register preFilter plugin (%s)", tt.preFilterPlugin.name)
				}
				cfgPls.PreFilter.Enabled = append(
					cfgPls.PreFilter.Enabled,
					config.Plugin{Name: tt.preFilterPlugin.name},
				)
			}
			if tt.filterPlugin != nil {
				if err := registry.Register(tt.filterPlugin.name,
					func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
						return tt.filterPlugin, nil
					}); err != nil {
					t.Fatalf("fail to register filter plugin (%s)", tt.filterPlugin.name)
				}
				cfgPls.Filter.Enabled = append(
					cfgPls.Filter.Enabled,
					config.Plugin{Name: tt.filterPlugin.name},
				)
			}

			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0)
			podInformer := informerFactory.Core().V1().Pods().Informer()
			err := podInformer.GetStore().Add(tt.pod)
			if err != nil {
				t.Fatalf("Error adding pod to podInformer: %s", err)
			}
			if tt.nominatedPod != nil {
				err = podInformer.GetStore().Add(tt.nominatedPod)
				if err != nil {
					t.Fatalf("Error adding nominated pod to podInformer: %s", err)
				}
			}

			podNominator := internalqueue.NewSchedulingQueue(nil, informerFactory)
			if tt.nominatedPod != nil {
				podNominator.AddNominatedPod(
					logger,
					mustNewPodInfo(t, tt.nominatedPod),
					&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName})
			}
			profile := config.KubeSchedulerProfile{Plugins: cfgPls}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile, WithPodNominator(podNominator))
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()
			tt.nodeInfo.SetNode(tt.node)
			gotStatus := f.RunFilterPluginsWithNominatedPods(ctx, state, tt.pod, tt.nodeInfo)
			if diff := cmp.Diff(tt.wantStatus, gotStatus, statusCmpOpts...); diff != "" {
				t.Errorf("Unexpected status: (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPreBindPlugins(t *testing.T) {
	tests := []struct {
		name       string
		plugins    []*TestPlugin
		wantStatus *fwk.Status
	}{
		{
			name:       "NoPreBindPlugin",
			plugins:    []*TestPlugin{},
			wantStatus: nil,
		},
		{
			name: "SuccessPreBindPlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "UnschedulablePreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "ErrorPreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running PreBind plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "UnschedulablePreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "SuccessErrorPreBindPlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PreBindStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running PreBind plugin "TestPlugin 1": %w`, errInjectedStatus)),
		},
		{
			name: "ErrorSuccessPreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running PreBind plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "SuccessSuccessPreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "ErrorAndErrorPlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PreBindStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running PreBind plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "UnschedulableAndSuccessPreBindPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PreBindStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectReason).WithPlugin("TestPlugin"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			configPlugins := &config.Plugins{}

			for _, pl := range tt.plugins {
				tmpPl := pl
				if err := registry.Register(pl.name, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return tmpPl, nil
				}); err != nil {
					t.Fatalf("Unable to register pre bind plugins: %s", pl.name)
				}

				configPlugins.PreBind.Enabled = append(
					configPlugins.PreBind.Enabled,
					config.Plugin{Name: pl.name},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: configPlugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()

			status := f.RunPreBindPlugins(ctx, state, pod, "")

			if diff := cmp.Diff(tt.wantStatus, status, statusCmpOpts...); diff != "" {
				t.Errorf("Wrong status code (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestGetPreBindPluginGroups(t *testing.T) {
	tests := []struct {
		name            string
		plugins         []fwk.PreBindPlugin
		skippedPlugins  sets.Set[string]
		parallelPlugins sets.Set[string]
		expectedGroups  [][]string
	}{
		{
			name:           "No plugins",
			plugins:        []fwk.PreBindPlugin{},
			expectedGroups: nil,
		},
		{
			name: "Single plugin, not skipped, not parallel",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
			},
			expectedGroups: [][]string{{"p1"}},
		},
		{
			name: "Single plugin, skipped",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
			},
			skippedPlugins: sets.New("p1"),
			expectedGroups: nil,
		},
		{
			name: "Single plugin, parallel",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
			},
			parallelPlugins: sets.New("p1"),
			expectedGroups:  [][]string{{"p1"}},
		},
		{
			name: "Single plugin, parallel & skipped",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
			},
			parallelPlugins: sets.New("p1"),
			skippedPlugins:  sets.New("p1"),
			expectedGroups:  nil,
		},
		{
			name: "Multiple plugins, consecutive serial",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
				&TestPlugin{name: "p2"},
			},
			expectedGroups: [][]string{
				{"p1"},
				{"p2"},
			},
		},
		{
			name: "Multiple plugins, consecutive parallel",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"},
				&TestPlugin{name: "p2"},
			},
			parallelPlugins: sets.New("p1", "p2"),
			expectedGroups: [][]string{
				{"p1", "p2"},
			},
		},
		{
			name: "Multiple plugins, mixed parallel and serial",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // serial
				&TestPlugin{name: "p2"}, // parallel
				&TestPlugin{name: "p3"}, // parallel
				&TestPlugin{name: "p4"}, // serial
			},
			parallelPlugins: sets.New("p2", "p3"),
			expectedGroups: [][]string{
				{"p1"},
				{"p2", "p3"},
				{"p4"},
			},
		},
		{
			name: "Parallel plugins separated by serial",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // parallel
				&TestPlugin{name: "p2"}, // serial
				&TestPlugin{name: "p3"}, // parallel
			},
			parallelPlugins: sets.New("p1", "p3"),
			expectedGroups: [][]string{
				{"p1"},
				{"p2"},
				{"p3"},
			},
		},
		{
			name: "Skipped plugins",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // serial
				&TestPlugin{name: "p2"}, // serial & skipped
				&TestPlugin{name: "p3"}, // serial
			},
			skippedPlugins: sets.New("p2"),
			expectedGroups: [][]string{
				{"p1"},
				{"p3"},
			},
		},
		{
			name: "Parallel plugins with one skipped in between",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // parallel
				&TestPlugin{name: "p2"}, // parallel & skipped
				&TestPlugin{name: "p3"}, // parallel
			},
			parallelPlugins: sets.New("p1", "p2", "p3"),
			skippedPlugins:  sets.New("p2"),
			expectedGroups:  [][]string{{"p1", "p3"}},
		},
		{
			name: "Parallel plugins with one serial & skipped in between",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // parallel
				&TestPlugin{name: "p2"}, // serial & skipped
				&TestPlugin{name: "p3"}, // parallel
			},
			parallelPlugins: sets.New("p1", "p3"),
			skippedPlugins:  sets.New("p2"),
			expectedGroups: [][]string{
				{"p1"},
				{"p3"},
			},
		},
		{
			name: "Complex mix",
			plugins: []fwk.PreBindPlugin{
				&TestPlugin{name: "p1"}, // serial
				&TestPlugin{name: "p2"}, // parallel
				&TestPlugin{name: "p3"}, // serial & skipped
				&TestPlugin{name: "p4"}, // parallel
				&TestPlugin{name: "p5"}, // parallel & skipped
				&TestPlugin{name: "p6"}, // parallel
				&TestPlugin{name: "p7"}, // serial
				&TestPlugin{name: "p8"}, // serial
			},
			parallelPlugins: sets.New("p2", "p4", "p5", "p6"),
			skippedPlugins:  sets.New("p3", "p5"),
			expectedGroups: [][]string{
				{"p1"},
				{"p2"},
				{"p4", "p6"},
				{"p7"},
				{"p8"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &frameworkImpl{
				preBindPlugins: tt.plugins,
			}
			state := framework.NewCycleState()

			groups := f.getPreBindPluginGroups(state, tt.skippedPlugins, tt.parallelPlugins)

			var gotGroups [][]string
			for _, g := range groups {
				var groupNames []string
				for _, p := range g {
					groupNames = append(groupNames, p.Name())
				}
				gotGroups = append(gotGroups, groupNames)
			}

			if diff := cmp.Diff(tt.expectedGroups, gotGroups); diff != "" {
				t.Errorf("getPreBindPluginGroups() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPreBindPreFlightPlugins(t *testing.T) {
	tests := []struct {
		name       string
		plugins    []*TestPlugin
		wantStatus *fwk.Status
	}{
		{
			name:       "Skip when there's no PreBind Plugin",
			plugins:    []*TestPlugin{},
			wantStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			name: "Success when PreBindPreFlight returns Success",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Skip)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "Skip when all PreBindPreFlight returns Skip",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Skip)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Skip)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			name: "Error when PreBindPreFlight returns Error",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin1",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Skip)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running PreBindPreFlight "TestPlugin2": %w`, errInjectedStatus)),
		},
		{
			name: "Error when PreBindPreFlight returns Unschedulable",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PreBindPreFlightStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Error, "PreBindPreFlight TestPlugin returned \"Unschedulable\", which is unsupported. It is supposed to return Success, Skip, or Error status"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			configPlugins := &config.Plugins{}

			for _, pl := range tt.plugins {
				tmpPl := pl
				if err := registry.Register(pl.name, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return tmpPl, nil
				}); err != nil {
					t.Fatalf("Unable to register pre bind plugins: %s", pl.name)
				}

				configPlugins.PreBind.Enabled = append(
					configPlugins.PreBind.Enabled,
					config.Plugin{Name: pl.name},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: configPlugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}
			defer func() {
				_ = f.Close()
			}()

			status := f.RunPreBindPreFlights(ctx, state, pod, "")

			if diff := cmp.Diff(tt.wantStatus, status, statusCmpOpts...); diff != "" {
				t.Errorf("Wrong status code (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestReservePlugins(t *testing.T) {
	tests := []struct {
		name       string
		plugins    []*TestPlugin
		wantStatus *fwk.Status
	}{
		{
			name:       "NoReservePlugin",
			plugins:    []*TestPlugin{},
			wantStatus: nil,
		},
		{
			name: "SuccessReservePlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "UnschedulableReservePlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "ErrorReservePlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Reserve plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "UnschedulableReservePlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "SuccessSuccessReservePlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "ErrorErrorReservePlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{ReserveStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Reserve plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "SuccessErrorReservePlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{ReserveStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Reserve plugin "TestPlugin 1": %w`, errInjectedStatus)),
		},
		{
			name: "ErrorSuccessReservePlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Reserve plugin "TestPlugin": %w`, errInjectedStatus)),
		},
		{
			name: "UnschedulableAndSuccessReservePlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{ReserveStatus: int(fwk.Unschedulable)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{ReserveStatus: int(fwk.Success)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectReason).WithPlugin("TestPlugin"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			configPlugins := &config.Plugins{}

			for _, pl := range tt.plugins {
				tmpPl := pl
				if err := registry.Register(pl.name, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return tmpPl, nil
				}); err != nil {
					t.Fatalf("Unable to register pre bind plugins: %s", pl.name)
				}

				configPlugins.Reserve.Enabled = append(
					configPlugins.Reserve.Enabled,
					config.Plugin{Name: pl.name},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: configPlugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			defer func() {
				_ = f.Close()
			}()
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}

			status := f.RunReservePluginsReserve(ctx, state, pod, "")

			if diff := cmp.Diff(tt.wantStatus, status, statusCmpOpts...); diff != "" {
				t.Errorf("Wrong status code (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPermitPlugins(t *testing.T) {
	tests := []struct {
		name                string
		plugins             []*TestPlugin
		wantPluginsWaitTime map[string]time.Duration
		wantStatus          *fwk.Status
	}{
		{
			name:       "NilPermitPlugin",
			plugins:    []*TestPlugin{},
			wantStatus: nil,
		},
		{
			name: "SuccessPermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "UnschedulablePermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Unschedulable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.Unschedulable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "ErrorPermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Permit plugin "TestPlugin": %w`, errInjectedStatus)).WithPlugin("TestPlugin"),
		},
		{
			name: "UnschedulableAndUnresolvablePermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.UnschedulableAndUnresolvable)},
				},
			},
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, injectReason).WithPlugin("TestPlugin"),
		},
		{
			name: "WaitPermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Wait)},
				},
				{
					name: "TestPlugin2",
					inj:  injectedResult{PermitStatus: int(fwk.Wait), PermitTimeout: time.Second},
				},
			},
			wantPluginsWaitTime: map[string]time.Duration{
				"TestPlugin":  0,
				"TestPlugin2": time.Second,
			},
			wantStatus: fwk.NewStatus(fwk.Wait, "injected status, one or more plugins asked to wait and no plugin rejected pod"),
		},
		{
			name: "SuccessSuccessPermitPlugin",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Success)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PermitStatus: int(fwk.Success)},
				},
			},
			wantStatus: nil,
		},
		{
			name: "ErrorAndErrorPlugins",
			plugins: []*TestPlugin{
				{
					name: "TestPlugin",
					inj:  injectedResult{PermitStatus: int(fwk.Error)},
				},
				{
					name: "TestPlugin 1",
					inj:  injectedResult{PermitStatus: int(fwk.Error)},
				},
			},
			wantStatus: fwk.AsStatus(fmt.Errorf(`running Permit plugin "TestPlugin": %w`, errInjectedStatus)).WithPlugin("TestPlugin"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			registry := Registry{}
			configPlugins := &config.Plugins{}

			for _, pl := range tt.plugins {
				tmpPl := pl
				if err := registry.Register(pl.name, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return tmpPl, nil
				}); err != nil {
					t.Fatalf("Unable to register Permit plugin: %s", pl.name)
				}

				configPlugins.Permit.Enabled = append(
					configPlugins.Permit.Enabled,
					config.Plugin{Name: pl.name},
				)
			}
			profile := config.KubeSchedulerProfile{Plugins: configPlugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile,
				WithWaitingPods(NewWaitingPodsMap()),
			)
			defer func() {
				_ = f.Close()
			}()
			if err != nil {
				t.Fatalf("fail to create framework: %s", err)
			}

			pluginsWaitTime, status := f.RunPermitPlugins(ctx, state, pod, "")
			if diff := cmp.Diff(tt.wantStatus, status, statusCmpOpts...); diff != "" {
				t.Errorf("Wrong status code (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantPluginsWaitTime, pluginsWaitTime); diff != "" {
				t.Errorf("Wrong plugins wait time map (-want,+got):\n%s", diff)
			}
		})
	}
}

// withMetricsRecorder set metricsRecorder for the scheduling frameworkImpl.
func withMetricsRecorder(recorder *metrics.MetricAsyncRecorder) Option {
	return func(o *frameworkOptions) {
		o.metricsRecorder = recorder
	}
}

func TestRecordingMetrics(t *testing.T) {
	state.SetRecordPluginMetrics(true)
	tests := []struct {
		name               string
		action             func(ctx context.Context, f framework.Framework)
		inject             injectedResult
		wantExtensionPoint string
		wantStatus         fwk.Code
	}{
		{
			name:               "PreFilter - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreFilterPlugins(ctx, state, pod) },
			wantExtensionPoint: "PreFilter",
			wantStatus:         fwk.Success,
		},
		{
			name:               "PreScore - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreScorePlugins(ctx, state, pod, nil) },
			wantExtensionPoint: "PreScore",
			wantStatus:         fwk.Success,
		},
		{
			name: "Score - Success",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunScorePlugins(ctx, state, pod, BuildNodeInfos(nodes))
			},
			wantExtensionPoint: "Score",
			wantStatus:         fwk.Success,
		},
		{
			name:               "Reserve - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunReservePluginsReserve(ctx, state, pod, "") },
			wantExtensionPoint: "Reserve",
			wantStatus:         fwk.Success,
		},
		{
			name:               "Unreserve - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunReservePluginsUnreserve(ctx, state, pod, "") },
			wantExtensionPoint: "Unreserve",
			wantStatus:         fwk.Success,
		},
		{
			name:               "PreBind - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreBindPlugins(ctx, state, pod, "") },
			wantExtensionPoint: "PreBind",
			wantStatus:         fwk.Success,
		},
		{
			name:               "Bind - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunBindPlugins(ctx, state, pod, "") },
			wantExtensionPoint: "Bind",
			wantStatus:         fwk.Success,
		},
		{
			name:               "PostBind - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPostBindPlugins(ctx, state, pod, "") },
			wantExtensionPoint: "PostBind",
			wantStatus:         fwk.Success,
		},
		{
			name:               "Permit - Success",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPermitPlugins(ctx, state, pod, "") },
			wantExtensionPoint: "Permit",
			wantStatus:         fwk.Success,
		},
		{
			name: "PlacementGenerate - Success",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunPlacementGeneratePlugins(ctx, state, nil, []fwk.NodeInfo{framework.NewNodeInfo()})
			},
			inject:             injectedResult{GeneratePlacementsResult: []*fwk.Placement{{}}},
			wantExtensionPoint: "PlacementGenerate",
			wantStatus:         fwk.Success,
		},
		{
			name: "PlacementScore - Success",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunPlacementScorePlugins(ctx, state, nil, []*fwk.PodGroupAssignments{{Placement: &fwk.Placement{}}})
			},
			wantExtensionPoint: "PlacementScore",
			wantStatus:         fwk.Success,
		},

		{
			name:               "PreFilter - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreFilterPlugins(ctx, state, pod) },
			inject:             injectedResult{PreFilterStatus: int(fwk.Error)},
			wantExtensionPoint: "PreFilter",
			wantStatus:         fwk.Error,
		},
		{
			name:               "PreScore - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreScorePlugins(ctx, state, pod, nil) },
			inject:             injectedResult{PreScoreStatus: int(fwk.Error)},
			wantExtensionPoint: "PreScore",
			wantStatus:         fwk.Error,
		},
		{
			name: "Score - Error",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunScorePlugins(ctx, state, pod, BuildNodeInfos(nodes))
			},
			inject:             injectedResult{ScoreStatus: int(fwk.Error)},
			wantExtensionPoint: "Score",
			wantStatus:         fwk.Error,
		},
		{
			name:               "Reserve - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunReservePluginsReserve(ctx, state, pod, "") },
			inject:             injectedResult{ReserveStatus: int(fwk.Error)},
			wantExtensionPoint: "Reserve",
			wantStatus:         fwk.Error,
		},
		{
			name:               "PreBind - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPreBindPlugins(ctx, state, pod, "") },
			inject:             injectedResult{PreBindStatus: int(fwk.Error)},
			wantExtensionPoint: "PreBind",
			wantStatus:         fwk.Error,
		},
		{
			name:               "Bind - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunBindPlugins(ctx, state, pod, "") },
			inject:             injectedResult{BindStatus: int(fwk.Error)},
			wantExtensionPoint: "Bind",
			wantStatus:         fwk.Error,
		},
		{
			name:               "Permit - Error",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPermitPlugins(ctx, state, pod, "") },
			inject:             injectedResult{PermitStatus: int(fwk.Error)},
			wantExtensionPoint: "Permit",
			wantStatus:         fwk.Error,
		},
		{
			name:               "Permit - Wait",
			action:             func(ctx context.Context, f framework.Framework) { f.RunPermitPlugins(ctx, state, pod, "") },
			inject:             injectedResult{PermitStatus: int(fwk.Wait)},
			wantExtensionPoint: "Permit",
			wantStatus:         fwk.Wait,
		},
		{
			name: "PlacementGenerate - Error",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunPlacementGeneratePlugins(ctx, state, nil, []fwk.NodeInfo{framework.NewNodeInfo()})
			},
			inject:             injectedResult{GeneratePlacementsStatus: int(fwk.Error)},
			wantExtensionPoint: "PlacementGenerate",
			wantStatus:         fwk.Error,
		},
		{
			name: "PlacementScore - Error",
			action: func(ctx context.Context, f framework.Framework) {
				f.RunPlacementScorePlugins(ctx, state, nil, []*fwk.PodGroupAssignments{{Placement: &fwk.Placement{}}})
			},
			inject:             injectedResult{PlacementScoreStatus: int(fwk.Error)},
			wantExtensionPoint: "PlacementScore",
			wantStatus:         fwk.Error,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			metrics.FrameworkExtensionPointDuration.Reset()
			metrics.PluginExecutionDuration.Reset()

			plugin := &TestPlugin{name: testPlugin, inj: tt.inject}
			r := make(Registry)
			r.Register(testPlugin,
				func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return plugin, nil
				})
			pluginSet := config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}}
			plugins := &config.Plugins{
				Score:             pluginSet,
				PreFilter:         pluginSet,
				Filter:            pluginSet,
				PreScore:          pluginSet,
				Reserve:           pluginSet,
				Permit:            pluginSet,
				PreBind:           pluginSet,
				Bind:              pluginSet,
				PostBind:          pluginSet,
				PlacementGenerate: pluginSet,
				PlacementScore:    pluginSet,
			}

			recorder := metrics.NewMetricsAsyncRecorder(100, time.Nanosecond, ctx.Done())
			profile := config.KubeSchedulerProfile{
				PercentageOfNodesToScore: ptr.To[int32](testPercentageOfNodesToScore),
				SchedulerName:            testProfileName,
				Plugins:                  plugins,
			}
			f, err := newFrameworkWithQueueSortAndBind(ctx, r, profile,
				withMetricsRecorder(recorder),
				WithWaitingPods(NewWaitingPodsMap()),
				WithSnapshotSharedLister(cache.NewEmptySnapshot()),
			)
			if err != nil {
				cancel()
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			tt.action(ctx, f)

			// Stop the goroutine which records metrics and ensure it's stopped.
			cancel()
			<-recorder.IsStoppedCh
			// Try to clean up the metrics buffer again in case it's not empty.
			recorder.FlushMetrics()

			collectAndCompareFrameworkMetrics(t, tt.wantExtensionPoint, tt.wantStatus)
			collectAndComparePluginMetrics(t, tt.wantExtensionPoint, testPlugin, tt.wantStatus)
		})
	}
}

func TestRunBindPlugins(t *testing.T) {
	tests := []struct {
		name       string
		injects    []fwk.Code
		wantStatus fwk.Code
	}{
		{
			name:       "simple success",
			injects:    []fwk.Code{fwk.Success},
			wantStatus: fwk.Success,
		},
		{
			name:       "error on second",
			injects:    []fwk.Code{fwk.Skip, fwk.Error, fwk.Success},
			wantStatus: fwk.Error,
		},
		{
			name:       "all skip",
			injects:    []fwk.Code{fwk.Skip, fwk.Skip, fwk.Skip},
			wantStatus: fwk.Skip,
		},
		{
			name:       "error on third, but not reached",
			injects:    []fwk.Code{fwk.Skip, fwk.Success, fwk.Error},
			wantStatus: fwk.Success,
		},
		{
			name:       "no bind plugin, returns default binder",
			injects:    []fwk.Code{},
			wantStatus: fwk.Success,
		},
		{
			name:       "invalid status",
			injects:    []fwk.Code{fwk.Unschedulable},
			wantStatus: fwk.Unschedulable,
		},
		{
			name:       "simple error",
			injects:    []fwk.Code{fwk.Error},
			wantStatus: fwk.Error,
		},
		{
			name:       "success on second, returns success",
			injects:    []fwk.Code{fwk.Skip, fwk.Success},
			wantStatus: fwk.Success,
		},
		{
			name:       "invalid status, returns error",
			injects:    []fwk.Code{fwk.Skip, fwk.UnschedulableAndUnresolvable},
			wantStatus: fwk.UnschedulableAndUnresolvable,
		},
		{
			name:       "error after success status, returns success",
			injects:    []fwk.Code{fwk.Success, fwk.Error},
			wantStatus: fwk.Success,
		},
		{
			name:       "success before invalid status, returns success",
			injects:    []fwk.Code{fwk.Success, fwk.Error},
			wantStatus: fwk.Success,
		},
		{
			name:       "success after error status, returns error",
			injects:    []fwk.Code{fwk.Error, fwk.Success},
			wantStatus: fwk.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.FrameworkExtensionPointDuration.Reset()
			metrics.PluginExecutionDuration.Reset()

			pluginSet := config.PluginSet{}
			r := make(Registry)
			for i, inj := range tt.injects {
				name := fmt.Sprintf("bind-%d", i)
				plugin := &TestPlugin{name: name, inj: injectedResult{BindStatus: int(inj)}}
				r.Register(name,
					func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
						return plugin, nil
					})
				pluginSet.Enabled = append(pluginSet.Enabled, config.Plugin{Name: name})
			}
			plugins := &config.Plugins{Bind: pluginSet}
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			recorder := metrics.NewMetricsAsyncRecorder(100, time.Nanosecond, ctx.Done())
			profile := config.KubeSchedulerProfile{
				SchedulerName:            testProfileName,
				PercentageOfNodesToScore: ptr.To[int32](testPercentageOfNodesToScore),
				Plugins:                  plugins,
			}
			fwk, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, withMetricsRecorder(recorder))
			if err != nil {
				cancel()
				t.Fatal(err)
			}
			defer func() {
				_ = fwk.Close()
			}()

			st := fwk.RunBindPlugins(ctx, state, pod, "")
			if st.Code() != tt.wantStatus {
				t.Errorf("got status code %s, want %s", st.Code(), tt.wantStatus)
			}

			// Stop the goroutine which records metrics and ensure it's stopped.
			cancel()
			<-recorder.IsStoppedCh
			// Try to clean up the metrics buffer again in case it's not empty.
			recorder.FlushMetrics()
			collectAndCompareFrameworkMetrics(t, "Bind", tt.wantStatus)
		})
	}
}

func TestPermitWaitDurationMetric(t *testing.T) {
	tests := []struct {
		name    string
		inject  injectedResult
		wantRes string
	}{
		{
			name: "WaitOnPermit - No Wait",
		},
		{
			name:    "WaitOnPermit - Wait Timeout",
			inject:  injectedResult{PermitStatus: int(fwk.Wait)},
			wantRes: "Unschedulable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			metrics.PermitWaitDuration.Reset()

			plugin := &TestPlugin{name: testPlugin, inj: tt.inject}
			r := make(Registry)
			err := r.Register(testPlugin,
				func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return plugin, nil
				})
			if err != nil {
				t.Fatal(err)
			}
			plugins := &config.Plugins{
				Permit: config.PluginSet{Enabled: []config.Plugin{{Name: testPlugin, Weight: 1}}},
			}
			profile := config.KubeSchedulerProfile{Plugins: plugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, r, profile,
				WithWaitingPods(NewWaitingPodsMap()),
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			pluginsWaitTime, status := f.RunPermitPlugins(ctx, state, pod, "")
			if status.IsWait() {
				f.AddWaitingPod(pod, pluginsWaitTime)
			} else if !status.IsSuccess() {
				t.Fatalf("Failed to run permit plugins: %v", status)
			}
			status = f.WaitOnPermit(ctx, pod)
			if !status.IsSuccess() && tt.wantRes != "Unschedulable" {
				t.Fatalf("Failed to wait on permit: %v", status)
			}

			collectAndComparePermitWaitDuration(t, tt.wantRes)
		})
	}
}

func TestWaitOnPermit(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod",
			UID:  types.UID("pod"),
		},
	}

	tests := []struct {
		name   string
		action func(f framework.Framework)
		want   *fwk.Status
	}{
		{
			name: "Reject Waiting Pod",
			action: func(f framework.Framework) {
				f.GetWaitingPod(pod.UID).Reject(permitPlugin, "reject message")
			},
			want: fwk.NewStatus(fwk.Unschedulable, "reject message").WithPlugin(permitPlugin),
		},
		{
			name: "Allow Waiting Pod",
			action: func(f framework.Framework) {
				f.GetWaitingPod(pod.UID).Allow(permitPlugin)
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			testPermitPlugin := &TestPermitPlugin{}
			r := make(Registry)
			r.Register(permitPlugin,
				func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return testPermitPlugin, nil
				})
			plugins := &config.Plugins{
				Permit: config.PluginSet{Enabled: []config.Plugin{{Name: permitPlugin, Weight: 1}}},
			}
			profile := config.KubeSchedulerProfile{Plugins: plugins}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, r, profile,
				WithWaitingPods(NewWaitingPodsMap()),
			)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()

			pluginsWaitTime, runPermitPluginsStatus := f.RunPermitPlugins(ctx, state, pod, "")
			if runPermitPluginsStatus.Code() != fwk.Wait {
				t.Fatalf("Expected RunPermitPlugins to return status %v, but got %v",
					fwk.Wait, runPermitPluginsStatus.Code())
			}
			f.AddWaitingPod(pod, pluginsWaitTime)

			go tt.action(f)

			got := f.WaitOnPermit(ctx, pod)
			if diff := cmp.Diff(tt.want, got, statusCmpOpts...); diff != "" {
				t.Errorf("Unexpected status (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestListPlugins(t *testing.T) {
	tests := []struct {
		name    string
		plugins *config.Plugins
		want    *config.Plugins
	}{
		{
			name:    "Add empty plugin",
			plugins: &config.Plugins{},
			want: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
				Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
			},
		},
		{
			name: "Add multiple plugins",
			plugins: &config.Plugins{
				Score: config.PluginSet{Enabled: []config.Plugin{{Name: scorePlugin1, Weight: 3}, {Name: scoreWithNormalizePlugin1}}},
			},
			want: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: queueSortPlugin}}},
				Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: bindPlugin}}},
				Score:     config.PluginSet{Enabled: []config.Plugin{{Name: scorePlugin1, Weight: 3}, {Name: scoreWithNormalizePlugin1, Weight: 1}}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := config.KubeSchedulerProfile{Plugins: tt.plugins}
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
			defer func() {
				_ = f.Close()
			}()
			got := f.ListPlugins()
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Unexpected plugins (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestClose(t *testing.T) {
	tests := []struct {
		name    string
		plugins *config.Plugins
		wantErr error
	}{
		{
			name: "close doesn't return error",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin, Weight: 5},
					},
				},
			},
		},
		{
			name: "close returns error",
			plugins: &config.Plugins{
				MultiPoint: config.PluginSet{
					Enabled: []config.Plugin{
						{Name: testPlugin, Weight: 5},
						{Name: testCloseErrorPlugin},
					},
				},
			},
			wantErr: errClose,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fw, err := NewFramework(ctx, registry, &config.KubeSchedulerProfile{Plugins: tc.plugins})
			if err != nil {
				t.Fatalf("Unexpected error during calling NewFramework, got %v", err)
			}
			err = fw.Close()
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Unexpected error from Close(), got: %v, want: %v", err, tc.wantErr)
			}
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
	return &config.Plugins{Score: config.PluginSet{Enabled: plugins}}
}

type injectedResult struct {
	ScoreRes                 int64                `json:"scoreRes,omitempty"`
	NormalizeRes             int64                `json:"normalizeRes,omitempty"`
	ScoreStatus              int                  `json:"scoreStatus,omitempty"`
	NormalizeStatus          int                  `json:"normalizeStatus,omitempty"`
	PreFilterResult          *fwk.PreFilterResult `json:"preFilterResult,omitempty"`
	PreFilterStatus          int                  `json:"preFilterStatus,omitempty"`
	PreFilterAddPodStatus    int                  `json:"preFilterAddPodStatus,omitempty"`
	PreFilterRemovePodStatus int                  `json:"preFilterRemovePodStatus,omitempty"`
	FilterStatus             int                  `json:"filterStatus,omitempty"`
	PostFilterStatus         int                  `json:"postFilterStatus,omitempty"`
	PreScoreStatus           int                  `json:"preScoreStatus,omitempty"`
	ReserveStatus            int                  `json:"reserveStatus,omitempty"`
	PreBindPreFlightStatus   int                  `json:"preBindPreFlightStatus,omitempty"`
	PreBindStatus            int                  `json:"preBindStatus,omitempty"`
	BindStatus               int                  `json:"bindStatus,omitempty"`
	PermitStatus             int                  `json:"permitStatus,omitempty"`
	PermitTimeout            time.Duration        `json:"permitTimeout,omitempty"`
	GeneratePlacementsResult []*fwk.Placement     `json:"generatePlacementsResult,omitempty"`
	GeneratePlacementsStatus int                  `json:"generatePlacementsStatus,omitempty"`
	PlacementScoreStatus     int                  `json:"placementScoreStatus,omitempty"`
}

func setScoreRes(inj injectedResult) (int64, *fwk.Status) {
	if fwk.Code(inj.ScoreStatus) != fwk.Success {
		return 0, fwk.NewStatus(fwk.Code(inj.ScoreStatus), "injecting failure.")
	}
	return inj.ScoreRes, nil
}

func injectNormalizeRes(inj injectedResult, scores fwk.NodeScoreList) *fwk.Status {
	if fwk.Code(inj.NormalizeStatus) != fwk.Success {
		return fwk.NewStatus(fwk.Code(inj.NormalizeStatus), "injecting failure.")
	}
	for i := range scores {
		scores[i].Score = inj.NormalizeRes
	}
	return nil
}

func collectAndComparePluginMetrics(t *testing.T, wantExtensionPoint, wantPlugin string, wantStatus fwk.Code) {
	t.Helper()
	m := metrics.PluginExecutionDuration.WithLabelValues(wantPlugin, wantExtensionPoint, wantStatus.String())

	count, err := testutil.GetHistogramMetricCount(m)
	if err != nil {
		t.Errorf("Failed to get %s sampleCount, err: %v", metrics.PluginExecutionDuration.Name, err)
	}
	if count == 0 {
		t.Error("Expect at least 1 sample")
	}
	value, err := testutil.GetHistogramMetricValue(m)
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metrics.PluginExecutionDuration.Name, err)
	}
	checkLatency(t, value)
}

func collectAndCompareFrameworkMetrics(t *testing.T, wantExtensionPoint string, wantStatus fwk.Code) {
	t.Helper()
	m := metrics.FrameworkExtensionPointDuration.WithLabelValues(wantExtensionPoint, wantStatus.String(), testProfileName)

	count, err := testutil.GetHistogramMetricCount(m)
	if err != nil {
		t.Errorf("Failed to get %s sampleCount, err: %v", metrics.FrameworkExtensionPointDuration.Name, err)
	}
	if count != 1 {
		t.Errorf("Expect 1 sample, got: %v", count)
	}
	value, err := testutil.GetHistogramMetricValue(m)
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metrics.FrameworkExtensionPointDuration.Name, err)
	}
	checkLatency(t, value)
}

func collectAndComparePermitWaitDuration(t *testing.T, wantRes string) {
	m := metrics.PermitWaitDuration.WithLabelValues(wantRes)
	count, err := testutil.GetHistogramMetricCount(m)
	if err != nil {
		t.Errorf("Failed to get %s sampleCount, err: %v", metrics.PermitWaitDuration.Name, err)
	}
	if wantRes == "" {
		if count != 0 {
			t.Errorf("Expect 0 sample, got: %v", count)
		}
	} else {
		if count != 1 {
			t.Errorf("Expect 1 sample, got: %v", count)
		}
		value, err := testutil.GetHistogramMetricValue(m)
		if err != nil {
			t.Errorf("Failed to get %s value, err: %v", metrics.PermitWaitDuration.Name, err)
		}
		checkLatency(t, value)
	}
}

func mustNewPodInfo(t *testing.T, pod *v1.Pod) *framework.PodInfo {
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatal(err)
	}
	return podInfo
}

// BuildNodeInfos build NodeInfo slice from a v1.Node slice
func BuildNodeInfos(nodes []*v1.Node) []fwk.NodeInfo {
	res := make([]fwk.NodeInfo, len(nodes))
	for i := 0; i < len(nodes); i++ {
		res[i] = framework.NewNodeInfo()
		res[i].SetNode(nodes[i])
	}
	return res
}

func TestRunPlacementGeneratePlugins(t *testing.T) {
	nodeResources := []*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
	}
	nodesInCluster := make([]fwk.NodeInfo, len(nodeResources))
	for i, node := range nodeResources {
		nodesInCluster[i] = framework.NewNodeInfo()
		nodesInCluster[i].SetNode(node)
	}
	tests := map[string]struct {
		pluginResults  []injectedResult
		wantPlacements []*fwk.Placement
		wantStatusCode fwk.Code
	}{
		"When no plugins provided, returns a single placement with initial nodes": {
			pluginResults: []injectedResult{},
			wantPlacements: []*fwk.Placement{
				{
					Name:  "",
					Nodes: nodesInCluster,
				},
			},
			wantStatusCode: fwk.Success,
		},
		"When one plugin provided, returns placements generated by the plugin": {
			pluginResults: []injectedResult{
				{
					GeneratePlacementsResult: []*fwk.Placement{
						{
							Name: "foo",
							Nodes: []fwk.NodeInfo{
								nodesInCluster[0],
								nodesInCluster[1],
							},
						},
						{
							Name: "bar",
							Nodes: []fwk.NodeInfo{
								nodesInCluster[1],
								nodesInCluster[2],
							},
						},
					},
				},
			},
			wantPlacements: []*fwk.Placement{
				{
					Name: "foo",
					Nodes: []fwk.NodeInfo{
						nodesInCluster[0],
						nodesInCluster[1],
					},
				},
				{
					Name: "bar",
					Nodes: []fwk.NodeInfo{
						nodesInCluster[1],
						nodesInCluster[2],
					},
				},
			},
			wantStatusCode: fwk.Success,
		},
		"When plugin fails, returns error status": {
			pluginResults: []injectedResult{
				{GeneratePlacementsStatus: int(fwk.Error)},
			},
			wantStatusCode: fwk.Error,
		},
		"When plugin returns no placements, returns unschedulable": {
			pluginResults: []injectedResult{
				{GeneratePlacementsResult: []*fwk.Placement{}},
			},
			wantStatusCode: fwk.Unschedulable,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			r := make(Registry)
			pluginSet := config.PluginSet{}
			plugins := make([]*TestPlacementGeneratePlugin, len(tt.pluginResults))
			for i, inj := range tt.pluginResults {
				pluginName := fmt.Sprintf("plugin[%d]", i)
				pluginSet.Enabled = append(pluginSet.Enabled, config.Plugin{Name: pluginName})
				plugins[i] = &TestPlacementGeneratePlugin{
					name: pluginName,
					inj:  inj,
				}
				err := r.Register(pluginName, func(ctx context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return plugins[i], nil
				})
				if err != nil {
					t.Fatalf("failed to register PlacementGeneratePlugin")
				}
			}

			profile := config.KubeSchedulerProfile{Plugins: &config.Plugins{PlacementGenerate: pluginSet}}
			fw, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
			if err != nil {
				t.Fatalf("Unexpected error during calling NewFramework, got %v", err)
			}

			result, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, nodesInCluster)

			if status.Code() != tt.wantStatusCode {
				t.Errorf("Unexpected status code, want %v, got %v", tt.wantStatusCode, status.Code())
			}
			if diff := cmp.Diff(tt.wantPlacements, result, cmp.AllowUnexported(framework.NodeInfo{})); diff != "" {
				t.Errorf("Unexpected placements (-want,+got):\n%s", diff)
			}
		})
	}
}

type placementScoreResult struct {
	score  int64
	status *fwk.Status
}

type testPlacementScorePlugin struct {
	name        string
	weight      int32
	results     map[*fwk.Placement]placementScoreResult
	normalizeFn func(scores []fwk.PlacementScore) *fwk.Status
}

func (pl *testPlacementScorePlugin) Name() string {
	return pl.name
}

func (pl *testPlacementScorePlugin) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	r := pl.results[placement.Placement]
	return r.score, r.status
}

func (pl *testPlacementScorePlugin) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	if pl.normalizeFn == nil {
		return nil
	}
	return pl
}

func (pl *testPlacementScorePlugin) NormalizePlacementScore(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, scores []fwk.PlacementScore) *fwk.Status {
	return pl.normalizeFn(scores)
}

func TestRunPlacementScorePlugins(t *testing.T) {
	// 3 placements, the content doesn't matter as they're not used in the test
	placements := []*fwk.Placement{{}, {}, {}}

	tests := []struct {
		name           string
		plugins        []testPlacementScorePlugin
		wantScore      []fwk.PlacementPluginScores
		wantStatusCode fwk.Code
	}{
		{
			name: "all success",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
				{
					name:   "plugin2",
					weight: 2,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 0, status: nil},
					},
				},
			},
			wantScore: []fwk.PlacementPluginScores{
				{
					Placement: placements[0],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 0},
						{Name: "plugin2", Score: 0},
					},
					TotalScore: 0,
				},
				{
					Placement: placements[1],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 50},
						{Name: "plugin2", Score: 100},
					},
					TotalScore: 150,
				},
				{
					Placement: placements[2],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 100},
						{Name: "plugin2", Score: 0},
					},
					TotalScore: 100,
				},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "any error",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
				{
					name:   "plugin2",
					weight: 2,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: fwk.NewStatus(fwk.Error, "error for test")},
						placements[2]: {score: 0, status: nil},
					},
				},
			},
			wantStatusCode: fwk.Error,
		},
		{
			name:    "no plugins",
			plugins: []testPlacementScorePlugin{},
			wantScore: []fwk.PlacementPluginScores{
				{Placement: placements[0]},
				{Placement: placements[1]},
				{Placement: placements[2]},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "normalize success",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
				{
					name:   "plugin2",
					weight: 2,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 100, status: nil},
						placements[1]: {score: 200, status: nil},
						placements[2]: {score: 400, status: nil},
					},
					normalizeFn: func(scores []fwk.PlacementScore) *fwk.Status {
						for i := range scores {
							scores[i].Score = int64(float32(scores[i].Score) / float32(scores[2].Score) * 100)
						}
						return nil
					},
				}},
			wantScore: []fwk.PlacementPluginScores{
				{
					Placement: placements[0],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 0},
						{Name: "plugin2", Score: 50},
					},
					TotalScore: 50,
				},
				{
					Placement: placements[1],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 50},
						{Name: "plugin2", Score: 100},
					},
					TotalScore: 150,
				},
				{
					Placement: placements[2],
					Scores: []fwk.PluginScore{
						{Name: "plugin1", Score: 100},
						{Name: "plugin2", Score: 200},
					},
					TotalScore: 300,
				},
			},
			wantStatusCode: fwk.Success,
		},
		{
			name: "normalize failure",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
				{
					name:   "plugin2",
					weight: 2,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 50, status: nil},
						placements[2]: {score: 100, status: nil},
					},
					normalizeFn: func(scores []fwk.PlacementScore) *fwk.Status {
						return fwk.NewStatus(fwk.Error, "error for test")
					},
				}},
			wantStatusCode: fwk.Error,
		},
		{
			name: "plugin result is greater than max score",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: 101, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
			},
			wantStatusCode: fwk.Error,
		},
		{
			name: "plugin result is less than min score",
			plugins: []testPlacementScorePlugin{
				{
					name:   "plugin1",
					weight: 1,
					results: map[*fwk.Placement]placementScoreResult{
						placements[0]: {score: 0, status: nil},
						placements[1]: {score: -1, status: nil},
						placements[2]: {score: 100, status: nil},
					},
				},
			},
			wantStatusCode: fwk.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			r := make(Registry)
			pluginSet := config.PluginSet{}
			for _, p := range tt.plugins {
				pluginSet.Enabled = append(pluginSet.Enabled, config.Plugin{Name: p.name, Weight: p.weight})
				err := r.Register(p.name, func(ctx context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
					return &p, nil
				})
				if err != nil {
					t.Fatalf("Unexpected error during call to Register, got %v", err)
				}
			}
			profile := config.KubeSchedulerProfile{Plugins: &config.Plugins{PlacementScore: pluginSet}}
			fw, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
			if err != nil {
				t.Fatalf("Unexpected error during calling NewFramework, got %v", err)
			}
			assumedPlacements := make([]*fwk.PodGroupAssignments, len(placements))
			for i := range placements {
				assumedPlacements[i] = &fwk.PodGroupAssignments{Placement: placements[i]}
			}

			result, status := fw.RunPlacementScorePlugins(ctx, framework.NewCycleState(), nil, assumedPlacements)
			if status.Code() != tt.wantStatusCode {
				t.Errorf("got status code %s, want %s", status.Code(), tt.wantStatusCode)
			}
			if diff := cmp.Diff(tt.wantScore, result, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Unexpected placement score (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPluginEvaluationTotalMetric(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	metrics.PluginEvaluationTotal.Reset()

	registry := Registry{}

	const (
		preFilterPluginName = "plugin-eval-prefilter"
		filterPluginNameA   = "plugin-eval-filter-a"
		filterPluginNameB   = "plugin-eval-filter-b"
		preScorePluginName  = "plugin-eval-prescore"
		scorePluginName     = "plugin-eval-score"
		profileName2        = "test-profile-2"
	)

	preFilterPl := &TestPlugin{name: preFilterPluginName, inj: injectedResult{PreFilterStatus: int(fwk.Success)}}
	if err := registry.Register(preFilterPluginName, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return preFilterPl, nil
	}); err != nil {
		t.Fatalf("failed to register prefilter plugin %q: %v", preFilterPluginName, err)
	}

	filterPlA := &TestPlugin{name: filterPluginNameA, inj: injectedResult{FilterStatus: int(fwk.Success)}}
	if err := registry.Register(filterPluginNameA, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return filterPlA, nil
	}); err != nil {
		t.Fatalf("failed to register filter plugin %q: %v", filterPluginNameA, err)
	}

	filterPlB := &TestPlugin{name: filterPluginNameB, inj: injectedResult{FilterStatus: int(fwk.Success)}}
	if err := registry.Register(filterPluginNameB, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return filterPlB, nil
	}); err != nil {
		t.Fatalf("failed to register filter plugin %q: %v", filterPluginNameB, err)
	}

	preScorePl := &TestPlugin{name: preScorePluginName, inj: injectedResult{PreScoreStatus: int(fwk.Success)}}
	if err := registry.Register(preScorePluginName, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return preScorePl, nil
	}); err != nil {
		t.Fatalf("failed to register prescore plugin %q: %v", preScorePluginName, err)
	}

	scorePl := &TestPlugin{name: scorePluginName, inj: injectedResult{}}
	if err := registry.Register(scorePluginName, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return scorePl, nil
	}); err != nil {
		t.Fatalf("failed to register score plugin %q: %v", scorePluginName, err)
	}

	// Profile 1: exercise PreFilter, Filter, PreScore and Score extension points.
	cfgPls1 := &config.Plugins{}
	cfgPls1.PreFilter.Enabled = append(cfgPls1.PreFilter.Enabled, config.Plugin{Name: preFilterPluginName})
	cfgPls1.Filter.Enabled = append(cfgPls1.Filter.Enabled, config.Plugin{Name: filterPluginNameA})
	cfgPls1.PreScore.Enabled = append(cfgPls1.PreScore.Enabled, config.Plugin{Name: preScorePluginName})
	cfgPls1.Score.Enabled = append(cfgPls1.Score.Enabled, config.Plugin{Name: scorePluginName})
	profile1 := config.KubeSchedulerProfile{
		SchedulerName: testProfileName,
		Plugins:       cfgPls1,
	}

	f1, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile1, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
	if err != nil {
		t.Fatalf("failed to create framework (profile=%q): %v", testProfileName, err)
	}
	defer func() { _ = f1.Close() }()

	state1 := framework.NewCycleState()
	if _, st, _ := f1.RunPreFilterPlugins(ctx, state1, pod); st != nil && !st.IsSuccess() {
		t.Fatalf("RunPreFilterPlugins returned unexpected status: %v", st)
	}
	if st := f1.RunFilterPlugins(ctx, state1, pod, nil); st != nil && !st.IsSuccess() {
		t.Fatalf("RunFilterPlugins returned unexpected status: %v", st)
	}
	if st := f1.RunPreScorePlugins(ctx, state1, pod, nil); st != nil && !st.IsSuccess() {
		t.Fatalf("RunPreScorePlugins returned unexpected status: %v", st)
	}
	if _, st := f1.RunScorePlugins(ctx, state1, pod, BuildNodeInfos(nodes)); st != nil && !st.IsSuccess() {
		t.Fatalf("RunScorePlugins returned unexpected status: %v", st)
	}

	// Profile 2: exercise a different plugin and profile label on Filter.
	cfgPls2 := &config.Plugins{}
	cfgPls2.Filter.Enabled = append(cfgPls2.Filter.Enabled, config.Plugin{Name: filterPluginNameB})
	profile2 := config.KubeSchedulerProfile{
		SchedulerName: profileName2,
		Plugins:       cfgPls2,
	}

	f2, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile2, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
	if err != nil {
		t.Fatalf("failed to create framework (profile=%q): %v", profileName2, err)
	}
	defer func() { _ = f2.Close() }()

	state2 := framework.NewCycleState()
	if st := f2.RunFilterPlugins(ctx, state2, pod, nil); st != nil && !st.IsSuccess() {
		t.Fatalf("RunFilterPlugins returned unexpected status: %v", st)
	}

	want := `# HELP scheduler_plugin_evaluation_total Number of attempts to schedule pods by each plugin and the extension point (available only in PreFilter, Filter, PreScore, and Score).
# TYPE scheduler_plugin_evaluation_total counter
scheduler_plugin_evaluation_total{extension_point="Filter",plugin="plugin-eval-filter-a",profile="test-profile"} 1
scheduler_plugin_evaluation_total{extension_point="Filter",plugin="plugin-eval-filter-b",profile="test-profile-2"} 1
scheduler_plugin_evaluation_total{extension_point="PreFilter",plugin="plugin-eval-prefilter",profile="test-profile"} 1
scheduler_plugin_evaluation_total{extension_point="PreScore",plugin="plugin-eval-prescore",profile="test-profile"} 1
scheduler_plugin_evaluation_total{extension_point="Score",plugin="plugin-eval-score",profile="test-profile"} 1
`
	if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(want), metrics.PluginEvaluationTotal.Name); err != nil {
		t.Fatalf("unexpected plugin_evaluation_total metric output:\n%v", err)
	}
}
