/*
Copyright 2018 The Kubernetes Authors.

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

package scheduler

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

type PreFilterPlugin struct {
	numPreFilterCalled int
	failPreFilter      bool
	rejectPreFilter    bool
}

type ScorePlugin struct {
	failScore      bool
	numScoreCalled int32
	highScoreNode  string
}

type ScoreWithNormalizePlugin struct {
	numScoreCalled          int
	numNormalizeScoreCalled int
}

type FilterPlugin struct {
	numFilterCalled int
	failFilter      bool
	rejectFilter    bool
}

type PostFilterPlugin struct {
	fh                  framework.FrameworkHandle
	numPostFilterCalled int
	failPostFilter      bool
	rejectPostFilter    bool
}

type ReservePlugin struct {
	name                  string
	numReserveCalled      int
	failReserve           bool
	numUnreserveCalled    int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PreScorePlugin struct {
	numPreScoreCalled int
	failPreScore      bool
}

type PreBindPlugin struct {
	numPreBindCalled int
	failPreBind      bool
	rejectPreBind    bool
}

type BindPlugin struct {
	numBindCalled         int
	PluginName            string
	bindStatus            *framework.Status
	client                *clientset.Clientset
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PostBindPlugin struct {
	name                  string
	numPostBindCalled     int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PermitPlugin struct {
	name                string
	numPermitCalled     int
	failPermit          bool
	rejectPermit        bool
	timeoutPermit       bool
	waitAndRejectPermit bool
	waitAndAllowPermit  bool
	cancelled           bool
	waitingPod          string
	rejectingPod        string
	allowingPod         string
	fh                  framework.FrameworkHandle
}

const (
	prefilterPluginName          = "prefilter-plugin"
	postfilterPluginName         = "postfilter-plugin"
	scorePluginName              = "score-plugin"
	scoreWithNormalizePluginName = "score-with-normalize-plugin"
	filterPluginName             = "filter-plugin"
	preScorePluginName           = "prescore-plugin"
	reservePluginName            = "reserve-plugin"
	preBindPluginName            = "prebind-plugin"
	postBindPluginName           = "postbind-plugin"
	permitPluginName             = "permit-plugin"
)

var _ framework.PreFilterPlugin = &PreFilterPlugin{}
var _ framework.PostFilterPlugin = &PostFilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.FilterPlugin = &FilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.ScorePlugin = &ScoreWithNormalizePlugin{}
var _ framework.ReservePlugin = &ReservePlugin{}
var _ framework.PreScorePlugin = &PreScorePlugin{}
var _ framework.PreBindPlugin = &PreBindPlugin{}
var _ framework.BindPlugin = &BindPlugin{}
var _ framework.PostBindPlugin = &PostBindPlugin{}
var _ framework.PermitPlugin = &PermitPlugin{}

// newPlugin returns a plugin factory with specified Plugin.
func newPlugin(plugin framework.Plugin) frameworkruntime.PluginFactory {
	return func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
		return plugin, nil
	}
}

// newPlugin returns a plugin factory with specified Plugin.
func newPostFilterPlugin(plugin *PostFilterPlugin) frameworkruntime.PluginFactory {
	return func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
		plugin.fh = fh
		return plugin, nil
	}
}

// Name returns name of the score plugin.
func (sp *ScorePlugin) Name() string {
	return scorePluginName
}

// reset returns name of the score plugin.
func (sp *ScorePlugin) reset() {
	sp.failScore = false
	sp.numScoreCalled = 0
	sp.highScoreNode = ""
}

// Score returns the score of scheduling a pod on a specific node.
func (sp *ScorePlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	curCalled := atomic.AddInt32(&sp.numScoreCalled, 1)
	if sp.failScore {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", p.Name))
	}

	score := int64(1)
	if curCalled == 1 {
		// The first node is scored the highest, the rest is scored lower.
		sp.highScoreNode = nodeName
		score = framework.MaxNodeScore
	}
	return score, nil
}

func (sp *ScorePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Name returns name of the score plugin.
func (sp *ScoreWithNormalizePlugin) Name() string {
	return scoreWithNormalizePluginName
}

// reset returns name of the score plugin.
func (sp *ScoreWithNormalizePlugin) reset() {
	sp.numScoreCalled = 0
	sp.numNormalizeScoreCalled = 0
}

// Score returns the score of scheduling a pod on a specific node.
func (sp *ScoreWithNormalizePlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	sp.numScoreCalled++
	score := int64(10)
	return score, nil
}

func (sp *ScoreWithNormalizePlugin) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	sp.numNormalizeScoreCalled++
	return nil
}

func (sp *ScoreWithNormalizePlugin) ScoreExtensions() framework.ScoreExtensions {
	return sp
}

// Name returns name of the plugin.
func (fp *FilterPlugin) Name() string {
	return filterPluginName
}

// reset is used to reset filter plugin.
func (fp *FilterPlugin) reset() {
	fp.numFilterCalled = 0
	fp.failFilter = false
}

// Filter is a test function that returns an error or nil, depending on the
// value of "failFilter".
func (fp *FilterPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	fp.numFilterCalled++

	if fp.failFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if fp.rejectFilter {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}

	return nil
}

// Name returns name of the plugin.
func (rp *ReservePlugin) Name() string {
	return rp.name
}

// Reserve is a test function that increments an intenral counter and returns
// an error or nil, depending on the value of "failReserve".
func (rp *ReservePlugin) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	rp.numReserveCalled++
	if rp.failReserve {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	return nil
}

// Unreserve is a test function that increments an internal counter and emits
// an event to a channel. While Unreserve implementations should normally be
// idempotent, we relax that requirement here for testing purposes.
func (rp *ReservePlugin) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	rp.numUnreserveCalled++
	if rp.pluginInvokeEventChan != nil {
		rp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: rp.Name(), val: rp.numUnreserveCalled}
	}
}

// reset used to reset internal counters.
func (rp *ReservePlugin) reset() {
	rp.numReserveCalled = 0
	rp.numUnreserveCalled = 0
	rp.failReserve = false
}

// Name returns name of the plugin.
func (*PreScorePlugin) Name() string {
	return preScorePluginName
}

// PreScore is a test function.
func (pfp *PreScorePlugin) PreScore(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, _ []*v1.Node) *framework.Status {
	pfp.numPreScoreCalled++
	if pfp.failPreScore {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// reset used to reset prescore plugin.
func (pfp *PreScorePlugin) reset() {
	pfp.numPreScoreCalled = 0
	pfp.failPreScore = false
}

// Name returns name of the plugin.
func (pp *PreBindPlugin) Name() string {
	return preBindPluginName
}

// PreBind is a test function that returns (true, nil) or errors for testing.
func (pp *PreBindPlugin) PreBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	pp.numPreBindCalled++
	if pp.failPreBind {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPreBind {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil
}

// reset used to reset prebind plugin.
func (pp *PreBindPlugin) reset() {
	pp.numPreBindCalled = 0
	pp.failPreBind = false
	pp.rejectPreBind = false
}

const bindPluginAnnotation = "bindPluginName"

func (bp *BindPlugin) Name() string {
	return bp.PluginName
}

func (bp *BindPlugin) Bind(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) *framework.Status {
	bp.numBindCalled++
	if bp.pluginInvokeEventChan != nil {
		bp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: bp.Name(), val: bp.numBindCalled}
	}
	if bp.bindStatus.IsSuccess() {
		if err := bp.client.CoreV1().Pods(p.Namespace).Bind(context.TODO(), &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Namespace: p.Namespace, Name: p.Name, UID: p.UID, Annotations: map[string]string{bindPluginAnnotation: bp.Name()}},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: nodeName,
			},
		}, metav1.CreateOptions{}); err != nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("bind failed: %v", err))
		}
	}
	return bp.bindStatus
}

// reset used to reset numBindCalled.
func (bp *BindPlugin) reset() {
	bp.numBindCalled = 0
}

// Name returns name of the plugin.
func (pp *PostBindPlugin) Name() string {
	return pp.name
}

// PostBind is a test function, which counts the number of times called.
func (pp *PostBindPlugin) PostBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	pp.numPostBindCalled++
	if pp.pluginInvokeEventChan != nil {
		pp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: pp.Name(), val: pp.numPostBindCalled}
	}
}

// reset used to reset postbind plugin.
func (pp *PostBindPlugin) reset() {
	pp.numPostBindCalled = 0
}

// Name returns name of the plugin.
func (pp *PreFilterPlugin) Name() string {
	return prefilterPluginName
}

// Extensions returns the PreFilterExtensions interface.
func (pp *PreFilterPlugin) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// PreFilter is a test function that returns (true, nil) or errors for testing.
func (pp *PreFilterPlugin) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	pp.numPreFilterCalled++
	if pp.failPreFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPreFilter {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil
}

// reset used to reset prefilter plugin.
func (pp *PreFilterPlugin) reset() {
	pp.numPreFilterCalled = 0
	pp.failPreFilter = false
	pp.rejectPreFilter = false
}

// Name returns name of the plugin.
func (pp *PostFilterPlugin) Name() string {
	return postfilterPluginName
}

func (pp *PostFilterPlugin) PostFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, _ framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
	pp.numPostFilterCalled++
	nodeInfos, err := pp.fh.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, framework.NewStatus(framework.Error, err.Error())
	}
	ph := pp.fh.PreemptHandle()
	for _, nodeInfo := range nodeInfos {
		ph.RunFilterPlugins(ctx, state, pod, nodeInfo)
	}
	if pp.failPostFilter {
		return nil, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPostFilter {
		return nil, framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil, framework.NewStatus(framework.Success, fmt.Sprintf("make room for pod %v to be schedulable", pod.Name))
}

// Name returns name of the plugin.
func (pp *PermitPlugin) Name() string {
	return pp.name
}

// Permit implements the permit test plugin.
func (pp *PermitPlugin) Permit(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	pp.numPermitCalled++
	if pp.failPermit {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name)), 0
	}
	if pp.rejectPermit {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
	}
	if pp.timeoutPermit {
		go func() {
			select {
			case <-ctx.Done():
				pp.cancelled = true
			}
		}()
		return framework.NewStatus(framework.Wait, ""), 3 * time.Second
	}
	if pp.waitAndRejectPermit || pp.waitAndAllowPermit {
		if pp.waitingPod == "" || pp.waitingPod == pod.Name {
			pp.waitingPod = pod.Name
			return framework.NewStatus(framework.Wait, ""), 30 * time.Second
		}
		if pp.waitAndRejectPermit {
			pp.rejectingPod = pod.Name
			pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) {
				wp.Reject(fmt.Sprintf("reject pod %v", wp.GetPod().Name))
			})
			return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
		}
		if pp.waitAndAllowPermit {
			pp.allowingPod = pod.Name
			pp.allowAllPods()
			return nil, 0
		}
	}
	return nil, 0
}

// allowAllPods allows all waiting pods.
func (pp *PermitPlugin) allowAllPods() {
	pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { wp.Allow(pp.name) })
}

// rejectAllPods rejects all waiting pods.
func (pp *PermitPlugin) rejectAllPods() {
	pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { wp.Reject("rejectAllPods") })
}

// reset used to reset permit plugin.
func (pp *PermitPlugin) reset() {
	pp.numPermitCalled = 0
	pp.failPermit = false
	pp.rejectPermit = false
	pp.timeoutPermit = false
	pp.waitAndRejectPermit = false
	pp.waitAndAllowPermit = false
	pp.cancelled = false
	pp.waitingPod = ""
	pp.allowingPod = ""
	pp.rejectingPod = ""
}

// newPermitPlugin returns a factory for permit plugin with specified PermitPlugin.
func newPermitPlugin(permitPlugin *PermitPlugin) frameworkruntime.PluginFactory {
	return func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
		permitPlugin.fh = fh
		return permitPlugin, nil
	}
}

// TestPreFilterPlugin tests invocation of prefilter plugins.
func TestPreFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a pre-filter plugin.
	preFilterPlugin := &PreFilterPlugin{}
	registry := frameworkruntime.Registry{prefilterPluginName: newPlugin(preFilterPlugin)}

	// Setup initial prefilter plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			PreFilter: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{Name: prefilterPluginName},
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "prefilter-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name   string
		fail   bool
		reject bool
	}{
		{
			name:   "disable fail and reject flags",
			fail:   false,
			reject: false,
		},
		{
			name:   "enable fail and disable reject flags",
			fail:   true,
			reject: false,
		},
		{
			name:   "disable fail and enable reject flags",
			fail:   false,
			reject: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			preFilterPlugin.failPreFilter = test.fail
			preFilterPlugin.rejectPreFilter = test.reject
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.reject || test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if preFilterPlugin.numPreFilterCalled == 0 {
				t.Errorf("Expected the prefilter plugin to be called.")
			}

			preFilterPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPostFilterPlugin tests invocation of postfilter plugins.
func TestPostFilterPlugin(t *testing.T) {
	numNodes := 1
	tests := []struct {
		name                      string
		rejectFilter              bool
		rejectPostFilter          bool
		expectFilterNumCalled     int
		expectPostFilterNumCalled int
	}{
		{
			name:                      "Filter passed",
			rejectFilter:              false,
			rejectPostFilter:          false,
			expectFilterNumCalled:     numNodes,
			expectPostFilterNumCalled: 0,
		},
		{
			name:                      "Filter failed and PostFilter passed",
			rejectFilter:              true,
			rejectPostFilter:          false,
			expectFilterNumCalled:     numNodes * 2,
			expectPostFilterNumCalled: 1,
		},
		{
			name:                      "Filter failed and PostFilter failed",
			rejectFilter:              true,
			rejectPostFilter:          true,
			expectFilterNumCalled:     numNodes * 2,
			expectPostFilterNumCalled: 1,
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a plugin registry for testing. Register a combination of filter and postFilter plugin.
			var (
				filterPlugin     = &FilterPlugin{}
				postFilterPlugin = &PostFilterPlugin{}
			)
			filterPlugin.rejectFilter = tt.rejectFilter
			postFilterPlugin.rejectPostFilter = tt.rejectPostFilter
			registry := frameworkruntime.Registry{
				filterPluginName:     newPlugin(filterPlugin),
				postfilterPluginName: newPostFilterPlugin(postFilterPlugin),
			}

			// Setup plugins for testing.
			prof := schedulerconfig.KubeSchedulerProfile{
				SchedulerName: v1.DefaultSchedulerName,
				Plugins: &schedulerconfig.Plugins{
					Filter: &schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: filterPluginName},
						},
					},
					PostFilter: &schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: postfilterPluginName},
						},
						// Need to disable default in-tree PostFilter plugins, as they will
						// call RunFilterPlugins and hence impact the "numFilterCalled".
						Disabled: []schedulerconfig.Plugin{
							{Name: "*"},
						},
					},
				},
			}

			// Create the master and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				testutils.InitTestMaster(t, fmt.Sprintf("postfilter%v-", i), nil),
				numNodes,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
			)
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if tt.rejectFilter {
				if err = wait.Poll(10*time.Millisecond, 10*time.Second, podUnschedulable(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled.")
				}
				if filterPlugin.numFilterCalled < tt.expectFilterNumCalled {
					t.Errorf("Expected the filter plugin to be called at least %v times, but got %v.", tt.expectFilterNumCalled, filterPlugin.numFilterCalled)
				}
				if postFilterPlugin.numPostFilterCalled < tt.expectPostFilterNumCalled {
					t.Errorf("Expected the postfilter plugin to be called at least %v times, but got %v.", tt.expectPostFilterNumCalled, postFilterPlugin.numPostFilterCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				if filterPlugin.numFilterCalled != tt.expectFilterNumCalled {
					t.Errorf("Expected the filter plugin to be called %v times, but got %v.", tt.expectFilterNumCalled, filterPlugin.numFilterCalled)
				}
				if postFilterPlugin.numPostFilterCalled != tt.expectPostFilterNumCalled {
					t.Errorf("Expected the postfilter plugin to be called %v times, but got %v.", tt.expectPostFilterNumCalled, postFilterPlugin.numPostFilterCalled)
				}
			}
		})
	}
}

// TestScorePlugin tests invocation of score plugins.
func TestScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a score plugin.
	scorePlugin := &ScorePlugin{}
	registry := frameworkruntime.Registry{
		scorePluginName: newPlugin(scorePlugin),
	}

	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Score: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{Name: scorePluginName},
				},
			},
		},
	}

	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "score-plugin", nil), 10,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail score plugin",
			fail: true,
		},
		{
			name: "do not fail score plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorePlugin.failScore = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Fatalf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				} else {
					p, err := getPod(testCtx.ClientSet, pod.Name, pod.Namespace)
					if err != nil {
						t.Errorf("Failed to retrieve the pod. error: %v", err)
					} else if p.Spec.NodeName != scorePlugin.highScoreNode {
						t.Errorf("Expected the pod to be scheduled on node %q, got %q", scorePlugin.highScoreNode, p.Spec.NodeName)
					}
				}
			}

			if scorePlugin.numScoreCalled == 0 {
				t.Errorf("Expected the score plugin to be called.")
			}

			scorePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestNormalizeScorePlugin tests invocation of normalize score plugins.
func TestNormalizeScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a normalize score plugin.
	scoreWithNormalizePlugin := &ScoreWithNormalizePlugin{}
	registry := frameworkruntime.Registry{
		scoreWithNormalizePluginName: newPlugin(scoreWithNormalizePlugin),
	}

	// Setup initial score plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Score: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{Name: scoreWithNormalizePluginName},
				},
			},
		},
	}
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "score-plugin", nil), 10,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))

	defer testutils.CleanupTest(t, testCtx)

	// Create a best effort pod.
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Fatalf("Error while creating a test pod: %v", err)
	}

	if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if scoreWithNormalizePlugin.numScoreCalled == 0 {
		t.Errorf("Expected the score plugin to be called.")
	}
	if scoreWithNormalizePlugin.numNormalizeScoreCalled == 0 {
		t.Error("Expected the normalize score plugin to be called")
	}

	scoreWithNormalizePlugin.reset()
}

// TestReservePlugin tests invocation of reserve plugins.
func TestReservePluginReserve(t *testing.T) {
	// Create a plugin registry for testing. Register only a reserve plugin.
	reservePlugin := &ReservePlugin{}
	registry := frameworkruntime.Registry{reservePluginName: newPlugin(reservePlugin)}

	// Setup initial reserve plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Reserve: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{
						Name: reservePluginName,
					},
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "reserve-plugin-reserve", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail reserve plugin",
			fail: true,
		},
		{
			name: "do not fail reserve plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reservePlugin.failReserve = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second,
					podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if reservePlugin.numReserveCalled == 0 {
				t.Errorf("Expected the reserve plugin to be called.")
			}

			reservePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPrebindPlugin tests invocation of prebind plugins.
func TestPrebindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a prebind plugin.
	preBindPlugin := &PreBindPlugin{}
	registry := frameworkruntime.Registry{preBindPluginName: newPlugin(preBindPlugin)}

	// Setup initial prebind plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			PreBind: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{
						Name: preBindPluginName,
					},
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "prebind-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name   string
		fail   bool
		reject bool
	}{
		{
			name:   "disable fail and reject flags",
			fail:   false,
			reject: false,
		},
		{
			name:   "enable fail and disable reject flags",
			fail:   true,
			reject: false,
		},
		{
			name:   "disable fail and enable reject flags",
			fail:   false,
			reject: true,
		},
		{
			name:   "enable fail and reject flags",
			fail:   true,
			reject: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			preBindPlugin.failPreBind = test.fail
			preBindPlugin.rejectPreBind = test.reject
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail || test.reject {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
			} else if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}

			if preBindPlugin.numPreBindCalled == 0 {
				t.Errorf("Expected the prebind plugin to be called.")
			}

			preBindPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestUnreserveReservePlugin tests invocation of the Unreserve operation in
// reserve plugins through failures in execution points such as pre-bind. Also
// tests that the order of invocation of Unreserve operation is executed in the
// reverse order of invocation of the Reserve operation.
func TestReservePluginUnreserve(t *testing.T) {
	tests := []struct {
		name             string
		failReserve      bool
		failReserveIndex int
		failPreBind      bool
	}{
		{
			name:             "fail reserve",
			failReserve:      true,
			failReserveIndex: 1,
		},
		{
			name:        "fail preBind",
			failPreBind: true,
		},
		{
			name: "pass everything",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			numReservePlugins := 3
			pluginInvokeEventChan := make(chan pluginInvokeEvent, numReservePlugins)

			preBindPlugin := &PreBindPlugin{
				failPreBind: true,
			}
			var reservePlugins []*ReservePlugin
			for i := 0; i < numReservePlugins; i++ {
				reservePlugins = append(reservePlugins, &ReservePlugin{
					name:                  fmt.Sprintf("%s-%d", reservePluginName, i),
					pluginInvokeEventChan: pluginInvokeEventChan,
				})
			}

			registry := frameworkruntime.Registry{
				// TODO(#92229): test more failure points that would trigger Unreserve in
				// reserve plugins than just one pre-bind plugin.
				preBindPluginName: newPlugin(preBindPlugin),
			}
			for _, pl := range reservePlugins {
				registry[pl.Name()] = newPlugin(pl)
			}

			// Setup initial reserve and prebind plugin for testing.
			prof := schedulerconfig.KubeSchedulerProfile{
				SchedulerName: v1.DefaultSchedulerName,
				Plugins: &schedulerconfig.Plugins{
					Reserve: &schedulerconfig.PluginSet{
						// filled by looping over reservePlugins
					},
					PreBind: &schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{
								Name: preBindPluginName,
							},
						},
					},
				},
			}
			for _, pl := range reservePlugins {
				prof.Plugins.Reserve.Enabled = append(prof.Plugins.Reserve.Enabled, schedulerconfig.Plugin{
					Name: pl.Name(),
				})
			}

			// Create the master and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "reserve-plugin-unreserve", nil), 2,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			preBindPlugin.failPreBind = test.failPreBind
			if test.failReserve {
				reservePlugins[test.failReserveIndex].failReserve = true
			}
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.failPreBind || test.failReserve {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it: %v", err)
				}
				for i := numReservePlugins - 1; i >= 0; i-- {
					select {
					case event := <-pluginInvokeEventChan:
						expectedPluginName := reservePlugins[i].Name()
						if expectedPluginName != event.pluginName {
							t.Errorf("event.pluginName = %s, want %s", event.pluginName, expectedPluginName)
						}
					case <-time.After(time.Second * 30):
						t.Errorf("pluginInvokeEventChan receive timed out")
					}
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled, got an error: %v", err)
				}
				for i, pl := range reservePlugins {
					if pl.numUnreserveCalled != 0 {
						t.Errorf("reservePlugins[%d].numUnreserveCalled = %d, want 0", i, pl.numUnreserveCalled)
					}
				}
			}
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

type pluginInvokeEvent struct {
	pluginName string
	val        int
}

// TestBindPlugin tests invocation of bind plugins.
func TestBindPlugin(t *testing.T) {
	testContext := testutils.InitTestMaster(t, "bind-plugin", nil)
	bindPlugin1 := &BindPlugin{PluginName: "bind-plugin-1", client: testContext.ClientSet}
	bindPlugin2 := &BindPlugin{PluginName: "bind-plugin-2", client: testContext.ClientSet}
	reservePlugin := &ReservePlugin{name: "mock-reserve-plugin"}
	postBindPlugin := &PostBindPlugin{name: "mock-post-bind-plugin"}
	// Create a plugin registry for testing. Register reserve, bind, and
	// postBind plugins.
	registry := frameworkruntime.Registry{
		reservePlugin.Name(): func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return reservePlugin, nil
		},
		bindPlugin1.Name(): func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return bindPlugin1, nil
		},
		bindPlugin2.Name(): func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return bindPlugin2, nil
		},
		postBindPlugin.Name(): func(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return postBindPlugin, nil
		},
	}

	// Setup initial unreserve and bind plugins for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Reserve: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{{Name: reservePlugin.Name()}},
			},
			Bind: &schedulerconfig.PluginSet{
				// Put DefaultBinder last.
				Enabled:  []schedulerconfig.Plugin{{Name: bindPlugin1.Name()}, {Name: bindPlugin2.Name()}, {Name: defaultbinder.Name}},
				Disabled: []schedulerconfig.Plugin{{Name: defaultbinder.Name}},
			},
			PostBind: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{{Name: postBindPlugin.Name()}},
			},
		},
	}

	// Create the scheduler with the test plugin set.
	testCtx := testutils.InitTestSchedulerWithOptions(t, testContext, nil, time.Second,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	defer testutils.CleanupTest(t, testCtx)

	// Add a few nodes.
	_, err := createNodes(testCtx.ClientSet, "test-node", st.MakeNode(), 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	tests := []struct {
		name                   string
		bindPluginStatuses     []*framework.Status
		expectBoundByScheduler bool   // true means this test case expecting scheduler would bind pods
		expectBoundByPlugin    bool   // true means this test case expecting a plugin would bind pods
		expectBindPluginName   string // expecting plugin name to bind pods
		expectInvokeEvents     []pluginInvokeEvent
	}{
		{
			name:                   "bind plugins skipped to bind the pod and scheduler bond the pod",
			bindPluginStatuses:     []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Skip, "")},
			expectBoundByScheduler: true,
			expectInvokeEvents:     []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:                 "bindplugin2 succeeded to bind the pod",
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin2.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:                 "bindplugin1 succeeded to bind the pod",
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Success, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin1.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:               "bind plugin fails to bind the pod",
			bindPluginStatuses: []*framework.Status{framework.NewStatus(framework.Error, "failed to bind"), framework.NewStatus(framework.Success, "")},
			expectInvokeEvents: []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: reservePlugin.Name(), val: 1}, {pluginName: bindPlugin1.Name(), val: 2}, {pluginName: reservePlugin.Name(), val: 2}},
		},
	}

	var pluginInvokeEventChan chan pluginInvokeEvent
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginInvokeEventChan = make(chan pluginInvokeEvent, 10)

			bindPlugin1.bindStatus = test.bindPluginStatuses[0]
			bindPlugin2.bindStatus = test.bindPluginStatuses[1]

			bindPlugin1.pluginInvokeEventChan = pluginInvokeEventChan
			bindPlugin2.pluginInvokeEventChan = pluginInvokeEventChan
			reservePlugin.pluginInvokeEventChan = pluginInvokeEventChan
			postBindPlugin.pluginInvokeEventChan = pluginInvokeEventChan

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.expectBoundByScheduler || test.expectBoundByPlugin {
				// bind plugins skipped to bind the pod
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Fatalf("Expected the pod to be scheduled. error: %v", err)
				}
				pod, err = testCtx.ClientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
				if err != nil {
					t.Errorf("can't get pod: %v", err)
				}
				if test.expectBoundByScheduler {
					if pod.Annotations[bindPluginAnnotation] != "" {
						t.Errorf("Expected the pod to be bound by scheduler instead of by bindplugin %s", pod.Annotations[bindPluginAnnotation])
					}
					if bindPlugin1.numBindCalled != 1 || bindPlugin2.numBindCalled != 1 {
						t.Errorf("Expected each bind plugin to be called once, was called %d and %d times.", bindPlugin1.numBindCalled, bindPlugin2.numBindCalled)
					}
				} else {
					if pod.Annotations[bindPluginAnnotation] != test.expectBindPluginName {
						t.Errorf("Expected the pod to be bound by bindplugin %s instead of by bindplugin %s", test.expectBindPluginName, pod.Annotations[bindPluginAnnotation])
					}
					if bindPlugin1.numBindCalled != 1 {
						t.Errorf("Expected %s to be called once, was called %d times.", bindPlugin1.Name(), bindPlugin1.numBindCalled)
					}
					if test.expectBindPluginName == bindPlugin1.Name() && bindPlugin2.numBindCalled > 0 {
						// expect bindplugin1 succeeded to bind the pod and bindplugin2 should not be called.
						t.Errorf("Expected %s not to be called, was called %d times.", bindPlugin2.Name(), bindPlugin1.numBindCalled)
					}
				}
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, func() (done bool, err error) {
					return postBindPlugin.numPostBindCalled == 1, nil
				}); err != nil {
					t.Errorf("Expected the postbind plugin to be called once, was called %d times.", postBindPlugin.numPostBindCalled)
				}
				if reservePlugin.numUnreserveCalled != 0 {
					t.Errorf("Expected unreserve to not be called, was called %d times.", reservePlugin.numUnreserveCalled)
				}
			} else {
				// bind plugin fails to bind the pod
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
				if postBindPlugin.numPostBindCalled > 0 {
					t.Errorf("Didn't expect the postbind plugin to be called %d times.", postBindPlugin.numPostBindCalled)
				}
			}
			for j := range test.expectInvokeEvents {
				expectEvent := test.expectInvokeEvents[j]
				select {
				case event := <-pluginInvokeEventChan:
					if event.pluginName != expectEvent.pluginName {
						t.Errorf("Expect invoke event %d from plugin %s instead of %s", j, expectEvent.pluginName, event.pluginName)
					}
					if event.val != expectEvent.val {
						t.Errorf("Expect val of invoke event %d to be %d instead of %d", j, expectEvent.val, event.val)
					}
				case <-time.After(time.Second * 30):
					t.Errorf("Waiting for invoke event %d timeout.", j)
				}
			}
			postBindPlugin.reset()
			bindPlugin1.reset()
			bindPlugin2.reset()
			reservePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPostBindPlugin tests invocation of postbind plugins.
func TestPostBindPlugin(t *testing.T) {
	tests := []struct {
		name        string
		preBindFail bool
	}{
		{
			name:        "plugin preBind fail",
			preBindFail: true,
		},
		{
			name:        "plugin preBind do not fail",
			preBindFail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a plugin registry for testing. Register a prebind and a postbind plugin.
			preBindPlugin := &PreBindPlugin{
				failPreBind: test.preBindFail,
			}
			postBindPlugin := &PostBindPlugin{
				name:                  postBindPluginName,
				pluginInvokeEventChan: make(chan pluginInvokeEvent, 1),
			}
			registry := frameworkruntime.Registry{
				preBindPluginName:  newPlugin(preBindPlugin),
				postBindPluginName: newPlugin(postBindPlugin),
			}

			// Setup initial prebind and postbind plugin for testing.
			prof := schedulerconfig.KubeSchedulerProfile{
				SchedulerName: v1.DefaultSchedulerName,
				Plugins: &schedulerconfig.Plugins{
					PreBind: &schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{
								Name: preBindPluginName,
							},
						},
					},
					PostBind: &schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{
								Name: postBindPluginName,
							},
						},
					},
				},
			}

			// Create the master and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "postbind-plugin", nil), 2,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.preBindFail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
				if postBindPlugin.numPostBindCalled > 0 {
					t.Errorf("Didn't expect the postbind plugin to be called %d times.", postBindPlugin.numPostBindCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				select {
				case <-postBindPlugin.pluginInvokeEventChan:
				case <-time.After(time.Second * 15):
					t.Errorf("pluginInvokeEventChan timed out")
				}
				if postBindPlugin.numPostBindCalled == 0 {
					t.Errorf("Expected the postbind plugin to be called, was called %d times.", postBindPlugin.numPostBindCalled)
				}
			}

			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPermitPlugin tests invocation of permit plugins.
func TestPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{name: permitPluginName}
	registry, prof := initRegistryAndConfig(perPlugin)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name    string
		fail    bool
		reject  bool
		timeout bool
	}{
		{
			name:    "disable fail, reject and timeout flags",
			fail:    false,
			reject:  false,
			timeout: false,
		},
		{
			name:    "enable fail, disable reject and timeout flags",
			fail:    true,
			reject:  false,
			timeout: false,
		},
		{
			name:    "disable fail and timeout, enable reject flags",
			fail:    false,
			reject:  true,
			timeout: false,
		},
		{
			name:    "enable fail and reject, disable timeout flags",
			fail:    true,
			reject:  true,
			timeout: false,
		},
		{
			name:    "disable fail and reject, disable timeout flags",
			fail:    false,
			reject:  false,
			timeout: true,
		},
		{
			name:    "disable fail and reject, enable timeout flags",
			fail:    false,
			reject:  false,
			timeout: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			perPlugin.failPermit = test.fail
			perPlugin.rejectPermit = test.reject
			perPlugin.timeoutPermit = test.timeout
			perPlugin.waitAndRejectPermit = false
			perPlugin.waitAndAllowPermit = false

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}
			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
			} else {
				if test.reject || test.timeout {
					if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
						t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
					}
				} else {
					if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
						t.Errorf("Expected the pod to be scheduled. error: %v", err)
					}
				}
			}

			if perPlugin.numPermitCalled == 0 {
				t.Errorf("Expected the permit plugin to be called.")
			}

			perPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestMultiplePermitPlugins tests multiple permit plugins returning wait for a same pod.
func TestMultiplePermitPlugins(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, prof := initRegistryAndConfig(perPlugin1, perPlugin2)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "multi-permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&pausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Errorf("Error while creating a test pod: %v", err)
	}

	var waitingPod framework.WaitingPod
	// Wait until the test pod is actually waiting.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		waitingPod = perPlugin1.fh.GetWaitingPod(pod.UID)
		return waitingPod != nil, nil
	})

	// Check the number of pending permits
	if l := len(waitingPod.GetPendingPlugins()); l != 2 {
		t.Errorf("Expected the number of pending plugins is 2, but got %d", l)
	}

	perPlugin1.allowAllPods()
	// Check the number of pending permits
	if l := len(waitingPod.GetPendingPlugins()); l != 1 {
		t.Errorf("Expected the number of pending plugins is 1, but got %d", l)
	}

	perPlugin2.allowAllPods()
	if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if perPlugin1.numPermitCalled == 0 || perPlugin2.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
}

// TestPermitPluginsCancelled tests whether all permit plugins are cancelled when pod is rejected.
func TestPermitPluginsCancelled(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, prof := initRegistryAndConfig(perPlugin1, perPlugin2)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "permit-plugins", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&pausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Errorf("Error while creating a test pod: %v", err)
	}

	var waitingPod framework.WaitingPod
	// Wait until the test pod is actually waiting.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		waitingPod = perPlugin1.fh.GetWaitingPod(pod.UID)
		return waitingPod != nil, nil
	})

	perPlugin1.rejectAllPods()
	// Wait some time for the permit plugins to be cancelled
	err = wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		return perPlugin1.cancelled && perPlugin2.cancelled, nil
	})
	if err != nil {
		t.Errorf("Expected all permit plugins to be cancelled")
	}
}

// TestCoSchedulingWithPermitPlugin tests invocation of permit plugins.
func TestCoSchedulingWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	permitPlugin := &PermitPlugin{name: permitPluginName}
	registry, prof := initRegistryAndConfig(permitPlugin)

	// Create the master and the scheduler with the test plugin set.
	// TODO Make the subtests not share scheduler instances.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name       string
		waitReject bool
		waitAllow  bool
	}{
		{
			name:       "having wait reject true and wait allow false",
			waitReject: true,
			waitAllow:  false,
		},
		{
			name:       "having wait reject false and wait allow true",
			waitReject: false,
			waitAllow:  true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			permitPlugin.failPermit = false
			permitPlugin.rejectPermit = false
			permitPlugin.timeoutPermit = false
			permitPlugin.waitAndRejectPermit = test.waitReject
			permitPlugin.waitAndAllowPermit = test.waitAllow

			// Create two pods. First pod to enter Permit() will wait and a second one will either
			// reject or allow first one.
			podA, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "pod-a", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating the first pod: %v", err)
			}
			podB, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "pod-b", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating the second pod: %v", err)
			}

			if test.waitReject {
				if err = waitForPodUnschedulable(testCtx.ClientSet, podA); err != nil {
					t.Errorf("Didn't expect the first pod to be scheduled. error: %v", err)
				}
				if err = waitForPodUnschedulable(testCtx.ClientSet, podB); err != nil {
					t.Errorf("Didn't expect the second pod to be scheduled. error: %v", err)
				}
				if !((permitPlugin.waitingPod == podA.Name && permitPlugin.rejectingPod == podB.Name) ||
					(permitPlugin.waitingPod == podB.Name && permitPlugin.rejectingPod == podA.Name)) {
					t.Errorf("Expect one pod to wait and another pod to reject instead %s waited and %s rejected.",
						permitPlugin.waitingPod, permitPlugin.rejectingPod)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, podA); err != nil {
					t.Errorf("Expected the first pod to be scheduled. error: %v", err)
				}
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, podB); err != nil {
					t.Errorf("Expected the second pod to be scheduled. error: %v", err)
				}
				if !((permitPlugin.waitingPod == podA.Name && permitPlugin.allowingPod == podB.Name) ||
					(permitPlugin.waitingPod == podB.Name && permitPlugin.allowingPod == podA.Name)) {
					t.Errorf("Expect one pod to wait and another pod to allow instead %s waited and %s allowed.",
						permitPlugin.waitingPod, permitPlugin.allowingPod)
				}
			}

			if permitPlugin.numPermitCalled == 0 {
				t.Errorf("Expected the permit plugin to be called.")
			}

			permitPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{podA, podB})
		})
	}
}

// TestFilterPlugin tests invocation of filter plugins.
func TestFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a filter plugin.
	filterPlugin := &FilterPlugin{}
	registry := frameworkruntime.Registry{filterPluginName: newPlugin(filterPlugin)}

	// Setup initial filter plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			Filter: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{
						Name: filterPluginName,
					},
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "filter-plugin", nil), 1,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail filter plugin",
			fail: true,
		},
		{
			name: "do not fail filter plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filterPlugin.failFilter = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podUnschedulable(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled.")
				}
				if filterPlugin.numFilterCalled < 1 {
					t.Errorf("Expected the filter plugin to be called at least 1 time, but got %v.", filterPlugin.numFilterCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				if filterPlugin.numFilterCalled != 1 {
					t.Errorf("Expected the filter plugin to be called 1 time, but got %v.", filterPlugin.numFilterCalled)
				}
			}

			filterPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPreScorePlugin tests invocation of pre-score plugins.
func TestPreScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a pre-score plugin.
	preScorePlugin := &PreScorePlugin{}
	registry := frameworkruntime.Registry{preScorePluginName: newPlugin(preScorePlugin)}

	// Setup initial pre-score plugin for testing.
	prof := schedulerconfig.KubeSchedulerProfile{
		SchedulerName: v1.DefaultSchedulerName,
		Plugins: &schedulerconfig.Plugins{
			PreScore: &schedulerconfig.PluginSet{
				Enabled: []schedulerconfig.Plugin{
					{
						Name: preScorePluginName,
					},
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "pre-score-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail preScore plugin",
			fail: true,
		},
		{
			name: "do not fail preScore plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			preScorePlugin.failPreScore = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&pausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if preScorePlugin.numPreScoreCalled == 0 {
				t.Errorf("Expected the pre-score plugin to be called.")
			}

			preScorePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPreemptWithPermitPlugin tests preempt with permit plugins.
func TestPreemptWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	permitPlugin := &PermitPlugin{}
	registry, prof := initRegistryAndConfig(permitPlugin)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestMaster(t, "preempt-with-permit-plugin", nil), 0,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Add one node.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createNodes(testCtx.ClientSet, "test-node", st.MakeNode().Capacity(nodeRes), 1)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	permitPlugin.failPermit = false
	permitPlugin.rejectPermit = false
	permitPlugin.timeoutPermit = false
	permitPlugin.waitAndRejectPermit = false
	permitPlugin.waitAndAllowPermit = true

	lowPriority, highPriority := int32(100), int32(300)
	resourceRequest := v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI)},
	}

	// First pod will go waiting.
	waitingPod := initPausePod(&pausePodConfig{Name: "waiting-pod", Namespace: testCtx.NS.Name, Priority: &lowPriority, Resources: &resourceRequest})
	waitingPod.Spec.TerminationGracePeriodSeconds = new(int64)
	waitingPod, err = createPausePod(testCtx.ClientSet, waitingPod)
	if err != nil {
		t.Errorf("Error while creating the waiting pod: %v", err)
	}
	// Wait until the waiting-pod is actually waiting, then create a preemptor pod to preempt it.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		w := false
		permitPlugin.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
		return w, nil
	})

	// Create second pod which should preempt first pod.
	preemptorPod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&pausePodConfig{Name: "preemptor-pod", Namespace: testCtx.NS.Name, Priority: &highPriority, Resources: &resourceRequest}))
	if err != nil {
		t.Errorf("Error while creating the preemptor pod: %v", err)
	}

	if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, preemptorPod); err != nil {
		t.Errorf("Expected the preemptor pod to be scheduled. error: %v", err)
	}

	if _, err := getPod(testCtx.ClientSet, waitingPod.Name, waitingPod.Namespace); err == nil {
		t.Error("Expected the waiting pod to get preempted and deleted")
	}

	if permitPlugin.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	permitPlugin.reset()
	testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{waitingPod, preemptorPod})
}

func initTestSchedulerForFrameworkTest(t *testing.T, testCtx *testutils.TestContext, nodeCount int, opts ...scheduler.Option) *testutils.TestContext {
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, nil, time.Second, opts...)
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	if nodeCount > 0 {
		_, err := createNodes(testCtx.ClientSet, "test-node", st.MakeNode(), nodeCount)
		if err != nil {
			t.Fatalf("Cannot create nodes: %v", err)
		}
	}
	return testCtx
}

// initRegistryAndConfig returns registry and plugins config based on give plugins.
// TODO: refactor it to a more generic functions that accepts all kinds of Plugins as arguments
func initRegistryAndConfig(pp ...*PermitPlugin) (registry frameworkruntime.Registry, prof schedulerconfig.KubeSchedulerProfile) {
	if len(pp) == 0 {
		return
	}

	registry = frameworkruntime.Registry{}
	var plugins []schedulerconfig.Plugin
	for _, p := range pp {
		registry.Register(p.Name(), newPermitPlugin(p))
		plugins = append(plugins, schedulerconfig.Plugin{Name: p.Name()})
	}

	prof.SchedulerName = v1.DefaultSchedulerName
	prof.Plugins = &schedulerconfig.Plugins{
		Permit: &schedulerconfig.PluginSet{
			Enabled: plugins,
		},
	}
	return
}
