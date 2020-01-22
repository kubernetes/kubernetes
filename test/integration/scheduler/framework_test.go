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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	scheduler "k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

type PreFilterPlugin struct {
	numPreFilterCalled int
	failPreFilter      bool
	rejectPreFilter    bool
}

type ScorePlugin struct {
	failScore      bool
	numScoreCalled int
	highScoreNode  string
}

type ScoreWithNormalizePlugin struct {
	numScoreCalled          int
	numNormalizeScoreCalled int
}

type FilterPlugin struct {
	numFilterCalled int
	failFilter      bool
}

type ReservePlugin struct {
	numReserveCalled int
	failReserve      bool
}

type PostFilterPlugin struct {
	numPostFilterCalled int
	failPostFilter      bool
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

type UnreservePlugin struct {
	name                  string
	numUnreserveCalled    int
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
	allowPermit         bool
	cancelled           bool
	fh                  framework.FrameworkHandle
}

const (
	prefilterPluginName          = "prefilter-plugin"
	scorePluginName              = "score-plugin"
	scoreWithNormalizePluginName = "score-with-normalize-plugin"
	filterPluginName             = "filter-plugin"
	postFilterPluginName         = "postfilter-plugin"
	reservePluginName            = "reserve-plugin"
	preBindPluginName            = "prebind-plugin"
	unreservePluginName          = "unreserve-plugin"
	postBindPluginName           = "postbind-plugin"
	permitPluginName             = "permit-plugin"
)

var _ framework.PreFilterPlugin = &PreFilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.FilterPlugin = &FilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.ScorePlugin = &ScoreWithNormalizePlugin{}
var _ framework.ReservePlugin = &ReservePlugin{}
var _ framework.PostFilterPlugin = &PostFilterPlugin{}
var _ framework.PreBindPlugin = &PreBindPlugin{}
var _ framework.BindPlugin = &BindPlugin{}
var _ framework.PostBindPlugin = &PostBindPlugin{}
var _ framework.UnreservePlugin = &UnreservePlugin{}
var _ framework.PermitPlugin = &PermitPlugin{}

// newPlugin returns a plugin factory with specified Plugin.
func newPlugin(plugin framework.Plugin) framework.PluginFactory {
	return func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
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
	sp.numScoreCalled++
	if sp.failScore {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", p.Name))
	}

	score := int64(1)
	if sp.numScoreCalled == 1 {
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
func (fp *FilterPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	fp.numFilterCalled++

	if fp.failFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// Name returns name of the plugin.
func (rp *ReservePlugin) Name() string {
	return reservePluginName
}

// Reserve is a test function that returns an error or nil, depending on the
// value of "failReserve".
func (rp *ReservePlugin) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	rp.numReserveCalled++
	if rp.failReserve {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	return nil
}

// reset used to reset reserve plugin.
func (rp *ReservePlugin) reset() {
	rp.numReserveCalled = 0
}

// Name returns name of the plugin.
func (*PostFilterPlugin) Name() string {
	return postFilterPluginName
}

// PostFilter is a test function.
func (pfp *PostFilterPlugin) PostFilter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, _ []*v1.Node, _ framework.NodeToStatusMap) *framework.Status {
	pfp.numPostFilterCalled++
	if pfp.failPostFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// reset used to reset postfilter plugin.
func (pfp *PostFilterPlugin) reset() {
	pfp.numPostFilterCalled = 0
	pfp.failPostFilter = false
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
		if err := bp.client.CoreV1().Pods(p.Namespace).Bind(&v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Namespace: p.Namespace, Name: p.Name, UID: p.UID, Annotations: map[string]string{bindPluginAnnotation: bp.Name()}},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: nodeName,
			},
		}); err != nil {
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
func (up *UnreservePlugin) Name() string {
	return up.name
}

// Unreserve is a test function that returns an error or nil, depending on the
// value of "failUnreserve".
func (up *UnreservePlugin) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	up.numUnreserveCalled++
	if up.pluginInvokeEventChan != nil {
		up.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: up.Name(), val: up.numUnreserveCalled}
	}
}

// reset used to reset numUnreserveCalled.
func (up *UnreservePlugin) reset() {
	up.numUnreserveCalled = 0
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

	if pp.allowPermit && pod.Name != "waiting-pod" {
		return nil, 0
	}
	if pp.waitAndRejectPermit || pp.waitAndAllowPermit {
		if pod.Name == "waiting-pod" {
			return framework.NewStatus(framework.Wait, ""), 30 * time.Second
		}
		// This is the signalling pod, wait until the waiting-pod is actually waiting and then either reject or allow it.
		wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
			w := false
			pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
			return w, nil
		})
		if pp.waitAndRejectPermit {
			pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) {
				wp.Reject(fmt.Sprintf("reject pod %v", wp.GetPod().Name))
			})
			return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
		}
		if pp.waitAndAllowPermit {
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
	pp.allowPermit = false
	pp.cancelled = false
}

// newPermitPlugin returns a factory for permit plugin with specified PermitPlugin.
func newPermitPlugin(permitPlugin *PermitPlugin) framework.PluginFactory {
	return func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
		permitPlugin.fh = fh
		return permitPlugin, nil
	}
}

// TestPreFilterPlugin tests invocation of prefilter plugins.
func TestPreFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a pre-filter plugin.
	preFilterPlugin := &PreFilterPlugin{}
	registry := framework.Registry{prefilterPluginName: newPlugin(preFilterPlugin)}

	// Setup initial prefilter plugin for testing.
	plugins := &schedulerconfig.Plugins{
		PreFilter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: prefilterPluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "prefilter-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		fail   bool
		reject bool
	}{
		{
			fail:   false,
			reject: false,
		},
		{
			fail:   true,
			reject: false,
		},
		{
			fail:   false,
			reject: true,
		},
	}

	for i, test := range tests {
		preFilterPlugin.failPreFilter = test.fail
		preFilterPlugin.rejectPreFilter = test.reject
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.reject || test.fail {
			if err = waitForPodUnschedulable(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
			}
		}

		if preFilterPlugin.numPreFilterCalled == 0 {
			t.Errorf("Expected the prefilter plugin to be called.")
		}

		preFilterPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestScorePlugin tests invocation of score plugins.
func TestScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a score plugin.
	scorePlugin := &ScorePlugin{}
	registry := framework.Registry{
		scorePluginName: newPlugin(scorePlugin),
	}

	// Setup initial score plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Score: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: scorePluginName,
				},
			},
		},
	}

	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "score-plugin", nil), 10,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	for i, fail := range []bool{false, true} {
		scorePlugin.failScore = fail
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Fatalf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = waitForPodUnschedulable(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			} else {
				p, err := getPod(testCtx.clientSet, pod.Name, pod.Namespace)
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
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestNormalizeScorePlugin tests invocation of normalize score plugins.
func TestNormalizeScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a normalize score plugin.
	scoreWithNormalizePlugin := &ScoreWithNormalizePlugin{}
	registry := framework.Registry{
		scoreWithNormalizePluginName: newPlugin(scoreWithNormalizePlugin),
	}

	// Setup initial score plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Score: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: scoreWithNormalizePluginName,
				},
			},
		},
	}
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "score-plugin", nil), 10,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))

	defer cleanupTest(t, testCtx)

	// Create a best effort pod.
	pod, err := createPausePod(testCtx.clientSet,
		initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
	if err != nil {
		t.Fatalf("Error while creating a test pod: %v", err)
	}

	if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
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
func TestReservePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a reserve plugin.
	reservePlugin := &ReservePlugin{}
	registry := framework.Registry{reservePluginName: newPlugin(reservePlugin)}

	// Setup initial reserve plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Reserve: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: reservePluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "reserve-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	for _, fail := range []bool{false, true} {
		reservePlugin.failReserve = fail
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second,
				podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if reservePlugin.numReserveCalled == 0 {
			t.Errorf("Expected the reserve plugin to be called.")
		}

		reservePlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestPrebindPlugin tests invocation of prebind plugins.
func TestPrebindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a prebind plugin.
	preBindPlugin := &PreBindPlugin{}
	registry := framework.Registry{preBindPluginName: newPlugin(preBindPlugin)}

	// Setup initial prebind plugin for testing.
	plugins := &schedulerconfig.Plugins{
		PreBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: preBindPluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "prebind-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		fail   bool
		reject bool
	}{
		{
			fail:   false,
			reject: false,
		},
		{
			fail:   true,
			reject: false,
		},
		{
			fail:   false,
			reject: true,
		},
		{
			fail:   true,
			reject: true,
		},
	}

	for i, test := range tests {
		preBindPlugin.failPreBind = test.fail
		preBindPlugin.rejectPreBind = test.reject
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.fail || test.reject {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
		} else if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
			t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
		}

		if preBindPlugin.numPreBindCalled == 0 {
			t.Errorf("Expected the prebind plugin to be called.")
		}

		preBindPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestUnreservePlugin tests invocation of un-reserve plugin
func TestUnreservePlugin(t *testing.T) {
	// TODO: register more plugin which would trigger un-reserve plugin
	preBindPlugin := &PreBindPlugin{}
	unreservePlugin := &UnreservePlugin{name: unreservePluginName}
	registry := framework.Registry{
		unreservePluginName: newPlugin(unreservePlugin),
		preBindPluginName:   newPlugin(preBindPlugin),
	}

	// Setup initial unreserve and prebind plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Unreserve: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: unreservePluginName,
				},
			},
		},
		PreBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: preBindPluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "unreserve-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		preBindFail bool
	}{
		{
			preBindFail: false,
		},
		{
			preBindFail: true,
		},
	}

	for i, test := range tests {
		preBindPlugin.failPreBind = test.preBindFail

		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.preBindFail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if unreservePlugin.numUnreserveCalled == 0 || unreservePlugin.numUnreserveCalled != preBindPlugin.numPreBindCalled {
				t.Errorf("test #%v: Expected the unreserve plugin to be called %d times, was called %d times.", i, preBindPlugin.numPreBindCalled, unreservePlugin.numUnreserveCalled)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
			}
			if unreservePlugin.numUnreserveCalled > 0 {
				t.Errorf("test #%v: Didn't expected the unreserve plugin to be called, was called %d times.", i, unreservePlugin.numUnreserveCalled)
			}
		}

		unreservePlugin.reset()
		preBindPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

type pluginInvokeEvent struct {
	pluginName string
	val        int
}

// TestBindPlugin tests invocation of bind plugins.
func TestBindPlugin(t *testing.T) {
	testContext := initTestMaster(t, "bind-plugin", nil)
	bindPlugin1 := &BindPlugin{PluginName: "bind-plugin-1", client: testContext.clientSet}
	bindPlugin2 := &BindPlugin{PluginName: "bind-plugin-2", client: testContext.clientSet}
	unreservePlugin := &UnreservePlugin{name: "mock-unreserve-plugin"}
	postBindPlugin := &PostBindPlugin{name: "mock-post-bind-plugin"}
	// Create a plugin registry for testing. Register an unreserve, a bind plugin and a postBind plugin.
	registry := framework.Registry{
		unreservePlugin.Name(): func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return unreservePlugin, nil
		},
		bindPlugin1.Name(): func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return bindPlugin1, nil
		},
		bindPlugin2.Name(): func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return bindPlugin2, nil
		},
		postBindPlugin.Name(): func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return postBindPlugin, nil
		},
	}

	// Setup initial unreserve and bind plugins for testing.
	plugins := &schedulerconfig.Plugins{
		Unreserve: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{{Name: unreservePlugin.Name()}},
		},
		Bind: &schedulerconfig.PluginSet{
			// Put DefaultBinder last.
			Enabled:  []schedulerconfig.Plugin{{Name: bindPlugin1.Name()}, {Name: bindPlugin2.Name()}, {Name: defaultbinder.Name}},
			Disabled: []schedulerconfig.Plugin{{Name: defaultbinder.Name}},
		},
		PostBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{{Name: postBindPlugin.Name()}},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerWithOptions(t, testContext, false, nil, time.Second,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	// Add a few nodes.
	_, err := createNodes(testCtx.clientSet, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	tests := []struct {
		bindPluginStatuses     []*framework.Status
		expectBoundByScheduler bool   // true means this test case expecting scheduler would bind pods
		expectBoundByPlugin    bool   // true means this test case expecting a plugin would bind pods
		expectBindPluginName   string // expecting plugin name to bind pods
		expectInvokeEvents     []pluginInvokeEvent
	}{
		// bind plugins skipped to bind the pod and scheduler bond the pod
		{
			bindPluginStatuses:     []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Skip, "")},
			expectBoundByScheduler: true,
			expectInvokeEvents:     []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		// bindplugin2 succeeded to bind the pod
		{
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin2.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		// bindplugin1 succeeded to bind the pod
		{
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Success, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin1.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		// bind plugin fails to bind the pod
		{
			bindPluginStatuses: []*framework.Status{framework.NewStatus(framework.Error, "failed to bind"), framework.NewStatus(framework.Success, "")},
			expectInvokeEvents: []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: unreservePlugin.Name(), val: 1}, {pluginName: bindPlugin1.Name(), val: 2}, {pluginName: unreservePlugin.Name(), val: 2}},
		},
	}

	var pluginInvokeEventChan chan pluginInvokeEvent
	for i, test := range tests {
		bindPlugin1.bindStatus = test.bindPluginStatuses[0]
		bindPlugin2.bindStatus = test.bindPluginStatuses[1]

		pluginInvokeEventChan = make(chan pluginInvokeEvent, 10)
		bindPlugin1.pluginInvokeEventChan = pluginInvokeEventChan
		bindPlugin2.pluginInvokeEventChan = pluginInvokeEventChan
		unreservePlugin.pluginInvokeEventChan = pluginInvokeEventChan
		postBindPlugin.pluginInvokeEventChan = pluginInvokeEventChan

		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.expectBoundByScheduler || test.expectBoundByPlugin {
			// bind plugins skipped to bind the pod
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				continue
			}
			pod, err = testCtx.clientSet.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("can't get pod: %v", err)
			}
			if test.expectBoundByScheduler {
				if pod.Annotations[bindPluginAnnotation] != "" {
					t.Errorf("test #%v: Expected the pod to be bound by scheduler instead of by bindplugin %s", i, pod.Annotations[bindPluginAnnotation])
				}
				if bindPlugin1.numBindCalled != 1 || bindPlugin2.numBindCalled != 1 {
					t.Errorf("test #%v: Expected each bind plugin to be called once, was called %d and %d times.", i, bindPlugin1.numBindCalled, bindPlugin2.numBindCalled)
				}
			} else {
				if pod.Annotations[bindPluginAnnotation] != test.expectBindPluginName {
					t.Errorf("test #%v: Expected the pod to be bound by bindplugin %s instead of by bindplugin %s", i, test.expectBindPluginName, pod.Annotations[bindPluginAnnotation])
				}
				if bindPlugin1.numBindCalled != 1 {
					t.Errorf("test #%v: Expected %s to be called once, was called %d times.", i, bindPlugin1.Name(), bindPlugin1.numBindCalled)
				}
				if test.expectBindPluginName == bindPlugin1.Name() && bindPlugin2.numBindCalled > 0 {
					// expect bindplugin1 succeeded to bind the pod and bindplugin2 should not be called.
					t.Errorf("test #%v: Expected %s not to be called, was called %d times.", i, bindPlugin2.Name(), bindPlugin1.numBindCalled)
				}
			}
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, func() (done bool, err error) {
				return postBindPlugin.numPostBindCalled == 1, nil
			}); err != nil {
				t.Errorf("test #%v: Expected the postbind plugin to be called once, was called %d times.", i, postBindPlugin.numPostBindCalled)
			}
			if unreservePlugin.numUnreserveCalled != 0 {
				t.Errorf("test #%v: Expected the unreserve plugin not to be called, was called %d times.", i, unreservePlugin.numUnreserveCalled)
			}
		} else {
			// bind plugin fails to bind the pod
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if postBindPlugin.numPostBindCalled > 0 {
				t.Errorf("test #%v: Didn't expected the postbind plugin to be called %d times.", i, postBindPlugin.numPostBindCalled)
			}
		}
		for j := range test.expectInvokeEvents {
			expectEvent := test.expectInvokeEvents[j]
			select {
			case event := <-pluginInvokeEventChan:
				if event.pluginName != expectEvent.pluginName {
					t.Errorf("test #%v: Expect invoke event %d from plugin %s instead of %s", i, j, expectEvent.pluginName, event.pluginName)
				}
				if event.val != expectEvent.val {
					t.Errorf("test #%v: Expect val of invoke event %d to be %d instead of %d", i, j, expectEvent.val, event.val)
				}
			case <-time.After(time.Second * 30):
				t.Errorf("test #%v: Waiting for invoke event %d timeout.", i, j)
			}
		}
		postBindPlugin.reset()
		bindPlugin1.reset()
		bindPlugin2.reset()
		unreservePlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestPostBindPlugin tests invocation of postbind plugins.
func TestPostBindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register a prebind and a postbind plugin.
	preBindPlugin := &PreBindPlugin{}
	postBindPlugin := &PostBindPlugin{name: postBindPluginName}
	registry := framework.Registry{
		preBindPluginName:  newPlugin(preBindPlugin),
		postBindPluginName: newPlugin(postBindPlugin),
	}

	// Setup initial prebind and postbind plugin for testing.
	plugins := &schedulerconfig.Plugins{
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
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "postbind-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		preBindFail   bool
		preBindReject bool
	}{
		{
			preBindFail: false,
		},
		{
			preBindFail: true,
		},
	}

	for i, test := range tests {
		preBindPlugin.failPreBind = test.preBindFail

		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.preBindFail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if postBindPlugin.numPostBindCalled > 0 {
				t.Errorf("test #%v: Didn't expected the postbind plugin to be called %d times.", i, postBindPlugin.numPostBindCalled)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
			}
			if postBindPlugin.numPostBindCalled == 0 {
				t.Errorf("test #%v: Expected the postbind plugin to be called, was called %d times.", i, postBindPlugin.numPostBindCalled)
			}
		}

		postBindPlugin.reset()
		preBindPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestPermitPlugin tests invocation of permit plugins.
func TestPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{name: permitPluginName}
	registry, plugins := initRegistryAndConfig(perPlugin)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "permit-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		fail    bool
		reject  bool
		timeout bool
	}{
		{
			fail:    false,
			reject:  false,
			timeout: false,
		},
		{
			fail:    true,
			reject:  false,
			timeout: false,
		},
		{
			fail:    false,
			reject:  true,
			timeout: false,
		},
		{
			fail:    true,
			reject:  true,
			timeout: false,
		},
		{
			fail:    false,
			reject:  false,
			timeout: true,
		},
		{
			fail:    false,
			reject:  false,
			timeout: true,
		},
	}

	for i, test := range tests {
		perPlugin.failPermit = test.fail
		perPlugin.rejectPermit = test.reject
		perPlugin.timeoutPermit = test.timeout
		perPlugin.waitAndRejectPermit = false
		perPlugin.waitAndAllowPermit = false

		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}
		if test.fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
		} else {
			if test.reject || test.timeout {
				if err = waitForPodUnschedulable(testCtx.clientSet, pod); err != nil {
					t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
				}
			} else {
				if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
					t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				}
			}
		}

		if perPlugin.numPermitCalled == 0 {
			t.Errorf("Expected the permit plugin to be called.")
		}

		perPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestMultiplePermitPlugins tests multiple permit plugins returning wait for a same pod.
func TestMultiplePermitPlugins(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, plugins := initRegistryAndConfig(perPlugin1, perPlugin2)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "multi-permit-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.clientSet,
		initPausePod(testCtx.clientSet, &pausePodConfig{Name: podName, Namespace: testCtx.ns.Name}))
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
	if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if perPlugin1.numPermitCalled == 0 || perPlugin2.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
}

// TestPermitPluginsCancelled tests whether all permit plugins are cancelled when pod is rejected.
func TestPermitPluginsCancelled(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, plugins := initRegistryAndConfig(perPlugin1, perPlugin2)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "permit-plugins", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.clientSet,
		initPausePod(testCtx.clientSet, &pausePodConfig{Name: podName, Namespace: testCtx.ns.Name}))
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
	registry, plugins := initRegistryAndConfig(permitPlugin)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "permit-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	tests := []struct {
		waitReject bool
		waitAllow  bool
	}{
		{
			waitReject: true,
			waitAllow:  false,
		},
		{
			waitReject: false,
			waitAllow:  true,
		},
	}

	for i, test := range tests {
		permitPlugin.failPermit = false
		permitPlugin.rejectPermit = false
		permitPlugin.timeoutPermit = false
		permitPlugin.waitAndRejectPermit = test.waitReject
		permitPlugin.waitAndAllowPermit = test.waitAllow

		// Create two pods.
		waitingPod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "waiting-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating the waiting pod: %v", err)
		}
		signallingPod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "signalling-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating the signalling pod: %v", err)
		}

		if test.waitReject {
			if err = waitForPodUnschedulable(testCtx.clientSet, waitingPod); err != nil {
				t.Errorf("test #%v: Didn't expect the waiting pod to be scheduled. error: %v", i, err)
			}
			if err = waitForPodUnschedulable(testCtx.clientSet, signallingPod); err != nil {
				t.Errorf("test #%v: Didn't expect the signalling pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, waitingPod); err != nil {
				t.Errorf("test #%v: Expected the waiting pod to be scheduled. error: %v", i, err)
			}
			if err = waitForPodToSchedule(testCtx.clientSet, signallingPod); err != nil {
				t.Errorf("test #%v: Expected the signalling pod to be scheduled. error: %v", i, err)
			}
		}

		if permitPlugin.numPermitCalled == 0 {
			t.Errorf("Expected the permit plugin to be called.")
		}

		permitPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{waitingPod, signallingPod})
	}
}

// TestFilterPlugin tests invocation of filter plugins.
func TestFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a filter plugin.
	filterPlugin := &FilterPlugin{}
	registry := framework.Registry{filterPluginName: newPlugin(filterPlugin)}

	// Setup initial filter plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Filter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: filterPluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "filter-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	for _, fail := range []bool{false, true} {
		filterPlugin.failFilter = fail
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podUnschedulable(testCtx.clientSet, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled.")
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if filterPlugin.numFilterCalled == 0 {
			t.Errorf("Expected the filter plugin to be called.")
		}

		filterPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestPostFilterPlugin tests invocation of post-filter plugins.
func TestPostFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a post-filter plugin.
	postFilterPlugin := &PostFilterPlugin{}
	registry := framework.Registry{postFilterPluginName: newPlugin(postFilterPlugin)}

	// Setup initial post-filter plugin for testing.
	plugins := &schedulerconfig.Plugins{
		PostFilter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: postFilterPluginName,
				},
			},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "post-filter-plugin", nil), 2,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	for _, fail := range []bool{false, true} {
		postFilterPlugin.failPostFilter = fail
		// Create a best effort pod.
		pod, err := createPausePod(testCtx.clientSet,
			initPausePod(testCtx.clientSet, &pausePodConfig{Name: "test-pod", Namespace: testCtx.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = waitForPodUnschedulable(testCtx.clientSet, pod); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
			}
		} else {
			if err = waitForPodToSchedule(testCtx.clientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if postFilterPlugin.numPostFilterCalled == 0 {
			t.Errorf("Expected the post-filter plugin to be called.")
		}

		postFilterPlugin.reset()
		cleanupPods(testCtx.clientSet, t, []*v1.Pod{pod})
	}
}

// TestPreemptWithPermitPlugin tests preempt with permit plugins.
func TestPreemptWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	permitPlugin := &PermitPlugin{}
	registry, plugins := initRegistryAndConfig(permitPlugin)

	// Create the master and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, initTestMaster(t, "preempt-with-permit-plugin", nil), 0,
		scheduler.WithFrameworkPlugins(plugins),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer cleanupTest(t, testCtx)

	// Add one node.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNodes(testCtx.clientSet, "test-node", nodeRes, 1)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	permitPlugin.failPermit = false
	permitPlugin.rejectPermit = false
	permitPlugin.timeoutPermit = false
	permitPlugin.waitAndRejectPermit = false
	permitPlugin.waitAndAllowPermit = true
	permitPlugin.allowPermit = true

	lowPriority, highPriority := int32(100), int32(300)
	resourceRequest := v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI)},
	}

	// Create two pods.
	waitingPod := initPausePod(testCtx.clientSet, &pausePodConfig{Name: "waiting-pod", Namespace: testCtx.ns.Name, Priority: &lowPriority, Resources: &resourceRequest})
	waitingPod.Spec.TerminationGracePeriodSeconds = new(int64)
	waitingPod, err = createPausePod(testCtx.clientSet, waitingPod)
	if err != nil {
		t.Errorf("Error while creating the waiting pod: %v", err)
	}
	// Wait until the waiting-pod is actually waiting, then create a preemptor pod to preempt it.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		w := false
		permitPlugin.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
		return w, nil
	})

	preemptorPod, err := createPausePod(testCtx.clientSet,
		initPausePod(testCtx.clientSet, &pausePodConfig{Name: "preemptor-pod", Namespace: testCtx.ns.Name, Priority: &highPriority, Resources: &resourceRequest}))
	if err != nil {
		t.Errorf("Error while creating the preemptor pod: %v", err)
	}

	if err = waitForPodToSchedule(testCtx.clientSet, preemptorPod); err != nil {
		t.Errorf("Expected the preemptor pod to be scheduled. error: %v", err)
	}

	if _, err := getPod(testCtx.clientSet, waitingPod.Name, waitingPod.Namespace); err == nil {
		t.Error("Expected the waiting pod to get preempted and deleted")
	}

	if permitPlugin.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	permitPlugin.reset()
	cleanupPods(testCtx.clientSet, t, []*v1.Pod{waitingPod, preemptorPod})
}

func initTestSchedulerForFrameworkTest(t *testing.T, testCtx *testContext, nodeCount int, opts ...scheduler.Option) *testContext {
	c := initTestSchedulerWithOptions(t, testCtx, false, nil, time.Second, opts...)
	if nodeCount > 0 {
		_, err := createNodes(c.clientSet, "test-node", nil, nodeCount)
		if err != nil {
			t.Fatalf("Cannot create nodes: %v", err)
		}
	}
	return c
}

// initRegistryAndConfig returns registry and plugins config based on give plugins.
// TODO: refactor it to a more generic functions that accepts all kinds of Plugins as arguments
func initRegistryAndConfig(pp ...*PermitPlugin) (registry framework.Registry, plugins *schedulerconfig.Plugins) {
	if len(pp) == 0 {
		return
	}

	registry = framework.Registry{}
	plugins = &schedulerconfig.Plugins{
		Permit: &schedulerconfig.PluginSet{},
	}

	for _, p := range pp {
		registry.Register(p.Name(), newPermitPlugin(p))
		plugins.Permit.Enabled = append(plugins.Permit.Enabled, schedulerconfig.Plugin{Name: p.Name()})
	}
	return
}
