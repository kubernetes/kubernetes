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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

type PrefilterPlugin struct {
	numPrefilterCalled int
	failPrefilter      bool
	rejectPrefilter    bool
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

type PrebindPlugin struct {
	numPrebindCalled int
	failPrebind      bool
	rejectPrebind    bool
}

type BindPlugin struct {
	numBindCalled         int
	PluginName            string
	bindStatus            *framework.Status
	client                *clientset.Clientset
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PostbindPlugin struct {
	name                  string
	numPostbindCalled     int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type UnreservePlugin struct {
	name                  string
	numUnreserveCalled    int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PermitPlugin struct {
	numPermitCalled     int
	failPermit          bool
	rejectPermit        bool
	timeoutPermit       bool
	waitAndRejectPermit bool
	waitAndAllowPermit  bool
	allowPermit         bool
	fh                  framework.FrameworkHandle
}

const (
	prefilterPluginName          = "prefilter-plugin"
	scorePluginName              = "score-plugin"
	scoreWithNormalizePluginName = "score-with-normalize-plugin"
	filterPluginName             = "filter-plugin"
	postFilterPluginName         = "postfilter-plugin"
	reservePluginName            = "reserve-plugin"
	prebindPluginName            = "prebind-plugin"
	unreservePluginName          = "unreserve-plugin"
	postbindPluginName           = "postbind-plugin"
	permitPluginName             = "permit-plugin"
)

var _ = framework.PrefilterPlugin(&PrefilterPlugin{})
var _ = framework.ScorePlugin(&ScorePlugin{})
var _ = framework.FilterPlugin(&FilterPlugin{})
var _ = framework.ScorePlugin(&ScorePlugin{})
var _ = framework.ScoreWithNormalizePlugin(&ScoreWithNormalizePlugin{})
var _ = framework.ReservePlugin(&ReservePlugin{})
var _ = framework.PostFilterPlugin(&PostFilterPlugin{})
var _ = framework.PrebindPlugin(&PrebindPlugin{})
var _ = framework.BindPlugin(&BindPlugin{})
var _ = framework.PostbindPlugin(&PostbindPlugin{})
var _ = framework.UnreservePlugin(&UnreservePlugin{})
var _ = framework.PermitPlugin(&PermitPlugin{})

var scPlugin = &ScorePlugin{}

// NewScorePlugin is the factory for score plugin.
func NewScorePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return scPlugin, nil
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
func (sp *ScorePlugin) Score(pc *framework.PluginContext, p *v1.Pod, nodeName string) (int, *framework.Status) {
	sp.numScoreCalled++
	if sp.failScore {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", p.Name))
	}

	score := 1
	if sp.numScoreCalled == 1 {
		// The first node is scored the highest, the rest is scored lower.
		sp.highScoreNode = nodeName
		score = framework.MaxNodeScore
	}
	return score, nil
}

var scoreWithNormalizePlguin = &ScoreWithNormalizePlugin{}

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
func (sp *ScoreWithNormalizePlugin) Score(pc *framework.PluginContext, p *v1.Pod, nodeName string) (int, *framework.Status) {
	sp.numScoreCalled++
	score := 10
	return score, nil
}

func (sp *ScoreWithNormalizePlugin) NormalizeScore(pc *framework.PluginContext, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	sp.numNormalizeScoreCalled++
	return nil
}

// NewScoreWithNormalizePlugin is the factory for score with normalize plugin.
func NewScoreWithNormalizePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return scoreWithNormalizePlguin, nil
}

var filterPlugin = &FilterPlugin{}

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
func (fp *FilterPlugin) Filter(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	fp.numFilterCalled++

	if fp.failFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// NewFilterPlugin is the factory for filtler plugin.
func NewFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return filterPlugin, nil
}

// Name returns name of the plugin.
func (rp *ReservePlugin) Name() string {
	return reservePluginName
}

var resPlugin = &ReservePlugin{}

// Reserve is a test function that returns an error or nil, depending on the
// value of "failReserve".
func (rp *ReservePlugin) Reserve(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
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

// NewReservePlugin is the factory for reserve plugin.
func NewReservePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return resPlugin, nil
}

// Name returns name of the plugin.
func (*PostFilterPlugin) Name() string {
	return postFilterPluginName
}

var postFilterPlugin = &PostFilterPlugin{}

// PostFilter is a test function.
func (pfp *PostFilterPlugin) PostFilter(_ *framework.PluginContext, pod *v1.Pod, _ []*v1.Node, _ framework.NodeToStatusMap) *framework.Status {
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

// NewPostFilterPlugin is the factory for post-filter plugin.
func NewPostFilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return postFilterPlugin, nil
}

var pbdPlugin = &PrebindPlugin{}

// Name returns name of the plugin.
func (pp *PrebindPlugin) Name() string {
	return prebindPluginName
}

// Prebind is a test function that returns (true, nil) or errors for testing.
func (pp *PrebindPlugin) Prebind(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	pp.numPrebindCalled++
	if pp.failPrebind {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPrebind {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil
}

// reset used to reset prebind plugin.
func (pp *PrebindPlugin) reset() {
	pp.numPrebindCalled = 0
	pp.failPrebind = false
	pp.rejectPrebind = false
}

const bindPluginAnnotation = "bindPluginName"

// NewPrebindPlugin is the factory for prebind plugin.
func NewPrebindPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return pbdPlugin, nil
}

func (bp *BindPlugin) Name() string {
	return bp.PluginName
}

func (bp *BindPlugin) Bind(pc *framework.PluginContext, p *v1.Pod, nodeName string) *framework.Status {
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

var ptbdPlugin = &PostbindPlugin{name: postbindPluginName}

// Name returns name of the plugin.
func (pp *PostbindPlugin) Name() string {
	return pp.name
}

// Postbind is a test function, which counts the number of times called.
func (pp *PostbindPlugin) Postbind(pc *framework.PluginContext, pod *v1.Pod, nodeName string) {
	pp.numPostbindCalled++
	if pp.pluginInvokeEventChan != nil {
		pp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: pp.Name(), val: pp.numPostbindCalled}
	}
}

// reset used to reset postbind plugin.
func (pp *PostbindPlugin) reset() {
	pp.numPostbindCalled = 0
}

// NewPostbindPlugin is the factory for postbind plugin.
func NewPostbindPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return ptbdPlugin, nil
}

var pfPlugin = &PrefilterPlugin{}

// Name returns name of the plugin.
func (pp *PrefilterPlugin) Name() string {
	return prefilterPluginName
}

// Prefilter is a test function that returns (true, nil) or errors for testing.
func (pp *PrefilterPlugin) Prefilter(pc *framework.PluginContext, pod *v1.Pod) *framework.Status {
	pp.numPrefilterCalled++
	if pp.failPrefilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPrefilter {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil
}

// reset used to reset prefilter plugin.
func (pp *PrefilterPlugin) reset() {
	pp.numPrefilterCalled = 0
	pp.failPrefilter = false
	pp.rejectPrefilter = false
}

// NewPrebindPlugin is the factory for prebind plugin.
func NewPrefilterPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return pfPlugin, nil
}

var unresPlugin = &UnreservePlugin{name: unreservePluginName}

// Name returns name of the plugin.
func (up *UnreservePlugin) Name() string {
	return up.name
}

// Unreserve is a test function that returns an error or nil, depending on the
// value of "failUnreserve".
func (up *UnreservePlugin) Unreserve(pc *framework.PluginContext, pod *v1.Pod, nodeName string) {
	up.numUnreserveCalled++
	if up.pluginInvokeEventChan != nil {
		up.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: up.Name(), val: up.numUnreserveCalled}
	}
}

// reset used to reset numUnreserveCalled.
func (up *UnreservePlugin) reset() {
	up.numUnreserveCalled = 0
}

// NewUnreservePlugin is the factory for unreserve plugin.
func NewUnreservePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return unresPlugin, nil
}

// Name returns name of the plugin.
func (pp *PermitPlugin) Name() string {
	return permitPluginName
}

// Permit implements the permit test plugin.
func (pp *PermitPlugin) Permit(pc *framework.PluginContext, pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	pp.numPermitCalled++
	if pp.failPermit {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name)), 0
	}
	if pp.rejectPermit {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
	}
	if pp.timeoutPermit {
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
			pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { wp.Allow() })
			return nil, 0
		}
	}
	return nil, 0
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
}

// NewPermitPlugin returns a factory for permit plugin with specified PermitPlugin.
func NewPermitPlugin(permitPlugin *PermitPlugin) framework.PluginFactory {
	return func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
		permitPlugin.fh = fh
		return permitPlugin, nil
	}
}

// TestPrefilterPlugin tests invocation of prefilter plugins.
func TestPrefilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a reserve plugin.
	registry := framework.Registry{prefilterPluginName: NewPrefilterPlugin}

	// Setup initial prefilter plugin for testing.
	prefilterPlugin := &schedulerconfig.Plugins{
		PreFilter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: prefilterPluginName,
				},
			},
		},
	}
	// Set empty plugin config for testing
	emptyPluginConfig := []schedulerconfig.PluginConfig{}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "prefilter-plugin", nil),
		false, nil, registry, prefilterPlugin, emptyPluginConfig, false, time.Second)

	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

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
		pfPlugin.failPrefilter = test.fail
		pfPlugin.rejectPrefilter = test.reject
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.reject || test.fail {
			if err = waitForPodUnschedulable(cs, pod); err != nil {
				t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
			}
		}

		if pfPlugin.numPrefilterCalled == 0 {
			t.Errorf("Expected the prefilter plugin to be called.")
		}

		pfPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestScorePlugin tests invocation of score plugins.
func TestScorePlugin(t *testing.T) {
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
	context, cs := initTestContextForScorePlugin(t, plugins)
	defer cleanupTest(t, context)

	for i, fail := range []bool{false, true} {
		scPlugin.failScore = fail
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Fatalf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = waitForPodUnschedulable(cs, pod); err != nil {
				t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			} else {
				p, err := getPod(cs, pod.Name, pod.Namespace)
				if err != nil {
					t.Errorf("Failed to retrieve the pod. error: %v", err)
				} else if p.Spec.NodeName != scPlugin.highScoreNode {
					t.Errorf("Expected the pod to be scheduled on node %q, got %q", scPlugin.highScoreNode, p.Spec.NodeName)
				}
			}
		}

		if scPlugin.numScoreCalled == 0 {
			t.Errorf("Expected the score plugin to be called.")
		}

		scPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestNormalizeScorePlugin tests invocation of normalize score plugins.
func TestNormalizeScorePlugin(t *testing.T) {
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
	context, cs := initTestContextForScorePlugin(t, plugins)
	defer cleanupTest(t, context)

	// Create a best effort pod.
	pod, err := createPausePod(cs,
		initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
	if err != nil {
		t.Fatalf("Error while creating a test pod: %v", err)
	}

	if err = waitForPodToSchedule(cs, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if scoreWithNormalizePlguin.numScoreCalled == 0 {
		t.Errorf("Expected the score plugin to be called.")
	}
	if scoreWithNormalizePlguin.numNormalizeScoreCalled == 0 {
		t.Error("Expected the normalize score plugin to be called")
	}

	scoreWithNormalizePlguin.reset()
}

// TestReservePlugin tests invocation of reserve plugins.
func TestReservePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a reserve plugin.
	registry := framework.Registry{reservePluginName: NewReservePlugin}

	// Setup initial reserve plugin for testing.
	reservePlugin := &schedulerconfig.Plugins{
		Reserve: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: reservePluginName,
				},
			},
		},
	}
	// Set empty plugin config for testing
	emptyPluginConfig := []schedulerconfig.PluginConfig{}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "reserve-plugin", nil),
		false, nil, registry, reservePlugin, emptyPluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	for _, fail := range []bool{false, true} {
		resPlugin.failReserve = fail
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if resPlugin.numReserveCalled == 0 {
			t.Errorf("Expected the reserve plugin to be called.")
		}

		resPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPrebindPlugin tests invocation of prebind plugins.
func TestPrebindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a prebind plugin.
	registry := framework.Registry{prebindPluginName: NewPrebindPlugin}

	// Setup initial prebind plugin for testing.
	preBindPlugin := &schedulerconfig.Plugins{
		PreBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: prebindPluginName,
				},
			},
		},
	}
	// Set reserve prebind config for testing
	preBindPluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: prebindPluginName,
			Args: runtime.Unknown{},
		},
	}
	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "prebind-plugin", nil),
		false, nil, registry, preBindPlugin, preBindPluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

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
		pbdPlugin.failPrebind = test.fail
		pbdPlugin.rejectPrebind = test.reject
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
		} else {
			if test.reject {
				if err = waitForPodUnschedulable(cs, pod); err != nil {
					t.Errorf("test #%v: Didn't expected the pod to be scheduled. error: %v", i, err)
				}
			} else {
				if err = waitForPodToSchedule(cs, pod); err != nil {
					t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				}
			}
		}

		if pbdPlugin.numPrebindCalled == 0 {
			t.Errorf("Expected the prebind plugin to be called.")
		}

		pbdPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestUnreservePlugin tests invocation of un-reserve plugin
func TestUnreservePlugin(t *testing.T) {
	// TODO: register more plugin which would trigger un-reserve plugin
	registry := framework.Registry{
		unreservePluginName: NewUnreservePlugin,
		prebindPluginName:   NewPrebindPlugin,
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
					Name: prebindPluginName,
				},
			},
		},
	}
	// Set unreserve and prebind plugin config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: unreservePluginName,
			Args: runtime.Unknown{},
		},
		{
			Name: prebindPluginName,
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "unreserve-plugin", nil),
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	tests := []struct {
		prebindFail   bool
		prebindReject bool
	}{
		{
			prebindFail:   false,
			prebindReject: false,
		},
		{
			prebindFail:   true,
			prebindReject: false,
		},
		{
			prebindFail:   false,
			prebindReject: true,
		},
		{
			prebindFail:   true,
			prebindReject: true,
		},
	}

	for i, test := range tests {
		pbdPlugin.failPrebind = test.prebindFail
		pbdPlugin.rejectPrebind = test.prebindReject

		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.prebindFail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if unresPlugin.numUnreserveCalled == 0 || unresPlugin.numUnreserveCalled != pbdPlugin.numPrebindCalled {
				t.Errorf("test #%v: Expected the unreserve plugin to be called %d times, was called %d times.", i, pbdPlugin.numPrebindCalled, unresPlugin.numUnreserveCalled)
			}
		} else {
			if test.prebindReject {
				if err = waitForPodUnschedulable(cs, pod); err != nil {
					t.Errorf("test #%v: Didn't expected the pod to be scheduled. error: %v", i, err)
				}
				if unresPlugin.numUnreserveCalled == 0 {
					t.Errorf("test #%v: Expected the unreserve plugin to be called %d times, was called %d times.", i, pbdPlugin.numPrebindCalled, unresPlugin.numUnreserveCalled)
				}
			} else {
				if err = waitForPodToSchedule(cs, pod); err != nil {
					t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				}
				if unresPlugin.numUnreserveCalled > 0 {
					t.Errorf("test #%v: Didn't expected the unreserve plugin to be called, was called %d times.", i, unresPlugin.numUnreserveCalled)
				}
			}
		}

		unresPlugin.reset()
		pbdPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
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
	postbindPlugin := &PostbindPlugin{name: "mock-post-bind-plugin"}
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
		postbindPlugin.Name(): func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return postbindPlugin, nil
		},
	}

	// Setup initial unreserve and bind plugins for testing.
	plugins := &schedulerconfig.Plugins{
		Unreserve: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{{Name: unreservePlugin.Name()}},
		},
		Bind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{{Name: bindPlugin1.Name()}, {Name: bindPlugin2.Name()}},
		},
		PostBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{{Name: postbindPlugin.Name()}},
		},
	}
	// Set reserve and bind config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: unreservePlugin.Name(),
			Args: runtime.Unknown{},
		},
		{
			Name: bindPlugin1.Name(),
			Args: runtime.Unknown{},
		},
		{
			Name: bindPlugin2.Name(),
			Args: runtime.Unknown{},
		},
		{
			Name: postbindPlugin.Name(),
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t, testContext,
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
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
			expectInvokeEvents:     []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postbindPlugin.Name(), val: 1}},
		},
		// bindplugin2 succeeded to bind the pod
		{
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin2.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postbindPlugin.Name(), val: 1}},
		},
		// bindplugin1 succeeded to bind the pod
		{
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Success, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin1.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: postbindPlugin.Name(), val: 1}},
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
		postbindPlugin.pluginInvokeEventChan = pluginInvokeEventChan

		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.expectBoundByScheduler || test.expectBoundByPlugin {
			// bind plugins skipped to bind the pod
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				continue
			}
			pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("can't get pod: %v", err)
			}
			if test.expectBoundByScheduler {
				if pod.Annotations[bindPluginAnnotation] != "" {
					t.Errorf("test #%v: Expected the pod to be binded by scheduler instead of by bindplugin %s", i, pod.Annotations[bindPluginAnnotation])
				}
				if bindPlugin1.numBindCalled != 1 || bindPlugin2.numBindCalled != 1 {
					t.Errorf("test #%v: Expected each bind plugin to be called once, was called %d and %d times.", i, bindPlugin1.numBindCalled, bindPlugin2.numBindCalled)
				}
			} else {
				if pod.Annotations[bindPluginAnnotation] != test.expectBindPluginName {
					t.Errorf("test #%v: Expected the pod to be binded by bindplugin %s instead of by bindplugin %s", i, test.expectBindPluginName, pod.Annotations[bindPluginAnnotation])
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
				return postbindPlugin.numPostbindCalled == 1, nil
			}); err != nil {
				t.Errorf("test #%v: Expected the postbind plugin to be called once, was called %d times.", i, postbindPlugin.numPostbindCalled)
			}
			if unreservePlugin.numUnreserveCalled != 0 {
				t.Errorf("test #%v: Expected the unreserve plugin not to be called, was called %d times.", i, unreservePlugin.numUnreserveCalled)
			}
		} else {
			// bind plugin fails to bind the pod
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if postbindPlugin.numPostbindCalled > 0 {
				t.Errorf("test #%v: Didn't expected the postbind plugin to be called %d times.", i, postbindPlugin.numPostbindCalled)
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
		postbindPlugin.reset()
		bindPlugin1.reset()
		bindPlugin2.reset()
		unreservePlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPostbindPlugin tests invocation of postbind plugins.
func TestPostbindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register a prebind and a postbind plugin.
	registry := framework.Registry{
		prebindPluginName:  NewPrebindPlugin,
		postbindPluginName: NewPostbindPlugin,
	}

	// Setup initial prebind and postbind plugin for testing.
	plugins := &schedulerconfig.Plugins{
		PreBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: prebindPluginName,
				},
			},
		},
		PostBind: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: postbindPluginName,
				},
			},
		},
	}
	// Set reserve prebind and postbind config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: prebindPluginName,
			Args: runtime.Unknown{},
		},
		{
			Name: postbindPluginName,
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "postbind-plugin", nil),
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	tests := []struct {
		prebindFail   bool
		prebindReject bool
	}{
		{
			prebindFail:   false,
			prebindReject: false,
		},
		{
			prebindFail:   true,
			prebindReject: false,
		},
		{
			prebindFail:   false,
			prebindReject: true,
		},
		{
			prebindFail:   true,
			prebindReject: true,
		},
	}

	for i, test := range tests {
		pbdPlugin.failPrebind = test.prebindFail
		pbdPlugin.rejectPrebind = test.prebindReject

		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if test.prebindFail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
			if ptbdPlugin.numPostbindCalled > 0 {
				t.Errorf("test #%v: Didn't expected the postbind plugin to be called %d times.", i, ptbdPlugin.numPostbindCalled)
			}
		} else {
			if test.prebindReject {
				if err = waitForPodUnschedulable(cs, pod); err != nil {
					t.Errorf("test #%v: Didn't expected the pod to be scheduled. error: %v", i, err)
				}
				if ptbdPlugin.numPostbindCalled > 0 {
					t.Errorf("test #%v: Didn't expected the postbind plugin to be called %d times.", i, ptbdPlugin.numPostbindCalled)
				}
			} else {
				if err = waitForPodToSchedule(cs, pod); err != nil {
					t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				}
				if ptbdPlugin.numPostbindCalled == 0 {
					t.Errorf("test #%v: Expected the postbind plugin to be called, was called %d times.", i, ptbdPlugin.numPostbindCalled)
				}
			}
		}

		ptbdPlugin.reset()
		pbdPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPermitPlugin tests invocation of permit plugins.
func TestPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{}
	registry := framework.Registry{permitPluginName: NewPermitPlugin(perPlugin)}

	// Setup initial permit plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Permit: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: permitPluginName,
				},
			},
		},
	}
	// Set permit plugin config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: permitPluginName,
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "permit-plugin", nil),
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

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
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}
		if test.fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("test #%v: Expected a scheduling error, but didn't get it. error: %v", i, err)
			}
		} else {
			if test.reject || test.timeout {
				if err = waitForPodUnschedulable(cs, pod); err != nil {
					t.Errorf("test #%v: Didn't expect the pod to be scheduled. error: %v", i, err)
				}
			} else {
				if err = waitForPodToSchedule(cs, pod); err != nil {
					t.Errorf("test #%v: Expected the pod to be scheduled. error: %v", i, err)
				}
			}
		}

		if perPlugin.numPermitCalled == 0 {
			t.Errorf("Expected the permit plugin to be called.")
		}

		perPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestCoSchedulingWithPermitPlugin tests invocation of permit plugins.
func TestCoSchedulingWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{}
	registry := framework.Registry{permitPluginName: NewPermitPlugin(perPlugin)}

	// Setup initial permit plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Permit: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: permitPluginName,
				},
			},
		},
	}
	// Set permit plugin config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: permitPluginName,
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "permit-plugin", nil),
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

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
		perPlugin.failPermit = false
		perPlugin.rejectPermit = false
		perPlugin.timeoutPermit = false
		perPlugin.waitAndRejectPermit = test.waitReject
		perPlugin.waitAndAllowPermit = test.waitAllow

		// Create two pods.
		waitingPod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "waiting-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating the waiting pod: %v", err)
		}
		signallingPod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "signalling-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating the signalling pod: %v", err)
		}

		if test.waitReject {
			if err = waitForPodUnschedulable(cs, waitingPod); err != nil {
				t.Errorf("test #%v: Didn't expect the waiting pod to be scheduled. error: %v", i, err)
			}
			if err = waitForPodUnschedulable(cs, signallingPod); err != nil {
				t.Errorf("test #%v: Didn't expect the signalling pod to be scheduled. error: %v", i, err)
			}
		} else {
			if err = waitForPodToSchedule(cs, waitingPod); err != nil {
				t.Errorf("test #%v: Expected the waiting pod to be scheduled. error: %v", i, err)
			}
			if err = waitForPodToSchedule(cs, signallingPod); err != nil {
				t.Errorf("test #%v: Expected the signalling pod to be scheduled. error: %v", i, err)
			}
		}

		if perPlugin.numPermitCalled == 0 {
			t.Errorf("Expected the permit plugin to be called.")
		}

		perPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{waitingPod, signallingPod})
	}
}

// TestFilterPlugin tests invocation of filter plugins.
func TestFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a filter plugin.
	registry := framework.Registry{filterPluginName: NewFilterPlugin}

	// Setup initial filter plugin for testing.
	plugin := &schedulerconfig.Plugins{
		Filter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: filterPluginName,
				},
			},
		},
	}
	// Set empty plugin config for testing
	emptyPluginConfig := []schedulerconfig.PluginConfig{}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "filter-plugin", nil),
		false, nil, registry, plugin, emptyPluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	for _, fail := range []bool{false, true} {
		filterPlugin.failFilter = fail
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = wait.Poll(10*time.Millisecond, 30*time.Second, podUnschedulable(cs, pod.Namespace, pod.Name)); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled.")
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if filterPlugin.numFilterCalled == 0 {
			t.Errorf("Expected the filter plugin to be called.")
		}

		filterPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPostFilterPlugin tests invocation of post-filter plugins.
func TestPostFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a post-filter plugin.
	registry := framework.Registry{postFilterPluginName: NewPostFilterPlugin}

	// Setup initial post-filter plugin for testing.
	pluginsConfig := &schedulerconfig.Plugins{
		PostFilter: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: postFilterPluginName,
				},
			},
		},
	}
	// Set empty plugin config for testing
	emptyPluginConfig := []schedulerconfig.PluginConfig{}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "post-filter-plugin", nil),
		false, nil, registry, pluginsConfig, emptyPluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	for _, fail := range []bool{false, true} {
		postFilterPlugin.failPostFilter = fail
		// Create a best effort pod.
		pod, err := createPausePod(cs,
			initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
		if err != nil {
			t.Errorf("Error while creating a test pod: %v", err)
		}

		if fail {
			if err = waitForPodUnschedulable(cs, pod); err != nil {
				t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if postFilterPlugin.numPostFilterCalled == 0 {
			t.Errorf("Expected the post-filter plugin to be called.")
		}

		postFilterPlugin.reset()
		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPreemptWithPermitPlugin tests preempt with permit plugins.
func TestPreemptWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{}
	registry := framework.Registry{permitPluginName: NewPermitPlugin(perPlugin)}

	// Setup initial permit plugin for testing.
	plugins := &schedulerconfig.Plugins{
		Permit: &schedulerconfig.PluginSet{
			Enabled: []schedulerconfig.Plugin{
				{
					Name: permitPluginName,
				},
			},
		},
	}
	// Set permit plugin config for testing
	pluginConfig := []schedulerconfig.PluginConfig{
		{
			Name: permitPluginName,
			Args: runtime.Unknown{},
		},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "preempt-with-permit-plugin", nil),
		false, nil, registry, plugins, pluginConfig, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add one node.
	nodeRes := &v1.ResourceList{
		v1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
		v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
	}
	_, err := createNodes(cs, "test-node", nodeRes, 1)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	perPlugin.failPermit = false
	perPlugin.rejectPermit = false
	perPlugin.timeoutPermit = false
	perPlugin.waitAndRejectPermit = false
	perPlugin.waitAndAllowPermit = true
	perPlugin.allowPermit = true

	lowPriority, highPriority := int32(100), int32(300)
	resourceRequest := v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI)},
	}

	// Create two pods.
	waitingPod, err := createPausePod(cs,
		initPausePod(cs, &pausePodConfig{Name: "waiting-pod", Namespace: context.ns.Name, Priority: &lowPriority, Resources: &resourceRequest}))
	if err != nil {
		t.Errorf("Error while creating the waiting pod: %v", err)
	}
	// Wait until the waiting-pod is actually waiting, then create a preemptor pod to preempt it.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		w := false
		perPlugin.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
		return w, nil
	})

	preemptorPod, err := createPausePod(cs,
		initPausePod(cs, &pausePodConfig{Name: "preemptor-pod", Namespace: context.ns.Name, Priority: &highPriority, Resources: &resourceRequest}))
	if err != nil {
		t.Errorf("Error while creating the preemptor pod: %v", err)
	}

	if err = waitForPodToSchedule(cs, preemptorPod); err != nil {
		t.Errorf("Expected the preemptor pod to be scheduled. error: %v", err)
	}

	if _, err := getPod(cs, waitingPod.Name, waitingPod.Namespace); err == nil {
		t.Error("Expected the waiting pod to get preempted and deleted")
	}

	if perPlugin.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	perPlugin.reset()
	cleanupPods(cs, t, []*v1.Pod{waitingPod, preemptorPod})
}

func initTestContextForScorePlugin(t *testing.T, plugins *schedulerconfig.Plugins) (*testContext, *clientset.Clientset) {
	// Create a plugin registry for testing. Register only a score plugin.
	registry := framework.Registry{
		scorePluginName:              NewScorePlugin,
		scoreWithNormalizePluginName: NewScoreWithNormalizePlugin,
	}

	// Set empty plugin config for testing
	emptyPluginConfig := []schedulerconfig.PluginConfig{}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "score-plugin", nil),
		false, nil, registry, plugins, emptyPluginConfig, false, time.Second)

	cs := context.clientSet
	_, err := createNodes(cs, "test-node", nil, 10)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	return context, cs
}
