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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// TesterPlugin is common ancestor for a test plugin that allows injection of
// failures and some other test functionalities.
type TesterPlugin struct {
	numReserveCalled    int
	numPrebindCalled    int
	numPostbindCalled   int
	numUnreserveCalled  int
	failReserve         bool
	failPrebind         bool
	rejectPrebind       bool
	numPermitCalled     int
	failPermit          bool
	rejectPermit        bool
	timeoutPermit       bool
	waitAndRejectPermit bool
	waitAndAllowPermit  bool
}

type ReservePlugin struct {
	TesterPlugin
}

type PrebindPlugin struct {
	TesterPlugin
}

type PostbindPlugin struct {
	TesterPlugin
}

type UnreservePlugin struct {
	TesterPlugin
}

type PermitPlugin struct {
	TesterPlugin
	fh framework.FrameworkHandle
}

const (
	reservePluginName   = "reserve-plugin"
	prebindPluginName   = "prebind-plugin"
	unreservePluginName = "unreserve-plugin"
	postbindPluginName  = "postbind-plugin"
	permitPluginName    = "permit-plugin"
)

var _ = framework.ReservePlugin(&ReservePlugin{})
var _ = framework.PrebindPlugin(&PrebindPlugin{})
var _ = framework.PostbindPlugin(&PostbindPlugin{})
var _ = framework.UnreservePlugin(&UnreservePlugin{})
var _ = framework.PermitPlugin(&PermitPlugin{})

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

// NewReservePlugin is the factory for reserve plugin.
func NewReservePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return resPlugin, nil
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

// reset used to reset numPrebindCalled.
func (pp *PrebindPlugin) reset() {
	pp.numPrebindCalled = 0
}

// NewPrebindPlugin is the factory for prebind plugin.
func NewPrebindPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return pbdPlugin, nil
}

var ptbdPlugin = &PostbindPlugin{}

// Name returns name of the plugin.
func (pp *PostbindPlugin) Name() string {
	return postbindPluginName
}

// Postbind is a test function, which counts the number of times called.
func (pp *PostbindPlugin) Postbind(pc *framework.PluginContext, pod *v1.Pod, nodeName string) {
	pp.numPostbindCalled++
}

// reset used to reset numPostbindCalled.
func (pp *PostbindPlugin) reset() {
	pp.numPostbindCalled = 0
}

// NewPostbindPlugin is the factory for postbind plugin.
func NewPostbindPlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return ptbdPlugin, nil
}

var unresPlugin = &UnreservePlugin{}

// Name returns name of the plugin.
func (up *UnreservePlugin) Name() string {
	return unreservePluginName
}

// Unreserve is a test function that returns an error or nil, depending on the
// value of "failUnreserve".
func (up *UnreservePlugin) Unreserve(pc *framework.PluginContext, pod *v1.Pod, nodeName string) {
	up.numUnreserveCalled++
}

// reset used to reset numUnreserveCalled.
func (up *UnreservePlugin) reset() {
	up.numUnreserveCalled = 0
}

// NewUnreservePlugin is the factory for unreserve plugin.
func NewUnreservePlugin(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return unresPlugin, nil
}

var perPlugin = &PermitPlugin{}

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

// NewPermitPlugin is the factory for permit plugin.
func NewPermitPlugin(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
	perPlugin.fh = fh
	return perPlugin, nil
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
				t.Errorf("Didn't expected the pod to be scheduled. error: %v", err)
			}
		} else {
			if err = waitForPodToSchedule(cs, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}
		}

		if resPlugin.numReserveCalled == 0 {
			t.Errorf("Expected the reserve plugin to be called.")
		}

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
				if unresPlugin.numUnreserveCalled == 0 || unresPlugin.numUnreserveCalled != pbdPlugin.numPrebindCalled {
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
	registry := framework.Registry{permitPluginName: NewPermitPlugin}

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

		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestCoSchedulingWithPermitPlugin tests invocation of permit plugins.
func TestCoSchedulingWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	registry := framework.Registry{permitPluginName: NewPermitPlugin}

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

		cleanupPods(cs, t, []*v1.Pod{waitingPod, signallingPod})
	}
}
