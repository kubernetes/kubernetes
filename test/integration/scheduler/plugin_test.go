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
	"k8s.io/apimachinery/pkg/util/wait"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
)

// StatefulMultipointExample is an example plugin that is executed at multiple extension points.
// This plugin is stateful. It receives arguments at initialization (NewMultipointPlugin)
// and changes its state when it is executed.
type TesterPlugin struct {
	numReserveCalled int
	numPrebindCalled int
	failReserve      bool
	failPrebind      bool
	rejectPrebind    bool
}

var _ = plugins.ReservePlugin(&TesterPlugin{})
var _ = plugins.PrebindPlugin(&TesterPlugin{})

// Name returns name of the plugin.
func (tp *TesterPlugin) Name() string {
	return "tester-plugin"
}

// Reserve is a test function that returns an error or nil, depending on the
// value of "failReserve".
func (tp *TesterPlugin) Reserve(ps plugins.PluginSet, pod *v1.Pod, nodeName string) error {
	tp.numReserveCalled++
	if tp.failReserve {
		return fmt.Errorf("injecting failure for pod %v", pod.Name)
	}
	return nil
}

// Prebind is a test function that returns (true, nil) or errors for testing.
func (tp *TesterPlugin) Prebind(ps plugins.PluginSet, pod *v1.Pod, nodeName string) (bool, error) {
	var err error
	tp.numPrebindCalled++
	if tp.failPrebind {
		err = fmt.Errorf("injecting failure for pod %v", pod.Name)
	}
	if tp.rejectPrebind {
		return false, err
	}
	return true, err
}

// TestPluginSet is a plugin set used for testing purposes.
type TestPluginSet struct {
	data           *plugins.PluginData
	reservePlugins []plugins.ReservePlugin
	prebindPlugins []plugins.PrebindPlugin
}

var _ = plugins.PluginSet(&TestPluginSet{})

// ReservePlugins returns a slice of default reserve plugins.
func (r *TestPluginSet) ReservePlugins() []plugins.ReservePlugin {
	return r.reservePlugins
}

// PrebindPlugins returns a slice of default prebind plugins.
func (r *TestPluginSet) PrebindPlugins() []plugins.PrebindPlugin {
	return r.prebindPlugins
}

// Data returns a pointer to PluginData.
func (r *TestPluginSet) Data() *plugins.PluginData {
	return r.data
}

// TestReservePlugin tests invocation of reserve plugins.
func TestReservePlugin(t *testing.T) {
	// Create a plugin set for testing. Register only a reserve plugin.
	testerPlugin := &TesterPlugin{}
	testPluginSet := &TestPluginSet{
		data: &plugins.PluginData{
			Ctx: plugins.NewPluginContext(),
		},
		reservePlugins: []plugins.ReservePlugin{testerPlugin},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "reserve-plugin", nil),
		false, nil, testPluginSet, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	for _, fail := range []bool{false, true} {
		testerPlugin.failReserve = fail
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

		if testerPlugin.numReserveCalled == 0 {
			t.Errorf("Expected the reserve plugin to be called.")
		}

		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestPrebindPlugin tests invocation of prebind plugins.
func TestPrebindPlugin(t *testing.T) {
	// Create a plugin set for testing. Register only a prebind plugin.
	testerPlugin := &TesterPlugin{}
	testPluginSet := &TestPluginSet{
		data: &plugins.PluginData{
			Ctx: plugins.NewPluginContext(),
		},
		prebindPlugins: []plugins.PrebindPlugin{testerPlugin},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "prebind-plugin", nil),
		false, nil, testPluginSet, false, time.Second)
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
		testerPlugin.failPrebind = test.fail
		testerPlugin.rejectPrebind = test.reject
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

		if testerPlugin.numPrebindCalled == 0 {
			t.Errorf("Expected the prebind plugin to be called.")
		}

		cleanupPods(cs, t, []*v1.Pod{pod})
	}
}

// TestContextCleanup tests that data inserted in the pluginContext is removed
// after a scheduling cycle is over.
func TestContextCleanup(t *testing.T) {
	// Create a plugin set for testing.
	testerPlugin := &TesterPlugin{}
	testPluginSet := &TestPluginSet{
		data: &plugins.PluginData{
			Ctx: plugins.NewPluginContext(),
		},
		reservePlugins: []plugins.ReservePlugin{testerPlugin},
		prebindPlugins: []plugins.PrebindPlugin{testerPlugin},
	}

	// Create the master and the scheduler with the test plugin set.
	context := initTestSchedulerWithOptions(t,
		initTestMaster(t, "plugin-context-cleanup", nil),
		false, nil, testPluginSet, false, time.Second)
	defer cleanupTest(t, context)

	cs := context.clientSet
	// Add a few nodes.
	_, err := createNodes(cs, "test-node", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}

	// Insert something in the plugin context.
	testPluginSet.Data().Ctx.Write("test", "foo")

	// Create and schedule a best effort pod.
	pod, err := runPausePod(cs,
		initPausePod(cs, &pausePodConfig{Name: "test-pod", Namespace: context.ns.Name}))
	if err != nil {
		t.Errorf("Error while creating or scheduling a test pod: %v", err)
	}

	// Make sure the data inserted in the plugin context is removed.
	_, err = testPluginSet.Data().Ctx.Read("test")
	if err == nil || err.Error() != plugins.NotFound {
		t.Errorf("Expected the plugin context to be cleaned up after a scheduling cycle. error: %v", err)
	}

	cleanupPods(cs, t, []*v1.Pod{pod})
}
