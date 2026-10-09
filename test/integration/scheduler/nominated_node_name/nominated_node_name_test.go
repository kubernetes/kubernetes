/*
Copyright 2025 The Kubernetes Authors.

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

package nominatednodename

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

type FakePermitPlugin struct {
	code fwk.Code
}

type RunForeverPreBindPlugin struct {
	cancel <-chan struct{}
}

type NoNNNPostBindPlugin struct {
	t      *testing.T
	cancel <-chan struct{}
}

func (bp *NoNNNPostBindPlugin) Name() string {
	return "NoNNNPostBindPlugin"
}

func (bp *NoNNNPostBindPlugin) PostBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if p.Status.NominatedNodeName != "" {
		bp.t.Fatalf("PostBind should not set .status.nominatedNodeName for pod %v/%v, but it was set to %v", p.Namespace, p.Name, p.Status.NominatedNodeName)
	}
}

// Name returns name of the plugin.
func (pp *FakePermitPlugin) Name() string {
	return "FakePermitPlugin"
}

// Permit implements the permit test plugin.
func (pp *FakePermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if pp.code == fwk.Wait {
		return fwk.NewStatus(pp.code, ""), 10 * time.Minute
	}
	return fwk.NewStatus(pp.code, ""), 0
}

// Name returns name of the plugin.
func (pp *RunForeverPreBindPlugin) Name() string {
	return "RunForeverPreBindPlugin"
}

// PreBindPreFlight is a test function that returns nil for testing.
func (pp *RunForeverPreBindPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return &fwk.PreBindPreFlightResult{AllowParallel: false}, nil
}

// PreBind is a test function that returns (true, nil) or errors for testing.
func (pp *RunForeverPreBindPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	select {
	case <-ctx.Done():
		return fwk.NewStatus(fwk.Error, "context cancelled")
	case <-pp.cancel:
		return fwk.NewStatus(fwk.Error, "pre-bind cancelled")
	}
}

// TestNominatedNodeNameIsSetBeforePreBindAndWaitOnPermit makes sure that nominatedNodeName is set in the binding cycle
// when the PreBind or Permit plugin (WaitOnPermit) is going to work.
func TestNominatedNodeNameIsSetBeforePreBindAndWaitOnPermit(t *testing.T) {
	tests := []struct {
		name                    string
		plugin                  fwk.Plugin
		expectNominatedNodeName bool
	}{
		{
			name:                    "NominatedNodeName is put if PreBindPlugin will run",
			plugin:                  &RunForeverPreBindPlugin{},
			expectNominatedNodeName: true,
		},
		{
			name:                    "NominatedNodeName is put if PermitPlugin will run at WaitOnPermit",
			expectNominatedNodeName: true,
			plugin: &FakePermitPlugin{
				code: fwk.Wait,
			},
		},
		{
			name: "NominatedNodeName is not put if PermitPlugin won't run at WaitOnPermit",
			plugin: &FakePermitPlugin{
				code: fwk.Success,
			},
			expectNominatedNodeName: false,
		},
		{
			name:                    "NominatedNodeName is not put if PermitPlugin nor PreBindPlugin will run",
			plugin:                  nil,
			expectNominatedNodeName: false,
		},
	}

	for _, test := range tests {
		for _, nnnForExpectationEnabled := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s (NominatedNodeName for expectation: %v)", test.name, nnnForExpectationEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, nnnForExpectationEnabled)

				testContext := testutils.InitTestAPIServer(t, "nnn-test", nil)

				pf := func(plugin fwk.Plugin) frameworkruntime.PluginFactory {
					return func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
						return plugin, nil
					}
				}

				plugins := []fwk.Plugin{&NoNNNPostBindPlugin{cancel: testContext.Ctx.Done(), t: t}}
				if test.plugin != nil {
					// This code makes sure that each test case (for each value of feature gate) uses a separate cancel channel.
					runForeverPlugin, ok := test.plugin.(*RunForeverPreBindPlugin)
					if ok {
						cancel := make(chan struct{})
						runForeverPlugin.cancel = cancel
						defer func() {
							close(cancel)
						}()
					}

					plugins = append(plugins, test.plugin)
				}

				registry, prof := schedulerutils.InitRegistryAndConfig(t, pf, plugins...)

				testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true,
					scheduler.WithProfiles(prof),
					scheduler.WithFrameworkOutOfTreeRegistry(registry))
				defer teardown()

				pod, err := testutils.CreatePausePod(testCtx.ClientSet,
					testutils.InitPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
				if err != nil {
					t.Fatalf("Error while creating a test pod: %v", err)
				}

				if test.expectNominatedNodeName {
					err := testutils.WaitForNominatedNodeNameWithTimeout(testCtx.Ctx, testCtx.ClientSet, pod, time.Second)
					if nnnForExpectationEnabled {
						if err != nil {
							t.Errorf(".status.nominatedNodeName was not set in pod %v/%v: %v", pod.Namespace, pod.Name, err)
						}
					} else {
						// If the feature is disabled, the expectation is that pod will be pending but NNN won't be set.
						if err == nil {
							t.Errorf("expected .status.nominatedNodeName not to be set in pod %v/%v: %v", pod.Namespace, pod.Name, err)
						}
					}
				} else {
					if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
						t.Errorf("Pod %v/%v was not scheduled: %v", pod.Namespace, pod.Name, err)
					}
				}
			})
		}
	}
}

type createPods struct {
	pods []*v1.Pod
	// nominatedNodeName to be set on each pod (same for all pods).
	nominatedNodeName string
}

type createPodGroup struct {
	podGroup *schedulingv1beta1.PodGroup
}

type createCompositePodGroup struct {
	compositePodGroup *schedulingv1alpha3.CompositePodGroup
}

type schedulePod struct {
	podName               string
	expectSuccess         bool
	expectUnschedulable   bool
	expectedScheduledNode string
}

type schedulePodGroup struct {
	podGroupName          string
	podNames              []string
	expectSuccess         bool
	expectUnschedulable   bool
	expectedScheduledNode string
	validateNNNCleared    bool
}

type checkNNN struct {
	podNames    []string
	expectedNNN string
}

type scenario struct {
	// name is this step's name, just for the debugging purpose.
	name string

	// Only one of the following actions should be set.

	// createPods creates Pods.
	createPods *createPods
	// createPodGroup creates a PodGroup.
	createPodGroup *createPodGroup
	// createCompositePodGroup creates a CompositePodGroup.
	createCompositePodGroup *createCompositePodGroup
	// createNode creates an additional Node.
	createNode string
	// schedulePod schedules one Pod that is at the top of the activeQ.
	// You should give a Pod name that is supposed to be scheduled.
	schedulePod *schedulePod
	// schedulePodGroup schedules one PodGroup or CompositePodGroup that is at the top of the activeQ.
	schedulePodGroup *schedulePodGroup
	// completePreemption completes the preemption that is currently on-going.
	// completePreemption waits until Pod or PodGroup completed preemption. It is no-op when AsyncPreemption is disabled.
	// You should give a Pod or PodGroup/CompositePodGroup name.
	completePreemption string
	// checkNNN checks that NominatedNodeName is set as expected.
	checkNNN *checkNNN
}

func makePod(name string, priority int32, cpu string) *v1.Pod {
	return st.MakePod().Name(name).Req(map[v1.ResourceName]string{v1.ResourceCPU: cpu}).Container("image").ZeroTerminationGracePeriod().Priority(priority).Obj()
}

func makeScheduledPod(name, node string, priority int32, cpu string) *v1.Pod {
	return st.MakePod().Name(name).Node(node).Req(map[v1.ResourceName]string{v1.ResourceCPU: cpu}).Container("image").ZeroTerminationGracePeriod().Priority(priority).Obj()
}

func makePodGroupPod(name, podgroupName string, priority int32, cpu string) *v1.Pod {
	return st.MakePod().Name(name).PodGroupName(podgroupName).Req(map[v1.ResourceName]string{v1.ResourceCPU: cpu}).Container("image").ZeroTerminationGracePeriod().Priority(priority).Obj()
}

func makeGangPodGroup(name string, priority, minCount int32) *schedulingv1beta1.PodGroup {
	return st.MakePodGroup().Name(name).Priority(priority).MinCount(minCount).Obj()
}

func makeBasicPodGroup(name string, priority int32) *schedulingv1beta1.PodGroup {
	return st.MakePodGroup().Name(name).Priority(priority).BasicPolicy().Obj()
}

func makeCompositePodGroup(name string, priority int32) *schedulingv1alpha3.CompositePodGroup {
	return st.MakeCompositePodGroup().Name(name).Priority(priority).BasicPolicy().WorkloadRef("wl1", "t1").Obj()
}

func makeChildGangPodGroup(name, parentCPGName string, priority, minCount int32) *schedulingv1beta1.PodGroup {
	return st.MakePodGroup().Name(name).Priority(priority).MinCount(minCount).ParentCompositePodGroup(parentCPGName).WorkloadRef("wl1", "t1").Obj()
}

const (
	lowPriority  = 1
	highPriority = 100
)

// TestPreemptionAndNominatedNodeNameScenarios tests setting/clearing NominatedNodeName in scenarios with preemption.
func TestPreemptionAndNominatedNodeNameScenarios(t *testing.T) {
	tests := []struct {
		name string
		// scenarios after the first attempt of scheduling the pod.
		scenarios []scenario
	}{
		{
			name: "basic preemption sets NominatedNodeName",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("victim", "node", lowPriority, "4"),
						},
					},
				},
				{
					name: "create a preemptor Pod",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("preemptor", highPriority, "2"),
						},
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name: "check NNN is set in preemptor",
					checkNNN: &checkNNN{
						podNames:    []string{"preemptor"},
						expectedNNN: "node",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again",
					schedulePod: &schedulePod{
						podName:               "preemptor",
						expectSuccess:         true,
						expectedScheduledNode: "node",
					},
				},
			},
		},
		{
			name: "Overwrite NominatedNodeName with preemption",
			scenarios: []scenario{
				{
					name:       "create node2",
					createNode: "node2",
				},
				{
					name: "create pods on node",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("victim-1", "node", lowPriority, "2"),
							makeScheduledPod("victim-2", "node", lowPriority, "2"),
						},
					},
				},
				{
					name: "create pod on node2",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("victim-3", "node2", lowPriority, "3"),
						},
					},
				},
				{
					name: "create a preemptor Pod with NNN set to node",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("preemptor", highPriority, "4"),
						},
						nominatedNodeName: "node",
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name: "check NNN in preemptor gets changed to node2 (because preemption of only 1 victim is needed on this node)",
					checkNNN: &checkNNN{
						podNames:    []string{"preemptor"},
						expectedNNN: "node2",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again",
					schedulePod: &schedulePod{
						podName:               "preemptor",
						expectSuccess:         true,
						expectedScheduledNode: "node2",
					},
				},
			},
		},
		{
			name: "NNN is ignored on pod that cannot fit on the node",
			scenarios: []scenario{
				{
					name: "create pod with NNN that exceeds capacity of the node",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("pod-exceeds-capacity", lowPriority, "5"),
						},
						nominatedNodeName: "node",
					},
				},
				{
					name: "create pod without NNN that fits on the node",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("pod-fits", lowPriority, "3"),
						},
					},
				},
				{
					name: "schedule pod-exceeds-capacity",
					schedulePod: &schedulePod{
						podName: "pod-exceeds-capacity",
					},
				},
				{
					name: "check NNN in pod-exceeds-capacity gets cleared upon scheduling failure",
					checkNNN: &checkNNN{
						podNames:    []string{"pod-exceeds-capacity"},
						expectedNNN: "",
					},
				},
				{
					name: "schedule pod-fits",
					schedulePod: &schedulePod{
						podName:               "pod-fits",
						expectSuccess:         true,
						expectedScheduledNode: "node",
					},
				},
			},
		},
		{
			name: "NNN is ignored on lower priority pod when higher priority pod without NNN is being scheduled",
			scenarios: []scenario{
				{
					name: "create low priority pod with NNN",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("low-priority", lowPriority, "4"),
						},
						nominatedNodeName: "node",
					},
				},
				{
					name: "create high priority pod without NNN",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("high-priority", highPriority, "4"),
						},
					},
				},
				{
					name: "schedule high-priority",
					schedulePod: &schedulePod{
						podName:               "high-priority",
						expectSuccess:         true,
						expectedScheduledNode: "node",
					},
				},
				{
					name: "schedule low-priority",
					schedulePod: &schedulePod{
						podName: "low-priority",
					},
				},
				{
					name: "check NNN in low priority pod gets cleared upon scheduling failure",
					checkNNN: &checkNNN{
						podNames:    []string{"low-priority"},
						expectedNNN: "",
					},
				},
			},
		},
		{
			name: "No preemption, NNN is cleared after binding",
			scenarios: []scenario{
				{
					name: "create pod with NNN",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePod("pod", lowPriority, "4"),
						},
						nominatedNodeName: "node",
					},
				},
				{
					name: "schedule pod (this step also verifies is NNN is cleared after binding, depending on the enabled feature)",
					schedulePod: &schedulePod{
						podName:               "pod",
						expectSuccess:         true,
						expectedScheduledNode: "node",
					},
				},
			},
		},
	}
	// All test cases run on the same node.
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	for _, nnnForExpectationEnabled := range []bool{true, false} {
		for _, clearNNNAfterBindingEnabled := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (NominatedNodeName for expectation: %v, Clearing NNN: %v)", test.name, nnnForExpectationEnabled, clearNNNAfterBindingEnabled), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.NominatedNodeNameForExpectation:       nnnForExpectationEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearNNNAfterBindingEnabled,
						features.SchedulerAsyncPreemption:              true,
					})
					runScenarios(t, test.scenarios, node, 0 /* disable backoff */)
				})
			}
		}
	}
}

// TestPodGroupPreemptionAndNominatedNodeNameScenarios tests setting/clearing NominatedNodeName in scenarios with workload aware preemption.
func TestPodGroupPreemptionAndNominatedNodeNameScenarios(t *testing.T) {
	tests := []struct {
		name      string
		scenarios []scenario
	}{
		{
			name: "full gang podgroup preemption sets NominatedNodeName on all preemptor pods",
			scenarios: []scenario{
				{
					name: "create preemptor gang pg1",
					createPodGroup: &createPodGroup{
						podGroup: makeGangPodGroup("pg1", highPriority, 3),
					},
				},
				{
					name: "create scheduled pods on node1",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("low-1", "node1", lowPriority, "1"),
							makeScheduledPod("low-2", "node1", lowPriority, "1"),
							makeScheduledPod("low-3", "node1", lowPriority, "1"),
						},
					},
				},
				{
					name: "create preemptor pods in gang pg1",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePodGroupPod("high-1", "pg1", highPriority, "1"),
							makePodGroupPod("high-2", "pg1", highPriority, "1"),
							makePodGroupPod("high-3", "pg1", highPriority, "1"),
						},
					},
				},
				{
					name: "schedule the preemptor PodGroup",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:        "pg1",
						podNames:            []string{"high-1", "high-2", "high-3"},
						expectUnschedulable: true,
					},
				},
				{
					name: "check NNN is set in all preemptor pods",
					checkNNN: &checkNNN{
						podNames:    []string{"high-1", "high-2", "high-3"},
						expectedNNN: "node1",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "pg1",
				},
				{
					name: "re-enter pg1 in the scheduling cycle",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:          "pg1",
						podNames:              []string{"high-1", "high-2", "high-3"},
						expectSuccess:         true,
						expectedScheduledNode: "node1",
						validateNNNCleared:    true,
					},
				},
			},
		},
		{
			name: "full basic podgroup preemption sets NominatedNodeName on all preemptor pods",
			scenarios: []scenario{
				{
					name: "create preemptor basic pg1",
					createPodGroup: &createPodGroup{
						podGroup: makeBasicPodGroup("pg1", highPriority),
					},
				},
				{
					name: "create scheduled pods on node1",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("low-1", "node1", lowPriority, "1"),
							makeScheduledPod("low-2", "node1", lowPriority, "1"),
							makeScheduledPod("low-3", "node1", lowPriority, "1"),
						},
					},
				},
				{
					name: "create preemptor pods in basic pg1",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePodGroupPod("high-1", "pg1", highPriority, "1"),
							makePodGroupPod("high-2", "pg1", highPriority, "1"),
							makePodGroupPod("high-3", "pg1", highPriority, "1"),
						},
					},
				},
				{
					// BasicPolicy does not wait for all pods in PreEnqueue, so a live scheduler may
					// start preemption before all pods arrive in activeQ. Waiting for all pods here simplifies
					// testing NNN; preemption behavior is covered by tests in preemption/podgroup/podgrouppreemption_test.go.
					name: "schedule the preemptor PodGroup",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:        "pg1",
						podNames:            []string{"high-1", "high-2", "high-3"},
						expectUnschedulable: true,
					},
				},
				{
					name: "check NNN is set in preemptor",
					checkNNN: &checkNNN{
						podNames:    []string{"high-1", "high-2", "high-3"},
						expectedNNN: "node1",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "pg1",
				},
				{
					name: "schedule the preemptor PodGroup again",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:          "pg1",
						podNames:              []string{"high-1", "high-2", "high-3"},
						expectSuccess:         true,
						expectedScheduledNode: "node1",
						validateNNNCleared:    true,
					},
				},
			},
		},
		{
			name: "composite podgroup preemption sets NominatedNodeName on all preemptor pods",
			scenarios: []scenario{
				{
					name: "create preemptor composite pod group cpg1",
					createCompositePodGroup: &createCompositePodGroup{
						compositePodGroup: makeCompositePodGroup("cpg1", highPriority),
					},
				},
				{
					name: "create child gang pg1",
					createPodGroup: &createPodGroup{
						podGroup: makeChildGangPodGroup("pg1", "cpg1", highPriority, 2),
					},
				},
				{
					name: "create child gang pg2",
					createPodGroup: &createPodGroup{
						podGroup: makeChildGangPodGroup("pg2", "cpg1", highPriority, 1),
					},
				},
				{
					name: "create scheduled pods on node1",
					createPods: &createPods{
						pods: []*v1.Pod{
							makeScheduledPod("low-1", "node1", lowPriority, "1"),
							makeScheduledPod("low-2", "node1", lowPriority, "1"),
							makeScheduledPod("low-3", "node1", lowPriority, "1"),
						},
					},
				},
				{
					name: "create preemptor pods in pg1 and pg2",
					createPods: &createPods{
						pods: []*v1.Pod{
							makePodGroupPod("high-1", "pg1", highPriority, "1"),
							makePodGroupPod("high-2", "pg1", highPriority, "1"),
							makePodGroupPod("high-3", "pg2", highPriority, "1"),
						},
					},
				},
				{
					name: "schedule the preemptor CompositePodGroup",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:        "cpg1",
						podNames:            []string{"high-1", "high-2", "high-3"},
						expectUnschedulable: true,
					},
				},
				{
					name: "check NNN is set in all preemptor pods",
					checkNNN: &checkNNN{
						podNames:    []string{"high-1", "high-2", "high-3"},
						expectedNNN: "node1",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "cpg1",
				},
				{
					name: "re-enter cpg1 in the scheduling cycle",
					schedulePodGroup: &schedulePodGroup{
						podGroupName:          "cpg1",
						podNames:              []string{"high-1", "high-2", "high-3"},
						expectSuccess:         true,
						expectedScheduledNode: "node1",
						validateNNNCleared:    true,
					},
				},
			},
		},
	}
	// All test cases run on the same node.
	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj()
	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, nnnForExpectationEnabled := range []bool{true, false} {
			for _, clearNNNAfterBindingEnabled := range []bool{true, false} {
				for _, test := range tests {
					t.Run(fmt.Sprintf("%s (AsyncPreemption: %v, NominatedNodeName for expectation: %v, Clearing NNN: %v)", test.name, asyncPreemptionEnabled, nnnForExpectationEnabled, clearNNNAfterBindingEnabled), func(t *testing.T) {
						featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
							features.GenericWorkload:                       true,
							features.TopologyAwareWorkloadScheduling:       true,
							features.PodGroupPreemptionPolicy:              true,
							features.CompositePodGroup:                     true,
							features.NominatedNodeNameForExpectation:       nnnForExpectationEnabled,
							features.ClearingNominatedNodeNameAfterBinding: clearNNNAfterBindingEnabled,
							features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						})
						// Set PodMaxBackoff to 1 second to turn on backoff and allow apiCacher to get information about
						// pod NNN. Without this we might have a race between starting binding and update of apiCacher.
						runScenarios(t, test.scenarios, node, 1 /* enable backoff */)
					})
				}
			}
		}
	}
}

func runScenarios(t *testing.T, scenarios []scenario, node *v1.Node, maxPodBackoffSeconds int64) {
	// We need to use a custom preemption plugin to test async preemption behavior
	delayedPreemptionPluginName := "delay-preemption"
	var lock sync.Mutex
	// keyed by the pod or podgroup name
	preemptionDoneChannels := make(map[string]chan struct{})
	defer func() {
		lock.Lock()
		defer lock.Unlock()
		for _, ch := range preemptionDoneChannels {
			close(ch)
		}
	}()
	registry := make(frameworkruntime.Registry)
	var preemptionPlugin *defaultpreemption.DefaultPreemption
	err := registry.Register(delayedPreemptionPluginName, func(c context.Context, r runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		p, err := frameworkruntime.FactoryAdapter(plfeature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate), defaultpreemption.New)(c, &config.DefaultPreemptionArgs{
			// Set default values to pass the validation at the initialization, not related to the test.
			MinCandidateNodesPercentage: 10,
			MinCandidateNodesAbsolute:   100,
		}, fh)
		if err != nil {
			return nil, fmt.Errorf("error creating default preemption plugin: %w", err)
		}

		var ok bool
		preemptionPlugin, ok = p.(*defaultpreemption.DefaultPreemption)
		if !ok {
			return nil, fmt.Errorf("unexpected plugin type %T", p)
		}

		executor, ok := preemptionPlugin.Executor.(*preemption.Executor)
		if !ok {
			return nil, fmt.Errorf("unexpected executor type %T", preemptionPlugin.Executor)
		}
		preemptPodFn := executor.PreemptPod
		executor.PreemptPod = func(ctx context.Context, c fwk.PreemptionCandidate, preemptor preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error) {
			// block the preemption goroutine to complete until the test case allows it to proceed.
			lock.Lock()
			ch, ok := preemptionDoneChannels[preemptor.GetName()]
			lock.Unlock()
			if ok {
				<-ch
			}
			return preemptPodFn(ctx, c, preemptor, victim, pluginName)
		}

		return preemptionPlugin, nil
	})
	if err != nil {
		t.Fatalf("Error registering a filter: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				MultiPoint: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: delayedPreemptionPluginName},
					},
					Disabled: []configv1.Plugin{
						{Name: names.DefaultPreemption},
					},
				},
			},
		}},
	})

	// It initializes the scheduler, but doesn't start.
	// We manually trigger the scheduling cycle.
	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "preemption", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodMaxBackoffSeconds(maxPodBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(0),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	cs := testCtx.ClientSet

	if preemptionPlugin == nil {
		t.Fatalf("the preemption plugin should be initialized")
	}

	logger, _ := ktesting.NewTestContext(t)
	if testCtx.Scheduler.APIDispatcher != nil {
		testCtx.Scheduler.APIDispatcher.Run(logger)
		defer testCtx.Scheduler.APIDispatcher.Close()
	}
	testCtx.Scheduler.SchedulingQueue.Run(logger)
	defer testCtx.Scheduler.SchedulingQueue.Close()

	createdPods := []*v1.Pod{}
	defer func() {
		testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if _, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create an initial Node %q: %v", node.Name, err)
	}
	defer func() {
		if err := cs.CoreV1().Nodes().Delete(ctx, node.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to delete the Node %q: %v", node.Name, err)
		}
	}()

	for _, scenario := range scenarios {
		t.Logf("Running scenario: %s", scenario.name)
		switch {
		case scenario.createNode != "":
			newNode := st.MakeNode().Name(scenario.createNode).Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
			if _, err := cs.CoreV1().Nodes().Create(ctx, newNode, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create an initial Node %q: %v", newNode.Name, err)
			}
			defer func() {
				if err := cs.CoreV1().Nodes().Delete(ctx, newNode.Name, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("Failed to delete the Node %q: %v", newNode.Name, err)
				}
			}()
		case scenario.createPodGroup != nil:
			pg := scenario.createPodGroup.podGroup.DeepCopy()
			pg.Namespace = testCtx.NS.Name
			if _, err := cs.SchedulingV1beta1().PodGroups(testCtx.NS.Name).Create(ctx, pg, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create PodGroup %q: %v", pg.Name, err)
			}
		case scenario.createCompositePodGroup != nil:
			cpg := scenario.createCompositePodGroup.compositePodGroup.DeepCopy()
			cpg.Namespace = testCtx.NS.Name
			if _, err := cs.SchedulingV1alpha3().CompositePodGroups(testCtx.NS.Name).Create(ctx, cpg, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create CompositePodGroup %q: %v", cpg.Name, err)
			}
		case scenario.createPods != nil:
			for _, p := range scenario.createPods.pods {
				pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(ctx, p, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create a Pod %q: %v", p.Name, err)
				}

				if scenario.createPods.nominatedNodeName != "" {
					patch := []byte(fmt.Sprintf(`{"status":{"nominatedNodeName":"%s"}}`, scenario.createPods.nominatedNodeName))
					pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Patch(ctx, pod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "status")
					if err != nil {
						t.Fatalf("update pod %s status with NNN: %v", p.Name, err)
					}
					// Wait until the scheduler picks up the NNN set on the pod.
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
						nominatedPods := testCtx.Scheduler.SchedulingQueue.NominatedPodsForNode(klog.FromContext(ctx), scenario.createPods.nominatedNodeName)
						if contains(nominatedPods, pod.Name) {
							return true, nil
						}
						return false, nil
					}); err != nil {
						t.Fatalf("scheduler does not see NNN set on pod %s: %v", p.Name, err)
					}
				}
				createdPods = append(createdPods, pod)
			}
		case scenario.schedulePod != nil:
			lastFailure := ""
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				if len(testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()) == 0 {
					lastFailure = fmt.Sprintf("Expected the pod %s to be scheduled, but no pod arrives at the activeQ", scenario.schedulePod.podName)
					return false, nil
				}

				if testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()[0].Name != scenario.schedulePod.podName {
					// need to wait more because maybe the queue will get another Pod that higher priority than the current top pod.
					lastFailure = fmt.Sprintf("The pod %s is expected to be scheduled, but the top Pod is %s", scenario.schedulePod.podName, testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()[0].Name)
					return false, nil
				}

				return true, nil
			}); err != nil {
				t.Fatal(lastFailure)
			}

			lock.Lock()
			preemptionDoneChannels[scenario.schedulePod.podName] = make(chan struct{})
			lock.Unlock()
			testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

			if scenario.schedulePod.expectSuccess {
				lastFailure := ""
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					pod, err2 := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, scenario.schedulePod.podName, metav1.GetOptions{})
					if err2 != nil {
						// This could be a connection error so we want to retry.
						return false, nil
					}
					if pod.Spec.NodeName == "" {
						// Pod is not scheduled yet.
						return false, nil
					}
					if scenario.schedulePod.expectedScheduledNode != pod.Spec.NodeName {
						lastFailure = fmt.Sprintf("Expected pod %s to be scheduled on node %s but got %v", scenario.schedulePod.podName, scenario.schedulePod.expectedScheduledNode, pod.Spec.NodeName)
						return false, err2
					}
					if utilfeature.DefaultFeatureGate.Enabled(features.ClearingNominatedNodeNameAfterBinding) && pod.Status.NominatedNodeName != "" {
						lastFailure = fmt.Sprintf("Expected pod %s to have NNN cleared after binding but got \"%v\"", scenario.schedulePod.podName, pod.Status.NominatedNodeName)
						return false, err2
					}
					return true, nil
				}); err != nil {
					t.Fatal(lastFailure)
				}

			} else if scenario.schedulePod.expectUnschedulable {
				// Wait some time for the scheduling operation to finish and move the pod to unschedulable or backoff.
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, 2*time.Second, false, func(ctx context.Context) (bool, error) {
					if podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.schedulePod.podName) {
						return true, nil
					}
					return false, nil
				}); err != nil {
					t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", scenario.schedulePod.podName)
				}
			}
		case scenario.schedulePodGroup != nil:
			lastFailure := ""
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				podsInActiveQ := testCtx.Scheduler.SchedulingQueue.PodsInActiveQ()
				if len(podsInActiveQ) < len(scenario.schedulePodGroup.podNames) {
					lastFailure = fmt.Sprintf("Expected pods %v to be at the top of the activeQ, but only %d pods are in the activeQ", scenario.schedulePodGroup.podNames, len(podsInActiveQ))
					return false, nil
				}
				for _, pod := range podsInActiveQ[:len(scenario.schedulePodGroup.podNames)] {
					if !slices.Contains(scenario.schedulePodGroup.podNames, pod.Name) {
						// need to wait more because maybe the queue will get another Pod that higher priority than the current top pod.
						lastFailure = fmt.Sprintf("Pods %v are expected to be found in activeQ, but one of the top Pods is %s", scenario.schedulePodGroup.podNames, pod.Name)
						return false, nil
					}
				}

				return true, nil
			}); err != nil {
				t.Fatal(lastFailure)
			}

			// If async preemption is disabled, there is no need for blocking until preemption is completed.
			if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption) {
				lock.Lock()
				preemptionDoneChannels[scenario.schedulePodGroup.podGroupName] = make(chan struct{})
				lock.Unlock()
			}
			testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

			for _, podName := range scenario.schedulePodGroup.podNames {
				if scenario.schedulePodGroup.expectSuccess {
					lastFailure := ""
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
						pod, err2 := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, podName, metav1.GetOptions{})
						if err2 != nil {
							// This could be a connection error so we want to retry.
							return false, nil
						}
						if pod.Spec.NodeName == "" {
							// Pod is not scheduled yet.
							return false, nil
						}
						if scenario.schedulePodGroup.expectedScheduledNode != pod.Spec.NodeName {
							lastFailure = fmt.Sprintf("Expected pod %s to be scheduled on node %s but got %v", podName, scenario.schedulePodGroup.expectedScheduledNode, pod.Spec.NodeName)
							return false, err2
						}
						if scenario.schedulePodGroup.validateNNNCleared {
							if utilfeature.DefaultFeatureGate.Enabled(features.ClearingNominatedNodeNameAfterBinding) {
								if pod.Status.NominatedNodeName != "" {
									lastFailure = fmt.Sprintf("Expected pod %s to have NNN cleared after binding but got %q", podName, pod.Status.NominatedNodeName)
									return false, err2
								}
							} else {
								if pod.Status.NominatedNodeName == "" {
									lastFailure = fmt.Sprintf("Expected pod %s to not have NNN cleared after binding, but it was cleared", podName)
									return false, err2
								}
							}
						}
						return true, nil
					}); err != nil {
						t.Fatal(lastFailure)
					}

				} else if scenario.schedulePodGroup.expectUnschedulable && utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption) {
					// Wait some time for the scheduling operation to finish and move the pod to unschedulable or backoff.
					// This is not applicable when async preemption is disabled, because the pod may return to activeQ.
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, 2*time.Second, false, func(ctx context.Context) (bool, error) {
						if podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, podName) {
							return true, nil
						}
						return false, nil
					}); err != nil {
						t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", podName)
					}
				}
			}
		case scenario.completePreemption != "":
			if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption) {
				lock.Lock()
				if _, ok := preemptionDoneChannels[scenario.completePreemption]; !ok {
					t.Fatalf("The preemptor %q is not running preemption", scenario.completePreemption)
				}

				close(preemptionDoneChannels[scenario.completePreemption])
				delete(preemptionDoneChannels, scenario.completePreemption)
				lock.Unlock()
			}
		case scenario.checkNNN != nil:
			for _, podName := range scenario.checkNNN.podNames {
				lastFailure := ""
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
					pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, podName, metav1.GetOptions{})

					if err != nil {
						lastFailure = fmt.Sprintf("Cannot retrieve pod %v", podName)
						return false, err
					}
					if scenario.checkNNN.expectedNNN != pod.Status.NominatedNodeName {
						lastFailure = fmt.Sprintf("Expected .status.nominatedNodeName %v for pod \"%v\" but got \"%v\"", scenario.checkNNN.expectedNNN, podName, pod.Status.NominatedNodeName)
						return false, nil
					}

					return true, nil
				}); err != nil {
					t.Fatal(lastFailure, err)
				}
			}
		}
	}
}

func podInUnschedulablePodPool(t *testing.T, queue queue.SchedulingQueue, podName string) bool {
	t.Helper()
	// First, look for the pod in the activeQ.
	for _, pod := range queue.PodsInActiveQ() {
		if pod.Name == podName {
			return false
		}
	}

	pending, _ := queue.PendingPods()
	for _, pod := range pending {
		if pod.Name == podName {
			return true
		}
	}

	return false
}

func contains(pods []fwk.PodInfo, podName string) bool {
	for _, p := range pods {
		if podName == p.GetPod().GetName() {
			return true
		}
	}
	return false
}

type mockPreBindPlugin struct{}

var _ fwk.PreBindPlugin = &mockPreBindPlugin{}

func (p *mockPreBindPlugin) Name() string {
	return "mockPreBindPlugin"
}

func (p *mockPreBindPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	return nil
}

func (p *mockPreBindPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return &fwk.PreBindPreFlightResult{AllowParallel: false}, nil
}

type mockBindPlugin struct {
	bindFn func() *fwk.Status
}

var _ fwk.BindPlugin = &mockBindPlugin{}

func (p *mockBindPlugin) Name() string {
	return "mockBindPlugin"
}

func (p *mockBindPlugin) Bind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	return p.bindFn()
}

type mockQueueSortPlugin struct {
	t     *testing.T
	order map[string]int
}

var _ fwk.QueueSortPlugin = &mockQueueSortPlugin{}

func (p *mockQueueSortPlugin) Name() string {
	return "mockQueueSortPlugin"
}

func (p *mockQueueSortPlugin) Less(entity1, entity2 fwk.QueuedEntityInfo) bool {
	name1 := entity1.(interface{ GetName() string }).GetName()
	name2 := entity2.(interface{ GetName() string }).GetName()
	o1, ok1 := p.order[name1]
	if !ok1 {
		p.t.Errorf("order doesn't contain pod with the specified name: %s", name1)
	}
	o2, ok2 := p.order[name2]
	if !ok2 {
		p.t.Errorf("order doesn't contain pod with the specified name: %s", name2)
	}
	return o1 < o2
}

// TestSchedulerRestartWithNominatedNode checks that NNN properly reserves the pod's node despite scheduler restart during binding cycle.
func TestSchedulerRestartWithNominatedNode(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, true)
	testContext := testutils.InitTestAPIServer(t, "nnn-test", nil)
	ctx, cancel := context.WithCancel(testContext.Ctx)
	cs := testContext.ClientSet
	defer cancel()

	// nodePreferred can only hold a single pod
	nodePreferred := st.MakeNode().Name("node-preferred").Capacity(map[v1.ResourceName]string{v1.ResourcePods: "1"}).Obj()
	// nodeOther has taint with PreferNoSchedule
	nodeOther := st.MakeNode().Name("node-other").Taints([]v1.Taint{{Key: "foo", Effect: v1.TaintEffectPreferNoSchedule}}).Obj()

	for _, n := range []*v1.Node{nodePreferred, nodeOther} {
		if _, err := testutils.CreateNode(cs, n); err != nil {
			t.Fatalf("Failed to create node %s: %v", n.Name, err)
		}
	}

	createPod := func(name string) *v1.Pod {
		t.Helper()
		pod, err := testutils.CreatePausePod(cs, testutils.InitPausePod(&testutils.PausePodConfig{
			Name: name, Namespace: testContext.NS.Name}))
		if err != nil {
			t.Fatalf("Failed to create %s: %v", name, err)
		}
		return pod
	}

	podA := createPod("pod-a")

	// Ensures desired pod order without affecting priorities, which could impact NNN logic.
	mockQueueSort := &mockQueueSortPlugin{t: t, order: map[string]int{}}
	// Ensures NNN is set
	mockPreBind := &mockPreBindPlugin{}

	initSched := func(additionalPlugins ...fwk.Plugin) (*testutils.TestContext, testutils.ShutdownFunc) {
		var pls []fwk.Plugin
		pls = append(pls, mockQueueSort, mockPreBind)
		pls = append(pls, additionalPlugins...)
		registry, prof := schedulerutils.InitRegistryAndConfig(t, nil, pls...)
		testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 0,
			false, /* do not run scheduler immediately after init */
			scheduler.WithProfiles(prof),
			scheduler.WithFrameworkOutOfTreeRegistry(registry),
		)
		// Ensure both nodes are in the cache before proceeding
		if err := testutils.WaitForNodesInCache(testCtx.Ctx, testCtx.Scheduler, 2); err != nil {
			t.Fatalf("Failed to wait for nodes in cache: %v", err)
		}
		return testCtx, teardown
	}

	// To allow triggering scheduler restart in binding phase, we need to initialize the scheduler
	// and pass the testCtx.SchedulerCloseFn to the bind plugin before calling Scheduler.Run.
	mockBind := &mockBindPlugin{}
	testCtx, _ := initSched(mockBind)

	// This will fail the binding cycle and stop the scheduler
	// Scheduler should set NNN for podA before this step.
	mockBind.bindFn = func() *fwk.Status {
		testCtx.SchedulerCloseFn()
		return fwk.NewStatus(fwk.Error, "simulated bind failure")
	}
	mockQueueSort.order[podA.Name] = 1

	// Start scheduler
	go testCtx.Scheduler.Run(testCtx.SchedulerCtx)

	// We expect the scheduler will stop due to SchedulerCloseFn being called in bind
	<-testCtx.SchedulerCtx.Done()

	podA, err := testutils.GetPod(cs, podA.Name, podA.Namespace)
	if err != nil {
		t.Fatalf("Failed to get pod %s: %v", podA.Name, err)
	}
	if podA.Status.NominatedNodeName != nodePreferred.Name {
		t.Errorf("Unexpected NominatedNodeName after scheduler abort for pod %s, got %s, want %s", podA.Name, podA.Status.NominatedNodeName, nodePreferred.Name)
	}
	if podA.Spec.NodeName != "" {
		t.Errorf("Unexpected NodeName after scheduler abort for pod %s, got %s, want unset", podA.Name, podA.Spec.NodeName)
	}

	podB := createPod("pod-b")

	// Make sure podB is evaluated before podA after scheduler restart
	// The scheduler should see podA's nominated node and schedule podB elsewhere
	mockQueueSort.order[podB.Name] = mockQueueSort.order[podA.Name] - 1

	// This time use the default bind plugin
	testCtx, teardown := initSched()
	defer teardown()

	// Start scheduler again
	go testCtx.Scheduler.Run(testCtx.SchedulerCtx)

	for _, pod := range []*v1.Pod{podA, podB} {
		if err := testutils.WaitForPodToSchedule(ctx, cs, pod); err != nil {
			t.Errorf("Failed waiting for pod %s: %v", pod.Name, err)
		}
	}

	checkNode := func(pod *v1.Pod, expected string) {
		t.Helper()
		p, _ := testutils.GetPod(cs, pod.Name, pod.Namespace)
		if p.Spec.NodeName != expected {
			t.Errorf("%s scheduled on %s, wanted %s", pod.Name, p.Spec.NodeName, expected)
		}
	}
	checkNode(podA, nodePreferred.Name)
	checkNode(podB, nodeOther.Name)
}
