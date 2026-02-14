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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
	"k8s.io/kubernetes/test/utils/ktesting"
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

// TestPreemptionAndNominatedNodeNameScenarios tests setting/clearing NominatedNodeName in scenarios with preemption.
func TestPreemptionAndNominatedNodeNameScenarios(t *testing.T) {
	type createPod struct {
		pod               *v1.Pod
		nominatedNodeName string
		// count is the number of times the pod should be created by this action.
		// i.e., if you use it, you have to use GenerateName.
		// By default, it's 1.
		count *int
	}

	type schedulePod struct {
		podName               string
		expectSuccess         bool
		expectUnschedulable   bool
		expectedScheduledNode string
	}

	type checkNNN struct {
		podName     string
		expectedNNN string
	}

	type scenario struct {
		// name is this step's name, just for the debugging purpose.
		name string

		// Only one of the following actions should be set.

		// createPod creates a Pod.
		createPod *createPod
		// createNode creates an additional Node.
		createNode string
		// schedulePod schedules one Pod that is at the top of the activeQ.
		// You should give a Pod name that is supposed to be scheduled.
		schedulePod *schedulePod
		// completePreemption completes the preemption that is currently on-going.
		// You should give a Pod name.
		completePreemption string
		// checkNNN checks that NominatedNodeName is set as expected.
		checkNNN *checkNNN
	}

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
					createPod: &createPod{
						pod: st.MakePod().Name("victim").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
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
						podName:     "preemptor",
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
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(2),
					},
				},
				{
					name: "create pod on node2",
					createPod: &createPod{
						pod: st.MakePod().Name("victim-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Node("node2").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "create a preemptor Pod with NNN set to node",
					createPod: &createPod{
						pod:               st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
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
						podName:     "preemptor",
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
					createPod: &createPod{
						pod:               st.MakePod().Name("pod-exceeds-capacity").Req(map[v1.ResourceName]string{v1.ResourceCPU: "5"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						nominatedNodeName: "node",
					},
				},
				{
					name: "create pod without NNN that fits on the node",
					createPod: &createPod{
						pod: st.MakePod().Name("pod-fits").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
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
						podName:     "pod-exceeds-capacity",
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
					createPod: &createPod{
						pod:               st.MakePod().Name("low-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						nominatedNodeName: "node",
					},
				},
				{
					name: "create high priority pod without NNN",
					createPod: &createPod{
						pod: st.MakePod().Name("high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
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
						podName:     "low-priority",
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
					createPod: &createPod{
						pod:               st.MakePod().Name("pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
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

					// We need to use a custom preemption plugin to test async preemption behavior
					delayedPreemptionPluginName := "delay-preemption"
					var lock sync.Mutex
					// keyed by the pod name
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
						p, err := frameworkruntime.FactoryAdapter(plfeature.Features{EnableAsyncPreemption: true}, defaultpreemption.New)(c, &config.DefaultPreemptionArgs{
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

						preemptPodFn := preemptionPlugin.Evaluator.PreemptPod
						preemptionPlugin.Evaluator.PreemptPod = func(ctx context.Context, c preemption.Candidate, preemptor preemption.Preemptor, victim *v1.Pod, pluginName string) error {
							// block the preemption goroutine to complete until the test case allows it to proceed.
							preemptorName := preemptor.Members()[0].Name
							lock.Lock()
							ch, ok := preemptionDoneChannels[preemptorName]
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
						// disable backoff
						scheduler.WithPodMaxBackoffSeconds(0),
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
					defer testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)

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

					for _, scenario := range test.scenarios {
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
						case scenario.createPod != nil:
							if scenario.createPod.count == nil {
								scenario.createPod.count = ptr.To(1)
							}

							for i := 0; i < *scenario.createPod.count; i++ {
								pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(ctx, scenario.createPod.pod, metav1.CreateOptions{})
								if err != nil {
									t.Fatalf("Failed to create a Pod %q: %v", scenario.createPod.pod.Name, err)
								}

								if scenario.createPod.nominatedNodeName != "" {
									patch := []byte(fmt.Sprintf(`{"status":{"nominatedNodeName":"%s"}}`, scenario.createPod.nominatedNodeName))
									pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Patch(ctx, pod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "status")
									if err != nil {
										t.Fatalf("update pod %s status with NNN: %v", scenario.createPod.pod.Name, err)
									}
									// Wait until the scheduler picks up the NNN set on the pod.
									if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
										nominatedPods := testCtx.Scheduler.SchedulingQueue.NominatedPodsForNode(scenario.createPod.nominatedNodeName)
										if contains(nominatedPods, pod.Name) {
											return true, nil
										}
										return false, nil
									}); err != nil {
										t.Fatalf("scheduler does not see NNN set on pod %s: %v", scenario.createPod.pod.Name, err)
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
									if clearNNNAfterBindingEnabled && pod.Status.NominatedNodeName != "" {
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
						case scenario.completePreemption != "":
							lock.Lock()
							if _, ok := preemptionDoneChannels[scenario.completePreemption]; !ok {
								t.Fatalf("The preemptor Pod %q is not running preemption", scenario.completePreemption)
							}

							close(preemptionDoneChannels[scenario.completePreemption])
							delete(preemptionDoneChannels, scenario.completePreemption)
							lock.Unlock()
						case scenario.checkNNN != nil:
							lastFailure := ""
							if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
								pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, scenario.checkNNN.podName, metav1.GetOptions{})

								if err != nil {
									lastFailure = fmt.Sprintf("Cannot retrieve pod %v", scenario.checkNNN.podName)
									return false, err
								}
								if scenario.checkNNN.expectedNNN != pod.Status.NominatedNodeName {
									lastFailure = fmt.Sprintf("Expected .status.nominatedNodeName %v for pod \"%v\" but got \"%v\"", scenario.checkNNN.expectedNNN, scenario.checkNNN.podName, pod.Status.NominatedNodeName)
									return false, nil
								}

								return true, nil
							}); err != nil {
								t.Fatal(lastFailure, err)
							}
						}
					}
				})
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
