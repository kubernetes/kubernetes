/*
Copyright The Kubernetes Authors.

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

package preemption

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const podBlockedInBindingName = "pod-blocked-in-binding"
const reservingPodName = "reserving-pod"

type createPodGroup struct {
	podGroup *schedulingapi.PodGroup
}

type createPod struct {
	pod *v1.Pod
	// count is the number of times the pod should be created by this action.
	// i.e., if you use it, you have to use GenerateName.
	// By default, it's 1.
	count *int
}

type schedulePod struct {
	podName             string
	expectSuccess       bool
	expectUnschedulable bool
}

type scenario struct {
	// name is this step's name, just for the debugging purpose.
	name string

	// Only one of the following actions should be set.

	createPodGroup *createPodGroup
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
	// podGatedInQueue checks if the given Pod is in the scheduling queue and gated by the preemption.
	// You should give a Pod name.
	podGatedInQueue string
	// podRunningPreemption checks if the given Pod is running preemption.
	// You should give a Pod index representing the order of Pod creation.
	// e.g., if you want to check the Pod created first in the test case, you should give 0.
	podRunningPreemption *int
	// activatePod moves the pod from unschedulable to active or backoff.
	// The value is the name of the pod to activate.
	activatePod string
	// resumeBind resumes the binding operation that keeps the pod blocked.
	// Note: The pod will only become blocked in the first place, if pod name matches string defined in podBlockedInBinding.
	resumeBind bool
	// verifyPodInUnschedulable waits for some time and confirms that the given pod is in the unschedulable pool.
	// The value is the name of the checked pod.
	verifyPodInUnschedulable string
}

type asyncPreemptionTest struct {
	name                  string
	initialBackoffSeconds int64
	maxBackoffSeconds     int64
	scenarios             []scenario
}

func runAsyncPreemptionScenarios(t *testing.T, tests []asyncPreemptionTest, enableWAP bool) {

	// All test cases have the same node.
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuresOverrides := featuregatetesting.FeatureOverrides{
				features.SchedulerAsyncAPICalls:   true,
				features.SchedulerAsyncPreemption: true,
			}
			if enableWAP {
				featuresOverrides[features.GangScheduling] = true
				featuresOverrides[features.GenericWorkload] = true
				featuresOverrides[features.WorkloadAwarePreemption] = true
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuresOverrides)

			// We need to use a custom preemption plugin to test async preemption behavior
			delayedPreemptionPluginName := "delay-preemption"
			var lock sync.Mutex
			// keyed by the pod name
			preemptionDoneChannels := make(map[string]chan struct{})
			defer func() {
				lock.Lock()
				defer lock.Unlock()
				for _, ch := range preemptionDoneChannels {
					select {
					case <-ch:
					default:
						close(ch)
					}
				}
			}()
			registry := make(frameworkruntime.Registry)
			var preemptionPlugin *defaultpreemption.DefaultPreemption
			err := registry.Register(delayedPreemptionPluginName, func(c context.Context, r runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				p, err := frameworkruntime.FactoryAdapter(plfeature.Features{EnableAsyncPreemption: true, EnableWorkloadAwarePreemption: enableWAP, EnableGenericWorkload: enableWAP, EnableGangScheduling: enableWAP}, defaultpreemption.New)(c, &config.DefaultPreemptionArgs{
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

				preemptPodFn := preemptionPlugin.Executor.PreemptPod
				preemptionPlugin.Executor.PreemptPod = func(ctx context.Context, c preemption.Candidate, preemptor preemption.ExecutorPreemptor, victim *v1.Pod, pluginName string) error {
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

			// Register fake bind plugin that will block on binding for the specified pod name, until it receives a resume signal via the blockBindingChannel.
			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			blockingBindPluginName := "blockingBindPlugin"
			err = registry.Register(blockingBindPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				db, err := defaultbinder.New(ctx, o, fh)
				if err != nil {
					t.Fatalf("Error creating a default binder plugin: %v", err)
				}
				var bindPlugin = blockingBindPlugin{
					name:                blockingBindPluginName,
					nameOfPodToBlock:    podBlockedInBindingName,
					realPlugin:          db.(fwk.BindPlugin),
					blockBindingChannel: blockBindingChannel,
				}
				return &bindPlugin, nil
			})
			if err != nil {
				t.Fatalf("Error registering a bind plugin: %v", err)
			}

			// Register fake plugin that will reserve some fake resources for one pod.
			// This could be used to check scheduler's behavior when the victim has to unreserve these resources to let the preemptor schedule.
			reservingPluginName := "reservingPlugin"
			err = registry.Register(reservingPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
				return &reservingPlugin{
					name:               reservingPluginName,
					nameOfPodToReserve: reservingPodName,
				}, nil
			})
			if err != nil {
				t.Fatalf("Error registering a reserving plugin: %v", err)
			}

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: ptr.To(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						MultiPoint: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: blockingBindPluginName},
								{Name: delayedPreemptionPluginName},
								{Name: reservingPluginName},
							},
							Disabled: []configv1.Plugin{
								{Name: names.DefaultPreemption},
								{Name: names.DefaultBinder},
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
				scheduler.WithPodMaxBackoffSeconds(test.maxBackoffSeconds),
				scheduler.WithPodInitialBackoffSeconds(test.initialBackoffSeconds),
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

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, true)

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
				case scenario.createPodGroup != nil:
					_, err := cs.SchedulingV1alpha2().PodGroups(testCtx.NS.Name).Create(ctx, scenario.createPodGroup.podGroup, metav1.CreateOptions{})
					if err != nil && !apierrors.IsAlreadyExists(err) {
						t.Fatalf("Failed to create a PodGroup %q: %v", scenario.createPodGroup.podGroup.Name, err)
					}
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, 2*time.Second, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
						_, err := testCtx.InformerFactory.Scheduling().V1alpha2().PodGroups().Lister().PodGroups(testCtx.NS.Name).Get(scenario.createPodGroup.podGroup.Name)
						if err != nil {
							if apierrors.IsNotFound(err) {
								return false, nil
							}
							return false, err
						}
						return true, nil
					}); err != nil {
						t.Fatalf("Failed to wait for PodGroup %q to sync: %v", scenario.createPodGroup.podGroup.Name, err)
					}
				case scenario.createPod != nil:
					if scenario.createPod.count == nil {
						scenario.createPod.count = ptr.To(1)
					}

					for i := 0; i < *scenario.createPod.count; i++ {
						pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(ctx, scenario.createPod.pod, metav1.CreateOptions{})
						if err != nil {
							t.Fatalf("Failed to create a Pod %q: %v", pod.Name, err)
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
					ch := make(chan struct{})
					preemptionDoneChannels[scenario.schedulePod.podName] = ch
					if pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, scenario.schedulePod.podName, metav1.GetOptions{}); err == nil && pod.Spec.SchedulingGroup != nil {
						preemptionDoneChannels[*pod.Spec.SchedulingGroup.PodGroupName] = ch
					}
					lock.Unlock()
					testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

					if scenario.schedulePod.expectSuccess {
						if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, testCtx.NS.Name, scenario.schedulePod.podName)); err != nil {
							t.Fatalf("Expected the pod %s to be scheduled", scenario.schedulePod.podName)
						}
					} else if scenario.schedulePod.expectUnschedulable {
						if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
							return podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.schedulePod.podName), nil
						}); err != nil {
							t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", scenario.schedulePod.podName)
						}
					}
				case scenario.activatePod != "":
					pod := unschedulablePod(t, testCtx.Scheduler.SchedulingQueue, scenario.activatePod)
					if pod == nil {
						t.Fatalf("Expected the pod %s to be in unschedulable queue before activation phase", scenario.activatePod)
					}
					m := map[string]*v1.Pod{scenario.activatePod: pod}
					testCtx.Scheduler.SchedulingQueue.Activate(logger, m)
				case scenario.completePreemption != "":
					lock.Lock()
					if _, ok := preemptionDoneChannels[scenario.completePreemption]; !ok {
						t.Fatalf("The preemptor Pod %q is not running preemption", scenario.completePreemption)
					}

					close(preemptionDoneChannels[scenario.completePreemption])
					delete(preemptionDoneChannels, scenario.completePreemption)
					lock.Unlock()
				case scenario.podGatedInQueue != "":
					// make sure the Pod is in the queue in the first place.
					if !podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.podGatedInQueue) {
						t.Fatalf("Expected the pod %s to be in the queue", scenario.podGatedInQueue)
					}

					// Make sure this Pod is gated by the preemption at PreEnqueue extension point
					// by activating the Pod and see if it's still in the unsched pod pool.
					testCtx.Scheduler.SchedulingQueue.Activate(logger, map[string]*v1.Pod{scenario.podGatedInQueue: st.MakePod().Namespace(testCtx.NS.Name).Name(scenario.podGatedInQueue).Obj()})
					if !podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.podGatedInQueue) {
						t.Fatalf("Expected the pod %s to be in the queue even after the activation", scenario.podGatedInQueue)
					}
				case scenario.podRunningPreemption != nil:
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
						pod := createdPods[*scenario.podRunningPreemption]
						if pod.Spec.SchedulingGroup != nil && pod.Spec.SchedulingGroup.PodGroupName != nil {
							pg, err := testCtx.InformerFactory.Scheduling().V1alpha2().PodGroups().Lister().PodGroups(pod.Namespace).Get(*pod.Spec.SchedulingGroup.PodGroupName)
							if err == nil {
								return preemptionPlugin.Executor.IsPodGroupRunningPreemption(pg.UID), nil
							}
						}
						return preemptionPlugin.Executor.IsPodRunningPreemption(pod.GetUID()), nil
					}); err != nil {
						t.Fatalf("Expected the pod %s to be running preemption", createdPods[*scenario.podRunningPreemption].Name)
					}
				case scenario.resumeBind:
					blockBindingChannel <- struct{}{}
				case scenario.verifyPodInUnschedulable != "":
					if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 200*time.Millisecond, false, func(ctx context.Context) (bool, error) {
						if !podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.verifyPodInUnschedulable) {
							return false, fmt.Errorf("expected the pod %s to remain in the unschedulable queue after the scheduling attempt", scenario.verifyPodInUnschedulable)
						}
						// Continue polling to confirm that pod remains in unschedulable queue and does not get activated.
						return false, nil
					}); err != nil && !errors.Is(err, context.DeadlineExceeded) {
						// If timeout was reached or context was cancelled without finding that vanished from unschedulable, it means the state is as expected.
						// If a different error occurred, it means that the pod got unexpectedly activated, or something else went wrong.
						t.Fatalf("Error in scenario verifyPodInUnschedulable: %v", err)
					}
				}
			}
		})
	}
}

// podInUnschedulablePodPool checks if the given Pod is in the unschedulable pod pool.
func podInUnschedulablePodPool(t *testing.T, queue queue.SchedulingQueue, podName string) bool {
	t.Helper()
	// First, look for the pod in the activeQ.
	for _, pod := range queue.PodsInActiveQ() {
		if pod.Name == podName {
			return false
		}
	}

	pendingPods, _ := queue.PendingPods()
	for _, pod := range pendingPods {
		if pod.Name == podName {
			return true
		}
	}

	return false
}

// unschedulablePod checks if the given Pod is in the unschedulable queue and returns it.
func unschedulablePod(t *testing.T, queue queue.SchedulingQueue, podName string) *v1.Pod {
	t.Helper()
	unschedPods := queue.UnschedulablePods()
	for _, pod := range unschedPods {
		if pod.Name == podName {
			return pod
		}
	}
	return nil
}

// blockingBindPlugin is a fake plugin that simulates a long binding operation.
// Underneath it calls realPlugin.Bind(), after receiving a signal that binding can be unblocked.
type blockingBindPlugin struct {
	name                string
	nameOfPodToBlock    string
	realPlugin          fwk.BindPlugin
	blockBindingChannel chan struct{}
}

func (bp *blockingBindPlugin) Name() string {
	return bp.name
}

func (bp *blockingBindPlugin) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if strings.Contains(p.Name, bp.nameOfPodToBlock) {
		// block the bind goroutine to complete until the test case allows it to proceed.
		select {
		case <-bp.blockBindingChannel:
		case <-ctx.Done():
		}
	}
	return bp.realPlugin.Bind(ctx, state, p, nodeName)
}

var _ fwk.BindPlugin = &blockingBindPlugin{}

// reservingPlugin is a fake plugin that reserves some resource in memory for nameOfPodToReserve pod.
// Other pods won't be scheduled, unless the resources are unreserved.
type reservingPlugin struct {
	lock               sync.Mutex
	name               string
	nameOfPodToReserve string
	reserved           bool
}

func (rp *reservingPlugin) Name() string {
	return rp.name
}

func (rp *reservingPlugin) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		// Plugin will wake up the pod on any Pod/Delete event.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete}},
	}, nil
}

const reservingPluginStateKey = "PreFilterReserving"

type reservingPluginState struct {
	reserved bool
}

func (s reservingPluginState) Clone() fwk.StateData {
	return reservingPluginState{
		reserved: s.reserved,
	}
}

func (rp *reservingPlugin) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	rp.lock.Lock()
	state.Write(reservingPluginStateKey, reservingPluginState{reserved: rp.reserved})
	rp.lock.Unlock()
	return nil, nil
}

func (rp *reservingPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	s, err := state.Read(reservingPluginStateKey)
	if err != nil {
		return fwk.AsStatus(err)
	}
	if s.(reservingPluginState).reserved {
		return fwk.NewStatus(fwk.Unschedulable, "resources are reserved")
	}
	return nil
}

func (rp *reservingPlugin) Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if strings.Contains(p.Name, rp.nameOfPodToReserve) {
		rp.lock.Lock()
		rp.reserved = true
		rp.lock.Unlock()
	}
	return nil
}

func (rp *reservingPlugin) Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if strings.Contains(p.Name, rp.nameOfPodToReserve) {
		rp.lock.Lock()
		rp.reserved = false
		rp.lock.Unlock()
	}
}

func (rp *reservingPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return rp
}

func (rp *reservingPlugin) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if strings.Contains(podInfoToAdd.GetPod().Name, rp.nameOfPodToReserve) {
		state.Write(reservingPluginStateKey, reservingPluginState{reserved: true})
	}
	return nil
}

func (rp *reservingPlugin) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if strings.Contains(podInfoToRemove.GetPod().Name, rp.nameOfPodToReserve) {
		state.Write(reservingPluginStateKey, reservingPluginState{reserved: false})
	}
	return nil
}

var _ fwk.PreFilterPlugin = &reservingPlugin{}
var _ fwk.FilterPlugin = &reservingPlugin{}
var _ fwk.PreFilterExtensions = &reservingPlugin{}
var _ fwk.ReservePlugin = &reservingPlugin{}

type blockedPod struct {
	blocked chan struct{}
}

// blockingPermitPlugin is a Permit plugin that blocks until a signal is received.
type blockingPermitPlugin struct {
	podsToBlock map[string]*blockedPod
}

const blockingPermitPluginName = "blocking-permit-plugin"

var _ fwk.PermitPlugin = &blockingPermitPlugin{}

func newBlockingPermitPlugin(_ context.Context, _ runtime.Object, h fwk.Handle, podsToBlock map[string]*blockedPod) fwk.Plugin {
	return &blockingPermitPlugin{
		podsToBlock: podsToBlock,
	}
}

func (pl *blockingPermitPlugin) Name() string {
	return blockingPermitPluginName
}

func (pl *blockingPermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if p, ok := pl.podsToBlock[pod.Name]; ok {
		delete(pl.podsToBlock, pod.Name)
		p.blocked <- struct{}{}
		return fwk.NewStatus(fwk.Wait, "waiting"), time.Minute
	}
	return nil, 0
}
