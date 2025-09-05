/*
Copyright 2017 The Kubernetes Authors.

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

// This file tests preemption functionality of the scheduler.

package preemption

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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

// imported from testutils
var (
	initPausePod        = testutils.InitPausePod
	createNode          = testutils.CreateNode
	createPausePod      = testutils.CreatePausePod
	runPausePod         = testutils.RunPausePod
	podIsGettingEvicted = testutils.PodIsGettingEvicted
)

const filterPluginName = "filter-plugin"

var lowPriority, mediumPriority, highPriority = int32(100), int32(200), int32(300)

const tokenFilterName = "token-filter"

// tokenFilter is a fake plugin that implements PreFilter and Filter.
// `Token` simulates the allowed pods number a cluster can host.
// If `EnablePreFilter` is set to false or `Token` is positive, PreFilter passes; otherwise returns Unschedulable
// For each Filter() call, `Token` is decreased by one. When `Token` is positive, Filter passes; otherwise return
// Unschedulable or UnschedulableAndUnresolvable (when `Unresolvable` is set to true)
// AddPod()/RemovePod() adds/removes one token to the cluster to simulate the dryrun preemption
type tokenFilter struct {
	Tokens          int
	Unresolvable    bool
	EnablePreFilter bool
}

// Name returns name of the plugin.
func (fp *tokenFilter) Name() string {
	return tokenFilterName
}

func (fp *tokenFilter) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod,
	nodeInfo fwk.NodeInfo) *fwk.Status {
	if fp.Tokens > 0 {
		fp.Tokens--
		return nil
	}
	status := fwk.Unschedulable
	if fp.Unresolvable {
		status = fwk.UnschedulableAndUnresolvable
	}
	return fwk.NewStatus(status, fmt.Sprintf("can't fit %v", pod.Name))
}

func (fp *tokenFilter) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	if !fp.EnablePreFilter || fp.Tokens > 0 {
		return nil, nil
	}
	return nil, fwk.NewStatus(fwk.Unschedulable)
}

func (fp *tokenFilter) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod,
	podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	fp.Tokens--
	return nil
}

func (fp *tokenFilter) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod,
	podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	fp.Tokens++
	return nil
}

func (fp *tokenFilter) PreFilterExtensions() framework.PreFilterExtensions {
	return fp
}

var _ framework.FilterPlugin = &tokenFilter{}

// TestPreemption tests a few preemption scenarios.
func TestPreemption(t *testing.T) {
	// Initialize scheduler with a filter plugin.
	var filter tokenFilter
	registry := make(frameworkruntime.Registry)
	err := registry.Register(filterPluginName, func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		return &filter, nil
	})
	if err != nil {
		t.Fatalf("Error registering a filter: %v", err)
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Filter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: filterPluginName},
					},
				},
				PreFilter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: filterPluginName},
					},
				},
			},
		}},
	})

	defaultPodRes := &v1.ResourceRequirements{Requests: v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
	}

	maxTokens := 1000
	tests := []struct {
		name                string
		existingPods        []*v1.Pod
		pod                 *v1.Pod
		initTokens          int
		enablePreFilter     bool
		unresolvable        bool
		preemptedPodIndexes map[int]struct{}
	}{
		{
			name:       "basic pod preemption",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-pod",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			name:       "basic pod preemption with filter",
			initTokens: 1,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-pod",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		// This is identical with previous subtest except for setting enablePreFilter to true.
		// With this fake plugin returning Unschedulable in PreFilter, it's able to exercise the path
		// that in-tree plugins return Skip in PreFilter and their AddPod/RemovePod functions are also
		// skipped properly upon preemption.
		{
			name:            "basic pod preemption with preFilter",
			initTokens:      1,
			enablePreFilter: true,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-pod",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			// same as the previous test, but the filter is unresolvable.
			name:         "basic pod preemption with unresolvable filter",
			initTokens:   1,
			unresolvable: true,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-pod",
					Priority: &lowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &highPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
		{
			name:       "preemption is performed to satisfy anti-affinity",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-0",
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-1",
					Priority:  &lowPriority,
					Labels:    map[string]string{"pod": "p1"},
					Resources: defaultPodRes,
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "pod",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"preemptor"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				}),
			},
			// A higher priority pod with anti-affinity.
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Priority:  &highPriority,
				Labels:    map[string]string{"pod": "preemptor"},
				Resources: defaultPodRes,
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"p0"},
										},
									},
								},
								TopologyKey: "node",
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// This is similar to the previous case only pod-1 is high priority.
			name:       "preemption is not performed when anti-affinity is not satisfied",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-0",
					Priority:  &mediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-1",
					Priority:  &highPriority,
					Labels:    map[string]string{"pod": "p1"},
					Resources: defaultPodRes,
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "pod",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"preemptor"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				}),
			},
			// A higher priority pod with anti-affinity.
			pod: initPausePod(&testutils.PausePodConfig{
				Name:      "preemptor-pod",
				Priority:  &highPriority,
				Labels:    map[string]string{"pod": "preemptor"},
				Resources: defaultPodRes,
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"p0"},
										},
									},
								},
								TopologyKey: "node",
							},
						},
					},
				},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
	}

	// Create a node with some resources and a label.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Obj()

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, asyncAPICallsEnabled := range []bool{true, false} {
			for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
				for _, test := range tests {
					t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, Async API calls enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, asyncAPICallsEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncPreemption, asyncPreemptionEnabled)
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncAPICalls, asyncAPICallsEnabled)
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ClearingNominatedNodeNameAfterBinding, clearingNominatedNodeNameAfterBinding)

						testCtx := testutils.InitTestSchedulerWithOptions(t,
							testutils.InitTestAPIServer(t, "preemption", nil),
							0,
							scheduler.WithProfiles(cfg.Profiles...),
							scheduler.WithFrameworkOutOfTreeRegistry(registry))
						testutils.SyncSchedulerInformerFactory(testCtx)
						go testCtx.Scheduler.Run(testCtx.Ctx)

						if _, err := createNode(testCtx.ClientSet, nodeObject); err != nil {
							t.Fatalf("Error creating node: %v", err)
						}

						cs := testCtx.ClientSet

						filter.Tokens = test.initTokens
						filter.EnablePreFilter = test.enablePreFilter
						filter.Unresolvable = test.unresolvable
						pods := make([]*v1.Pod, len(test.existingPods))
						// Create and run existingPods.
						for i, p := range test.existingPods {
							p.Namespace = testCtx.NS.Name
							pods[i], err = runPausePod(cs, p)
							if err != nil {
								t.Fatalf("Error running pause pod: %v", err)
							}
						}
						// Create the "pod".
						test.pod.Namespace = testCtx.NS.Name
						preemptor, err := createPausePod(cs, test.pod)
						if err != nil {
							t.Errorf("Error while creating high priority pod: %v", err)
						}
						// Wait for preemption of pods and make sure the other ones are not preempted.
						for i, p := range pods {
							if _, found := test.preemptedPodIndexes[i]; found {
								if err = wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false,
									podIsGettingEvicted(cs, p.Namespace, p.Name)); err != nil {
									t.Errorf("Pod %v/%v is not getting evicted.", p.Namespace, p.Name)
								}
								pod, err := cs.CoreV1().Pods(p.Namespace).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
								if err != nil {
									t.Errorf("Error %v when getting the updated status for pod %v/%v ", err, p.Namespace, p.Name)
								}
								_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
								if cond == nil {
									t.Errorf("Pod %q does not have the expected condition: %q", klog.KObj(pod), v1.DisruptionTarget)
								}
							} else if p.DeletionTimestamp != nil {
								t.Errorf("Didn't expect pod %v to get preempted.", p.Name)
							}
						}
						// Also check that the preemptor pod gets the NominatedNodeName field set.
						if len(test.preemptedPodIndexes) > 0 && !clearingNominatedNodeNameAfterBinding {
							if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
								t.Errorf("NominatedNodeName field was not set for pod %v: %v", preemptor.Name, err)
							}
						}

						// Cleanup
						pods = append(pods, preemptor)
						testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
					})
				}
			}
		}
	}
}

func TestAsyncPreemption(t *testing.T) {
	type createPod struct {
		pod *v1.Pod
		// count is the number of times the pod should be created by this action.
		// i.e., if you use it, you have to use GenerateName.
		// By default, it's 1.
		count *int
	}

	type schedulePod struct {
		podName       string
		expectSuccess bool
	}

	type scenario struct {
		// name is this step's name, just for the debugging purpose.
		name string

		// Only one of the following actions should be set.

		// createPod creates a Pod.
		createPod *createPod
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
	}

	tests := []struct {
		name string
		// scenarios after the first attempt of scheduling the pod.
		scenarios []scenario
	}{
		{
			// Very basic test case: if it fails, the basic scenario is broken somewhere.
			name: "basic: async preemption happens expectedly",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(2),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor",
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(2),
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod after the preemption",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			name: "Lower priority Pod doesn't take over the place for higher priority Pod that is running the preemption",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(2),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-high-priority",
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor-high-priority",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(2),
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor Pod is nominated to the node, this Pod should be unschedulable.
					name: "create a second Pod that is lower priority than the first preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("pod-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					name: "schedule the mid-priority Pod",
					schedulePod: &schedulePod{
						podName: "pod-mid-priority",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-high-priority",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					name: "schedule the preemptor Pod again",
					schedulePod: &schedulePod{
						podName:       "preemptor-high-priority",
						expectSuccess: true,
					},
				},
				{
					name: "schedule the mid-priority Pod again",
					schedulePod: &schedulePod{
						podName: "pod-mid-priority",
					},
				},
			},
		},
		{
			name: "Higher priority Pod takes over the place for lower priority Pod that is running the preemption",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-high-priority",
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor-high-priority",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(4),
				},
				{
					// This Pod is higher priority than the preemptor Pod.
					// Even though the preemptor Pod is nominated to the node, this Pod can take over the place.
					name: "create a second Pod that is higher priority than the first preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-super-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(200).Obj(),
					},
				},
				{
					name: "schedule the super-high-priority Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-super-high-priority",
					},
				},
				{
					name:                 "check the super-high-priority Pod making the preemption API calls",
					podRunningPreemption: ptr.To(5),
				},
				{
					// the super-high-priority preemptor should enter the preemption
					// and select the place where the preemptor-high-priority selected.
					// So, basically both goroutines are preempting the same Pods.
					name:            "check the super-high-priority pod is in the queue and gated",
					podGatedInQueue: "preemptor-super-high-priority",
				},
				{
					name:               "complete the preemption API calls of super-high-priority",
					completePreemption: "preemptor-super-high-priority",
				},
				{
					name:               "complete the preemption API calls of high-priority",
					completePreemption: "preemptor-high-priority",
				},
				{
					name: "schedule the super-high-priority Pod",
					schedulePod: &schedulePod{
						podName:       "preemptor-super-high-priority",
						expectSuccess: true,
					},
				},
				{
					name: "schedule the high-priority Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-high-priority",
					},
				},
			},
		},
		{
			name: "Lower priority Pod can select the same place where the higher priority Pod is preempting if the node is big enough",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					// It will preempt two victims.
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-high-priority",
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor-high-priority",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(4),
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor Pod is nominated to the node, this Pod should be unschedulable.
					// This Pod will trigger the preemption to target the two victims that the first Pod doesn't target.
					name: "create a second Pod that is lower priority than the first preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					name: "schedule the mid-priority Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-mid-priority",
					},
				},
				{
					name:            "check the mid-priority pod is in the queue and gated",
					podGatedInQueue: "preemptor-mid-priority",
				},
				{
					name:                 "check the mid-priority Pod making the preemption API calls",
					podRunningPreemption: ptr.To(5),
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-high-priority",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					name: "schedule the preemptor Pod again",
					schedulePod: &schedulePod{
						podName:       "preemptor-high-priority",
						expectSuccess: true,
					},
				},
				{
					name: "schedule the mid-priority Pod again",
					schedulePod: &schedulePod{
						podName:       "preemptor-mid-priority",
						expectSuccess: true,
					},
				},
			},
		},
	}

	// All test cases have the same node.
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()
	for _, asyncAPICallsEnabled := range []bool{true, false} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("%s (Async API calls enabled: %v)", test.name, asyncAPICallsEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncAPICalls, asyncAPICallsEnabled)

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
				err := registry.Register(delayedPreemptionPluginName, func(c context.Context, r runtime.Object, fh framework.Handle) (framework.Plugin, error) {
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
					preemptionPlugin.Evaluator.PreemptPod = func(ctx context.Context, c preemption.Candidate, preemptor, victim *v1.Pod, pluginName string) error {
						// block the preemption goroutine to complete until the test case allows it to proceed.
						lock.Lock()
						ch, ok := preemptionDoneChannels[preemptor.Name]
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
						preemptionDoneChannels[scenario.schedulePod.podName] = make(chan struct{})
						lock.Unlock()
						testCtx.Scheduler.ScheduleOne(testCtx.Ctx)
						if scenario.schedulePod.expectSuccess {
							if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, testCtx.NS.Name, scenario.schedulePod.podName)); err != nil {
								t.Fatalf("Expected the pod %s to be scheduled", scenario.schedulePod.podName)
							}
						} else {
							if !podInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.schedulePod.podName) {
								t.Fatalf("Expected the pod %s to be in the queue after the scheduling attempt", scenario.schedulePod.podName)
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
							return preemptionPlugin.Evaluator.IsPodRunningPreemption(createdPods[*scenario.podRunningPreemption].GetUID()), nil
						}); err != nil {
							t.Fatalf("Expected the pod %s to be running preemption", createdPods[*scenario.podRunningPreemption].Name)
						}
					}
				}
			})
		}
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
