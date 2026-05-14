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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
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

func (fp *tokenFilter) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
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

func (fp *tokenFilter) PreFilterExtensions() fwk.PreFilterExtensions {
	return fp
}

var _ fwk.FilterPlugin = &tokenFilter{}

// TestPreemption tests a few preemption scenarios.
func TestPreemption(t *testing.T) {
	// Initialize scheduler with a filter plugin.
	var filter tokenFilter
	registry := make(frameworkruntime.Registry)
	err := registry.Register(filterPluginName, func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
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
						featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
							features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
							features.SchedulerAsyncAPICalls:                asyncAPICallsEnabled,
							features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
						})

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

	tests := []asyncPreemptionTest{
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
						podName:             "preemptor",
						expectUnschedulable: true,
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
			name: "basic async preemption with 1 victim, preemptor gated until preemption API call finishes",
			scenarios: []scenario{
				{
					name: "create victim",
					createPod: &createPod{
						pod: st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the preemptor Pod is in the queue and gated",
					podGatedInQueue: "preemptor",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(1),
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled",
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
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
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
						podName:             "pod-mid-priority",
						expectUnschedulable: true,
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
						podName:             "pod-mid-priority",
						expectUnschedulable: true,
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
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
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
						podName:             "preemptor-super-high-priority",
						expectUnschedulable: true,
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
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
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
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
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
						podName:             "preemptor-mid-priority",
						expectUnschedulable: true,
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
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Delete event and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted",
			scenarios: []scenario{
				{
					name: "create victim Pod that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			scenarios: []scenario{
				{
					name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).TerminationGracePeriodSeconds(1000).Container("image").Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			scenarios: []scenario{
				{
					name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is reserving some resources required by the preemptor:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted), but is still unschedulable, because victim has to unreserve its resources.
			// After resuming binding for a victim, it releases the resources in its failure handler, preemptor is woken up again and ultimately scheduled.
			name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted and its resources are unreserved",
			scenarios: []scenario{
				{
					name: "create victim Pod that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName + reservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName + reservingPodName,
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination and sis reserving some resources required by the preemptor:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted), but is still unschedulable, because victim has to unreserve its resources.
			// After resuming binding for a victim, it releases the resources in its failure handler, preemptor is woken up again and ultimately scheduled.
			name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is under graceful termination and its resources are unreserved",
			scenarios: []scenario{
				{
					name: "create victim Pod that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName + reservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName + reservingPodName,
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
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			// Expected test outcome: lower priority Pod switches to another node, does not get stuck in unschedulable queue forever. (This part is in comment due to test name length limit.)
			name: "While lower priority Pod is waiting for preemption, higher priority Pod takes its place on the node",
			scenarios: []scenario{
				{
					name: "create N-1 victim Pods on the first node",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(3),
					},
				},
				{
					name: "create the last victim Pod on the first node, that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "schedule the last victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
					},
				},
				{
					name: "create a mid-priority preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					name: "schedule the mid-priority preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-mid-priority",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name:            "check the mid-priority preemptor Pod is gated, waiting for the last victim to be preempted",
					podGatedInQueue: "preemptor-mid-priority",
				},
				{
					name:       "create node2",
					createNode: "node2",
				},
				{
					name: "create victim Pods on node2",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node2").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					name: "create a high-priority preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					name: "schedule the high-priority preemptor Pod and expect it to get scheduled on node1",
					// While we don't check explicitly that Pod is scheduled on node1, we can assume that because
					// Pod won't fit on node2 without preemption and there are enough resources on node1.
					schedulePod: &schedulePod{
						podName:       "preemptor-high-priority",
						expectSuccess: true,
					},
				},
				{
					name:       "allow the preemption of the last victim Pod on node1 to finish",
					resumeBind: true,
				},
				{
					name: "check that mid-priority preemptor Pod got activated by completed preemption and try scheduling it again",
					schedulePod: &schedulePod{
						podName: "preemptor-mid-priority",
						// Pod won't fit on node1 anymore and should trigger preemptions on node2.
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API calls on node2",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name: "check that mid-priority Pod got activated, schedule it on node2",
					schedulePod: &schedulePod{
						podName:       "preemptor-mid-priority",
						expectSuccess: true,
					},
				},
			},
		},
	}
	runAsyncPreemptionScenarios(t, tests, false)
}

func TestPreemptionRespectsWaitingPod(t *testing.T) {
	// 1. Create a "blocking" permit plugin that signals when it's running and waits for a specific close.
	// 2. Create a big node on which low-priority pod will be scheduled.
	// 3. Schedule a low-priority pod (victim) that hits this plugin (after being selected to run on a big node).
	// 4. While victim is blocked in WaitOnPermit, add a smaller node on which the victim should be rescheduled.
	// 5. Schedule a high-priority pod (preemptor), that can only fit on big node.
	// 6. High-priority pod should be scheduled on a big node and victim should be preempted.
	// 7. Victim should be rescheduled on a smaller node.

	// Create a node with resources for only one pod.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourceCPU:    "2",
		v1.ResourceMemory: "2Gi",
	}
	node := st.MakeNode().Name("big-node").Capacity(nodeRes).Obj()

	victim := st.MakePod().Name("victim").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).Obj()
	// Preemptor requires more resources than the small node has.
	preemptor := st.MakePod().Name("preemptor").Priority(highPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5", v1.ResourceMemory: "1.5Gi"}).Obj()

	// Register the blocking plugin
	victimToBlock := &blockedPod{
		blocked: make(chan struct{}),
	}
	podsToBlock := map[string]*blockedPod{
		victim.Name: victimToBlock,
	}

	registry := make(frameworkruntime.Registry)
	err := registry.Register(blockingPermitPluginName, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return newBlockingPermitPlugin(ctx, obj, fh, podsToBlock), nil
	})
	if err != nil {
		t.Fatalf("Error registering plugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Permit: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: blockingPermitPluginName},
					},
				},
			},
		}},
	})
	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "preemption-waiting", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet

	if _, err := createNode(cs, node); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	t.Logf("Creating victim pod")
	victim, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, victim, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating victim: %v", err)
	}

	t.Logf("Waiting for victim to reach WaitOnPermit")
	select {
	case <-victimToBlock.blocked:
		t.Logf("Victim reached WaitOnPermit")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timed out waiting for victim to reach WaitOnPermit")
	}

	smallNodeRes := map[v1.ResourceName]string{
		v1.ResourceCPU:    "1",
		v1.ResourceMemory: "1Gi",
	}
	smallNode := st.MakeNode().Name("small-node").Capacity(smallNodeRes).Obj()
	if _, err := createNode(cs, smallNode); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	t.Logf("Creating preemptor pod")
	_, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, preemptor, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating preemptor: %v", err)
	}

	// Preemptor should eventually be scheduled or cause victim preemption.
	// Since victim is in WaitingOnPermit, Preemptor's preemption logic (PostFilter) should find it.
	// It should call PreemptPod() on waiting victim.
	// The plugin returns error on preemption, so the victim scheduling fails.
	// The victim should NOT be deleted from API server.
	// Instead the victim  should go to the backoff queue and get rescheduled eventually.
	t.Logf("Waiting for preemptor to be scheduled")
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 15*time.Second, false, func(ctx context.Context) (bool, error) {
		// Ensure that victim is not deleted
		_, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, victim.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, fmt.Errorf("victim pod was deleted")
			}
			return false, err
		}
		// Check if preemptor was scheduled
		p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, preemptor.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, fmt.Errorf("preemptor pod was deleted")
			}
			return false, err
		}
		return p.Spec.NodeName != "", nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for preemptor validation: %v", err)
	}

	t.Logf("waiting for victim to be rescheduled")
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 15*time.Second, false, func(ctx context.Context) (bool, error) {
		v, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, victim.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return v.Spec.NodeName != "", nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for victim validation: %v", err)
	}

	// Check that preemptor and victim are scheduled on expected nodes: victim on a small node and preemptor on a big node.
	v, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, victim.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error getting victim: %v", err)
	}
	if v.Spec.NodeName != "small-node" {
		t.Fatalf("Victim should be scheduled on small-node, but was scheduled on %s", v.Spec.NodeName)
	}

	p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error getting preemptor: %v", err)
	}
	if p.Spec.NodeName != "big-node" {
		t.Fatalf("Preemptor should be scheduled on big-node, but was scheduled on %s", p.Spec.NodeName)
	}
}

type perPodBlockingPlugin struct {
	shouldBlock bool
	blocked     chan struct{}
	released    chan struct{}
}

// blockingPreBindPlugin is a PreBindPlugin that blocks until a signal is received.
type blockingPreBindPlugin struct {
	podToChannels map[string]*perPodBlockingPlugin
	handle        fwk.Handle
}

const blockingPreBindPluginName = "blocking-prebind-plugin"

var _ fwk.PreBindPlugin = &blockingPreBindPlugin{}

func newBlockingPreBindPlugin(_ context.Context, _ runtime.Object, h fwk.Handle, podToChannels map[string]*perPodBlockingPlugin) (fwk.Plugin, error) {
	return &blockingPreBindPlugin{
		podToChannels: podToChannels,
		handle:        h,
	}, nil
}

func (pl *blockingPreBindPlugin) Name() string {
	return blockingPreBindPluginName
}

func (pl *blockingPreBindPlugin) PreBind(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) *fwk.Status {
	podBlocks, ok := pl.podToChannels[pod.Name]
	if !ok {
		return fwk.NewStatus(fwk.Error, "pod was not prepared in test case")
	}
	if !podBlocks.shouldBlock {
		return nil
	}

	close(podBlocks.blocked)
	podBlocks.shouldBlock = false
	select {
	case <-podBlocks.released:
		return nil
	case <-ctx.Done():
		return fwk.AsStatus(ctx.Err())
	}
}

func (pl *blockingPreBindPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return &fwk.PreBindPreFlightResult{}, nil
}

func TestPreemptionRespectsBindingPod(t *testing.T) {
	// 1. Create a "blocking" prebind plugin that signals when it's running and waits for a specific close.
	// 2. Schedule a low-priority pod (victim) that hits this plugin.
	// 3. While victim is blocked in PreBind, add a small node and schedule a high-priority pod (preemptor) that fits only on a bigger node.
	// 4. Wait for preemptor to be scheduled.
	// 5. Verify that:
	//		- preemptor takes place on the bigger node
	//		- victim is NOT deleted, it's rescheduled on to a smaller node

	// Create a node with resources for only one pod.
	bigNode := st.MakeNode().Name("big-node").Capacity(map[v1.ResourceName]string{
		v1.ResourceCPU:    "2",
		v1.ResourceMemory: "2Gi",
	}).Obj()
	// Victim requires full node resources.
	victim := st.MakePod().Name("victim").Priority(lowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).Obj()
	// Preemptor also requires full node resources.
	preemptor := st.MakePod().Name("preemptor").Priority(highPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5", v1.ResourceMemory: "1.5Gi"}).Obj()

	// Register the blocking plugin.
	victimBlockingPlugin := &perPodBlockingPlugin{
		shouldBlock: true,
		blocked:     make(chan struct{}),
		released:    make(chan struct{}),
	}
	podToChannels := map[string]*perPodBlockingPlugin{
		victim.Name: victimBlockingPlugin,
		preemptor.Name: {
			shouldBlock: false,
			blocked:     make(chan struct{}),
			released:    make(chan struct{}),
		},
	}

	registry := make(frameworkruntime.Registry)
	err := registry.Register(blockingPreBindPluginName, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return newBlockingPreBindPlugin(ctx, obj, fh, podToChannels)
	})
	if err != nil {
		t.Fatalf("Error registering plugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				PreBind: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: blockingPreBindPluginName},
					},
				},
			},
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "preemption-binding", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet

	if _, err := createNode(cs, bigNode); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	// 1. Run victim.
	t.Logf("Creating victim pod")
	victim, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, victim, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating victim: %v", err)
	}

	// Wait for victim to reach PreBind.
	t.Logf("Waiting for victim to reach PreBind")
	select {
	case <-victimBlockingPlugin.blocked:
		t.Logf("Victim reached PreBind")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timed out waiting for victim to reach PreBind")
	}

	// 2. Add a small node that will fit victim once its preempted.
	smallNode := st.MakeNode().Name("small-node").Capacity(map[v1.ResourceName]string{
		v1.ResourceCPU:    "1",
		v1.ResourceMemory: "1Gi",
	}).Obj()
	if _, err := createNode(cs, smallNode); err != nil {
		t.Fatalf("Error creating node: %v", err)
	}

	// 3. Run preemptor pod.
	t.Logf("Creating preemptor pod")
	preemptor, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, preemptor, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating preemptor: %v", err)
	}

	// 4. Wait for victim to be rescheduled.
	// Preemptor should eventually be scheduled or cause victim preemption.
	// Since victim is in PreBind (Binding Cycle), Preemptor's preemption logic (PostFilter) should find it.
	// It should call CancelPod() on the victim's BindingPod, causing it to go to backoff queue.
	// The victim pod should NOT be deleted from API server.
	// Instead it should be rescheduled onto a smaller node.
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		// Check if victim is deleted
		v, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, victim.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, fmt.Errorf("victim pod was deleted")
			}
			return false, err
		}
		// Check if victim was rescheduled
		_, cond := podutil.GetPodCondition(&v.Status, v1.PodScheduled)
		if cond != nil && cond.Status == v1.ConditionTrue {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for victim validation: %v", err)
	}

	// 5. Wait for preemptor to be scheduled.
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(ctx, preemptor.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// Check if preemptor is scheduled
		_, cond := podutil.GetPodCondition(&p.Status, v1.PodScheduled)
		if cond != nil && cond.Status == v1.ConditionTrue {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for preemptor to be scheduled: %v", err)
	}

	// 6. Check that preemptor and victim are scheduled on expected nodes: victim on a small node and preemptor on a big node.
	v, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, victim.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error getting victim: %v", err)
	}

	p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, preemptor.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error getting preemptor: %v", err)
	}
	// Verify the assignments are correct
	if v.Spec.NodeName != "small-node" {
		t.Errorf("victim should be scheduled on small-node, but was scheduled on %s", v.Spec.NodeName)
	}
	if p.Spec.NodeName != "big-node" {
		t.Errorf("preemptor should be scheduled on big-node, but was scheduled on %s", p.Spec.NodeName)
	}
	// Start a goroutine to release the plugin just in case, ensuring clean teardown.
	close(victimBlockingPlugin.released)
}
