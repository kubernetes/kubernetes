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

package podgrouppreemption

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestPodGroupPreemption_ClearingNominatedNodeNameAfterBinding checks that NNN is cleared after preemptor pod gets bound.
func TestPodGroupPreemption_ClearingNominatedNodeNameAfterBinding(t *testing.T) {
	tests := []struct {
		name                            string
		nodes                           []*v1.Node
		compositePodGroups              []*schedulingv1alpha3.CompositePodGroup
		podGroups                       []*schedulingv1beta1.PodGroup
		initialPods                     []*v1.Pod // pods that should be scheduled before preemption starts
		preemptorPods                   []*v1.Pod // pods that belong to a group and should trigger preemption
		expectedScheduled               []string
		expectedCandidatesForPreemption []string
		expectedToHaveNNNInfo           []string
		expectedPodsPreemptedByWAP      int
	}{
		{
			name: "Full PodGroup Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      3,
		},
		{
			name: "Full PodGroup Preemption for basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      3,
		},
		{
			name: "CPG with PreemptLowerPriority Policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace("default").Priority(100).BasicPolicy().WorkloadRef("wl1", "t1").PreemptionPolicy(schedulingv1alpha3.PreemptLowerPriority).Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(100).MinCount(1).ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).PreemptionPolicy(v1.PreemptLowerPriority).Obj(),
			},
			expectedScheduled:               []string{"high-1", "high-2", "high-3"},
			expectedCandidatesForPreemption: []string{"low-1", "low-2", "low-3"},
			expectedToHaveNNNInfo:           []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP:      3,
		},
	}

	for _, tt := range tests {
		for _, cpgEnabled := range []bool{true, false} {
			// Only execute tests that create CPGs with CompositePodGroup feature gate enabled.
			if len(tt.compositePodGroups) > 0 && !cpgEnabled {
				continue
			}
			for _, clearNNNAfterBinding := range []bool{true, false} {
				for _, asyncPreemption := range []bool{true, false} {
					t.Run(fmt.Sprintf("%s (CPG enabled: %v, clear NNN after binding: %v, async preemption: %v)", tt.name, cpgEnabled, clearNNNAfterBinding, asyncPreemption), func(t *testing.T) {
						featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
							features.PodLevelResources:                     true,
							features.GenericWorkload:                       true,
							features.TopologyAwareWorkloadScheduling:       true,
							features.PodGroupPreemptionPolicy:              true,
							features.CompositePodGroup:                     cpgEnabled,
							features.ClearingNominatedNodeNameAfterBinding: clearNNNAfterBinding,
							features.SchedulerAsyncPreemption:              asyncPreemption,
						})
						registry := make(frameworkruntime.Registry)

						// Register mock bind plugin that will register NNN information during binding.
						mockBindPluginName := "mockBindPlugin"
						var bindPlugin = mockBindPlugin{
							name:       mockBindPluginName,
							realPlugin: nil,
							nnnInfo:    sync.Map{},
						}
						err := registry.Register(mockBindPluginName, func(ctx context.Context, o runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
							db, err := defaultbinder.New(ctx, o, fh)
							if err != nil {
								t.Fatalf("Error creating a default binder plugin: %v", err)
							}
							bindPlugin.realPlugin = db.(fwk.BindPlugin)
							return &bindPlugin, nil
						})
						if err != nil {
							t.Fatalf("Error registering a bind plugin: %v", err)
						}

						cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
							Profiles: []configv1.KubeSchedulerProfile{{
								SchedulerName: new(v1.DefaultSchedulerName),
								Plugins: &configv1.Plugins{
									MultiPoint: configv1.PluginSet{
										Enabled: []configv1.Plugin{
											{Name: mockBindPluginName},
										},
										Disabled: []configv1.Plugin{
											{Name: names.DefaultBinder},
										},
									},
								},
							}},
						})

						// Set PodMaxBackoff to 1 second to turn on backoff and allow apiCacher to get information about
						// pod NNN. Without this we might have a race between starting binding and update of apiCacher.
						testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-preemption",
							scheduler.WithProfiles(cfg.Profiles...),
							scheduler.WithFrameworkOutOfTreeRegistry(registry),
							scheduler.WithPodMaxBackoffSeconds(1),
							scheduler.WithPodInitialBackoffSeconds(0))
						cs, ns := testCtx.ClientSet, testCtx.NS.Name

						// Create nodes
						for _, n := range tt.nodes {
							if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create node %s: %v", n.Name, err)
							}
						}

						// 1. Create CompositePodGroups
						for _, cpg := range tt.compositePodGroups {
							cpg.Namespace = ns
							if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create CompositePodGroup %s: %v", cpg.Name, err)
							}
						}

						// 2. Create PodGroups
						for _, pg := range tt.podGroups {
							pg.Namespace = ns
							if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create PodGroup %s: %v", pg.Name, err)
							}
						}

						// 3. Create initial pods
						for _, p := range tt.initialPods {
							p.Namespace = ns
							if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create pod %s: %v", p.Name, err)
							}
						}

						// 4. Wait for initial pods to be scheduled
						for _, p := range tt.initialPods {
							if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
								testutils.PodScheduled(cs, ns, p.Name)); err != nil {
								t.Errorf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
							}
						}

						// 5. Create preemptor pods
						for _, p := range tt.preemptorPods {
							p.Namespace = ns
							if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create pod %s: %v", p.Name, err)
							}
						}

						// 6. Wait for preemption to complete if WAP calls are expected
						if tt.expectedPodsPreemptedByWAP > 0 {
							wapCalls := 0
							err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
								wapCalls = 0
								for _, podName := range tt.expectedCandidatesForPreemption {
									events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
										FieldSelector: "involvedObject.name=" + podName,
									})
									if err != nil {
										return false, err
									}
									for _, event := range events.Items {
										if event.Reason == "Preempted" && (strings.HasPrefix(event.Message, "Preempted by compositepodgroup") || strings.HasPrefix(event.Message, "Preempted by podgroup")) {
											wapCalls++
											break
										}
									}
								}
								return wapCalls == tt.expectedPodsPreemptedByWAP, nil
							})
							if err != nil {
								t.Errorf("WorkloadAwarePreemption was not called expected times within timeout: want=%d, got=%d", tt.expectedPodsPreemptedByWAP, wapCalls)
							}
						}

						// 7. Verify scheduled pods
						for _, podName := range tt.expectedScheduled {
							if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
								testutils.PodScheduled(cs, ns, podName)); err != nil {
								t.Errorf("Pod %s was expected to be scheduled but wasn't: %v", podName, err)
							}
						}

						// 8. Verify preempted pods
						if len(tt.expectedCandidatesForPreemption) > 0 {
							var preemptedCount int
							var notPreemptedPods []string
							// Subgroup of pods (might be all) in candidatesForPreemption is expected to be preempted.
							// Preemption has finished, because all expected pods were scheduled - checked in step 7.
							// Retry will be performed when there is an error or number of preempted pod do not match expectedPodsPreemptedByWAP.
							err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
								func(ctx context.Context) (bool, error) {
									preemptedCount = 0
									notPreemptedPods = nil
									for _, podName := range tt.expectedCandidatesForPreemption {
										pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
										if err != nil {
											if apierrors.IsNotFound(err) {
												preemptedCount++
												continue
											}
											return false, err
										}
										if pod.DeletionTimestamp != nil {
											preemptedCount++
											continue
										}
										if _, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget); cond != nil {
											preemptedCount++
											continue
										}
										notPreemptedPods = append(notPreemptedPods, podName)
									}
									return preemptedCount == tt.expectedPodsPreemptedByWAP, nil
								})
							if err != nil {
								t.Errorf("Expected exactly %d pods from %v to be preempted, but only %d pods were preempted, not preempted pods: %v. Error: %v", tt.expectedPodsPreemptedByWAP, tt.expectedCandidatesForPreemption, preemptedCount, notPreemptedPods, err)
							}
						}

						// 9. Verify preemptor pods have nominated node name
						for _, podName := range tt.expectedToHaveNNNInfo {
							if node, ok := bindPlugin.nnnInfo.Load(podName); !ok || node.(string) == "" {
								t.Errorf("Pod %s was expected to have nominated node name but didn't", podName)
							}
						}

						// 10. Verify that NominatedNodeName was cleared for the preemptor pods that got bound (depending on the feature gate setting).
						for _, podName := range tt.expectedToHaveNNNInfo {
							pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
							if err != nil {
								t.Fatalf("Failed to get preemptor pod: %v", err)
							}
							if clearNNNAfterBinding && len(pod.Status.NominatedNodeName) > 0 {
								t.Errorf("Expected NominatedNodeName field to be cleared for pod %v", pod.Name)
							}
							if !clearNNNAfterBinding && len(pod.Status.NominatedNodeName) == 0 {
								t.Errorf("NominatedNodeName field was unexpectedly cleared for pod %v", pod.Name)
							}
						}

						// 11. Dump the state of pods to ease debugging failed runs.
						if t.Failed() {
							t.Log("Dumping states of initial and preemptor pods:")
							var allPods []string
							for _, p := range tt.initialPods {
								allPods = append(allPods, p.Name)
							}
							for _, p := range tt.preemptorPods {
								allPods = append(allPods, p.Name)
							}
							for _, podName := range allPods {
								pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
								if err != nil {
									if apierrors.IsNotFound(err) {
										t.Logf("Pod %q: not present in cluster", podName)
									} else {
										t.Logf("Pod %q: failed to get: %v", podName, err)
									}
									continue
								}

								var statusStr string
								if pod.Spec.NodeName != "" {
									statusStr = "scheduled on node " + pod.Spec.NodeName
								} else {
									_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
									if cond != nil && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable {
										statusStr = "unschedulable"
									} else {
										statusStr = "pending"
									}
								}
								t.Logf("Pod %q: status=%s, phase=%s", podName, statusStr, pod.Status.Phase)
							}
						}
					})
				}
			}
		}
	}
}
