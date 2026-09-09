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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8suuid "k8s.io/apimachinery/pkg/util/uuid"
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
	testfwk "k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/ktesting"
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
			SchedulerName: new(v1.DefaultSchedulerName),
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
		name                     string
		existingPods             []*v1.Pod
		pod                      *v1.Pod
		initTokens               int
		enablePreFilter          bool
		unresolvable             bool
		preemptedPodIndexes      map[int]struct{}
		extraNodes               []*v1.Node
		podGroups                []*schedulingv1beta1.PodGroup
		compositePodGroups       []*schedulingv1alpha3.CompositePodGroup
		genericWorkloadEnabled   bool
		compositePodGroupEnabled bool
	}{
		{
			name:       "basic pod preemption",
			initTokens: maxTokens,
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:     "victim-pod",
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
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
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
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
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
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
					Priority: &asyncframework.LowPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI)},
					},
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:     "preemptor-pod",
				Priority: &asyncframework.HighPriority,
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
					Priority:  &asyncframework.MediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-1",
					Priority:  &asyncframework.LowPriority,
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
				Priority:  &asyncframework.HighPriority,
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
					Priority:  &asyncframework.MediumPriority,
					Labels:    map[string]string{"pod": "p0"},
					Resources: defaultPodRes,
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:      "pod-1",
					Priority:  &asyncframework.HighPriority,
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
				Priority:  &asyncframework.HighPriority,
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
		{
			// The PodGroup is treated as a single atomic victim, so both pods
			// are evicted as a unit even though pod-group-victim-2 lives on node2.
			name:                   "pod group victim across multiple nodes, pod-group-as-victim enabled",
			initTokens:             maxTokens,
			genericWorkloadEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg1",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg1",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			// The entire pod group is evicted as a unit: both index 0 and index 1.
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When all pods of a PodGroup reside on a single node and DisruptionModeAll is enabled,
			// preempting one member to satisfy resource requests triggers atomic eviction of the entire group.
			name:                   "pod group occupying a single node, pod-group-as-victim enabled",
			initTokens:             maxTokens,
			genericWorkloadEnabled: true,
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-single").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-single",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-single",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When multiple victim PodGroups exist across nodes, preemption selects only the lowest-priority
			// group needed to satisfy the request. Its members across all nodes are evicted while higher-priority groups remain intact.
			name:                   "multiple victim pod groups, only subset selected for preemption",
			initTokens:             maxTokens,
			genericWorkloadEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-low").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().Obj(),
				st.MakePodGroup().Name("pg-medium").Priority(asyncframework.MediumPriority).BasicPolicy().
					DisruptionModeAll().Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-low-node1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-low",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-low-node2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-low",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-medium-node1",
					Priority:     &asyncframework.MediumPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-medium",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-medium-node2",
					Priority:     &asyncframework.MediumPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-medium",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When evicting all lower-priority victim PodGroups on a node is still insufficient to satisfy the request
			// due to un-preemptible equal-priority pods, preemption aborts without evicting any members of the PodGroup.
			name:                   "unsuccessful pod group preemption when freeing lower priority group is insufficient",
			initTokens:             maxTokens,
			genericWorkloadEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "1000",
				}).Label("node", "node2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "unpreemptible-pod",
					Priority:     &asyncframework.HighPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-victim",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-victim",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node2"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(600, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
		{
			// When a PodGroup across multiple nodes uses DisruptionModeSingle, evicting a member on the target
			// node does not trigger gang eviction; members on other nodes remain running untouched.
			name:                   "pod group victim across multiple nodes, DisruptionModeSingle enabled",
			initTokens:             maxTokens,
			genericWorkloadEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-single-multi").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeSingle().Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-single-multi-node1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg-single-multi",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-single-multi-node2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg-single-multi",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			// Only index 0 on node1 is evicted; index 1 on node2 is NOT evicted.
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
		{
			// The CompositePodGroup is treated as a single atomic victim, so both pods
			// are evicted as a unit even though pod-group-victim-2 lives on node2.
			name:                     "CPG victim across multiple nodes, pod-group-as-victim enabled",
			initTokens:               maxTokens,
			genericWorkloadEnabled:   true,
			compositePodGroupEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Priority(asyncframework.LowPriority).BasicPolicy().ParentCompositePodGroup("cpg1").WorkloadRef("t1", "wl1").Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg1",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg1",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			// The entire pod group is evicted as a unit: both index 0 and index 1.
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When all pods of a CPG reside on a single node and DisruptionModeAll is enabled,
			// preempting one member to satisfy resource requests triggers atomic eviction of the entire group.
			name:                     "CPG occupying a single node, pod-group-as-victim enabled",
			initTokens:               maxTokens,
			genericWorkloadEnabled:   true,
			compositePodGroupEnabled: true,
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-single").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-single").Priority(asyncframework.LowPriority).BasicPolicy().ParentCompositePodGroup("cpg-single").WorkloadRef("t1", "wl1").Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-single",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pod-group-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-single",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When multiple victim CPGs exist across nodes, preemption selects only the lowest-priority
			// group needed to satisfy the request. Its members across all nodes are evicted while higher-priority groups remain intact.
			name:                     "multiple victim CPGs, only subset selected for preemption",
			initTokens:               maxTokens,
			genericWorkloadEnabled:   true,
			compositePodGroupEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-low").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-medium").Priority(asyncframework.MediumPriority).BasicPolicy().
					DisruptionModeAll().WorkloadRef("wl2", "t2").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-low").Priority(asyncframework.LowPriority).BasicPolicy().ParentCompositePodGroup("cpg-low").WorkloadRef("t1", "wl1").Obj(),
				st.MakePodGroup().Name("pg-medium").Priority(asyncframework.MediumPriority).BasicPolicy().ParentCompositePodGroup("cpg-medium").WorkloadRef("t2", "wl2").Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-low-node1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-low",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-low-node2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-low",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-medium-node1",
					Priority:     &asyncframework.MediumPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-medium",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-medium-node2",
					Priority:     &asyncframework.MediumPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-medium",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{0: {}, 1: {}},
		},
		{
			// When evicting all lower-priority victim CPGs on a node is still insufficient to satisfy the request
			// due to un-preemptible equal-priority pods, preemption aborts without evicting any members of the CPG.
			name:                     "unsuccessful CPG preemption when freeing lower priority group is insufficient",
			initTokens:               maxTokens,
			genericWorkloadEnabled:   true,
			compositePodGroupEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "1000",
				}).Label("node", "node2").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-victim").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeAll().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-victim").Priority(asyncframework.LowPriority).BasicPolicy().ParentCompositePodGroup("cpg-victim").WorkloadRef("t1", "wl1").Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "unpreemptible-pod",
					Priority:     &asyncframework.HighPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-victim-1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-victim",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-victim-2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(150, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI),
					}},
					PodGroupName: "pg-victim",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node2"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(600, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			preemptedPodIndexes: map[int]struct{}{},
		},
		{
			// When a CPG across multiple nodes uses DisruptionModeSingle, evicting a member on the target
			// node does not trigger gang eviction; members on other nodes remain running untouched.
			name:                     "CPG victim across multiple nodes, DisruptionModeSingle enabled",
			initTokens:               maxTokens,
			genericWorkloadEnabled:   true,
			compositePodGroupEnabled: true,
			extraNodes: []*v1.Node{
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "32",
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "500",
				}).Label("node", "node2").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-single-multi").Priority(asyncframework.LowPriority).BasicPolicy().
					DisruptionModeSingle().WorkloadRef("wl1", "t1").Obj(),
			},
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-single-multi").Priority(asyncframework.LowPriority).BasicPolicy().ParentCompositePodGroup("cpg-single-multi").WorkloadRef("t1", "wl1").Obj(),
			},
			existingPods: []*v1.Pod{
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-single-multi-node1",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node1"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg-single-multi",
				}),
				initPausePod(&testutils.PausePodConfig{
					Name:         "pg-single-multi-node2",
					Priority:     &asyncframework.LowPriority,
					NodeSelector: map[string]string{"node": "node2"},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
					}},
					PodGroupName: "pg-single-multi",
				}),
			},
			pod: initPausePod(&testutils.PausePodConfig{
				Name:         "preemptor-pod",
				Priority:     &asyncframework.HighPriority,
				NodeSelector: map[string]string{"node": "node1"},
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
				}},
			}),
			// Only index 0 on node1 is evicted; index 1 on node2 is NOT evicted.
			preemptedPodIndexes: map[int]struct{}{0: {}},
		},
	}

	// Create a node with some resources and a label.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Obj()

	// Group test indexes by genericWorkloadEnabled and compositePodGroupEnabled feature gates
	// so each group can share an API server started with the correct feature gate values.
	type featureGateKey struct {
		genericWorkloadEnabled   bool
		compositePodGroupEnabled bool
	}

	testsByWASFeatureGates := make(map[featureGateKey][]int)
	for i, test := range tests {
		key := featureGateKey{
			genericWorkloadEnabled:   test.genericWorkloadEnabled,
			compositePodGroupEnabled: test.compositePodGroupEnabled,
		}
		testsByWASFeatureGates[key] = append(testsByWASFeatureGates[key], i)
	}

	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, asyncAPICallsEnabled := range []bool{true, false} {
			for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
				for fgKey, testIndexes := range testsByWASFeatureGates {
					// One API server per full flag combination. All flags including
					// GenericWorkload and CompositePodGroup are consistent between API server and scheduler.
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
						features.SchedulerAsyncAPICalls:                asyncAPICallsEnabled,
						features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
						features.GenericWorkload:                       fgKey.genericWorkloadEnabled,
						features.CompositePodGroup:                     fgKey.compositePodGroupEnabled,
						features.TopologyAwareWorkloadScheduling:       fgKey.compositePodGroupEnabled,
					})
					sharedAPICtx := testutils.InitTestAPIServer(t, "preemption", nil)

					for _, i := range testIndexes {
						test := tests[i]
						t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, Async API calls enabled: %v, ClearingNominatedNodeNameAfterBinding: %v)", test.name, asyncPreemptionEnabled, asyncAPICallsEnabled, clearingNominatedNodeNameAfterBinding), func(t *testing.T) {
							testCtx := testutils.InitTestSchedulerWithOptions(t,
								withNewNamespace(t, sharedAPICtx, "preemption"),
								0,
								scheduler.WithProfiles(cfg.Profiles...),
								scheduler.WithFrameworkOutOfTreeRegistry(registry))
							testutils.SyncSchedulerInformerFactory(testCtx)
							go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
							defer testCtx.SchedulerCloseFn()

							if _, err := createNode(testCtx.ClientSet, nodeObject); err != nil {
								t.Fatalf("Error creating node: %v", err)
							}
							t.Cleanup(func() {
								if err := testCtx.ClientSet.CoreV1().Nodes().Delete(testCtx.Ctx, nodeObject.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
									t.Errorf("Error deleting node %s: %v", nodeObject.Name, err)
								}
							})
							for _, n := range test.extraNodes {
								if _, err := createNode(testCtx.ClientSet, n); err != nil {
									t.Fatalf("Error creating extra node %s: %v", n.Name, err)
								}
								n := n
								t.Cleanup(func() {
									if err := testCtx.ClientSet.CoreV1().Nodes().Delete(testCtx.Ctx, n.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
										t.Errorf("Error deleting node %s: %v", n.Name, err)
									}
								})
							}

							cs := testCtx.ClientSet
							ns := testCtx.NS.Name

							for _, cpg := range test.compositePodGroups {
								cpg.Namespace = ns
								if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpg, metav1.CreateOptions{}); err != nil {
									t.Fatalf("Error creating CompositePodGroup %s: %v", cpg.Name, err)
								}
							}

							for _, pg := range test.podGroups {
								pg.Namespace = ns
								if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
									t.Fatalf("Error creating PodGroup %s: %v", pg.Name, err)
								}
							}

							filter.Tokens = test.initTokens
							filter.EnablePreFilter = test.enablePreFilter
							filter.Unresolvable = test.unresolvable
							pods := make([]*v1.Pod, len(test.existingPods))
							// Create and run existingPods.
							for i, p := range test.existingPods {
								p.Namespace = ns
								pods[i], err = runPausePod(cs, p)
								if err != nil {
									t.Fatalf("Error running pause pod: %v", err)
								}
							}
							// Create the "pod".
							test.pod.Namespace = ns
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
								} else {
									// Re-fetch to get current state; the pod object from runPausePod
									// always has DeletionTimestamp=nil and cannot detect unexpected eviction.
									current, err := cs.CoreV1().Pods(p.Namespace).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
									if err != nil {
										t.Errorf("Error getting pod %v: %v", p.Name, err)
									} else if current.DeletionTimestamp != nil {
										t.Errorf("Pod %v was unexpectedly preempted", p.Name)
									}
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
}

// TestAsyncPreemption is equivalent for TestPodGroupAsyncPreemption
// in test/integration/scheduler/preemption/podgroup/podgrouppreemption_test.go
// When adding test here, add also test there.
func TestAsyncPreemption(t *testing.T) {
	tests := []struct {
		Name  string
		Steps []asyncframework.Step
	}{
		{
			// Very base test case: if it fails, the base scenario is broken somewhere.
			Name: "base: async preemption happens expectedly",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(2),
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod after the preemption",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "base async preemption with 1 victim, preemptor gated until preemption API call finishes",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the preemptor Pod is in the queue and gated",
					PodGatedInQueue: "preemptor",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(1),
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "Lower priority Pod doesn't take over the place for higher priority Pod that is running the preemption",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-high-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(2),
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor Pod is nominated to the node, this Pod should be unschedulable.
					Name: "create a second Pod that is lower priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("pod-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					Name: "schedule the mid-priority Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "pod-mid-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "preemptor-high-priority",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					Name: "schedule the preemptor Pod again",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-high-priority",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule the mid-priority Pod again",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "pod-mid-priority",
						ExpectUnschedulable: true,
					},
				},
			},
		},
		{
			Name: "Higher priority Pod takes over the place for lower priority Pod that is running the preemption",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-high-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(4),
				},
				{
					// This Pod is higher priority than the preemptor Pod.
					// Even though the preemptor Pod is nominated to the node, this Pod can take over the place.
					Name: "create a second Pod that is higher priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-super-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(200).Obj(),
					},
				},
				{
					Name: "schedule the super-high-priority Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-super-high-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "check the super-high-priority Pod making the preemption API calls",
					PodRunningPreemption: new(5),
				},
				{
					// the super-high-priority preemptor should enter the preemption
					// and select the place where the preemptor-high-priority selected.
					// So, basically both goroutines are preempting the same Pods.
					Name:            "check the super-high-priority pod is in the queue and gated",
					PodGatedInQueue: "preemptor-super-high-priority",
				},
				{
					Name:               "complete the preemption API calls of super-high-priority",
					CompletePreemption: "preemptor-super-high-priority",
				},
				{
					Name:               "complete the preemption API calls of high-priority",
					CompletePreemption: "preemptor-high-priority",
				},
				{
					Name: "schedule the super-high-priority Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-super-high-priority",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule the high-priority Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-high-priority",
						ExpectUnschedulable: true,
					},
				},
			},
		},
		{
			Name: "Lower priority Pod can select the same place where the higher priority Pod is preempting if the node is big enough",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create scheduled Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					// It will preempt two victims.
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-high-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the pod is in the queue and gated",
					PodGatedInQueue: "preemptor-high-priority",
				},
				{
					Name:                 "check the preemptor Pod making the preemption API calls",
					PodRunningPreemption: new(4),
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor Pod is nominated to the node, this Pod should be unschedulable.
					// This Pod will trigger the preemption to target the two victims that the first Pod doesn't target.
					Name: "create a second Pod that is lower priority than the first preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					Name: "schedule the mid-priority Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor-mid-priority",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:            "check the mid-priority pod is in the queue and gated",
					PodGatedInQueue: "preemptor-mid-priority",
				},
				{
					Name:                 "check the mid-priority Pod making the preemption API calls",
					PodRunningPreemption: new(5),
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "preemptor-mid-priority",
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "preemptor-high-priority",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					Name: "schedule the preemptor Pod again",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-high-priority",
						ExpectSuccess: true,
					},
				},
				{
					Name: "schedule the mid-priority Pod again",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-mid-priority",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Delete event and is being scheduled, even before the victim binding is terminated.
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			Name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is reserving some resources required by the preemptor:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted), but is still unschedulable, because victim has to unreserve its resources.
			// After resuming binding for a victim, it releases the resources in its failure handler, preemptor is woken up again and ultimately scheduled.
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted and its resources are unreserved",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName,
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			Name: "gated preemptor is eventually scheduled even if victim deletion doesn't raise queue hints",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim pods",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName(fmt.Sprintf("victim-%s-", asyncframework.BlockingPodName)).Node("node").Priority(1).Container("image").ZeroTerminationGracePeriod().Obj(),
						Count: new(2),
					},
				},
				{
					Name: "create preemptor",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Priority(100).Container("image").Obj(),
					},
				},
				{
					Name: "schedule preemptor",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:                 "verify preemptor running preemption",
					PodRunningPreemption: new(2),
				},
				{
					Name:            "gate preemptor",
					PodGatedInQueue: "preemptor",
				},
				{
					Name:               "complete preemption",
					CompletePreemption: "preemptor",
				},
				{
					Name:               "wait for victims to be deleted",
					WaitForPodsDeleted: []int{0, 1},
				},
				{
					Name:                     "verify preemptor is still in unschedulable queue",
					VerifyPodInUnschedulable: "preemptor",
				},
				{
					Name:               "flush scheduling queue",
					FlushUnschedulable: true,
				},
				{
					Name: "verify preemptor scheduled",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
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
			Name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is under graceful termination and its resources are unreserved",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create victim Pod that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					Name: "schedule victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName + asyncframework.ReservingPodName,
					},
				},
				{
					Name: "create a preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API call",
					CompletePreemption: "preemptor",
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be unschedulable (resources are still reserved by the victim)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:             "preemptor",
						ExpectUnschedulable: true,
					},
				},
				{
					Name:       "resume binding of the blocked pod",
					ResumeBind: true,
				},
				{
					Name: "schedule the preemptor Pod again and expect it to be scheduled (victim pod unreserved its resources)",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor",
						ExpectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies that when preemption is in progress and a higher priotiy pod comes it will take place created for lower priority preemptor.
			// The lower priority Pod switches to another node, does not get stuck in unschedulable queue forever.
			Name: "While lower priority Pod is waiting for preemption, higher priority Pod takes its place on the node",
			Steps: []asyncframework.Step{
				{
					Name:       "create Node",
					CreateNode: "node",
				},
				{
					Name: "create N-1 victim Pods on the first node",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(3),
					},
				},
				{
					Name: "create the last victim Pod on the first node, that is going to be blocked in binding",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name(asyncframework.PodBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					Name: "schedule the last victim Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: asyncframework.PodBlockedInBindingName,
					},
				},
				{
					Name: "create a mid-priority preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).Obj(),
					},
				},
				{
					Name: "schedule the mid-priority preemptor Pod",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: "preemptor-mid-priority",
					},
				},
				{
					Name:               "complete the preemption API calls",
					CompletePreemption: "preemptor-mid-priority",
				},
				{
					Name:            "check the mid-priority preemptor Pod is gated, waiting for the last victim to be preempted",
					PodGatedInQueue: "preemptor-mid-priority",
				},
				{
					Name:       "create node2",
					CreateNode: "node2",
				},
				{
					Name: "create victim Pods on node2",
					CreatePod: &asyncframework.CreatePod{
						Pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node2").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						Count: new(4),
					},
				},
				{
					Name: "create a high-priority preemptor Pod",
					CreatePod: &asyncframework.CreatePod{
						Pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name: "schedule the high-priority preemptor Pod and expect it to get scheduled on node1",
					// While we don't check explicitly that Pod is scheduled on node1, we can assume that because
					// Pod won't fit on node2 without preemption and there are enough resources on node1.
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-high-priority",
						ExpectSuccess: true,
					},
				},
				{
					Name:       "allow the preemption of the last victim Pod on node1 to finish",
					ResumeBind: true,
				},
				{
					Name: "check that mid-priority preemptor Pod got activated by completed preemption and try scheduling it again",
					SchedulePod: &asyncframework.SchedulePod{
						PodName: "preemptor-mid-priority",
						// Pod won't fit on node1 anymore and should trigger preemptions on node2.
						ExpectUnschedulable: true,
					},
				},
				{
					Name:               "complete the preemption API calls on node2",
					CompletePreemption: "preemptor-mid-priority",
				},
				{
					Name: "check that mid-priority Pod got activated, schedule it on node2",
					SchedulePod: &asyncframework.SchedulePod{
						PodName:       "preemptor-mid-priority",
						ExpectSuccess: true,
					},
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			// map[string]chan struct{}
			preemptionDoneChannels := &sync.Map{}
			defer func() {
				preemptionDoneChannels.Range(func(key, value any) bool {
					ch := value.(chan struct{})
					select {
					case <-ch:
					default:
						close(ch)
					}
					return true
				})
			}()

			blockBindingChannel := make(chan struct{})
			defer close(blockBindingChannel)
			preemptionConfig := asyncframework.AsyncPreemptionTestConfig{
				EnableGenericWorkload:  false,
				PreemptionDoneChannels: preemptionDoneChannels,
				BlockBindingChannel:    blockBindingChannel,
			}
			testCtx, preemptionPlugin, cs := asyncframework.InitTestForAsyncPreemption(t, preemptionConfig)

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

			config := asyncframework.AsyncPreemptionStepRunnerConfig{
				CreatedPods:            createdPods,
				ClientSet:              cs,
				PreemptionDoneChannels: preemptionDoneChannels,
				Logger:                 logger,
				PreemptionPlugin:       preemptionPlugin,
				BlockBindingChannel:    blockBindingChannel,
			}
			asyncframework.RunAsyncPreemptionSteps(testCtx, t, test.Steps, config)
		})
	}
}

// There is equivalent test for pod group preemption at: test/integration/scheduler/preemption/podgroup/podgrouppreemption_test.go
// When adding new test cases for pod group preemption with waiting pods, add them to this test.
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

	victim := st.MakePod().Name("victim").Priority(asyncframework.LowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).Obj()
	// Preemptor requires more resources than the small node has.
	preemptor := st.MakePod().Name("preemptor").Priority(asyncframework.HighPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5", v1.ResourceMemory: "1.5Gi"}).Obj()

	// Register the blocking plugin
	victimToBlock := &asyncframework.BlockedPod{
		Blocked: make(chan struct{}),
	}
	podsToBlock := map[string]*asyncframework.BlockedPod{
		victim.Name: victimToBlock,
	}

	registry := make(frameworkruntime.Registry)
	err := registry.Register(asyncframework.BlockingPermitPluginName, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return asyncframework.NewBlockingPermitPlugin(ctx, obj, fh, podsToBlock), nil
	})
	if err != nil {
		t.Fatalf("Error registering plugin: %v", err)
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: new(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Permit: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: asyncframework.BlockingPermitPluginName},
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
	defer testCtx.SchedulerCloseFn()
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
	case <-victimToBlock.Blocked:
		t.Logf("Victim reached WaitOnPermit")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timed out waiting for victim to reach WaitOnPermit")
	}
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
		return testCtx.Scheduler.Profiles[v1.DefaultSchedulerName].GetWaitingPod(victim.UID) != nil, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for victim to be recorded as a waiting pod: %v", err)
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
	victim := st.MakePod().Name("victim").Priority(asyncframework.LowPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "1Gi"}).Obj()
	// Preemptor also requires full node resources.
	preemptor := st.MakePod().Name("preemptor").Priority(asyncframework.HighPriority).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1.5", v1.ResourceMemory: "1.5Gi"}).Obj()

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
			SchedulerName: new(v1.DefaultSchedulerName),
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
	defer testCtx.SchedulerCloseFn()
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

var defaultPodRes = &v1.ResourceRequirements{Requests: v1.ResourceList{
	v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
	v1.ResourceMemory: *resource.NewQuantity(100, resource.DecimalSI)},
}

func makePod(name string, priority int32, labelKey, labelValue string) *v1.Pod {
	pod := st.MakePod().Name(name).Priority(priority).Label(labelKey, labelValue).Container("pause").Obj()
	pod.Spec.Containers[0].Resources = *defaultPodRes
	return pod
}

func makePodWithAntiAffinityNode(name string, priority int32, labelKey, labelValue, antiAffinityVal string) *v1.Pod {
	pod := st.MakePod().Name(name).Priority(priority).Label(labelKey, labelValue).Container("pause").
		PodAntiAffinityIn("pod", "node", []string{antiAffinityVal}, st.PodAntiAffinityWithRequiredReq).Obj()
	pod.Spec.Containers[0].Resources = *defaultPodRes
	return pod
}

func makePodWithAntiAffinityHostname(name string, priority int32, labelKey, labelValue, antiAffinityVal string) *v1.Pod {
	pod := st.MakePod().Name(name).Priority(priority).Label(labelKey, labelValue).Container("pause").
		PodAntiAffinityIn("pod", v1.LabelHostname, []string{antiAffinityVal}, st.PodAntiAffinityWithRequiredReq).Obj()
	pod.Spec.Containers[0].Resources = *defaultPodRes
	return pod
}

func TestInterPodAffinityPreemption(t *testing.T) {
	type createPod struct {
		pod *v1.Pod
	}

	type schedulePod struct {
		podName             string
		expectSuccess       bool
		expectUnschedulable bool
	}

	type scenario struct {
		name                string
		createPod           *createPod
		createNode          string
		schedulePod         *schedulePod
		verifyPodDeleted    string
		verifyPodNotDeleted string
		waitForNominated    string
	}

	// A test node can fit up to 5 pods with defaultPodRes.

	maxTokens := 1000

	tests := []struct {
		name                   string
		initTokens             int
		enablePreFilter        bool
		unresolvable           bool
		scenarios              []scenario
		genericWorkloadEnabled bool
	}{
		{
			name:       "preemption is performed to satisfy anti-affinity",
			initTokens: maxTokens,
			scenarios: []scenario{
				{
					name: "create pod-0",
					createPod: &createPod{
						pod: makePod("pod-0", asyncframework.MediumPriority, "pod", "p0"),
					},
				},
				{
					name: "schedule pod-0",
					schedulePod: &schedulePod{
						podName:       "pod-0",
						expectSuccess: true,
					},
				},
				{
					name: "create low priority pod-1 that can be preempted by preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityNode("pod-1", asyncframework.LowPriority, "pod", "p1", "preemptor"),
					},
				},
				{
					name: "schedule pod-1",
					schedulePod: &schedulePod{
						podName:       "pod-1",
						expectSuccess: true,
					},
				},
				{
					name: "create preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityNode("preemptor-pod", asyncframework.HighPriority, "pod", "preemptor", "p0"),
					},
				},
				{
					name: "schedule preemptor",
					schedulePod: &schedulePod{
						podName:             "preemptor-pod",
						expectUnschedulable: true,
					},
				},
				{
					name:             "verify pod-0 is preempted",
					verifyPodDeleted: "pod-0",
				},
				{
					name:             "verify pod-1 is preempted",
					verifyPodDeleted: "pod-1",
				},
				{
					name:             "verify nominated node name is set",
					waitForNominated: "preemptor-pod",
				},
			},
		},
		{
			name:       "preemption is not performed when anti-affinity is not satisfied",
			initTokens: maxTokens,
			scenarios: []scenario{
				{
					name: "create pod-0",
					createPod: &createPod{
						pod: makePod("pod-0", asyncframework.MediumPriority, "pod", "p0"),
					},
				},
				{
					name: "schedule pod-0",
					schedulePod: &schedulePod{
						podName:       "pod-0",
						expectSuccess: true,
					},
				},
				{
					name: "create high priority pod-1 that cannot be preempted by preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityNode("pod-1", asyncframework.HighPriority, "pod", "p1", "preemptor"),
					},
				},
				{
					name: "schedule pod-1",
					schedulePod: &schedulePod{
						podName:       "pod-1",
						expectSuccess: true,
					},
				},
				{
					name: "create preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityNode("preemptor-pod", asyncframework.HighPriority, "pod", "preemptor", "p0"),
					},
				},
				{
					name: "schedule preemptor",
					schedulePod: &schedulePod{
						podName:             "preemptor-pod",
						expectUnschedulable: true,
					},
				},
				{
					name:                "verify pod-0 is not preempted",
					verifyPodNotDeleted: "pod-0",
				},
				{
					name:                "verify pod-1 is not preempted",
					verifyPodNotDeleted: "pod-1",
				},
			},
		},
		{
			name:       "basic pod preemption with hostname anti-affinity (fast path)",
			initTokens: maxTokens,
			scenarios: []scenario{
				{
					name: "create pod-0",
					createPod: &createPod{
						pod: makePod("pod-0", asyncframework.MediumPriority, "pod", "p0"),
					},
				},
				{
					name: "schedule pod-0",
					schedulePod: &schedulePod{
						podName:       "pod-0",
						expectSuccess: true,
					},
				},
				{
					name: "create low priority pod-1 that can be preempted by preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityHostname("pod-1", asyncframework.LowPriority, "pod", "p1", "preemptor"),
					},
				},
				{
					name: "schedule pod-1",
					schedulePod: &schedulePod{
						podName:       "pod-1",
						expectSuccess: true,
					},
				},
				{
					name: "create preemptor",
					createPod: &createPod{
						pod: makePodWithAntiAffinityHostname("preemptor-pod", asyncframework.HighPriority, "pod", "preemptor", "p0"),
					},
				},
				{
					name: "schedule preemptor",
					schedulePod: &schedulePod{
						podName:             "preemptor-pod",
						expectUnschedulable: true,
					},
				},
				{
					name:             "verify pod-0 is preempted",
					verifyPodDeleted: "pod-0",
				},
				{
					name:             "verify pod-1 is preempted",
					verifyPodDeleted: "pod-1",
				},
				{
					name:             "verify nominated node name is set",
					waitForNominated: "preemptor-pod",
				},
			},
		},
	}

	for _, clearingNominatedNodeNameAfterBinding := range []bool{true, false} {
		for _, genericOpts := range []struct{ genericWorkloadEnabled, cpgEnabled bool }{
			{genericWorkloadEnabled: true, cpgEnabled: true},
			{genericWorkloadEnabled: true, cpgEnabled: false},
			{genericWorkloadEnabled: false, cpgEnabled: false},
		} {
			genericWorkloadEnabled := genericOpts.genericWorkloadEnabled
			cpgEnabled := genericOpts.cpgEnabled
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                       genericWorkloadEnabled,
				features.CompositePodGroup:                     cpgEnabled,
				features.TopologyAwareWorkloadScheduling:       cpgEnabled,
				features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
			})
			sharedAPICtx := testutils.InitTestAPIServer(t, "preemption", nil)

			for _, asyncPreemptionEnabled := range []bool{true, false} {
				for _, fpEnabled := range []bool{true, false} {
					for _, test := range tests {
						nameSuffix := fmt.Sprintf("Async preemption enabled: %v, ClearingNominatedNodeNameAfterBinding: %v, fpEnabled: %v, genericWorkloadEnabled: %v, cpgEnabled: %v", asyncPreemptionEnabled, clearingNominatedNodeNameAfterBinding, fpEnabled, genericWorkloadEnabled, cpgEnabled)
						t.Run(fmt.Sprintf("%s (%s)", test.name, nameSuffix), func(t *testing.T) {
							// Feature gates map to test combinations.
							featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
								features.SchedulerAsyncPreemption:              asyncPreemptionEnabled,
								features.ClearingNominatedNodeNameAfterBinding: clearingNominatedNodeNameAfterBinding,
								features.InterPodAffinityHostnameFastPath:      fpEnabled,
								features.GenericWorkload:                       genericWorkloadEnabled,
								features.CompositePodGroup:                     cpgEnabled,
								features.TopologyAwareWorkloadScheduling:       cpgEnabled,
							})

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
									SchedulerName: new(v1.DefaultSchedulerName),
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

							testCtx := testutils.InitTestSchedulerWithOptions(t,
								withNewNamespace(t, sharedAPICtx, "preemption"),
								0,
								scheduler.WithProfiles(cfg.Profiles...),
								scheduler.WithFrameworkOutOfTreeRegistry(registry),
								// disable backoff
								scheduler.WithPodMaxBackoffSeconds(0),
								scheduler.WithPodInitialBackoffSeconds(0),
							)
							defer testCtx.SchedulerCloseFn()
							testutils.SyncSchedulerInformerFactory(testCtx)

							cs := testCtx.ClientSet

							logger, _ := ktesting.NewTestContext(t)
							if testCtx.Scheduler.APIDispatcher != nil {
								testCtx.Scheduler.APIDispatcher.Run(logger)
								defer testCtx.Scheduler.APIDispatcher.Close()
							}
							testCtx.Scheduler.SchedulingQueue.Run(logger)
							defer testCtx.Scheduler.SchedulingQueue.Close()

							ctx, cancel := context.WithCancel(context.Background())
							defer cancel()
							defer testCtx.SchedulerCloseFn()

							// Create a node with some resources and a label.
							nodeRes := map[v1.ResourceName]string{
								v1.ResourcePods:   "32",
								v1.ResourceCPU:    "500m",
								v1.ResourceMemory: "500",
							}
							nodeObject := st.MakeNode().Name("node1").Capacity(nodeRes).Label("node", "node1").Label(v1.LabelHostname, "node1").Obj()

							if _, err := cs.CoreV1().Nodes().Create(ctx, nodeObject, metav1.CreateOptions{}); err != nil {
								t.Fatalf("Failed to create an initial Node %q: %v", nodeObject.Name, err)
							}
							defer func() {
								if err := cs.CoreV1().Nodes().Delete(ctx, nodeObject.Name, metav1.DeleteOptions{}); err != nil {
									t.Fatalf("Failed to delete the Node %q: %v", nodeObject.Name, err)
								}
							}()

							filter.Tokens = test.initTokens
							filter.EnablePreFilter = test.enablePreFilter
							filter.Unresolvable = test.unresolvable

							var createdPods []*v1.Pod
							defer func() {
								testutils.CleanupPods(testCtx.Ctx, cs, t, createdPods)
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
									pod, err := cs.CoreV1().Pods(testCtx.NS.Name).Create(ctx, scenario.createPod.pod, metav1.CreateOptions{})
									if err != nil {
										t.Fatalf("Failed to create a Pod %q: %v", pod.Name, err)
									}
									createdPods = append(createdPods, pod)
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

									testCtx.Scheduler.ScheduleOne(testCtx.Ctx)

									if scenario.schedulePod.expectSuccess {
										if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodScheduled(cs, testCtx.NS.Name, scenario.schedulePod.podName)); err != nil {
											t.Fatalf("Expected the pod %s to be scheduled", scenario.schedulePod.podName)
										}
									} else if scenario.schedulePod.expectUnschedulable {
										if !asyncframework.PodInUnschedulablePodPool(t, testCtx.Scheduler.SchedulingQueue, scenario.schedulePod.podName) {
											t.Fatalf("Expected the pod %s to be in the unschedulable queue after the scheduling attempt", scenario.schedulePod.podName)
										}
									}
								case scenario.verifyPodDeleted != "":
									if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, wait.ForeverTestTimeout, false, testutils.PodDeleted(testCtx.Ctx, cs, testCtx.NS.Name, scenario.verifyPodDeleted)); err != nil {
										t.Fatalf("Failed to wait for pod %s to be deleted: %v", scenario.verifyPodDeleted, err)
									}
								case scenario.verifyPodNotDeleted != "":
									// Wait a small bit to ensure it doesn't get evicted!
									time.Sleep(200 * time.Millisecond)
									p, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, scenario.verifyPodNotDeleted, metav1.GetOptions{})
									if err != nil {
										t.Errorf("Error getting pod %v: %v", scenario.verifyPodNotDeleted, err)
									} else if p.DeletionTimestamp != nil {
										t.Errorf("Pod %v was unexpectedly preempted", p.Name)
									} else {
										_, cond := podutil.GetPodCondition(&p.Status, v1.DisruptionTarget)
										if cond != nil {
											t.Errorf("Pod %q was evicted unexpectedly.", klog.KObj(p))
										}
									}
								case scenario.waitForNominated != "":
									if !clearingNominatedNodeNameAfterBinding {
										preemptor, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, scenario.waitForNominated, metav1.GetOptions{})
										if err != nil {
											t.Fatalf("Failed to get preemptor pod: %v", err)
										}
										if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, preemptor); err != nil {
											t.Errorf("NominatedNodeName field was not set for pod %v: %v", preemptor.Name, err)
										}
									}
								}
							}
						})
					}
				}
			}
		}
	}
}

// withNewNamespace creates a child TestContext that shares the API server from
// parent but gets a fresh namespace. Only the namespace is deleted on t.Cleanup;
// the API server lifecycle is managed by the caller. This is useful when
// multiple subtests share one API server to avoid the per-subtest startup cost.
func withNewNamespace(t *testing.T, parent *testutils.TestContext, nsPrefix string) *testutils.TestContext {
	t.Helper()
	ctx, cancel := context.WithCancel(parent.Ctx)
	child := &testutils.TestContext{
		ClientSet:  parent.ClientSet,
		KubeConfig: parent.KubeConfig,
		Ctx:        ctx,
		CloseFn:    func() {}, // API server is owned by parent; do not tear it down here.
	}
	child.NS = testfwk.CreateNamespaceOrDie(child.ClientSet, nsPrefix+string(k8suuid.NewUUID()), t)
	t.Cleanup(func() {
		cancel()
		testfwk.DeleteNamespaceOrDie(child.ClientSet, child.NS, t)
	})
	return child
}
