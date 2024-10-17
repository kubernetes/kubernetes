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

package taint

// This file tests the Taint feature.

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func newPod(nsName, name string, req, limit v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: nsName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "busybox",
					Image: "busybox",
					Resources: v1.ResourceRequirements{
						Requests: req,
						Limits:   limit,
					},
				},
			},
		},
	}
}

// TestTaintNodeByCondition tests related cases for TaintNodeByCondition feature.
func TestTaintNodeByCondition(t *testing.T) {
	// Build PodToleration Admission.
	admission := podtolerationrestriction.NewPodTolerationsPlugin(&pluginapi.Configuration{})

	testCtx := testutils.InitTestAPIServer(t, "taint-node-by-condition", admission)

	// Build clientset and informers for controllers.
	externalClientConfig := restclient.CopyConfig(testCtx.KubeConfig)
	externalClientConfig.QPS = -1
	externalClientset := kubernetes.NewForConfigOrDie(externalClientConfig)
	externalInformers := informers.NewSharedInformerFactory(externalClientset, 0)

	admission.SetExternalKubeClientSet(externalClientset)
	admission.SetExternalKubeInformerFactory(externalInformers)

	testCtx = testutils.InitTestScheduler(t, testCtx)

	cs := testCtx.ClientSet
	nsName := testCtx.NS.Name

	// Start NodeLifecycleController for taint.
	nc, err := nodelifecycle.NewNodeLifecycleController(
		testCtx.Ctx,
		externalInformers.Coordination().V1().Leases(),
		externalInformers.Core().V1().Pods(),
		externalInformers.Core().V1().Nodes(),
		cs,
		time.Hour,   // Node monitor grace period
		time.Second, // Node startup grace period
		time.Second, // Node monitor period
		100,         // Eviction limiter QPS
		100,         // Secondary eviction limiter QPS
		100,         // Large cluster threshold
		100,         // Unhealthy zone threshold
	)
	if err != nil {
		t.Errorf("Failed to create node controller: %v", err)
		return
	}

	// Waiting for all controllers to sync
	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())
	testutils.SyncSchedulerInformerFactory(testCtx)

	// Run all controllers
	go nc.Run(testCtx.Ctx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	// -------------------------------------------
	// Test TaintNodeByCondition feature.
	// -------------------------------------------
	nodeRes := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("4000m"),
		v1.ResourceMemory: resource.MustParse("16Gi"),
		v1.ResourcePods:   resource.MustParse("110"),
	}

	podRes := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("100m"),
		v1.ResourceMemory: resource.MustParse("100Mi"),
	}

	notReadyToleration := v1.Toleration{
		Key:      v1.TaintNodeNotReady,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	unschedulableToleration := v1.Toleration{
		Key:      v1.TaintNodeUnschedulable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	memoryPressureToleration := v1.Toleration{
		Key:      v1.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	diskPressureToleration := v1.Toleration{
		Key:      v1.TaintNodeDiskPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	networkUnavailableToleration := v1.Toleration{
		Key:      v1.TaintNodeNetworkUnavailable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	pidPressureToleration := v1.Toleration{
		Key:      v1.TaintNodePIDPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	bestEffortPod := newPod(nsName, "besteffort-pod", nil, nil)
	burstablePod := newPod(nsName, "burstable-pod", podRes, nil)
	guaranteePod := newPod(nsName, "guarantee-pod", podRes, podRes)

	type podCase struct {
		pod         *v1.Pod
		tolerations []v1.Toleration
		fits        bool
	}

	// switch to table driven testings
	tests := []struct {
		name           string
		existingTaints []v1.Taint
		nodeConditions []v1.NodeCondition
		unschedulable  bool
		expectedTaints []v1.Taint
		pods           []podCase
	}{
		{
			name: "not-ready node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionFalse,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeNotReady,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{notReadyToleration},
					fits:        true,
				},
			},
		},
		{
			name:          "unschedulable node",
			unschedulable: true, // node.spec.unschedulable = true
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeUnschedulable,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{unschedulableToleration},
					fits:        true,
				},
			},
		},
		{
			name: "memory pressure node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeMemoryPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeMemoryPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			// In MemoryPressure condition, both Burstable and Guarantee pods are scheduled;
			// BestEffort pod with toleration are also scheduled.
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{memoryPressureToleration},
					fits:        true,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{diskPressureToleration},
					fits:        false,
				},
				{
					pod:  burstablePod,
					fits: true,
				},
				{
					pod:  guaranteePod,
					fits: true,
				},
			},
		},
		{
			name: "disk pressure node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeDiskPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeDiskPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			// In DiskPressure condition, only pods with toleration can be scheduled.
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{diskPressureToleration},
					fits:        true,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{memoryPressureToleration},
					fits:        false,
				},
			},
		},
		{
			name: "network unavailable and node is ready",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeNetworkUnavailable,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeNetworkUnavailable,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod: burstablePod,
					tolerations: []v1.Toleration{
						networkUnavailableToleration,
					},
					fits: true,
				},
			},
		},
		{
			name: "network unavailable and node is not ready",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeNetworkUnavailable,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionFalse,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeNetworkUnavailable,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    v1.TaintNodeNotReady,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod: burstablePod,
					tolerations: []v1.Toleration{
						networkUnavailableToleration,
					},
					fits: false,
				},
				{
					pod: burstablePod,
					tolerations: []v1.Toleration{
						networkUnavailableToleration,
						notReadyToleration,
					},
					fits: true,
				},
			},
		},
		{
			name: "pid pressure node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodePIDPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodePIDPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:  bestEffortPod,
					fits: false,
				},
				{
					pod:  burstablePod,
					fits: false,
				},
				{
					pod:  guaranteePod,
					fits: false,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{pidPressureToleration},
					fits:        true,
				},
			},
		},
		{
			name: "multi taints on node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodePIDPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeMemoryPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeDiskPressure,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    v1.TaintNodeDiskPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    v1.TaintNodeMemoryPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    v1.TaintNodePIDPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-1",
				},
				Spec: v1.NodeSpec{
					Unschedulable: test.unschedulable,
					Taints:        test.existingTaints,
				},
				Status: v1.NodeStatus{
					Capacity:    nodeRes,
					Allocatable: nodeRes,
					Conditions:  test.nodeConditions,
				},
			}

			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create node, err: %v", err)
			}
			if err := testutils.WaitForNodeTaints(testCtx.Ctx, cs, node, test.expectedTaints); err != nil {
				node, err = cs.CoreV1().Nodes().Get(testCtx.Ctx, node.Name, metav1.GetOptions{})
				if err != nil {
					t.Errorf("Failed to get node <%s>", node.Name)
				}

				t.Errorf("Failed to taint node <%s>, expected: %v, got: %v, err: %v", node.Name, test.expectedTaints, node.Spec.Taints, err)
			}

			var pods []*v1.Pod
			for i, p := range test.pods {
				pod := p.pod.DeepCopy()
				pod.Name = fmt.Sprintf("%s-%d", pod.Name, i)
				pod.Spec.Tolerations = p.tolerations

				createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create pod %s/%s, error: %v",
						pod.Namespace, pod.Name, err)
				}

				pods = append(pods, createdPod)

				if p.fits {
					if err := testutils.WaitForPodToSchedule(cs, createdPod); err != nil {
						t.Errorf("Failed to schedule pod %s/%s on the node, err: %v",
							pod.Namespace, pod.Name, err)
					}
				} else {
					if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, cs, createdPod); err != nil {
						t.Errorf("Unschedulable pod %s/%s gets scheduled on the node, err: %v",
							pod.Namespace, pod.Name, err)
					}
				}
			}

			testutils.CleanupPods(testCtx.Ctx, cs, t, pods)
			testutils.CleanupNodes(cs, t)
			testutils.WaitForSchedulerCacheCleanup(testCtx.Ctx, testCtx.Scheduler, t)
		})
	}
}
