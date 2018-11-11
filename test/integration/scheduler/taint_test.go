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

package scheduler

// This file tests the Taint feature.

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
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
	enabled := utilfeature.DefaultFeatureGate.Enabled("TaintNodesByCondition")
	defer func() {
		if !enabled {
			utilfeature.DefaultFeatureGate.Set("TaintNodesByCondition=False")
		}
	}()
	// Enable TaintNodeByCondition
	utilfeature.DefaultFeatureGate.Set("TaintNodesByCondition=True")

	// Build PodToleration Admission.
	admission := podtolerationrestriction.NewPodTolerationsPlugin(&pluginapi.Configuration{})

	context := initTestMaster(t, "default", admission)

	// Build clientset and informers for controllers.
	externalClientset := kubernetes.NewForConfigOrDie(&restclient.Config{
		QPS:           -1,
		Host:          context.httpServer.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	externalInformers := informers.NewSharedInformerFactory(externalClientset, time.Second)

	admission.SetExternalKubeClientSet(externalClientset)
	admission.SetExternalKubeInformerFactory(externalInformers)

	// Apply feature gates to enable TaintNodesByCondition
	algorithmprovider.ApplyFeatureGates()

	context = initTestScheduler(t, context, false, nil)
	cs := context.clientSet
	informers := context.informerFactory
	nsName := context.ns.Name

	// Start NodeLifecycleController for taint.
	nc, err := nodelifecycle.NewNodeLifecycleController(
		informers.Coordination().V1beta1().Leases(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Extensions().V1beta1().DaemonSets(),
		nil, // CloudProvider
		cs,
		time.Hour,   // Node monitor grace period
		time.Second, // Node startup grace period
		time.Second, // Node monitor period
		time.Second, // Pod eviction timeout
		100,         // Eviction limiter QPS
		100,         // Secondary eviction limiter QPS
		100,         // Large cluster threshold
		100,         // Unhealthy zone threshold
		true,        // Run taint manager
		true,        // Use taint based evictions
		true,        // Enabled TaintNodeByCondition feature
	)
	if err != nil {
		t.Errorf("Failed to create node controller: %v", err)
		return
	}
	go nc.Run(context.stopCh)

	// Waiting for all controller sync.
	externalInformers.Start(context.stopCh)
	externalInformers.WaitForCacheSync(context.stopCh)
	informers.Start(context.stopCh)
	informers.WaitForCacheSync(context.stopCh)

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
		Key:      schedulerapi.TaintNodeNotReady,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	unreachableToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeUnreachable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	unschedulableToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeUnschedulable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	outOfDiskToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeOutOfDisk,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	memoryPressureToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	diskPressureToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeDiskPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	networkUnavailableToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodeNetworkUnavailable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	pidPressureToleration := v1.Toleration{
		Key:      schedulerapi.TaintNodePIDPressure,
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
					Key:    schedulerapi.TaintNodeNotReady,
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
			name: "unreachable node",
			existingTaints: []v1.Taint{
				{
					Key:    schedulerapi.TaintNodeUnreachable,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionUnknown, // node status is "Unknown"
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    schedulerapi.TaintNodeUnreachable,
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
					tolerations: []v1.Toleration{unreachableToleration},
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
					Key:    schedulerapi.TaintNodeUnschedulable,
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
			name: "out of disk node",
			nodeConditions: []v1.NodeCondition{
				{
					Type:   v1.NodeOutOfDisk,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    schedulerapi.TaintNodeOutOfDisk,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			// In OutOfDisk condition, only pods with toleration can be scheduled.
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
					tolerations: []v1.Toleration{outOfDiskToleration},
					fits:        true,
				},
				{
					pod:         bestEffortPod,
					tolerations: []v1.Toleration{diskPressureToleration},
					fits:        false,
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
					Key:    schedulerapi.TaintNodeMemoryPressure,
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
					Key:    schedulerapi.TaintNodeDiskPressure,
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
					Key:    schedulerapi.TaintNodeNetworkUnavailable,
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
					Key:    schedulerapi.TaintNodeNetworkUnavailable,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    schedulerapi.TaintNodeNotReady,
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
					Key:    schedulerapi.TaintNodePIDPressure,
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
					Key:    schedulerapi.TaintNodeDiskPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    schedulerapi.TaintNodeMemoryPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    schedulerapi.TaintNodePIDPressure,
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

			if _, err := cs.CoreV1().Nodes().Create(node); err != nil {
				t.Errorf("Failed to create node, err: %v", err)
			}
			if err := waitForNodeTaints(cs, node, test.expectedTaints); err != nil {
				t.Errorf("Failed to taint node <%s>, err: %v", node.Name, err)
			}

			var pods []*v1.Pod
			for i, p := range test.pods {
				pod := p.pod.DeepCopy()
				pod.Name = fmt.Sprintf("%s-%d", pod.Name, i)
				pod.Spec.Tolerations = p.tolerations

				createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
				if err != nil {
					t.Fatalf("Failed to create pod %s/%s, error: %v",
						pod.Namespace, pod.Name, err)
				}

				pods = append(pods, createdPod)

				if p.fits {
					if err := waitForPodToSchedule(cs, createdPod); err != nil {
						t.Errorf("Failed to schedule pod %s/%s on the node, err: %v",
							pod.Namespace, pod.Name, err)
					}
				} else {
					if err := waitForPodUnschedulable(cs, createdPod); err != nil {
						t.Errorf("Unschedulable pod %s/%s gets scheduled on the node, err: %v",
							pod.Namespace, pod.Name, err)
					}
				}
			}

			cleanupPods(cs, t, pods)
			cleanupNodes(cs, t)
			waitForSchedulerCacheCleanup(context.scheduler, t)
		})
	}
}
