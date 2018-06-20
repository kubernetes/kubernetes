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
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	internalinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
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

// TestTaintNodeByCondition tests related cases for TaintNodeByCondition features.
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
	internalClientset := internalclientset.NewForConfigOrDie(&restclient.Config{
		QPS:           -1,
		Host:          context.httpServer.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	internalInformers := internalinformers.NewSharedInformerFactory(internalClientset, time.Second)

	kubeadmission.WantsInternalKubeClientSet(admission).SetInternalKubeClientSet(internalClientset)
	kubeadmission.WantsInternalKubeInformerFactory(admission).SetInternalKubeInformerFactory(internalInformers)

	controllerCh := make(chan struct{})
	defer close(controllerCh)

	// Apply feature gates to enable TaintNodesByCondition
	algorithmprovider.ApplyFeatureGates()

	context = initTestScheduler(t, context, controllerCh, false, nil)
	cs := context.clientSet
	informers := context.informerFactory
	nsName := context.ns.Name

	// Start NodeLifecycleController for taint.
	nc, err := nodelifecycle.NewNodeLifecycleController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Extensions().V1beta1().DaemonSets(),
		nil, // CloudProvider
		cs,
		time.Second, // Node monitor grace period
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
	go nc.Run(controllerCh)

	// Waiting for all controller sync.
	internalInformers.Start(controllerCh)
	internalInformers.WaitForCacheSync(controllerCh)

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

	memoryPressureToleration := v1.Toleration{
		Key:      algorithm.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	networkUnavailableToleration := v1.Toleration{
		Key:      algorithm.TaintNodeNetworkUnavailable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	unschedulableToleration := v1.Toleration{
		Key:      algorithm.TaintNodeUnschedulable,
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

	tests := []struct {
		name           string
		nodeConditions []v1.NodeCondition
		unschedulable  bool
		expectedTaints []v1.Taint
		pods           []podCase
	}{
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
					Key:    algorithm.TaintNodeMemoryPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			// In MemoryPressure condition, both Burstable and Guarantee pods are scheduled;
			// BestEffort pods with toleration is also scheduled.
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
			name: "network unavailable node",
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
					Key:    algorithm.TaintNodeNetworkUnavailable,
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
					pod:         burstablePod,
					tolerations: []v1.Toleration{networkUnavailableToleration},
					fits:        true,
				},
			},
		},
		{
			name:          "unschedulable node",
			unschedulable: true,
			expectedTaints: []v1.Taint{
				{
					Key:    algorithm.TaintNodeUnschedulable,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			pods: []podCase{
				{
					pod:         burstablePod,
					tolerations: []v1.Toleration{unschedulableToleration},
					fits:        true,
				},
				{
					pod:  burstablePod,
					fits: false,
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
					Key:    algorithm.TaintNodeDiskPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    algorithm.TaintNodeMemoryPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    algorithm.TaintNodePIDPressure,
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	for i, test := range tests {
		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-1",
			},
			Spec: v1.NodeSpec{
				Unschedulable: test.unschedulable,
			},
			Status: v1.NodeStatus{
				Capacity:    nodeRes,
				Allocatable: nodeRes,
				Conditions:  test.nodeConditions,
			},
		}

		if _, err := cs.CoreV1().Nodes().Create(node); err != nil {
			t.Errorf("Case %d (%s): Failed to create node, err: %v", i, test.name, err)
		}

		if err := waitForNodeTaints(cs, node, test.expectedTaints); err != nil {
			t.Errorf("Case %d (%s): Failed to taint node, err: %v", i, test.name, err)
		}

		var pods []*v1.Pod

		for j, p := range test.pods {
			pod := p.pod.DeepCopy()
			pod.Name = fmt.Sprintf("%s-%d", pod.Name, j)
			pod.Spec.Tolerations = p.tolerations

			createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
			if err != nil {
				t.Fatalf("Case %d (%s): Failed to create pod %s/%s, error: %v",
					i, test.name, pod.Namespace, pod.Name, err)
			}

			pods = append(pods, createdPod)

			if p.fits {
				if err := waitForPodToSchedule(cs, createdPod); err != nil {
					t.Errorf("Case %d (%s): Failed to schedule pod %s/%s on the node, err: %v", i, test.name,
						pod.Namespace, pod.Name, err)
				}
			} else {
				if err := waitForPodUnschedulable(cs, createdPod); err != nil {
					t.Errorf("Case %d (%s): Unscheduble pod %s/%s gets scheduled on the node, err: %v", i, test.name,
						pod.Namespace, pod.Name, err)
				}
			}
		}

		cleanupPods(cs, t, pods)
		cleanupNodes(cs, t)
	}
}
