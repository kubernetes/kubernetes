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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	internalinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller/node"
	"k8s.io/kubernetes/pkg/controller/node/ipam"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestTaintNodeByCondition verifies:
//   1. MemoryPressure Toleration is added to non-BestEffort Pod by PodTolerationRestriction
//   2. NodeController taints nodes by node condition
//   3. Scheduler allows pod to tolerate node condition taints, e.g. network unavailabe
func TestTaintNodeByCondition(t *testing.T) {
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	// Enable TaintNodeByCondition
	utilfeature.DefaultFeatureGate.Set("TaintNodesByCondition=True")

	// Build clientset and informers for controllers.
	internalClientset := internalclientset.NewForConfigOrDie(&restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()}})
	internalInformers := internalinformers.NewSharedInformerFactory(internalClientset, time.Second)

	clientset := clientset.NewForConfigOrDie(&restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()}})
	informers := informers.NewSharedInformerFactory(clientset, time.Second)

	// Build PodToleration Admission.
	admission := podtolerationrestriction.NewPodTolerationsPlugin(&pluginapi.Configuration{})
	kubeadmission.WantsInternalKubeClientSet(admission).SetInternalKubeClientSet(internalClientset)
	kubeadmission.WantsInternalKubeInformerFactory(admission).SetInternalKubeInformerFactory(internalInformers)

	// Start master with admission.
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.AdmissionControl = admission
	_, _, closeFn := framework.RunAMasterUsingServer(masterConfig, s, h)
	defer closeFn()

	nsName := "default"
	controllerCh := make(chan struct{})
	defer close(controllerCh)

	// Start NodeController for taint.
	nc, err := node.NewNodeController(
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Extensions().V1beta1().DaemonSets(),
		nil, // CloudProvider
		clientset,
		time.Second, // Pod eviction timeout
		100,         // Eviction limiter QPS
		100,         // Secondary eviction limiter QPS
		100,         // Large cluster threshold
		100,         // Unhealthy zone threshold
		time.Second, // Node monitor grace period
		time.Second, // Node startup grace period
		time.Second, // Node monitor period
		nil,         // Cluster CIDR
		nil,         // Service CIDR
		0,           // Node CIDR mask size
		false,       // Allocate node CIDRs
		ipam.RangeAllocatorType, // Allocator type
		true, // Run taint manger
		true, // Enabled taint based eviction
		true, // Enabled TaintNodeByCondition feature
	)
	if err != nil {
		t.Errorf("Failed to create node controller: %v", err)
		return
	}
	go nc.Run(controllerCh)

	// Apply feature gates to enable TaintNodesByCondition
	algorithmprovider.ApplyFeatureGates()

	// Start scheduler
	configurator := factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		clientset,
		informers.Core().V1().Nodes(),
		informers.Core().V1().Pods(),
		informers.Core().V1().PersistentVolumes(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Core().V1().ReplicationControllers(),
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Apps().V1beta1().StatefulSets(),
		informers.Core().V1().Services(),
		informers.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		true, // Enable EqualCache by default.
	)

	sched, err := scheduler.NewFromConfigurator(configurator, func(cfg *scheduler.Config) {
		cfg.StopEverything = controllerCh
		cfg.Recorder = &record.FakeRecorder{}
	})
	if err != nil {
		t.Errorf("Failed to create scheduler: %v.", err)
		return
	}
	go sched.Run()

	// Waiting for all controller sync.
	informers.Start(controllerCh)
	internalInformers.Start(controllerCh)

	informers.WaitForCacheSync(controllerCh)
	internalInformers.WaitForCacheSync(controllerCh)

	// -------------------------------------------
	// Test TaintNodeByCondition feature.
	// -------------------------------------------
	memoryPressureToleration := v1.Toleration{
		Key:      algorithm.TaintNodeMemoryPressure,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoSchedule,
	}

	// Case 1: Add MememoryPressure Toleration for non-BestEffort pod.
	burstablePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "burstable-pod",
			Namespace: nsName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "busybox",
					Image: "busybox",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				},
			},
		},
	}

	burstablePodInServ, err := clientset.CoreV1().Pods(nsName).Create(burstablePod)
	if err != nil {
		t.Errorf("Case 1: Failed to create pod: %v", err)
	} else if !reflect.DeepEqual(burstablePodInServ.Spec.Tolerations, []v1.Toleration{memoryPressureToleration}) {
		t.Errorf("Case 1: Unexpected toleration of non-BestEffort pod, expected: %+v, got: %v",
			[]v1.Toleration{memoryPressureToleration},
			burstablePodInServ.Spec.Tolerations)
	}

	// Case 2: No MemoryPressure Toleration for BestEffort pod.
	besteffortPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "best-effort-pod",
			Namespace: nsName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "busybox",
					Image: "busybox",
				},
			},
		},
	}

	besteffortPodInServ, err := clientset.CoreV1().Pods(nsName).Create(besteffortPod)
	if err != nil {
		t.Errorf("Case 2: Failed to create pod: %v", err)
	} else if len(besteffortPodInServ.Spec.Tolerations) != 0 {
		t.Errorf("Case 2: Unexpected toleration # of BestEffort pod, expected: 0, got: %v",
			len(besteffortPodInServ.Spec.Tolerations))
	}

	// Case 3: Taint Node by NetworkUnavailable condition.
	networkUnavailableNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4000m"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4000m"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeNetworkUnavailable,
					Status: v1.ConditionTrue,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionFalse,
				},
			},
		},
	}

	nodeInformerCh := make(chan bool)
	nodeInformer := informers.Core().V1().Nodes().Informer()
	nodeInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, cur interface{}) {
			curNode := cur.(*v1.Node)
			for _, taint := range curNode.Spec.Taints {
				if taint.Key == algorithm.TaintNodeNetworkUnavailable &&
					taint.Effect == v1.TaintEffectNoSchedule {
					nodeInformerCh <- true
					break
				}
			}
		},
	})

	if _, err := clientset.CoreV1().Nodes().Create(networkUnavailableNode); err != nil {
		t.Errorf("Case 3: Failed to create node: %v", err)
	} else {
		select {
		case <-time.After(60 * time.Second):
			t.Errorf("Case 3: Failed to taint node after 60s.")
		case <-nodeInformerCh:
		}
	}

	// Case 4: Schedule Pod with NetworkUnavailable toleration.
	networkDaemonPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "network-daemon-pod",
			Namespace: nsName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "busybox",
					Image: "busybox",
				},
			},
			Tolerations: []v1.Toleration{
				{
					Key:      algorithm.TaintNodeNetworkUnavailable,
					Operator: v1.TolerationOpExists,
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	if _, err := clientset.CoreV1().Pods(nsName).Create(networkDaemonPod); err != nil {
		t.Errorf("Case 4: Failed to create pod for network daemon: %v", err)
	} else {
		if err := waitForPodToScheduleWithTimeout(clientset, networkDaemonPod, time.Second*60); err != nil {
			t.Errorf("Case 4: Failed to schedule network daemon pod in 60s.")
		}
	}
}
