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

// This file tests the VolumeScheduling feature.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	affinityLabelKey = "kubernetes.io/hostname"
)

func TestLocalPVNegativeAffinity(t *testing.T) {
	config := setupNodes(t, "volume-scheduling", 3)
	defer config.teardown()

	pv := makeHostBoundPV(t, "local-pv", classImmediate, "", "", "node-1")
	pvc := makePVC("local-pvc", config.ns, &classImmediate, "")

	// Create PV
	if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
		t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
	}

	// Create PVC
	if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
		t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
	}

	nodeMarkers := []interface{}{
		markNodeAffinity,
		markNodeSelector,
	}
	for i := 0; i < len(nodeMarkers); i++ {
		podName := "local-pod-" + strconv.Itoa(i+1)
		pod := makePod(podName, config.ns, []string{"local-pvc"})
		nodeMarkers[i].(func(*v1.Pod, string))(pod, "node-2")
		// Create Pod
		if _, err := config.client.CoreV1().Pods(config.ns).Create(pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
		// Give time to shceduler to attempt to schedule pod
		if err := waitForPodToSchedule(config.client, pod); err == nil {
			t.Errorf("Failed as Pod %s was scheduled sucessfully but expected to fail", pod.Name)
		}
		// Deleting test pod
		p, err := config.client.CoreV1().Pods(config.ns).Get(podName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to access Pod %s status: %v", podName, err)
		}
		if strings.Compare(string(p.Status.Phase), "Pending") != 0 {
			t.Fatalf("Failed as Pod %s was in: %s state and not in expected: Pending state", podName, p.Status.Phase)
		}
		if strings.Compare(p.Status.Conditions[0].Reason, "Unschedulable") != 0 {
			t.Fatalf("Failed as Pod %s reason was: %s but expected: Unschedulable", podName, p.Status.Conditions[0].Reason)
		}
		if !strings.Contains(p.Status.Conditions[0].Message, "MatchNodeSelector") || !strings.Contains(p.Status.Conditions[0].Message, "VolumeNodeAffinityConflict") {
			t.Fatalf("Failed as Pod's %s failure message does not contain expected keywords: MatchNodeSelector, VolumeNodeAffinityConflict", podName)
		}
		if err := config.client.CoreV1().Pods(config.ns).Delete(podName, &metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to delete Pod %s: %v", podName, err)
		}
	}
}

func setupNodes(t *testing.T, nsName string, numberOfNodes int) *testConfig {
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	// Enable feature gates
	utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true,PersistentLocalVolumes=true")

	// Build clientset and informers for controllers.
	clientset := clientset.NewForConfigOrDie(&restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()}})
	informers := informers.NewSharedInformerFactory(clientset, time.Second)

	// Start master
	masterConfig := framework.NewIntegrationTestMasterConfig()

	_, _, closeFn := framework.RunAMasterUsingServer(masterConfig, s, h)
	ns := framework.CreateTestingNamespace(nsName, s, t).Name

	controllerCh := make(chan struct{})

	// Start PV controller for volume binding.
	params := persistentvolume.ControllerParameters{
		KubeClient:                clientset,
		SyncPeriod:                time.Hour, // test shouldn't need to resync
		VolumePlugins:             nil,       // TODO; need later for dynamic provisioning
		Cloud:                     nil,
		ClusterName:               "volume-test-cluster",
		VolumeInformer:            informers.Core().V1().PersistentVolumes(),
		ClaimInformer:             informers.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             informers.Storage().V1().StorageClasses(),
		EventRecorder:             nil, // TODO: add one so we can test PV events
		EnableDynamicProvisioning: true,
	}
	ctrl, err := persistentvolume.NewController(params)
	if err != nil {
		t.Fatalf("Failed to create PV controller: %v", err)
	}
	go ctrl.Run(controllerCh)

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
		informers.Storage().V1().StorageClasses(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		true, // Enable EqualCache by default.
	)

	eventBroadcaster := record.NewBroadcaster()
	sched, err := scheduler.NewFromConfigurator(configurator, func(cfg *scheduler.Config) {
		cfg.StopEverything = controllerCh
		cfg.Recorder = eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: v1.DefaultSchedulerName})
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(clientset.CoreV1().RESTClient()).Events("")})
	})
	if err != nil {
		t.Fatalf("Failed to create scheduler: %v.", err)
	}

	go sched.Run()

	// Waiting for all controller sync.
	informers.Start(controllerCh)
	informers.WaitForCacheSync(controllerCh)

	// Create shared objects
	// Create nodes
	for i := 0; i < numberOfNodes; i++ {
		testNode := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   fmt.Sprintf("node-%d", i+1),
				Labels: map[string]string{affinityLabelKey: fmt.Sprintf("node-%d", i+1)},
			},
			Spec: v1.NodeSpec{Unschedulable: false},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourcePods: *resource.NewQuantity(podLimit, resource.DecimalSI),
				},
				Conditions: []v1.NodeCondition{
					{
						Type:              v1.NodeReady,
						Status:            v1.ConditionTrue,
						Reason:            fmt.Sprintf("schedulable condition"),
						LastHeartbeatTime: metav1.Time{Time: time.Now()},
					},
				},
			},
		}
		if _, err := clientset.CoreV1().Nodes().Create(testNode); err != nil {
			t.Fatalf("Failed to create Node %q: %v", testNode.Name, err)
		}
	}

	// Create SCs
	scs := []*storagev1.StorageClass{
		makeStorageClass(classImmediate, &modeImmediate),
	}
	for _, sc := range scs {
		if _, err := clientset.StorageV1().StorageClasses().Create(sc); err != nil {
			t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
		}
	}

	return &testConfig{
		client: clientset,
		ns:     ns,
		stop:   controllerCh,
		teardown: func() {
			clientset.CoreV1().Pods(ns).DeleteCollection(nil, metav1.ListOptions{})
			clientset.CoreV1().PersistentVolumeClaims(ns).DeleteCollection(nil, metav1.ListOptions{})
			clientset.CoreV1().PersistentVolumes().DeleteCollection(nil, metav1.ListOptions{})
			clientset.StorageV1().StorageClasses().DeleteCollection(nil, metav1.ListOptions{})
			clientset.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
			close(controllerCh)
			closeFn()
			utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false,LocalPersistentVolumes=false")
		},
	}
}

func makeHostBoundPV(t *testing.T, name, scName, pvcName, ns string, node string) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("5Gi"),
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			StorageClassName: scName,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{
					Path: "/tmp/" + node + "/test-path",
				},
			},
		},
	}

	if pvcName != "" {
		pv.Spec.ClaimRef = &v1.ObjectReference{Name: pvcName, Namespace: ns}
	}

	testNodeAffinity := &v1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      affinityLabelKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{node},
						},
					},
				},
			},
		},
	}
	err := helper.StorageNodeAffinityToAlphaAnnotation(pv.Annotations, testNodeAffinity)
	if err != nil {
		t.Fatalf("Setting storage node affinity failed: %v", err)
	}

	return pv
}

func markNodeAffinity(pod *v1.Pod, node string) {
	affinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{node},
							},
						},
					},
				},
			},
		},
	}
	pod.Spec.Affinity = affinity
}

func markNodeSelector(pod *v1.Pod, node string) {
	ns := map[string]string{
		"kubernetes.io/hostname": node,
	}
	pod.Spec.NodeSelector = ns
}

func printIndentedJson(data interface{}) string {
	var indentedJSON []byte

	indentedJSON, err := json.MarshalIndent(data, "", "\t")
	if err != nil {
		return fmt.Sprintf("JSON parse error: %v", err)
	}
	return string(indentedJSON)
}
