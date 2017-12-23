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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
)

type testConfig struct {
	client   clientset.Interface
	ns       string
	stop     <-chan struct{}
	teardown func()
}

var (
	// Delete API objects immediately
	deletePeriod = int64(0)
	deleteOption = &metav1.DeleteOptions{GracePeriodSeconds: &deletePeriod}

	modeWait      = storagev1.VolumeBindingWaitForFirstConsumer
	modeImmediate = storagev1.VolumeBindingImmediate

	classWait      = "wait"
	classImmediate = "immediate"
)

const (
	labelKey   = "test-label"
	labelValue = "test-value"
	nodeName   = "node1"
	podLimit   = 100
	volsPerPod = 5
)

func TestVolumeBinding(t *testing.T) {
	config := setup(t, "volume-scheduling")
	defer config.teardown()

	cases := map[string]struct {
		pod  *v1.Pod
		pvs  []*v1.PersistentVolume
		pvcs []*v1.PersistentVolumeClaim
	}{
		"immediate can bind": {
			pod:  makePod("pod-i-canbind", config.ns, []string{"pvc-i-canbind"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-canbind", classImmediate, "", "")},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-canbind", config.ns, &classImmediate, "")},
		},
		"immediate pvc prebound": {
			pod:  makePod("pod-i-pvc-prebound", config.ns, []string{"pvc-i-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-pvc-prebound", classImmediate, "", "")},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-prebound", config.ns, &classImmediate, "pv-i-pvc-prebound")},
		},
		"immediate pv prebound": {
			pod:  makePod("pod-i-pv-prebound", config.ns, []string{"pvc-i-pv-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-prebound", classImmediate, "pvc-i-pv-prebound", config.ns)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-pv-prebound", config.ns, &classImmediate, "")},
		},
		"wait can bind": {
			pod:  makePod("pod-w-canbind", config.ns, []string{"pvc-w-canbind"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-canbind", classWait, "", "")},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-canbind", config.ns, &classWait, "")},
		},
		"wait pvc prebound": {
			pod:  makePod("pod-w-pvc-prebound", config.ns, []string{"pvc-w-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-pvc-prebound", classWait, "", "")},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-prebound", config.ns, &classWait, "pv-w-pvc-prebound")},
		},
		"wait pv prebound": {
			pod:  makePod("pod-w-pv-prebound", config.ns, []string{"pvc-w-pv-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-prebound", classWait, "pvc-w-pv-prebound", config.ns)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-pv-prebound", config.ns, &classWait, "")},
		},
		"wait can bind two": {
			pod: makePod("pod-w-canbind-2", config.ns, []string{"pvc-w-canbind-2", "pvc-w-canbind-3"}),
			pvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-canbind-2", classWait, "", ""),
				makePV(t, "pv-w-canbind-3", classWait, "", ""),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-w-canbind-2", config.ns, &classWait, ""),
				makePVC("pvc-w-canbind-3", config.ns, &classWait, ""),
			},
		},
		"mix immediate and wait": {
			pod: makePod("pod-mix-bound", config.ns, []string{"pvc-w-canbind-4", "pvc-i-canbind-2"}),
			pvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-canbind-4", classWait, "", ""),
				makePV(t, "pv-i-canbind-2", classImmediate, "", ""),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-w-canbind-4", config.ns, &classWait, ""),
				makePVC("pvc-i-canbind-2", config.ns, &classImmediate, ""),
			},
		},
		// TODO:
		// immediate mode - PVC cannot bound
		// wait mode - PVC cannot bind
		// wait mode - 2 PVCs, 1 cannot bind
	}

	for name, test := range cases {
		glog.Infof("Running test %v", name)

		// Create PVs
		for _, pv := range test.pvs {
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		// Create PVCs
		for _, pvc := range test.pvcs {
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}

		// Create Pod
		if _, err := config.client.CoreV1().Pods(config.ns).Create(test.pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", test.pod.Name, err)
		}
		if err := waitForPodToSchedule(config.client, test.pod); err != nil {
			t.Errorf("Failed to schedule Pod %q: %v", test.pod.Name, err)
		}

		// Validate PVC/PV binding
		for _, pvc := range test.pvcs {
			validatePVCPhase(t, config.client, pvc, v1.ClaimBound)
		}
		for _, pv := range test.pvs {
			validatePVPhase(t, config.client, pv, v1.VolumeBound)
		}

		// TODO: validate events on Pods and PVCs

		config.client.CoreV1().Pods(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumeClaims(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumes().DeleteCollection(deleteOption, metav1.ListOptions{})
	}
}

// TestVolumeBindingStress creates <podLimit> pods, each with <volsPerPod> unbound PVCs.
func TestVolumeBindingStress(t *testing.T) {
	config := setup(t, "volume-binding-stress")
	defer config.teardown()

	// Create enough PVs and PVCs for all the pods
	pvs := []*v1.PersistentVolume{}
	pvcs := []*v1.PersistentVolumeClaim{}
	for i := 0; i < podLimit*volsPerPod; i++ {
		pv := makePV(t, fmt.Sprintf("pv-stress-%v", i), classWait, "", "")
		pvc := makePVC(fmt.Sprintf("pvc-stress-%v", i), config.ns, &classWait, "")

		if pv, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
			t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
		}
		if pvc, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
			t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
		}

		pvs = append(pvs, pv)
		pvcs = append(pvcs, pvc)
	}

	pods := []*v1.Pod{}
	for i := 0; i < podLimit; i++ {
		// Generate string of all the PVCs for the pod
		podPvcs := []string{}
		for j := i * volsPerPod; j < (i+1)*volsPerPod; j++ {
			podPvcs = append(podPvcs, pvcs[j].Name)
		}

		pod := makePod(fmt.Sprintf("pod%v", i), config.ns, podPvcs)
		if pod, err := config.client.CoreV1().Pods(config.ns).Create(pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
		pods = append(pods, pod)
	}

	// Validate Pods scheduled
	for _, pod := range pods {
		if err := waitForPodToSchedule(config.client, pod); err != nil {
			t.Errorf("Failed to schedule Pod %q: %v", pod.Name, err)
		}
	}

	// Validate PVC/PV binding
	for _, pvc := range pvcs {
		validatePVCPhase(t, config.client, pvc, v1.ClaimBound)
	}
	for _, pv := range pvs {
		validatePVPhase(t, config.client, pv, v1.VolumeBound)
	}

	// TODO: validate events on Pods and PVCs
}

func setup(t *testing.T, nsName string) *testConfig {
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

	sched, err := scheduler.NewFromConfigurator(configurator, func(cfg *scheduler.Config) {
		cfg.StopEverything = controllerCh
		cfg.Recorder = &record.FakeRecorder{}
	})
	if err != nil {
		t.Fatalf("Failed to create scheduler: %v.", err)
	}
	go sched.Run()

	// Waiting for all controller sync.
	informers.Start(controllerCh)
	informers.WaitForCacheSync(controllerCh)

	// Create shared objects
	// Create node
	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeName,
			Labels: map[string]string{labelKey: labelValue},
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

	// Create SCs
	scs := []*storagev1.StorageClass{
		makeStorageClass(classWait, &modeWait),
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

func makeStorageClass(name string, mode *storagev1.VolumeBindingMode) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Provisioner:       "kubernetes.io/no-provisioner",
		VolumeBindingMode: mode,
	}
}

func makePV(t *testing.T, name, scName, pvcName, ns string) *v1.PersistentVolume {
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
					Path: "/test-path",
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
							Key:      labelKey,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{labelValue},
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

func makePVC(name, ns string, scName *string, volumeName string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("5Gi"),
				},
			},
			StorageClassName: scName,
			VolumeName:       volumeName,
		},
	}
}

func makePod(name, ns string, pvcs []string) *v1.Pod {
	volumes := []v1.Volume{}
	for i, pvc := range pvcs {
		volumes = append(volumes, v1.Volume{
			Name: fmt.Sprintf("vol%v", i),
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvc,
				},
			},
		})
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "write-pod",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "while true; do sleep 1; done"},
				},
			},
			Volumes: volumes,
		},
	}
}

func validatePVCPhase(t *testing.T, client clientset.Interface, pvc *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase) {
	claim, err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get PVC %v/%v: %v", pvc.Namespace, pvc.Name, err)
	}

	if claim.Status.Phase != phase {
		t.Errorf("PVC %v/%v phase not %v, got %v", pvc.Namespace, pvc.Name, phase, claim.Status.Phase)
	}
}

func validatePVPhase(t *testing.T, client clientset.Interface, pv *v1.PersistentVolume, phase v1.PersistentVolumePhase) {
	pv, err := client.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get PV %v: %v", pv.Name, err)
	}

	if pv.Status.Phase != phase {
		t.Errorf("PV %v phase not %v, got %v", pv.Name, phase, pv.Status.Phase)
	}
}
