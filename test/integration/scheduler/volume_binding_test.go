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
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/factory"
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
	node1                = "node-1"
	node2                = "node-2"
	podLimit             = 100
	volsPerPod           = 5
	nodeAffinityLabelKey = "kubernetes.io/hostname"
)

func TestVolumeBinding(t *testing.T) {
	config := setupCluster(t, "volume-scheduling", 2)
	defer config.teardown()

	cases := map[string]struct {
		pod  *v1.Pod
		pvs  []*v1.PersistentVolume
		pvcs []*v1.PersistentVolumeClaim
		// Create these, but they should not be bound in the end
		unboundPvcs []*v1.PersistentVolumeClaim
		unboundPvs  []*v1.PersistentVolume
		shouldFail  bool
	}{
		"immediate can bind": {
			pod:  makePod("pod-i-canbind", config.ns, []string{"pvc-i-canbind"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-canbind", classImmediate, "", "", node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-canbind", config.ns, &classImmediate, "")},
		},
		"immediate cannot bind": {
			pod:         makePod("pod-i-cannotbind", config.ns, []string{"pvc-i-cannotbind"}),
			unboundPvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-cannotbind", config.ns, &classImmediate, "")},
			shouldFail:  true,
		},
		"immediate pvc prebound": {
			pod:  makePod("pod-i-pvc-prebound", config.ns, []string{"pvc-i-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-pvc-prebound", classImmediate, "", "", node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-prebound", config.ns, &classImmediate, "pv-i-pvc-prebound")},
		},
		"immediate pv prebound": {
			pod:  makePod("pod-i-pv-prebound", config.ns, []string{"pvc-i-pv-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-i-prebound", classImmediate, "pvc-i-pv-prebound", config.ns, node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-i-pv-prebound", config.ns, &classImmediate, "")},
		},
		"wait can bind": {
			pod:  makePod("pod-w-canbind", config.ns, []string{"pvc-w-canbind"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-canbind", classWait, "", "", node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-canbind", config.ns, &classWait, "")},
		},
		"wait cannot bind": {
			pod:         makePod("pod-w-cannotbind", config.ns, []string{"pvc-w-cannotbind"}),
			unboundPvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-cannotbind", config.ns, &classWait, "")},
			shouldFail:  true,
		},
		"wait pvc prebound": {
			pod:  makePod("pod-w-pvc-prebound", config.ns, []string{"pvc-w-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-pvc-prebound", classWait, "", "", node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-prebound", config.ns, &classWait, "pv-w-pvc-prebound")},
		},
		"wait pv prebound": {
			pod:  makePod("pod-w-pv-prebound", config.ns, []string{"pvc-w-pv-prebound"}),
			pvs:  []*v1.PersistentVolume{makePV(t, "pv-w-prebound", classWait, "pvc-w-pv-prebound", config.ns, node1)},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-w-pv-prebound", config.ns, &classWait, "")},
		},
		"wait can bind two": {
			pod: makePod("pod-w-canbind-2", config.ns, []string{"pvc-w-canbind-2", "pvc-w-canbind-3"}),
			pvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-canbind-2", classWait, "", "", node2),
				makePV(t, "pv-w-canbind-3", classWait, "", "", node2),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-w-canbind-2", config.ns, &classWait, ""),
				makePVC("pvc-w-canbind-3", config.ns, &classWait, ""),
			},
			unboundPvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-canbind-5", classWait, "", "", node1),
			},
		},
		"wait cannot bind two": {
			pod: makePod("pod-w-cannotbind-2", config.ns, []string{"pvc-w-cannotbind-1", "pvc-w-cannotbind-2"}),
			unboundPvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-w-cannotbind-1", config.ns, &classWait, ""),
				makePVC("pvc-w-cannotbind-2", config.ns, &classWait, ""),
			},
			unboundPvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-cannotbind-1", classWait, "", "", node2),
				makePV(t, "pv-w-cannotbind-2", classWait, "", "", node1),
			},
			shouldFail: true,
		},
		"mix immediate and wait": {
			pod: makePod("pod-mix-bound", config.ns, []string{"pvc-w-canbind-4", "pvc-i-canbind-2"}),
			pvs: []*v1.PersistentVolume{
				makePV(t, "pv-w-canbind-4", classWait, "", "", node1),
				makePV(t, "pv-i-canbind-2", classImmediate, "", "", node1),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-w-canbind-4", config.ns, &classWait, ""),
				makePVC("pvc-i-canbind-2", config.ns, &classImmediate, ""),
			},
		},
	}

	for name, test := range cases {
		glog.Infof("Running test %v", name)

		// Create PVs
		for _, pv := range test.pvs {
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		for _, pv := range test.unboundPvs {
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
		for _, pvc := range test.unboundPvcs {
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}

		// Create Pod
		if _, err := config.client.CoreV1().Pods(config.ns).Create(test.pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", test.pod.Name, err)
		}
		if test.shouldFail {
			if err := waitForPodUnschedulable(config.client, test.pod); err != nil {
				t.Errorf("Pod %q was not unschedulable: %v", test.pod.Name, err)
			}
		} else {
			if err := waitForPodToSchedule(config.client, test.pod); err != nil {
				t.Errorf("Failed to schedule Pod %q: %v", test.pod.Name, err)
			}
		}

		// Validate PVC/PV binding
		for _, pvc := range test.pvcs {
			validatePVCPhase(t, config.client, pvc, v1.ClaimBound)
		}
		for _, pvc := range test.unboundPvcs {
			validatePVCPhase(t, config.client, pvc, v1.ClaimPending)
		}
		for _, pv := range test.pvs {
			validatePVPhase(t, config.client, pv, v1.VolumeBound)
		}
		for _, pv := range test.unboundPvs {
			validatePVPhase(t, config.client, pv, v1.VolumeAvailable)
		}

		// TODO: validate events on Pods and PVCs

		config.client.CoreV1().Pods(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumeClaims(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumes().DeleteCollection(deleteOption, metav1.ListOptions{})
	}
}

// TestVolumeBindingStress creates <podLimit> pods, each with <volsPerPod> unbound PVCs.
func TestVolumeBindingStress(t *testing.T) {
	config := setupCluster(t, "volume-binding-stress", 1)
	defer config.teardown()

	// Create enough PVs and PVCs for all the pods
	pvs := []*v1.PersistentVolume{}
	pvcs := []*v1.PersistentVolumeClaim{}
	for i := 0; i < podLimit*volsPerPod; i++ {
		pv := makePV(t, fmt.Sprintf("pv-stress-%v", i), classWait, "", "", node1)
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

func TestPVAffinityConflict(t *testing.T) {
	config := setupCluster(t, "volume-scheduling", 3)
	defer config.teardown()

	pv := makePV(t, "local-pv", classImmediate, "", "", node1)
	pvc := makePVC("local-pvc", config.ns, &classImmediate, "")

	// Create PV
	if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
		t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
	}

	// Create PVC
	if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
		t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
	}

	// Wait for PVC bound
	if err := waitForPVCBound(config.client, pvc); err != nil {
		t.Fatalf("PVC %q failed to bind: %v", pvc.Name, err)
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
		if err := waitForPodUnschedulable(config.client, pod); err != nil {
			t.Errorf("Failed as Pod %s was not unschedulable: %v", pod.Name, err)
		}
		// Check pod conditions
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
		if !strings.Contains(p.Status.Conditions[0].Message, "node(s) didn't match node selector") || !strings.Contains(p.Status.Conditions[0].Message, "node(s) had volume node affinity conflict") {
			t.Fatalf("Failed as Pod's %s failure message does not contain expected message: node(s) didn't match node selector, node(s) had volume node affinity conflict. Got message %q", podName, p.Status.Conditions[0].Message)
		}
		// Deleting test pod
		if err := config.client.CoreV1().Pods(config.ns).Delete(podName, &metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to delete Pod %s: %v", podName, err)
		}
	}
}

func setupCluster(t *testing.T, nsName string, numberOfNodes int) *testConfig {
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
		PodInformer:               informers.Core().V1().Pods(),
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
				Labels: map[string]string{nodeAffinityLabelKey: fmt.Sprintf("node-%d", i+1)},
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
		makeStorageClass(classWait, &modeWait),
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

func makePV(t *testing.T, name, scName, pvcName, ns, node string) *v1.PersistentVolume {
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
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      nodeAffinityLabelKey,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{node},
								},
							},
						},
					},
				},
			},
		},
	}

	if pvcName != "" {
		pv.Spec.ClaimRef = &v1.ObjectReference{Name: pvcName, Namespace: ns}
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
					Image:   "k8s.gcr.io/busybox:1.24",
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

func waitForPVCBound(client clientset.Interface, pvc *v1.PersistentVolumeClaim) error {
	return wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		claim, err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if claim.Status.Phase == v1.ClaimBound {
			return true, nil
		}
		return false, nil
	})
}

func markNodeAffinity(pod *v1.Pod, node string) {
	affinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      nodeAffinityLabelKey,
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
		nodeAffinityLabelKey: node,
	}
	pod.Spec.NodeSelector = ns
}
