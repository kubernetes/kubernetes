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

package volumescheduling

// This file tests the VolumeScheduling feature.

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	persistentvolumeoptions "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/options"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	imageutils "k8s.io/kubernetes/test/utils/image"
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

	classWait         = "wait"
	classImmediate    = "immediate"
	classDynamic      = "dynamic"
	classTopoMismatch = "topomismatch"

	sharedClasses = map[string]*storagev1.StorageClass{
		classImmediate: makeStorageClass(classImmediate, &modeImmediate),
		classWait:      makeStorageClass(classWait, &modeWait),
	}
)

const (
	node1                 = "node-1"
	node2                 = "node-2"
	podLimit              = 100
	volsPerPod            = 5
	nodeAffinityLabelKey  = "kubernetes.io/hostname"
	provisionerPluginName = "kubernetes.io/mock-provisioner"
)

type testPV struct {
	name        string
	scName      string
	preboundPVC string
	node        string
}

type testPVC struct {
	name       string
	scName     string
	preboundPV string
}

func TestVolumeBinding(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-scheduling-", 2, 0, 0)
	defer config.teardown()

	cases := map[string]struct {
		pod  *v1.Pod
		pvs  []*testPV
		pvcs []*testPVC
		// Create these, but they should not be bound in the end
		unboundPvcs []*testPVC
		unboundPvs  []*testPV
		shouldFail  bool
	}{
		"immediate can bind": {
			pod:  makePod("pod-i-canbind", config.ns, []string{"pvc-i-canbind"}),
			pvs:  []*testPV{{"pv-i-canbind", classImmediate, "", node1}},
			pvcs: []*testPVC{{"pvc-i-canbind", classImmediate, ""}},
		},
		"immediate cannot bind": {
			pod:         makePod("pod-i-cannotbind", config.ns, []string{"pvc-i-cannotbind"}),
			unboundPvcs: []*testPVC{{"pvc-i-cannotbind", classImmediate, ""}},
			shouldFail:  true,
		},
		"immediate pvc prebound": {
			pod:  makePod("pod-i-pvc-prebound", config.ns, []string{"pvc-i-prebound"}),
			pvs:  []*testPV{{"pv-i-pvc-prebound", classImmediate, "", node1}},
			pvcs: []*testPVC{{"pvc-i-prebound", classImmediate, "pv-i-pvc-prebound"}},
		},
		"immediate pv prebound": {
			pod:  makePod("pod-i-pv-prebound", config.ns, []string{"pvc-i-pv-prebound"}),
			pvs:  []*testPV{{"pv-i-prebound", classImmediate, "pvc-i-pv-prebound", node1}},
			pvcs: []*testPVC{{"pvc-i-pv-prebound", classImmediate, ""}},
		},
		"wait can bind": {
			pod:  makePod("pod-w-canbind", config.ns, []string{"pvc-w-canbind"}),
			pvs:  []*testPV{{"pv-w-canbind", classWait, "", node1}},
			pvcs: []*testPVC{{"pvc-w-canbind", classWait, ""}},
		},
		"wait cannot bind": {
			pod:         makePod("pod-w-cannotbind", config.ns, []string{"pvc-w-cannotbind"}),
			unboundPvcs: []*testPVC{{"pvc-w-cannotbind", classWait, ""}},
			shouldFail:  true,
		},
		"wait pvc prebound": {
			pod:  makePod("pod-w-pvc-prebound", config.ns, []string{"pvc-w-prebound"}),
			pvs:  []*testPV{{"pv-w-pvc-prebound", classWait, "", node1}},
			pvcs: []*testPVC{{"pvc-w-prebound", classWait, "pv-w-pvc-prebound"}},
		},
		"wait pv prebound": {
			pod:  makePod("pod-w-pv-prebound", config.ns, []string{"pvc-w-pv-prebound"}),
			pvs:  []*testPV{{"pv-w-prebound", classWait, "pvc-w-pv-prebound", node1}},
			pvcs: []*testPVC{{"pvc-w-pv-prebound", classWait, ""}},
		},
		"wait can bind two": {
			pod: makePod("pod-w-canbind-2", config.ns, []string{"pvc-w-canbind-2", "pvc-w-canbind-3"}),
			pvs: []*testPV{
				{"pv-w-canbind-2", classWait, "", node2},
				{"pv-w-canbind-3", classWait, "", node2},
			},
			pvcs: []*testPVC{
				{"pvc-w-canbind-2", classWait, ""},
				{"pvc-w-canbind-3", classWait, ""},
			},
			unboundPvs: []*testPV{
				{"pv-w-canbind-5", classWait, "", node1},
			},
		},
		"wait cannot bind two": {
			pod: makePod("pod-w-cannotbind-2", config.ns, []string{"pvc-w-cannotbind-1", "pvc-w-cannotbind-2"}),
			unboundPvcs: []*testPVC{
				{"pvc-w-cannotbind-1", classWait, ""},
				{"pvc-w-cannotbind-2", classWait, ""},
			},
			unboundPvs: []*testPV{
				{"pv-w-cannotbind-1", classWait, "", node2},
				{"pv-w-cannotbind-2", classWait, "", node1},
			},
			shouldFail: true,
		},
		"mix immediate and wait": {
			pod: makePod("pod-mix-bound", config.ns, []string{"pvc-w-canbind-4", "pvc-i-canbind-2"}),
			pvs: []*testPV{
				{"pv-w-canbind-4", classWait, "", node1},
				{"pv-i-canbind-2", classImmediate, "", node1},
			},
			pvcs: []*testPVC{
				{"pvc-w-canbind-4", classWait, ""},
				{"pvc-i-canbind-2", classImmediate, ""},
			},
		},
	}

	for name, test := range cases {
		klog.Infof("Running test %v", name)

		// Create two StorageClasses
		suffix := rand.String(4)
		classes := map[string]*storagev1.StorageClass{}
		classes[classImmediate] = makeStorageClass(fmt.Sprintf("immediate-%v", suffix), &modeImmediate)
		classes[classWait] = makeStorageClass(fmt.Sprintf("wait-%v", suffix), &modeWait)
		for _, sc := range classes {
			if _, err := config.client.StorageV1().StorageClasses().Create(sc); err != nil {
				t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
			}
		}

		// Create PVs
		for _, pvConfig := range test.pvs {
			pv := makePV(pvConfig.name, classes[pvConfig.scName].Name, pvConfig.preboundPVC, config.ns, pvConfig.node)
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		for _, pvConfig := range test.unboundPvs {
			pv := makePV(pvConfig.name, classes[pvConfig.scName].Name, pvConfig.preboundPVC, config.ns, pvConfig.node)
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		// Create PVCs
		for _, pvcConfig := range test.pvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &classes[pvcConfig.scName].Name, pvcConfig.preboundPV)
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}
		for _, pvcConfig := range test.unboundPvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &classes[pvcConfig.scName].Name, pvcConfig.preboundPV)
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
			validatePVCPhase(t, config.client, pvc.name, config.ns, v1.ClaimBound, false)
		}
		for _, pvc := range test.unboundPvcs {
			validatePVCPhase(t, config.client, pvc.name, config.ns, v1.ClaimPending, false)
		}
		for _, pv := range test.pvs {
			validatePVPhase(t, config.client, pv.name, v1.VolumeBound)
		}
		for _, pv := range test.unboundPvs {
			validatePVPhase(t, config.client, pv.name, v1.VolumeAvailable)
		}

		// Force delete objects, but they still may not be immediately removed
		deleteTestObjects(config.client, config.ns, deleteOption)
	}
}

// TestVolumeBindingRescheduling tests scheduler will retry scheduling when needed.
func TestVolumeBindingRescheduling(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-scheduling-", 2, 0, 0)
	defer config.teardown()

	storageClassName := "local-storage"

	cases := map[string]struct {
		pod        *v1.Pod
		pvcs       []*testPVC
		pvs        []*testPV
		trigger    func(config *testConfig)
		shouldFail bool
	}{
		"reschedule on WaitForFirstConsumer dynamic storage class add": {
			pod: makePod("pod-reschedule-onclassadd-dynamic", config.ns, []string{"pvc-reschedule-onclassadd-dynamic"}),
			pvcs: []*testPVC{
				{"pvc-reschedule-onclassadd-dynamic", "", ""},
			},
			trigger: func(config *testConfig) {
				sc := makeDynamicProvisionerStorageClass(storageClassName, &modeWait, nil)
				if _, err := config.client.StorageV1().StorageClasses().Create(sc); err != nil {
					t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
				}
			},
			shouldFail: false,
		},
		"reschedule on WaitForFirstConsumer static storage class add": {
			pod: makePod("pod-reschedule-onclassadd-static", config.ns, []string{"pvc-reschedule-onclassadd-static"}),
			pvcs: []*testPVC{
				{"pvc-reschedule-onclassadd-static", "", ""},
			},
			trigger: func(config *testConfig) {
				sc := makeStorageClass(storageClassName, &modeWait)
				if _, err := config.client.StorageV1().StorageClasses().Create(sc); err != nil {
					t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
				}
				// Create pv for this class to mock static provisioner behavior.
				pv := makePV("pv-reschedule-onclassadd-static", storageClassName, "", "", node1)
				if pv, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
					t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
				}
			},
			shouldFail: false,
		},
		"reschedule on delay binding PVC add": {
			pod: makePod("pod-reschedule-onpvcadd", config.ns, []string{"pvc-reschedule-onpvcadd"}),
			pvs: []*testPV{
				{
					name:   "pv-reschedule-onpvcadd",
					scName: classWait,
					node:   node1,
				},
			},
			trigger: func(config *testConfig) {
				pvc := makePVC("pvc-reschedule-onpvcadd", config.ns, &classWait, "")
				if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
					t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
				}
			},
			shouldFail: false,
		},
	}

	for name, test := range cases {
		klog.Infof("Running test %v", name)

		if test.pod == nil {
			t.Fatal("pod is required for this test")
		}

		// Create unbound pvc
		for _, pvcConfig := range test.pvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &storageClassName, "")
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}

		// Create PVs
		for _, pvConfig := range test.pvs {
			pv := makePV(pvConfig.name, sharedClasses[pvConfig.scName].Name, pvConfig.preboundPVC, config.ns, pvConfig.node)
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		// Create pod
		if _, err := config.client.CoreV1().Pods(config.ns).Create(test.pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", test.pod.Name, err)
		}

		// Wait for pod is unschedulable.
		klog.Infof("Waiting for pod is unschedulable")
		if err := waitForPodUnschedulable(config.client, test.pod); err != nil {
			t.Errorf("Failed as Pod %s was not unschedulable: %v", test.pod.Name, err)
		}

		// Trigger
		test.trigger(config)

		// Wait for pod is scheduled or unscheduable.
		if !test.shouldFail {
			klog.Infof("Waiting for pod is scheduled")
			if err := waitForPodToSchedule(config.client, test.pod); err != nil {
				t.Errorf("Failed to schedule Pod %q: %v", test.pod.Name, err)
			}
		} else {
			klog.Infof("Waiting for pod is unschedulable")
			if err := waitForPodUnschedulable(config.client, test.pod); err != nil {
				t.Errorf("Failed as Pod %s was not unschedulable: %v", test.pod.Name, err)
			}
		}

		// Force delete objects, but they still may not be immediately removed
		deleteTestObjects(config.client, config.ns, deleteOption)
	}
}

// TestVolumeBindingStress creates <podLimit> pods, each with <volsPerPod> unbound or prebound PVCs.
// PVs are precreated.
func TestVolumeBindingStress(t *testing.T) {
	testVolumeBindingStress(t, 0, false, 0)
}

// Like TestVolumeBindingStress but with scheduler resync. In real cluster,
// scheduler will schedule failed pod frequently due to various events, e.g.
// service/node update events.
// This is useful to detect possible race conditions.
func TestVolumeBindingStressWithSchedulerResync(t *testing.T) {
	testVolumeBindingStress(t, time.Second, false, 0)
}

// Like TestVolumeBindingStress but with fast dynamic provisioning
func TestVolumeBindingDynamicStressFast(t *testing.T) {
	testVolumeBindingStress(t, 0, true, 0)
}

// Like TestVolumeBindingStress but with slow dynamic provisioning
func TestVolumeBindingDynamicStressSlow(t *testing.T) {
	testVolumeBindingStress(t, 0, true, 30)
}

func testVolumeBindingStress(t *testing.T, schedulerResyncPeriod time.Duration, dynamic bool, provisionDelaySeconds int) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-binding-stress-", 1, schedulerResyncPeriod, provisionDelaySeconds)
	defer config.teardown()

	// Set max volume limit to the number of PVCs the test will create
	// TODO: remove when max volume limit allows setting through storageclass
	if err := os.Setenv(predicates.KubeMaxPDVols, fmt.Sprintf("%v", podLimit*volsPerPod)); err != nil {
		t.Fatalf("failed to set max pd limit: %v", err)
	}
	defer os.Unsetenv(predicates.KubeMaxPDVols)

	scName := &classWait
	if dynamic {
		scName = &classDynamic
		sc := makeDynamicProvisionerStorageClass(*scName, &modeWait, nil)
		if _, err := config.client.StorageV1().StorageClasses().Create(sc); err != nil {
			t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
		}
	}

	// Create enough PVs and PVCs for all the pods
	pvs := []*v1.PersistentVolume{}
	pvcs := []*v1.PersistentVolumeClaim{}
	for i := 0; i < podLimit*volsPerPod; i++ {
		var (
			pv      *v1.PersistentVolume
			pvc     *v1.PersistentVolumeClaim
			pvName  = fmt.Sprintf("pv-stress-%v", i)
			pvcName = fmt.Sprintf("pvc-stress-%v", i)
		)
		// Don't create pvs for dynamic provisioning test
		if !dynamic {
			if rand.Int()%2 == 0 {
				// static unbound pvs
				pv = makePV(pvName, *scName, "", "", node1)
			} else {
				// static prebound pvs
				pv = makePV(pvName, classImmediate, pvcName, config.ns, node1)
			}
			if pv, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
			pvs = append(pvs, pv)
		}
		if pv != nil && pv.Spec.ClaimRef != nil && pv.Spec.ClaimRef.Name == pvcName {
			pvc = makePVC(pvcName, config.ns, &classImmediate, pv.Name)
		} else {
			pvc = makePVC(pvcName, config.ns, scName, "")
		}
		if pvc, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
			t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
		}
		pvcs = append(pvcs, pvc)
	}

	pods := []*v1.Pod{}
	for i := 0; i < podLimit; i++ {
		// Generate string of all the PVCs for the pod
		podPvcs := []string{}
		for j := i * volsPerPod; j < (i+1)*volsPerPod; j++ {
			podPvcs = append(podPvcs, pvcs[j].Name)
		}

		pod := makePod(fmt.Sprintf("pod%03d", i), config.ns, podPvcs)
		if pod, err := config.client.CoreV1().Pods(config.ns).Create(pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
		pods = append(pods, pod)
	}

	// Validate Pods scheduled
	for _, pod := range pods {
		// Use increased timeout for stress test because there is a higher chance of
		// PV sync error
		if err := waitForPodToScheduleWithTimeout(config.client, pod, 2*time.Minute); err != nil {
			t.Errorf("Failed to schedule Pod %q: %v", pod.Name, err)
		}
	}

	// Validate PVC/PV binding
	for _, pvc := range pvcs {
		validatePVCPhase(t, config.client, pvc.Name, config.ns, v1.ClaimBound, dynamic)
	}
	for _, pv := range pvs {
		validatePVPhase(t, config.client, pv.Name, v1.VolumeBound)
	}
}

func testVolumeBindingWithAffinity(t *testing.T, anti bool, numNodes, numPods, numPVsFirstNode int) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-pod-affinity-", numNodes, 0, 0)
	defer config.teardown()

	pods := []*v1.Pod{}
	pvcs := []*v1.PersistentVolumeClaim{}
	pvs := []*v1.PersistentVolume{}

	// Create PVs for the first node
	for i := 0; i < numPVsFirstNode; i++ {
		pv := makePV(fmt.Sprintf("pv-node1-%v", i), classWait, "", "", node1)
		if pv, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
			t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
		}
		pvs = append(pvs, pv)
	}

	// Create 1 PV per Node for the remaining nodes
	for i := 2; i <= numNodes; i++ {
		pv := makePV(fmt.Sprintf("pv-node%v-0", i), classWait, "", "", fmt.Sprintf("node-%v", i))
		if pv, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
			t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
		}
		pvs = append(pvs, pv)
	}

	// Create pods
	for i := 0; i < numPods; i++ {
		// Create one pvc per pod
		pvc := makePVC(fmt.Sprintf("pvc-%v", i), config.ns, &classWait, "")
		if pvc, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
			t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
		}
		pvcs = append(pvcs, pvc)

		// Create pod with pod affinity
		pod := makePod(fmt.Sprintf("pod%03d", i), config.ns, []string{pvc.Name})
		pod.Spec.Affinity = &v1.Affinity{}
		affinityTerms := []v1.PodAffinityTerm{
			{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "app",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"volume-binding-test"},
						},
					},
				},
				TopologyKey: nodeAffinityLabelKey,
			},
		}
		if anti {
			pod.Spec.Affinity.PodAntiAffinity = &v1.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: affinityTerms,
			}
		} else {
			pod.Spec.Affinity.PodAffinity = &v1.PodAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: affinityTerms,
			}
		}

		if pod, err := config.client.CoreV1().Pods(config.ns).Create(pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", pod.Name, err)
		}
		pods = append(pods, pod)
	}

	// Validate Pods scheduled
	scheduledNodes := sets.NewString()
	for _, pod := range pods {
		if err := waitForPodToSchedule(config.client, pod); err != nil {
			t.Errorf("Failed to schedule Pod %q: %v", pod.Name, err)
		} else {
			// Keep track of all the nodes that the Pods were scheduled on
			pod, err = config.client.CoreV1().Pods(config.ns).Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get Pod %q: %v", pod.Name, err)
			}
			if pod.Spec.NodeName == "" {
				t.Fatalf("Pod %q node name unset after scheduling", pod.Name)
			}
			scheduledNodes.Insert(pod.Spec.NodeName)
		}
	}

	// Validate the affinity policy
	if anti {
		// The pods should have been spread across different nodes
		if scheduledNodes.Len() != numPods {
			t.Errorf("Pods were scheduled across %v nodes instead of %v", scheduledNodes.Len(), numPods)
		}
	} else {
		// The pods should have been scheduled on 1 node
		if scheduledNodes.Len() != 1 {
			t.Errorf("Pods were scheduled across %v nodes instead of %v", scheduledNodes.Len(), 1)
		}
	}

	// Validate PVC binding
	for _, pvc := range pvcs {
		validatePVCPhase(t, config.client, pvc.Name, config.ns, v1.ClaimBound, false)
	}
}

func TestVolumeBindingWithAntiAffinity(t *testing.T) {
	numNodes := 10
	// Create as many pods as number of nodes
	numPods := numNodes
	// Create many more PVs on node1 to increase chance of selecting node1
	numPVsFirstNode := 10 * numNodes

	testVolumeBindingWithAffinity(t, true, numNodes, numPods, numPVsFirstNode)
}

func TestVolumeBindingWithAffinity(t *testing.T) {
	numPods := 10
	// Create many more nodes to increase chance of selecting a PV on a different node than node1
	numNodes := 10 * numPods
	// Create numPods PVs on the first node
	numPVsFirstNode := numPods

	testVolumeBindingWithAffinity(t, true, numNodes, numPods, numPVsFirstNode)
}

func TestPVAffinityConflict(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-scheduling-", 3, 0, 0)
	defer config.teardown()

	pv := makePV("local-pv", classImmediate, "", "", node1)
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

func TestVolumeProvision(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentLocalVolumes, true)()
	config := setupCluster(t, "volume-scheduling", 1, 0, 0)
	defer config.teardown()

	cases := map[string]struct {
		pod             *v1.Pod
		pvs             []*testPV
		boundPvcs       []*testPVC
		provisionedPvcs []*testPVC
		// Create these, but they should not be bound in the end
		unboundPvcs []*testPVC
		shouldFail  bool
	}{
		"wait provisioned": {
			pod:             makePod("pod-pvc-canprovision", config.ns, []string{"pvc-canprovision"}),
			provisionedPvcs: []*testPVC{{"pvc-canprovision", classWait, ""}},
		},
		"topolgy unsatisfied": {
			pod:         makePod("pod-pvc-topomismatch", config.ns, []string{"pvc-topomismatch"}),
			unboundPvcs: []*testPVC{{"pvc-topomismatch", classTopoMismatch, ""}},
			shouldFail:  true,
		},
		"wait one bound, one provisioned": {
			pod:             makePod("pod-pvc-canbind-or-provision", config.ns, []string{"pvc-w-canbind", "pvc-canprovision"}),
			pvs:             []*testPV{{"pv-w-canbind", classWait, "", node1}},
			boundPvcs:       []*testPVC{{"pvc-w-canbind", classWait, ""}},
			provisionedPvcs: []*testPVC{{"pvc-canprovision", classWait, ""}},
		},
		"one immediate pv prebound, one wait provisioned": {
			pod:             makePod("pod-i-pv-prebound-w-provisioned", config.ns, []string{"pvc-i-pv-prebound", "pvc-canprovision"}),
			pvs:             []*testPV{{"pv-i-prebound", classImmediate, "pvc-i-pv-prebound", node1}},
			boundPvcs:       []*testPVC{{"pvc-i-pv-prebound", classImmediate, ""}},
			provisionedPvcs: []*testPVC{{"pvc-canprovision", classWait, ""}},
		},
		"wait one pv prebound, one provisioned": {
			pod:             makePod("pod-w-pv-prebound-w-provisioned", config.ns, []string{"pvc-w-pv-prebound", "pvc-canprovision"}),
			pvs:             []*testPV{{"pv-w-prebound", classWait, "pvc-w-pv-prebound", node1}},
			boundPvcs:       []*testPVC{{"pvc-w-pv-prebound", classWait, ""}},
			provisionedPvcs: []*testPVC{{"pvc-canprovision", classWait, ""}},
		},
		"immediate provisioned by controller": {
			pod: makePod("pod-i-unbound", config.ns, []string{"pvc-controller-provisioned"}),
			// A pvc of immediate binding mode is expected to be provisioned by controller,
			// we treat it as "bound" here because it is supposed to be in same state
			// with bound claims, i.e. in bound status and has no selectedNode annotation.
			boundPvcs: []*testPVC{{"pvc-controller-provisioned", classImmediate, ""}},
		},
	}

	for name, test := range cases {
		klog.Infof("Running test %v", name)

		// Create StorageClasses
		suffix := rand.String(4)
		classes := map[string]*storagev1.StorageClass{}
		classes[classImmediate] = makeDynamicProvisionerStorageClass(fmt.Sprintf("immediate-%v", suffix), &modeImmediate, nil)
		classes[classWait] = makeDynamicProvisionerStorageClass(fmt.Sprintf("wait-%v", suffix), &modeWait, nil)
		topo := []v1.TopologySelectorTerm{
			{
				MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
					{
						Key:    nodeAffinityLabelKey,
						Values: []string{node2},
					},
				},
			},
		}
		classes[classTopoMismatch] = makeDynamicProvisionerStorageClass(fmt.Sprintf("topomismatch-%v", suffix), &modeWait, topo)
		for _, sc := range classes {
			if _, err := config.client.StorageV1().StorageClasses().Create(sc); err != nil {
				t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
			}
		}
		// Create PVs
		for _, pvConfig := range test.pvs {
			pv := makePV(pvConfig.name, classes[pvConfig.scName].Name, pvConfig.preboundPVC, config.ns, pvConfig.node)
			if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
				t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
			}
		}

		// Create PVCs
		for _, pvcConfig := range test.boundPvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &classes[pvcConfig.scName].Name, pvcConfig.preboundPV)
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}
		for _, pvcConfig := range test.unboundPvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &classes[pvcConfig.scName].Name, pvcConfig.preboundPV)
			if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
				t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
			}
		}
		for _, pvcConfig := range test.provisionedPvcs {
			pvc := makePVC(pvcConfig.name, config.ns, &classes[pvcConfig.scName].Name, pvcConfig.preboundPV)
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
		for _, pvc := range test.boundPvcs {
			validatePVCPhase(t, config.client, pvc.name, config.ns, v1.ClaimBound, false)
		}
		for _, pvc := range test.unboundPvcs {
			validatePVCPhase(t, config.client, pvc.name, config.ns, v1.ClaimPending, false)
		}
		for _, pvc := range test.provisionedPvcs {
			validatePVCPhase(t, config.client, pvc.name, config.ns, v1.ClaimBound, true)
		}
		for _, pv := range test.pvs {
			validatePVPhase(t, config.client, pv.name, v1.VolumeBound)
		}

		// Force delete objects, but they still may not be immediately removed
		deleteTestObjects(config.client, config.ns, deleteOption)
	}
}

// TestRescheduleProvisioning validate that PV controller will remove
// selectedNode annotation from a claim to reschedule volume provision
// on provision failure.
func TestRescheduleProvisioning(t *testing.T) {
	// Set feature gates
	controllerCh := make(chan struct{})

	context := initTestMaster(t, "reschedule-volume-provision", nil)

	clientset := context.clientSet
	ns := context.ns.Name

	defer func() {
		close(controllerCh)
		deleteTestObjects(clientset, ns, nil)
		context.clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
		context.closeFn()
	}()

	ctrl, informerFactory, err := initPVController(context, 0)
	if err != nil {
		t.Fatalf("Failed to create PV controller: %v", err)
	}

	// Prepare node and storage class.
	testNode := makeNode(0)
	if _, err := clientset.CoreV1().Nodes().Create(testNode); err != nil {
		t.Fatalf("Failed to create Node %q: %v", testNode.Name, err)
	}
	scName := "fail-provision"
	sc := makeDynamicProvisionerStorageClass(scName, &modeWait, nil)
	// Expect the storage class fail to provision.
	sc.Parameters[volumetest.ExpectProvisionFailureKey] = ""
	if _, err := clientset.StorageV1().StorageClasses().Create(sc); err != nil {
		t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
	}

	// Create a pvc with selected node annotation.
	pvcName := "pvc-fail-to-provision"
	pvc := makePVC(pvcName, ns, &scName, "")
	pvc.Annotations = map[string]string{"volume.kubernetes.io/selected-node": node1}
	pvc, err = clientset.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
	if err != nil {
		t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
	}
	// Validate selectedNode annotation exists on created claim.
	selectedNodeAnn, exist := pvc.Annotations["volume.kubernetes.io/selected-node"]
	if !exist || selectedNodeAnn != node1 {
		t.Fatalf("Created pvc is not annotated as expected")
	}

	// Start controller.
	go ctrl.Run(controllerCh)
	informerFactory.Start(controllerCh)
	informerFactory.WaitForCacheSync(controllerCh)

	// Validate that the annotation is removed by controller for provision reschedule.
	if err := waitForProvisionAnn(clientset, pvc, false); err != nil {
		t.Errorf("Expect to reschedule provision for PVC %v/%v, but still found selected-node annotation on it", ns, pvcName)
	}
}

func setupCluster(t *testing.T, nsName string, numberOfNodes int, resyncPeriod time.Duration, provisionDelaySeconds int) *testConfig {
	context := initTestSchedulerWithOptions(t, initTestMaster(t, nsName, nil), resyncPeriod)
	clientset := context.clientSet
	ns := context.ns.Name

	ctrl, informerFactory, err := initPVController(context, provisionDelaySeconds)
	if err != nil {
		t.Fatalf("Failed to create PV controller: %v", err)
	}
	go ctrl.Run(context.stopCh)
	// Start informer factory after all controllers are configured and running.
	informerFactory.Start(context.stopCh)
	informerFactory.WaitForCacheSync(context.stopCh)

	// Create shared objects
	// Create nodes
	for i := 0; i < numberOfNodes; i++ {
		testNode := makeNode(i)
		if _, err := clientset.CoreV1().Nodes().Create(testNode); err != nil {
			t.Fatalf("Failed to create Node %q: %v", testNode.Name, err)
		}
	}

	// Create SCs
	for _, sc := range sharedClasses {
		if _, err := clientset.StorageV1().StorageClasses().Create(sc); err != nil {
			t.Fatalf("Failed to create StorageClass %q: %v", sc.Name, err)
		}
	}

	return &testConfig{
		client: clientset,
		ns:     ns,
		stop:   context.stopCh,
		teardown: func() {
			deleteTestObjects(clientset, ns, nil)
			cleanupTest(t, context)
		},
	}
}

func initPVController(context *testContext, provisionDelaySeconds int) (*persistentvolume.PersistentVolumeController, informers.SharedInformerFactory, error) {
	clientset := context.clientSet
	// Informers factory for controllers, we disable resync period for testing.
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)

	// Start PV controller for volume binding.
	host := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugin := &volumetest.FakeVolumePlugin{
		PluginName:             provisionerPluginName,
		Host:                   host,
		Config:                 volume.VolumeConfig{},
		LastProvisionerOptions: volume.VolumeOptions{},
		ProvisionDelaySeconds:  provisionDelaySeconds,
		NewAttacherCallCount:   0,
		NewDetacherCallCount:   0,
		Mounters:               nil,
		Unmounters:             nil,
		Attachers:              nil,
		Detachers:              nil,
	}
	plugins := []volume.VolumePlugin{plugin}

	controllerOptions := persistentvolumeoptions.NewPersistentVolumeControllerOptions()
	params := persistentvolume.ControllerParameters{
		KubeClient:                clientset,
		SyncPeriod:                controllerOptions.PVClaimBinderSyncPeriod,
		VolumePlugins:             plugins,
		Cloud:                     nil,
		ClusterName:               "volume-test-cluster",
		VolumeInformer:            informerFactory.Core().V1().PersistentVolumes(),
		ClaimInformer:             informerFactory.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             informerFactory.Storage().V1().StorageClasses(),
		PodInformer:               informerFactory.Core().V1().Pods(),
		NodeInformer:              informerFactory.Core().V1().Nodes(),
		EnableDynamicProvisioning: true,
	}

	ctrl, err := persistentvolume.NewController(params)
	if err != nil {
		return nil, nil, err
	}

	return ctrl, informerFactory, nil
}

func deleteTestObjects(client clientset.Interface, ns string, option *metav1.DeleteOptions) {
	client.CoreV1().Pods(ns).DeleteCollection(option, metav1.ListOptions{})
	client.CoreV1().PersistentVolumeClaims(ns).DeleteCollection(option, metav1.ListOptions{})
	client.CoreV1().PersistentVolumes().DeleteCollection(option, metav1.ListOptions{})
	client.StorageV1().StorageClasses().DeleteCollection(option, metav1.ListOptions{})
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

func makeDynamicProvisionerStorageClass(name string, mode *storagev1.VolumeBindingMode, allowedTopologies []v1.TopologySelectorTerm) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Provisioner:       provisionerPluginName,
		VolumeBindingMode: mode,
		AllowedTopologies: allowedTopologies,
		Parameters:        map[string]string{},
	}
}

func makePV(name, scName, pvcName, ns, node string) *v1.PersistentVolume {
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
			Labels: map[string]string{
				"app": "volume-binding-test",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "write-pod",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "while true; do sleep 1; done"},
				},
			},
			Volumes: volumes,
		},
	}
}

func makeNode(index int) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   fmt.Sprintf("node-%d", index+1),
			Labels: map[string]string{nodeAffinityLabelKey: fmt.Sprintf("node-%d", index+1)},
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
}

func validatePVCPhase(t *testing.T, client clientset.Interface, pvcName string, ns string, phase v1.PersistentVolumeClaimPhase, isProvisioned bool) {
	claim, err := client.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get PVC %v/%v: %v", ns, pvcName, err)
	}

	if claim.Status.Phase != phase {
		t.Errorf("PVC %v/%v phase not %v, got %v", ns, pvcName, phase, claim.Status.Phase)
	}

	// Check whether the bound claim is provisioned/bound as expect.
	if phase == v1.ClaimBound {
		if err := validateProvisionAnn(claim, isProvisioned); err != nil {
			t.Errorf("Provisoning annotaion on PVC %v/%v not bahaviors as expected: %v", ns, pvcName, err)
		}
	}
}

func validateProvisionAnn(claim *v1.PersistentVolumeClaim, volIsProvisioned bool) error {
	selectedNode, provisionAnnoExist := claim.Annotations["volume.kubernetes.io/selected-node"]
	if volIsProvisioned {
		if !provisionAnnoExist {
			return fmt.Errorf("PVC %v/%v expected to be provisioned, but no selected-node annotation found", claim.Namespace, claim.Name)
		}
		if selectedNode != node1 {
			return fmt.Errorf("PVC %v/%v expected to be annotated as %v, but got %v", claim.Namespace, claim.Name, node1, selectedNode)
		}
	}
	if !volIsProvisioned && provisionAnnoExist {
		return fmt.Errorf("PVC %v/%v not expected to be provisioned, but found selected-node annotation", claim.Namespace, claim.Name)
	}

	return nil
}

func waitForProvisionAnn(client clientset.Interface, pvc *v1.PersistentVolumeClaim, annShouldExist bool) error {
	return wait.Poll(time.Second, 30*time.Second, func() (bool, error) {
		claim, err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if err := validateProvisionAnn(claim, annShouldExist); err == nil {
			return true, nil
		}
		return false, nil
	})
}

func validatePVPhase(t *testing.T, client clientset.Interface, pvName string, phase v1.PersistentVolumePhase) {
	pv, err := client.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get PV %v: %v", pvName, err)
	}

	if pv.Status.Phase != phase {
		t.Errorf("PV %v phase not %v, got %v", pvName, phase, pv.Status.Phase)
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
