/*
Copyright 2018 The Kubernetes Authors.

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
	"os"
	"strconv"
	"testing"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

const (
	testRegion     = "test-region"
	nonExistRegion = "non-exist-region"
)

func deletePVC(t *testing.T, pvcName string, config *testConfig) {
	if err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Delete(pvcName, nil); err != nil {
		t.Fatalf("Failed to delete PersistentVolumeClaim %q: %v", pvcName, err)
	}
}

func removePod(t *testing.T, podName string, config *testConfig) {
	if err := config.client.CoreV1().Pods(config.ns).Delete(podName, metav1.NewDeleteOptions(0)); err != nil {
		t.Fatalf("Failed to delete Pod %q: %v", podName, err)
	}
}

func deletePV(t *testing.T, pvName string, config *testConfig) {
	if err := config.client.CoreV1().PersistentVolumes().Delete(pvName, nil); err != nil {
		t.Fatalf("Failed to delete PersistentVolume %q: %v", pvName, err)
	}
}

func updatePVRegion(t *testing.T, pvName string, config *testConfig) {
	pv, err := config.client.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get PersistentVolume %q: %v", pvName, err)
	}
	// Update the PV with a invalid region.
	pv.SetLabels(map[string]string{kubeletapis.LabelZoneRegion: nonExistRegion})
	if _, err := config.client.CoreV1().PersistentVolumes().Update(pv); err != nil {
		t.Fatalf("Failed to update PersistentVolume %q: %v", pvName, err)
	}
}

// TestNoVolumeZoneConflictPred verifies NoVolumeZoneConflictPred will be properly invalidated after certain actions happen.
func TestNoVolumeZoneConflictPred(t *testing.T) {
	policy := &schedulerapi.Policy{
		Predicates: []schedulerapi.PredicatePolicy{
			{
				Name: predicates.NoVolumeZoneConflictPred,
			},
		},
	}
	policy.APIVersion = testapi.Groups[v1.GroupName].GroupVersion().String()
	config := setupClusterWithOptions(t, "volume-scheduling", 2, policy, "")
	defer config.teardown()

	// Create a test PV with region label.
	pv := makePV("pv-canbind-1", classImmediate, "", "", node1)
	pv.SetLabels(map[string]string{
		kubeletapis.LabelZoneRegion: testRegion,
	})

	cases := testCase{
		"On PVC delete": {
			pod:  makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			pvs:  []*v1.PersistentVolume{pv},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv.Name)},
			action: func() {
				deletePVC(t, "pvc-canbind-1", config)
			},
			equivPod: makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Scheduling should fail since PVC has been deleted.
			equivPodShouldFail: true,
		},
		"On PVC add": {
			pod:             makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			firstShouldFail: true,
			unboundPvs:      []*v1.PersistentVolume{pv},
			action: func() {
				pvc := makePVC("pvc-canbind-1", config.ns, &classImmediate, pv.Name)
				if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
					t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
				}
			},
			equivPod: makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Scheduling should not fail since the requested PVC has been added.
			equivPodShouldFail: false,
		},
		"On PV delete": {
			pod:  makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			pvs:  []*v1.PersistentVolume{pv},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv.Name)},
			action: func() {
				deletePV(t, pv.Name, config)
			},
			equivPod: makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Scheduling should fail since PV has been deleted .
			equivPodShouldFail: true,
		},
		"On PV add": {
			pod:             makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			firstShouldFail: true,
			unboundPvcs:     []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv.Name)},
			action: func() {
				// Add a PV to change pvc-canbind-1 to bound.
				if _, err := config.client.CoreV1().PersistentVolumes().Create(pv); err != nil {
					t.Fatalf("Failed to create PersistentVolume %q: %v", pv.Name, err)
				}
			},
			equivPod: makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Scheduling should not fail since pvc-canbind-1 has become to bound.
			equivPodShouldFail: false,
		},
		"On PV region update": {
			pod:  makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			pvs:  []*v1.PersistentVolume{pv},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv.Name)},
			action: func() {
				updatePVRegion(t, pv.Name, config)
			},
			equivPod: makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Scheduling should fail since PV has been updated with a invalid region.
			equivPodShouldFail: true,
		},
	}

	runTest(t, cases, config)
}

// TestMaxGCEPDVolumeCountPred verifies MaxGCEPDVolumeCountPred will be properly invalidated after certain actions happen.
func TestMaxGCEPDVolumeCountPred(t *testing.T) {
	// Setup a cluster with max PD volume count to 1.
	os.Setenv(predicates.KubeMaxPDVols, strconv.Itoa(1))

	policy := &schedulerapi.Policy{
		Predicates: []schedulerapi.PredicatePolicy{
			{
				Name: predicates.MaxGCEPDVolumeCountPred,
			},
		},
	}
	policy.APIVersion = testapi.Groups[v1.GroupName].GroupVersion().String()
	config := setupClusterWithOptions(t, "volume-scheduling", 1, policy, "")
	defer config.teardown()

	podPv1 := makePod("pod-pv1", config.ns, []string{})
	podPv1.Spec.Volumes = []v1.Volume{
		{
			Name: "pv-gcepd-2",
			VolumeSource: v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: "test-gcepd-2",
				},
			},
		},
	}

	podPv2 := makePod("pod-pv2", config.ns, []string{})
	podPv2.Spec.Volumes = []v1.Volume{
		{
			Name: "pv-gcepd-2",
			VolumeSource: v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: "test-gcepd-2",
				},
			},
		},
	}

	podPv := makePod("pod-pv", config.ns, []string{})
	podPv.Spec.Volumes = []v1.Volume{
		{
			Name: "pv-gcepd",
			VolumeSource: v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: "test-gcepd",
				},
			},
		},
	}

	// These test cases will create work-flow as below:
	// 1. Create first Pod, its schedule result is cached in ecache.
	// 2. Perform action to change the status of MaxGCEPDVolumeCountPred, i.e. abuse/release PD count limit.
	// 3. Create the equivalent Pod of first Pod, it should have opposite schedule result, since it is re-calculated with
	//    respect to step 2. We need to create opposite schedule result to verify ecache data is not reused.
	cases := testCase{
		"On Pod delete": {
			existingPods: []*v1.Pod{podPv},
			pod:          podPv1,
			// Fail because PD count limit is abused
			firstShouldFail: true,
			action: func() {
				removePod(t, podPv.Name, config)
			},
			// The equivalent Pod should not fail since PD count limit is released by Pod deletion.
			equivPod:           podPv2,
			equivPodShouldFail: false,
		},
		// TODO(harry): it's not clear how to test "On Pod add" because a equivalent Pod always has the same PVC
		// or PV with the original ones, so it will not abuse PV count limit and be bound with existing volumes.
		// TODO(harry): it's not clear how to test "On PV/PVC add" only because we always need to schedule a equivalent Pod
		// at the last step, but it will then falls into "On Pod add" case.
		// So maybe we should not invalidate ecache on Pod add also, we can then test On PV/PVC add.
		// NOTE: PVC cannot modify its VolumeName once its set, so we don't have to worry about this.
	}

	runTest(t, cases, config)
}

// TestCheckVolumeBindingPred verifies CheckVolumeBindingPred will be properly invalidated after certain actions happen.
func TestCheckVolumeBindingPred(t *testing.T) {
	policy := &schedulerapi.Policy{
		Predicates: []schedulerapi.PredicatePolicy{
			{
				Name: predicates.CheckVolumeBindingPred,
			},
		},
	}
	policy.APIVersion = testapi.Groups[v1.GroupName].GroupVersion().String()
	config := setupClusterWithOptions(
		t, "volume-scheduling", 2, policy, "VolumeScheduling=true")
	defer config.teardown()

	// Create test PV with GCEPersistentDisk.
	// TODO(harry): maybe with local disk so don't need to set PV name.
	pv1 := makePV("pv-gcepd-1", classImmediate, "", "", node1)

	// These test cases will create work-flow as below:
	// 1. Create first Pod, its schedule result is cached in ecache.
	// 2. Perform action to make schedule result of first Pod invalid,
	//    either change from schedulable to unschedulable or vice versa.
	// 3. Create the equivalent Pod of first Pod, it should have a opposite schedule result.
	cases := testCase{
		"On PV add": {
			pod: makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Should not schedule as pvc-canbind-1 is not bound for now.
			firstShouldFail: true,
			unboundPvcs:     []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv1.Name)},
			action: func() {
				// Add a PV to change pvc-canbind-1 to bound.
				if _, err := config.client.CoreV1().PersistentVolumes().Create(pv1); err != nil {
					t.Fatalf("Failed to create PersistentVolume %q: %v", pv1.Name, err)
				}
			},
			// The second pod claim pvc-canbind-1 , it should be scheduled as PVC became bound.
			equivPod:           makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			equivPodShouldFail: false,
		},
		"On PV delete": {
			pod:  makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			pvs:  []*v1.PersistentVolume{pv1},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv1.Name)},
			action: func() {
				// Delete PV.
				deletePV(t, pv1.Name, config)
			},
			// The second pod claim the previous PVC, it should not be scheduled as PV is deleted.
			equivPod:           makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			equivPodShouldFail: true,
		},
		"On PVC delete": {
			pod:  makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			pvs:  []*v1.PersistentVolume{pv1},
			pvcs: []*v1.PersistentVolumeClaim{makePVC("pvc-canbind-1", config.ns, &classImmediate, pv1.Name)},
			action: func() {
				// Delete PVC.
				deletePVC(t, "pvc-canbind-1", config)
			},
			// The second pod claim the previous PVC, it should not be scheduled as PVC is deleted.
			equivPod:           makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			equivPodShouldFail: true,
		},
		"On PVC add": {
			pod: makePod("pod-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			// Should not schedule as pvc-canbind-1 is not exists for now.
			firstShouldFail: true,
			unboundPvs:      []*v1.PersistentVolume{pv1},
			action: func() {
				pvc := makePVC("pvc-canbind-1", config.ns, &classImmediate, pv1.Name)
				if _, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(pvc); err != nil {
					t.Fatalf("Failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
				}
			},
			// The second pod claim the previous PVC, it should be scheduled as PVC is added.
			equivPod:           makePod("pod-2-pvc-1", config.ns, []string{"pvc-canbind-1"}),
			equivPodShouldFail: false,
		},
		// NOTE: PVC cannot modify its VolumeName once its set, so we don't have to worry about this. It's
		// the same case for NodeAffinity of PV as well.
	}

	runTest(t, cases, config)
}

type testCase map[string]struct {
	pod *v1.Pod

	existingPods []*v1.Pod

	pvs  []*v1.PersistentVolume
	pvcs []*v1.PersistentVolumeClaim
	// Create these, but they should not be bound in the end
	unboundPvcs     []*v1.PersistentVolumeClaim
	unboundPvs      []*v1.PersistentVolume
	firstShouldFail bool

	// Action happens after first pod is processed.
	action func()
	// The second pod comes after actions. It should be of first pod.
	equivPod           *v1.Pod
	equivPodShouldFail bool
}

func runTest(t *testing.T, cases testCase, config *testConfig) {
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

		// Create existing Pods
		for _, pod := range test.existingPods {
			if _, err := config.client.CoreV1().Pods(config.ns).Create(pod); err != nil {
				t.Fatalf("Failed to create existing Pod %q: %v", pod.Name, err)
			}
			if err := waitForPodToSchedule(config.client, pod); err != nil {
				t.Errorf("Failed to schedule existing Pod %q: %v", pod.Name, err)
			}
		}

		// Create first Pod
		if _, err := config.client.CoreV1().Pods(config.ns).Create(test.pod); err != nil {
			t.Fatalf("Failed to create Pod %q: %v", test.pod.Name, err)
		}
		if test.firstShouldFail {
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
			validatePVCPhase(t, config.client, pvc.GetName(), config.ns, v1.ClaimBound)
		}
		for _, pvc := range test.unboundPvcs {
			validatePVCPhase(t, config.client, pvc.GetName(), config.ns, v1.ClaimPending)
		}
		for _, pv := range test.pvs {
			validatePVPhase(t, config.client, pv.GetName(), v1.VolumeBound)
		}
		for _, pv := range test.unboundPvs {
			validatePVPhase(t, config.client, pv.GetName(), v1.VolumeAvailable)
		}

		// Do the action
		if test.action != nil {
			test.action()
			glog.Infof("Doing action to trigger ecache invalidation ...")
		}

		// TODO(harry): verify equivPod is equivalent to given Pod after we shipped the new equiv hash function.

		// Create equivalent Pod and validate it's status.
		if _, err := config.client.CoreV1().Pods(config.ns).Create(test.equivPod); err != nil {
			t.Fatalf("Failed to create equivalent Pod %q: %v", test.equivPod.Name, err)
		}
		if test.equivPodShouldFail {
			if err := waitForPodUnschedulable(config.client, test.equivPod); err != nil {
				t.Errorf("Pod %q was not unschedulable: %v", test.equivPod.Name, err)
			}
		} else {
			if err := waitForPodToSchedule(config.client, test.equivPod); err != nil {
				t.Errorf("Failed to schedule Pod %q: %v", test.equivPod.Name, err)
			}
		}

		config.client.CoreV1().Pods(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumeClaims(config.ns).DeleteCollection(deleteOption, metav1.ListOptions{})
		config.client.CoreV1().PersistentVolumes().DeleteCollection(deleteOption, metav1.ListOptions{})
	}
}
