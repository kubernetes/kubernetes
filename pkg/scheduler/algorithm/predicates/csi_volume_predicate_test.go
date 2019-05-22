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

package predicates

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestCSIVolumeCountPredicate(t *testing.T) {
	// for pods with CSI pvcs
	oneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs-0",
						},
					},
				},
			},
		},
	}
	twoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "cs-ebs-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs-2",
						},
					},
				},
			},
		},
	}

	runningPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs-3",
						},
					},
				},
			},
		},
	}

	pendingVolumePod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-4",
						},
					},
				},
			},
		},
	}

	// Different pod than pendingVolumePod, but using the same unbound PVC
	unboundPVCPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-4",
						},
					},
				},
			},
		},
	}

	missingPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-6",
						},
					},
				},
			},
		},
	}

	noSCPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-5",
						},
					},
				},
			},
		},
	}
	gceTwoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "cs-gce-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-gce-2",
						},
					},
				},
			},
		},
	}

	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		maxVols      int
		driverNames  []string
		fits         bool
		test         string
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{runningPod, twoVolPod},
			filterName:   "csi",
			maxVols:      4,
			driverNames:  []string{"ebs"},
			fits:         true,
			test:         "fits when node capacity >= new pods CSI volume",
		},
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{runningPod, twoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs"},
			fits:         false,
			test:         "doesn't when node capacity <= pods CSI volume",
		},
		// should count pending PVCs
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{pendingVolumePod, twoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs"},
			fits:         false,
			test:         "count pending PVCs towards capacity <= pods CSI volume",
		},
		// two same pending PVCs should be counted as 1
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{pendingVolumePod, unboundPVCPod2, twoVolPod},
			filterName:   "csi",
			maxVols:      3,
			driverNames:  []string{"ebs"},
			fits:         true,
			test:         "count multiple pending pvcs towards capacity >= pods CSI volume",
		},
		// should count PVCs with invalid PV name but valid SC
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{missingPVPod, twoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs"},
			fits:         false,
			test:         "should count PVCs with invalid PV name but valid SC",
		},
		// don't count a volume which has storageclass missing
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{runningPod, noSCPVCPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs"},
			fits:         true,
			test:         "don't count pvcs with missing SC towards capacity",
		},
		// don't count multiple volume types
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{gceTwoVolPod, twoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs", "gce"},
			fits:         true,
			test:         "don't count pvcs with different type towards capacity",
		},
		{
			newPod:       gceTwoVolPod,
			existingPods: []*v1.Pod{twoVolPod, runningPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{"ebs", "gce"},
			fits:         true,
			test:         "don't count pvcs with different type towards capacity",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()
	expectedFailureReasons := []PredicateFailureReason{ErrMaxVolumeCountExceeded}
	// running attachable predicate tests with feature gate and limit present on nodes
	for _, test := range tests {
		node := getNodeWithPodAndVolumeLimits(test.existingPods, int64(test.maxVols), test.driverNames...)
		pred := NewCSIMaxVolumeLimitPredicate(getFakeCSIPVInfo(test.filterName, test.driverNames...),
			getFakeCSIPVCInfo(test.filterName, "csi-sc", test.driverNames...),
			getFakeCSIStorageClassInfo("csi-sc", test.driverNames[0]))

		fits, reasons, err := pred(test.newPod, GetPredicateMetadata(test.newPod, nil), node)
		if err != nil {
			t.Errorf("Using allocatable [%s]%s: unexpected error: %v", test.filterName, test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("Using allocatable [%s]%s: unexpected failure reasons: %v, want: %v", test.filterName, test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("Using allocatable [%s]%s: expected %v, got %v", test.filterName, test.test, test.fits, fits)
		}
	}
}

func getFakeCSIPVInfo(volumeName string, driverNames ...string) FakePersistentVolumeInfo {
	pvInfos := FakePersistentVolumeInfo{}
	for _, driver := range driverNames {
		for j := 0; j < 4; j++ {
			volumeHandle := fmt.Sprintf("%s-%s-%d", volumeName, driver, j)
			pv := v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: volumeHandle},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       driver,
							VolumeHandle: volumeHandle,
						},
					},
				},
			}
			pvInfos = append(pvInfos, pv)
		}

	}
	return pvInfos
}

func getFakeCSIPVCInfo(volumeName, scName string, driverNames ...string) FakePersistentVolumeClaimInfo {
	pvcInfos := FakePersistentVolumeClaimInfo{}
	for _, driver := range driverNames {
		for j := 0; j < 4; j++ {
			v := fmt.Sprintf("%s-%s-%d", volumeName, driver, j)
			pvc := v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: v},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: v},
			}
			pvcInfos = append(pvcInfos, pvc)
		}
	}

	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-4"},
		Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &scName},
	})
	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-5"},
		Spec:       v1.PersistentVolumeClaimSpec{},
	})
	// a pvc with missing PV but available storageclass.
	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-6"},
		Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &scName, VolumeName: "missing-in-action"},
	})
	return pvcInfos
}

func getFakeCSIStorageClassInfo(scName, provisionerName string) FakeStorageClassInfo {
	return FakeStorageClassInfo{
		{
			ObjectMeta:  metav1.ObjectMeta{Name: scName},
			Provisioner: provisionerName,
		},
	}
}
