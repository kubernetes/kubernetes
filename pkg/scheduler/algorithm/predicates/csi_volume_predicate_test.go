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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
)

func TestCSIVolumeCountPredicate(t *testing.T) {
	// for pods with CSI pvcs
	oneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs",
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

	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		maxVols      int
		fits         bool
		test         string
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{runningPod, twoVolPod},
			filterName:   "csi-ebs",
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pods CSI volume",
		},
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{runningPod, twoVolPod},
			filterName:   "csi-ebs",
			maxVols:      2,
			fits:         false,
			test:         "doesn't when node capacity <= pods CSI volume",
		},
	}

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}
	// running attachable predicate tests with feature gate and limit present on nodes
	for _, test := range tests {
		node := getNodeWithPodAndVolumeLimits(test.existingPods, int64(test.maxVols), test.filterName)
		pred := NewCSIMaxVolumeLimitPredicate(getFakeCSIPVInfo("csi-ebs", "csi-ebs"), getFakeCSIPVCInfo("csi-ebs"))
		fits, reasons, err := pred(test.newPod, PredicateMetadata(test.newPod, nil), node)
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

func getFakeCSIPVInfo(volumeName, driverName string) FakePersistentVolumeInfo {
	return FakePersistentVolumeInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver:       driverName,
						VolumeHandle: volumeName,
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-2"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver:       driverName,
						VolumeHandle: volumeName + "-2",
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-3"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver:       driverName,
						VolumeHandle: volumeName + "-3",
					},
				},
			},
		},
	}
}

func getFakeCSIPVCInfo(volumeName string) FakePersistentVolumeClaimInfo {
	return FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: volumeName},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-2"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: volumeName + "-2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-3"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: volumeName + "-3"},
		},
	}
}
