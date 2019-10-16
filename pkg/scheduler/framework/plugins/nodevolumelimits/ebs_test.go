/*
Copyright 2019 The Kubernetes Authors.

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

package nodevolumelimits

import (
	"context"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	fakelisters "k8s.io/kubernetes/pkg/scheduler/listers/fake"
	utilpointer "k8s.io/utils/pointer"
)

func onePVCPod(filterName string) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "some" + filterName + "Vol",
						},
					},
				},
			},
		},
	}
}

func splitPVCPod(filterName string) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someNon" + filterName + "Vol",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "some" + filterName + "Vol",
						},
					},
				},
			},
		},
	}
}

func TestEBSLimits(t *testing.T) {
	oneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp"},
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
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp1"},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp2"},
					},
				},
			},
		},
	}
	unboundPVCwithInvalidSCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVCwithInvalidSCPod",
						},
					},
				},
			},
		},
	}
	unboundPVCwithDefaultSCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVCwithDefaultSCPod",
						},
					},
				},
			},
		},
	}
	splitVolsPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "svp"},
					},
				},
			},
		},
	}
	nonApplicablePod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{},
					},
				},
			},
		},
	}
	deletedPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "deletedPVC",
						},
					},
				},
			},
		},
	}
	twoDeletedPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "deletedPVC",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherDeletedPVC",
						},
					},
				},
			},
		},
	}
	deletedPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "pvcWithDeletedPV",
						},
					},
				},
			},
		},
	}
	// deletedPVPod2 is a different pod than deletedPVPod but using the same PVC
	deletedPVPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "pvcWithDeletedPV",
						},
					},
				},
			},
		},
	}
	// anotherDeletedPVPod is a different pod than deletedPVPod and uses another PVC
	anotherDeletedPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherPVCWithDeletedPV",
						},
					},
				},
			},
		},
	}
	emptyPod := &v1.Pod{
		Spec: v1.PodSpec{},
	}
	unboundPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVC",
						},
					},
				},
			},
		},
	}
	// Different pod than unboundPVCPod, but using the same unbound PVC
	unboundPVCPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVC",
						},
					},
				},
			},
		},
	}

	// pod with unbound PVC that's different to unboundPVC
	anotherUnboundPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherUnboundPVC",
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
		driverName   string
		maxVols      int
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      4,
			test:         "fits when node capacity >= new pod's EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "doesn't fit when node capacity < new pod's EBS volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count ignores non-EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "existing pods' counts ignore non-EBS volumes",
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count considers PVCs backed by EBS volumes",
		},
		{
			newPod:       splitPVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count ignores PVCs not backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(predicates.EBSVolumeFilterType)},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "existing pods' counts considers PVCs backed by EBS volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(predicates.EBSVolumeFilterType)},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      4,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(predicates.EBSVolumeFilterType)},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "the same EBS volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "missing PVC is not counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "missing PVC is not counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two missing PVCs are not counted towards the PV limit twice",
		},
		{
			newPod:       unboundPVCwithInvalidSCPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "unbound PVC with invalid SC is not counted towards the PV limit",
		},
		{
			newPod:       unboundPVCwithDefaultSCPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "unbound PVC from different provisioner is not counted towards the PV limit",
		},

		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "pod with missing PV is counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "pod with unbound PVC is counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
		{
			newPod:       onePVCPod(predicates.EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   predicates.EBSVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, predicates.ErrMaxVolumeCountExceeded.GetReason()),
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), test.filterName)
			p := &EBSLimits{
				predicate: predicates.NewMaxPDVolumeCountPredicate(test.filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(test.filterName, test.driverName), getFakePVLister(test.filterName), getFakePVCLister(test.filterName)),
			}
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func getFakePVCLister(filterName string) fakelisters.PersistentVolumeClaimLister {
	return fakelisters.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "some" + filterName + "Vol"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "some" + filterName + "Vol",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNon" + filterName + "Vol"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "someNon" + filterName + "Vol",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pvcWithDeletedPV"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "pvcWithDeletedPV",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherPVCWithDeletedPV"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "anotherPVCWithDeletedPV",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVC"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherUnboundPVC"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVCwithDefaultSCPod"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: utilpointer.StringPtr("standard-sc"),
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVCwithInvalidSCPod"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: utilpointer.StringPtr("invalid-sc"),
			},
		},
	}
}

func getFakePVLister(filterName string) fakelisters.PersistentVolumeLister {
	return fakelisters.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "some" + filterName + "Vol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: strings.ToLower(filterName) + "Vol"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNon" + filterName + "Vol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{},
			},
		},
	}
}
