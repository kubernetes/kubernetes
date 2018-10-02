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
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
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

func TestVolumeCountConflicts(t *testing.T) {
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
		maxVols      int
		fits         bool
		test         string
	}{
		// filterName:EBSVolumeFilterType
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pod's EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "doesn't fit when node capacity < new pod's EBS volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores non-EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts ignore non-EBS volumes",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count considers PVCs backed by EBS volumes",
		},
		{
			newPod:       splitPVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores PVCs not backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(EBSVolumeFilterType)},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         false,
			test:         "existing pods' counts considers PVCs backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(EBSVolumeFilterType)},
			filterName:   EBSVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(EBSVolumeFilterType)},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "the same EBS volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         false,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(EBSVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   EBSVolumeFilterType,
			maxVols:      2,
			fits:         false,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
		// filterName:GCEPDVolumeFilterType
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pod's GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "fit when node capacity < new pod's GCE volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores non-GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts ignore non-GCE volumes",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count considers PVCs backed by GCE volumes",
		},
		{
			newPod:       splitPVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores PVCs not backed by GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(GCEPDVolumeFilterType)},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts considers PVCs backed by GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(GCEPDVolumeFilterType)},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(GCEPDVolumeFilterType)},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "the same GCE volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(GCEPDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   GCEPDVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
		// filterName:AzureDiskVolumeFilterType
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pod's AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "fit when node capacity < new pod's AzureDisk volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores non-AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts ignore non-AzureDisk volumes",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count considers PVCs backed by AzureDisk volumes",
		},
		{
			newPod:       splitPVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores PVCs not backed by AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(AzureDiskVolumeFilterType)},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts considers PVCs backed by AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(AzureDiskVolumeFilterType)},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      4,
			fits:         true,
			test:         "already-mounted AzureDisk volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(AzureDiskVolumeFilterType)},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "the same AzureDisk volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(AzureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      3,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   AzureDiskVolumeFilterType,
			maxVols:      2,
			fits:         true,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
	}

	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}

	// running attachable predicate tests without feature gate and no limit present on nodes
	for _, test := range tests {
		os.Setenv(KubeMaxPDVols, strconv.Itoa(test.maxVols))
		pred := NewMaxPDVolumeCountPredicate(test.filterName, getFakePVInfo(test.filterName), getFakePVCInfo(test.filterName))
		fits, reasons, err := pred(test.newPod, PredicateMetadata(test.newPod, nil), schedulercache.NewNodeInfo(test.existingPods...))
		if err != nil {
			t.Errorf("[%s]%s: unexpected error: %v", test.filterName, test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("[%s]%s: unexpected failure reasons: %v, want: %v", test.filterName, test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("[%s]%s: expected %v, got %v", test.filterName, test.test, test.fits, fits)
		}
	}

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()

	// running attachable predicate tests with feature gate and limit present on nodes
	for _, test := range tests {
		node := getNodeWithPodAndVolumeLimits(test.existingPods, int64(test.maxVols), test.filterName)
		pred := NewMaxPDVolumeCountPredicate(test.filterName, getFakePVInfo(test.filterName), getFakePVCInfo(test.filterName))
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

func getFakePVInfo(filterName string) FakePersistentVolumeInfo {
	return FakePersistentVolumeInfo{
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

func getFakePVCInfo(filterName string) FakePersistentVolumeClaimInfo {
	return FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "some" + filterName + "Vol"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "some" + filterName + "Vol"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNon" + filterName + "Vol"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "someNon" + filterName + "Vol"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pvcWithDeletedPV"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "pvcWithDeletedPV"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherPVCWithDeletedPV"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "anotherPVCWithDeletedPV"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVC"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherUnboundPVC"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
		},
	}
}

func TestMaxVolumeFuncM5(t *testing.T) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-for-m5-instance",
			Labels: map[string]string{
				kubeletapis.LabelInstanceType: "m5.large",
			},
		},
	}
	os.Unsetenv(KubeMaxPDVols)
	maxVolumeFunc := getMaxVolumeFunc(EBSVolumeFilterType)
	maxVolume := maxVolumeFunc(node)
	if maxVolume != volumeutil.DefaultMaxEBSNitroVolumeLimit {
		t.Errorf("Expected max volume to be %d got %d", volumeutil.DefaultMaxEBSNitroVolumeLimit, maxVolume)
	}
}

func TestMaxVolumeFuncT3(t *testing.T) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-for-t3-instance",
			Labels: map[string]string{
				kubeletapis.LabelInstanceType: "t3.medium",
			},
		},
	}
	os.Unsetenv(KubeMaxPDVols)
	maxVolumeFunc := getMaxVolumeFunc(EBSVolumeFilterType)
	maxVolume := maxVolumeFunc(node)
	if maxVolume != volumeutil.DefaultMaxEBSNitroVolumeLimit {
		t.Errorf("Expected max volume to be %d got %d", volumeutil.DefaultMaxEBSNitroVolumeLimit, maxVolume)
	}
}

func TestMaxVolumeFuncR5(t *testing.T) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-for-r5-instance",
			Labels: map[string]string{
				kubeletapis.LabelInstanceType: "r5d.xlarge",
			},
		},
	}
	os.Unsetenv(KubeMaxPDVols)
	maxVolumeFunc := getMaxVolumeFunc(EBSVolumeFilterType)
	maxVolume := maxVolumeFunc(node)
	if maxVolume != volumeutil.DefaultMaxEBSNitroVolumeLimit {
		t.Errorf("Expected max volume to be %d got %d", volumeutil.DefaultMaxEBSNitroVolumeLimit, maxVolume)
	}
}

func TestMaxVolumeFuncM4(t *testing.T) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-for-m4-instance",
			Labels: map[string]string{
				kubeletapis.LabelInstanceType: "m4.2xlarge",
			},
		},
	}
	os.Unsetenv(KubeMaxPDVols)
	maxVolumeFunc := getMaxVolumeFunc(EBSVolumeFilterType)
	maxVolume := maxVolumeFunc(node)
	if maxVolume != volumeutil.DefaultMaxEBSVolumes {
		t.Errorf("Expected max volume to be %d got %d", volumeutil.DefaultMaxEBSVolumes, maxVolume)
	}
}

func getNodeWithPodAndVolumeLimits(pods []*v1.Pod, limit int64, filter string) *schedulercache.NodeInfo {
	nodeInfo := schedulercache.NewNodeInfo(pods...)
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-for-max-pd-test-1"},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				getVolumeLimitKey(filter): *resource.NewQuantity(limit, resource.DecimalSI),
			},
		},
	}
	nodeInfo.SetNode(node)
	return nodeInfo
}

func getVolumeLimitKey(filterType string) v1.ResourceName {
	switch filterType {
	case EBSVolumeFilterType:
		return v1.ResourceName(volumeutil.EBSVolumeLimitKey)
	case GCEPDVolumeFilterType:
		return v1.ResourceName(volumeutil.GCEVolumeLimitKey)
	case AzureDiskVolumeFilterType:
		return v1.ResourceName(volumeutil.AzureVolumeLimitKey)
	default:
		return v1.ResourceName(volumeutil.GetCSIAttachLimitKey(filterType))
	}
}
