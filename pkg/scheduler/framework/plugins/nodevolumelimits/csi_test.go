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
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csitrans "k8s.io/csi-translation-lib"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/utils/ptr"
)

const (
	ebsCSIDriverName = csilibplugins.AWSEBSDriverName
)

var (
	scName = "csi-sc"
)

func TestCSILimits(t *testing.T) {
	testCases := []struct {
		name                string
		newPod              *v1.Pod
		existingPods        []*v1.Pod
		nodeVolumeLimit     int32
		volumeAttachments   []*storagev1.VolumeAttachment
		expectedSchedulable bool
	}{
		{
			name:                "pod with no volumes",
			newPod:              st.MakePod().Name("pod-no-volumes").Obj(),
			nodeVolumeLimit:     10,
			expectedSchedulable: true,
		},
		{
			name:                "pod with one volume, no existing attachments",
			newPod:              st.MakePod().Name("pod-one-volume").PVC("pvc-1").Obj(),
			nodeVolumeLimit:     1,
			expectedSchedulable: true,
		},
		{
			name:   "pod with one volume, at volume limit",
			newPod: st.MakePod().Name("pod-one-volume").PVC("pvc-1").Obj(),
			volumeAttachments: []*storagev1.VolumeAttachment{
				makeVolumeAttachment("va-1", "node-1", ebsCSIDriverName, "pv-1"),
				makeVolumeAttachment("va-2", "node-1", ebsCSIDriverName, "pv-2"),
			},
			nodeVolumeLimit:     3,
			expectedSchedulable: true,
		},
		{
			name:   "pod with one volume, over volume limit",
			newPod: st.MakePod().Name("pod-one-volume").PVC("pvc-1").Obj(),
			volumeAttachments: []*storagev1.VolumeAttachment{
				makeVolumeAttachment("va-1", "node-1", ebsCSIDriverName, "pv-1"),
				makeVolumeAttachment("va-2", "node-1", ebsCSIDriverName, "pv-2"),
				makeVolumeAttachment("va-3", "node-1", ebsCSIDriverName, "pv-3"),
			},
			nodeVolumeLimit:     3,
			expectedSchedulable: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nodeInfo := framework.NewNodeInfo()
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
			}
			nodeInfo.SetNode(node)

			csiNode := &storagev1.CSINode{
				ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
				Spec: storagev1.CSINodeSpec{
					Drivers: []storagev1.CSINodeDriver{
						{
							Name:   ebsCSIDriverName,
							NodeID: "node-1",
							Allocatable: &storagev1.VolumeNodeResources{
								Count: ptr.To(tc.nodeVolumeLimit),
							},
						},
					},
				},
			}

			p := &CSILimits{
				csiNodeLister: getFakeCSINodeLister(csiNode),
				pvLister:      getFakeCSIPVLister(ebsCSIDriverName),
				pvcLister:     getFakeCSIPVCLister("pvc", scName, ebsCSIDriverName),
				scLister:      getFakeCSIStorageClassLister(scName, ebsCSIDriverName),
				vaLister:      getFakeVolumeAttachmentLister(tc.volumeAttachments),
				translator:    csitrans.New(),
			}

			gotStatus := p.Filter(context.Background(), nil, tc.newPod, nodeInfo)
			checkStatus(t, tc.name, tc.expectedSchedulable, gotStatus)
		})
	}
}

func checkStatus(t *testing.T, name string, expectedSchedulable bool, gotStatus *framework.Status) {
	if expectedSchedulable && gotStatus != nil {
		t.Errorf("%s: expected schedulable, got unschedulable: %v", name, gotStatus)
	}
	if !expectedSchedulable {
		if gotStatus == nil {
			t.Errorf("%s: expected unschedulable, got schedulable", name)
		} else {
			if gotStatus.Code() != framework.Unschedulable {
				t.Errorf("%s: expected status code %d, got %d", name, framework.Unschedulable, gotStatus.Code())
			}
			if gotStatus.Reasons()[0] != ErrReasonMaxVolumeCountExceeded {
				t.Errorf("%s: expected status reason %q, got %q", name, ErrReasonMaxVolumeCountExceeded, gotStatus.Reasons()[0])
			}
		}
	}
}

func getFakeCSINodeLister(csiNode *storagev1.CSINode) tf.CSINodeLister {
	csiNodeLister := tf.CSINodeLister{}
	if csiNode != nil {
		csiNodeLister = append(csiNodeLister, *csiNode.DeepCopy())
	}
	return csiNodeLister
}

func getFakeVolumeAttachmentLister(attachments []*storagev1.VolumeAttachment) tf.VolumeAttachmentLister {
	vaList := tf.VolumeAttachmentLister{}
	for _, va := range attachments {
		vaList = append(vaList, *va.DeepCopy())
	}
	return vaList
}

func getFakeCSIPVLister(driverName string) tf.PersistentVolumeLister {
	return tf.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pv-1"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver: driverName,
					},
				},
			},
		},
	}
}

func getFakeCSIPVCLister(_, scName, _ string) tf.PersistentVolumeClaimLister {
	return tf.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pvc-1"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "pv-1",
				StorageClassName: &scName,
			},
		},
	}
}

func getFakeCSIStorageClassLister(scName, provisionerName string) tf.StorageClassLister {
	return tf.StorageClassLister{
		{
			ObjectMeta:  metav1.ObjectMeta{Name: scName},
			Provisioner: provisionerName,
		},
	}
}

func makeVolumeAttachment(name, nodeName, driverName, pvName string) *storagev1.VolumeAttachment {
	return &storagev1.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: storagev1.VolumeAttachmentSpec{
			NodeName: nodeName,
			Attacher: driverName,
			Source: storagev1.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
		},
	}
}
