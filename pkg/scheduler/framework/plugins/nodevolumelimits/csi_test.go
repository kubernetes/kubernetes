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
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	fakeframework "k8s.io/kubernetes/pkg/scheduler/framework/fake"
	utilpointer "k8s.io/utils/pointer"
)

const (
	ebsCSIDriverName = csilibplugins.AWSEBSDriverName
	gceCSIDriverName = csilibplugins.GCEPDDriverName

	hostpathInTreePluginName = "kubernetes.io/hostpath"
)

var csiNode = &storagev1.CSINode{
	ObjectMeta: metav1.ObjectMeta{Name: "csi-node-for-max-pd-test-1"},
	Spec: storagev1.CSINodeSpec{
		Drivers: []storagev1.CSINodeDriver{},
	},
}
var nodeName = "node-name"

func TestCSILimits(t *testing.T) {
	noVolPod := &v1.Pod{
		Spec: v1.PodSpec{
		},
	}
	// pods with matching csi driver names
	csiEBSOneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-0",
						},
					},
				},
			},
		},
	}
	csiEBSTwoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-2",
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
		nodeName     string
		csiNode      *storagev1.CSINode
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{noVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			nodeName:     nodeName,
			csiNode:      csiNode,
			test:         "pod does not have any volume",
			wantStatus:   nil,
		},
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      3,
			driverNames:  []string{ebsCSIDriverName},
			nodeName:     fakeframework.NodeNameWithoutCSINode,
			csiNode:      nil,
			test:         "cannot get CSINode object",
			wantStatus:   framework.NewStatus(framework.Error, "Could not get a CSINode object for the node: csinode \"node-without-csi\" not found"),
		},
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      3,
			driverNames:  []string{ebsCSIDriverName},
			nodeName:     nodeName,
			csiNode:      csiNode,
			test:         "node has available csi volumes",
			wantStatus:   nil,
		},
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			nodeName:     nodeName,
			csiNode:      csiNode,
			test:         "node exceeds max csi volume count",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
	}

	// running attachable predicate tests with limit present on nodes
	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits(test.existingPods, int64(test.maxVols), test.nodeName, test.csiNode, test.driverNames...)

			p := &CSILimits{
				csiNodeLister:        getFakeCSINodeLister(csiNode),
				pvLister:             getFakeCSIPVLister(test.filterName, test.driverNames...),
				pvcLister:            getFakeCSIPVCLister(test.filterName, "csi-sc", test.driverNames...),
				scLister:             getFakeCSIStorageClassLister("csi-sc", test.driverNames[0]),
				randomVolumeIDPrefix: rand.String(32),
			}
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func getFakeCSIPVLister(volumeName string, driverNames ...string) fakeframework.PersistentVolumeLister {
	pvLister := fakeframework.PersistentVolumeLister{}
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

			pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       driver,
					VolumeHandle: volumeHandle,
				},
			}
			pvLister = append(pvLister, pv)
		}
	}
	return pvLister
}

func getFakeCSIPVCLister(volumeName, scName string, driverNames ...string) fakeframework.PersistentVolumeClaimLister {
	pvcLister := fakeframework.PersistentVolumeClaimLister{}
	for _, driver := range driverNames {
		for j := 0; j < 4; j++ {
			v := fmt.Sprintf("%s-%s-%d", volumeName, driver, j)
			pvc := v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: v},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: v},
			}
			pvcLister = append(pvcLister, pvc)
		}
	}

	// a pvc with missing PV but available storageclass.
	pvcLister = append(pvcLister, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-6"},
		Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &scName, VolumeName: "missing-in-action"},
	})
	return pvcLister
}

func getFakeCSIStorageClassLister(scName, provisionerName string) fakeframework.StorageClassLister {
	return fakeframework.StorageClassLister{
		{
			ObjectMeta:  metav1.ObjectMeta{Name: scName},
			Provisioner: provisionerName,
		},
	}
}

func getFakeCSINodeLister(csiNode *storagev1.CSINode) fakeframework.CSINodeLister {
	if csiNode != nil {
		return fakeframework.CSINodeLister(*csiNode)
	}
	return fakeframework.CSINodeLister{}
}

func getNodeWithPodAndVolumeLimits(pods []*v1.Pod, limit int64, nodeName string, csiNode *storagev1.CSINode, driverNames ...string) (*framework.NodeInfo, *storagev1.CSINode) {
	nodeInfo := framework.NewNodeInfo(pods...)
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{},
		},
	}
	if csiNode != nil {
		for _, driver := range driverNames {
			driver := storagev1.CSINodeDriver{
				Name:   driver,
				NodeID: nodeName,
			}
			driver.Allocatable = &storagev1.VolumeNodeResources{
				Count: utilpointer.Int32Ptr(int32(limit)),
			}
			csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, driver)
		}
	}

	_ = nodeInfo.SetNode(node)
	return nodeInfo, csiNode
}
