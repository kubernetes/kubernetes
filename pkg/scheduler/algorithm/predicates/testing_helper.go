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

package predicates

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
)

// FakePersistentVolumeClaimInfo declares a []v1.PersistentVolumeClaim type for testing.
type FakePersistentVolumeClaimInfo []v1.PersistentVolumeClaim

// GetPersistentVolumeClaimInfo gets PVC matching the namespace and PVC ID.
func (pvcs FakePersistentVolumeClaimInfo) GetPersistentVolumeClaimInfo(namespace string, pvcID string) (*v1.PersistentVolumeClaim, error) {
	for _, pvc := range pvcs {
		if pvc.Name == pvcID && pvc.Namespace == namespace {
			return &pvc, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume claim: %s/%s", namespace, pvcID)
}

// FakeNodeInfo declares a v1.Node type for testing.
type FakeNodeInfo v1.Node

// GetNodeInfo return a fake node info object.
func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	node := v1.Node(n)
	return &node, nil
}

// FakeNodeListInfo declares a []v1.Node type for testing.
type FakeNodeListInfo []v1.Node

// GetNodeInfo returns a fake node object in the fake nodes.
func (nodes FakeNodeListInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	for _, node := range nodes {
		if node.Name == nodeName {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("Unable to find node: %s", nodeName)
}

// FakeCSINodeInfo declares a storagev1beta1.CSINode type for testing.
type FakeCSINodeInfo storagev1beta1.CSINode

// GetCSINodeInfo returns a fake CSINode object.
func (n FakeCSINodeInfo) GetCSINodeInfo(name string) (*storagev1beta1.CSINode, error) {
	csiNode := storagev1beta1.CSINode(n)
	return &csiNode, nil
}

// FakePersistentVolumeInfo declares a []v1.PersistentVolume type for testing.
type FakePersistentVolumeInfo []v1.PersistentVolume

// GetPersistentVolumeInfo returns a fake PV object in the fake PVs by PV ID.
func (pvs FakePersistentVolumeInfo) GetPersistentVolumeInfo(pvID string) (*v1.PersistentVolume, error) {
	for _, pv := range pvs {
		if pv.Name == pvID {
			return &pv, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume: %s", pvID)
}

// FakeStorageClassInfo declares a []storagev1.StorageClass type for testing.
type FakeStorageClassInfo []storagev1.StorageClass

// GetStorageClassInfo returns a fake storage class object in the fake storage classes by name.
func (classes FakeStorageClassInfo) GetStorageClassInfo(name string) (*storagev1.StorageClass, error) {
	for _, sc := range classes {
		if sc.Name == name {
			return &sc, nil
		}
	}
	return nil, fmt.Errorf("Unable to find storage class: %s", name)
}
