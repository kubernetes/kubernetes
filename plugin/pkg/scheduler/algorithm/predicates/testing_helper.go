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

	"k8s.io/api/core/v1"
)

type FakePersistentVolumeClaimInfo []v1.PersistentVolumeClaim

func (pvcs FakePersistentVolumeClaimInfo) GetPersistentVolumeClaimInfo(namespace string, pvcID string) (*v1.PersistentVolumeClaim, error) {
	for _, pvc := range pvcs {
		if pvc.Name == pvcID && pvc.Namespace == namespace {
			return &pvc, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume claim: %s/%s", namespace, pvcID)
}

type FakeNodeInfo v1.Node

func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	node := v1.Node(n)
	return &node, nil
}

type FakeNodeListInfo []v1.Node

func (nodes FakeNodeListInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	for _, node := range nodes {
		if node.Name == nodeName {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("Unable to find node: %s", nodeName)
}

type FakePersistentVolumeInfo []v1.PersistentVolume

func (pvs FakePersistentVolumeInfo) GetPersistentVolumeInfo(pvID string) (*v1.PersistentVolume, error) {
	for _, pv := range pvs {
		if pv.Name == pvID {
			return &pv, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume: %s", pvID)
}
