/*
Copyright 2016 The Kubernetes Authors.

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

package vsphere

import (
	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
)

// Stores info about the kubernetes node
type NodeInfo struct {
	vm       *vclib.VirtualMachine
	vcServer string
}

type NodeManager struct {
	// Maps the VC server to VSphereInstance
	vsphereInstanceMap map[string]*VSphereInstance
	// Maps node name to node info.
	nodeInfoMap map[string]*NodeInfo
}

func (nm *NodeManager) registerNode(node *v1.Node) error {
	return nil
}

func (nm *NodeManager) unregisterNode(node *v1.Node) error {
	return nil
}

func (nm *NodeManager) getNodeInfo(nodeName k8stypes.NodeName) (*NodeInfo, error) {
	return nil, nil
}

func (nm *NodeManager) getVSphereInstance(nodeName k8stypes.NodeName) (*VSphereInstance, error) {
	return nil, nil
}
