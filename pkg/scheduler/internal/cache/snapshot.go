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

package cache

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// Snapshot is a snapshot of cache NodeInfo and NodeTree order. The scheduler takes a
// snapshot at the beginning of each scheduling cycle and uses it for its operations in that cycle.
type Snapshot struct {
	// nodeInfoMap a map of node name to a snapshot of its NodeInfo.
	nodeInfoMap map[string]*framework.NodeInfo
	// nodeInfoList is the list of nodes as ordered in the cache's nodeTree.
	nodeInfoList []*framework.NodeInfo
	// havePodsWithAffinityNodeInfoList is the list of nodes with at least one pod declaring affinity terms.
	havePodsWithAffinityNodeInfoList []*framework.NodeInfo
	generation                       int64
}

var _ framework.SharedLister = &Snapshot{}

// NewEmptySnapshot initializes a Snapshot struct and returns it.
func NewEmptySnapshot() *Snapshot {
	return &Snapshot{
		nodeInfoMap: make(map[string]*framework.NodeInfo),
	}
}

// NewSnapshot initializes a Snapshot struct and returns it.
func NewSnapshot(pods []*v1.Pod, nodes []*v1.Node) *Snapshot {
	nodeInfoMap := createNodeInfoMap(pods, nodes)
	nodeInfoList := make([]*framework.NodeInfo, 0, len(nodeInfoMap))
	havePodsWithAffinityNodeInfoList := make([]*framework.NodeInfo, 0, len(nodeInfoMap))
	for _, v := range nodeInfoMap {
		nodeInfoList = append(nodeInfoList, v)
		if len(v.PodsWithAffinity) > 0 {
			havePodsWithAffinityNodeInfoList = append(havePodsWithAffinityNodeInfoList, v)
		}
	}

	s := NewEmptySnapshot()
	s.nodeInfoMap = nodeInfoMap
	s.nodeInfoList = nodeInfoList
	s.havePodsWithAffinityNodeInfoList = havePodsWithAffinityNodeInfoList

	return s
}

// createNodeInfoMap obtains a list of pods and pivots that list into a map
// where the keys are node names and the values are the aggregated information
// for that node.
func createNodeInfoMap(pods []*v1.Pod, nodes []*v1.Node) map[string]*framework.NodeInfo {
	nodeNameToInfo := make(map[string]*framework.NodeInfo)
	for _, pod := range pods {
		nodeName := pod.Spec.NodeName
		if _, ok := nodeNameToInfo[nodeName]; !ok {
			nodeNameToInfo[nodeName] = framework.NewNodeInfo()
		}
		nodeNameToInfo[nodeName].AddPod(pod)
	}
	imageExistenceMap := createImageExistenceMap(nodes)

	for _, node := range nodes {
		if _, ok := nodeNameToInfo[node.Name]; !ok {
			nodeNameToInfo[node.Name] = framework.NewNodeInfo()
		}
		nodeInfo := nodeNameToInfo[node.Name]
		nodeInfo.SetNode(node)
		nodeInfo.ImageStates = getNodeImageStates(node, imageExistenceMap)
	}
	return nodeNameToInfo
}

// getNodeImageStates returns the given node's image states based on the given imageExistence map.
func getNodeImageStates(node *v1.Node, imageExistenceMap map[string]sets.String) map[string]*framework.ImageStateSummary {
	imageStates := make(map[string]*framework.ImageStateSummary)

	for _, image := range node.Status.Images {
		for _, name := range image.Names {
			imageStates[name] = &framework.ImageStateSummary{
				Size:     image.SizeBytes,
				NumNodes: len(imageExistenceMap[name]),
			}
		}
	}
	return imageStates
}

// createImageExistenceMap returns a map recording on which nodes the images exist, keyed by the images' names.
func createImageExistenceMap(nodes []*v1.Node) map[string]sets.String {
	imageExistenceMap := make(map[string]sets.String)
	for _, node := range nodes {
		for _, image := range node.Status.Images {
			for _, name := range image.Names {
				if _, ok := imageExistenceMap[name]; !ok {
					imageExistenceMap[name] = sets.NewString(node.Name)
				} else {
					imageExistenceMap[name].Insert(node.Name)
				}
			}
		}
	}
	return imageExistenceMap
}

// NodeInfos returns a NodeInfoLister.
func (s *Snapshot) NodeInfos() framework.NodeInfoLister {
	return s
}

// NumNodes returns the number of nodes in the snapshot.
func (s *Snapshot) NumNodes() int {
	return len(s.nodeInfoList)
}

// List returns the list of nodes in the snapshot.
func (s *Snapshot) List() ([]*framework.NodeInfo, error) {
	return s.nodeInfoList, nil
}

// HavePodsWithAffinityList returns the list of nodes with at least one pods with inter-pod affinity
func (s *Snapshot) HavePodsWithAffinityList() ([]*framework.NodeInfo, error) {
	return s.havePodsWithAffinityNodeInfoList, nil
}

// Get returns the NodeInfo of the given node name.
func (s *Snapshot) Get(nodeName string) (*framework.NodeInfo, error) {
	if v, ok := s.nodeInfoMap[nodeName]; ok && v.Node() != nil {
		return v, nil
	}
	return nil, fmt.Errorf("nodeinfo not found for node name %q", nodeName)
}
