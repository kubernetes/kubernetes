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

package nodeinfo

import (
	v1 "k8s.io/api/core/v1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Snapshot is a snapshot of cache NodeInfo and NodeTree order. The scheduler takes a
// snapshot at the beginning of each scheduling cycle and uses it for its operations in that cycle.
type Snapshot struct {
	// NodeInfoMap a map of node name to a snapshot of its NodeInfo.
	NodeInfoMap map[string]*schedulernodeinfo.NodeInfo
	// NodeInfoList is the list of nodes as ordered in the cache's nodeTree.
	NodeInfoList []*schedulernodeinfo.NodeInfo
	Generation   int64
}

// NewSnapshot initializes a Snapshot struct and returns it.
func NewSnapshot() *Snapshot {
	return &Snapshot{
		NodeInfoMap: make(map[string]*schedulernodeinfo.NodeInfo),
	}
}

// ListNodes returns the list of nodes in the snapshot.
func (s *Snapshot) ListNodes() []*v1.Node {
	nodes := make([]*v1.Node, 0, len(s.NodeInfoMap))
	for _, n := range s.NodeInfoList {
		if n.Node() != nil {
			nodes = append(nodes, n.Node())
		}
	}
	return nodes
}
