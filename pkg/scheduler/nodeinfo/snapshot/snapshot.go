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
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Snapshot is a snapshot of cache NodeInfo and NodeTree order. The scheduler takes a
// snapshot at the beginning of each scheduling cycle and uses it for its operations in that cycle.
type Snapshot struct {
	NodeInfoMap  map[string]*nodeinfo.NodeInfo
	SharedLister *listers.SharedLister
	// NodeInfoList is the list of nodes as ordered in the cache's nodeTree.
	NodeInfoList []*nodeinfo.NodeInfo
	Generation   int64
}

// NewSnapshot initializes a Snapshot struct and returns it.
func NewSnapshot() *Snapshot {
	s := &Snapshot{
		NodeInfoMap: make(map[string]*nodeinfo.NodeInfo),
	}
	s.SharedLister = listers.NewSharedLister(&podLister{snapshot: s}, &nodeInfoLister{snapshot: s})

	return s
}

// ListNodes returns the list of nodes in the snapshot.
func (s *Snapshot) ListNodes() []*v1.Node {
	nodes := make([]*v1.Node, 0, len(s.NodeInfoMap))
	for _, n := range s.NodeInfoMap {
		if n.Node() != nil {
			nodes = append(nodes, n.Node())
		}
	}
	return nodes
}

type podLister struct {
	snapshot *Snapshot
}

// List returns the list of pods in the snapshot.
func (p *podLister) List(selector labels.Selector) ([]*v1.Pod, error) {
	alwaysTrue := func(p *v1.Pod) bool { return true }
	return p.FilteredList(alwaysTrue, selector)
}

// FilteredList returns a filtered list of pods in the snapshot.
func (p *podLister) FilteredList(podFilter listers.PodFilter, selector labels.Selector) ([]*v1.Pod, error) {
	// podFilter is expected to return true for most or all of the pods. We
	// can avoid expensive array growth without wasting too much memory by
	// pre-allocating capacity.
	maxSize := 0
	for _, n := range p.snapshot.NodeInfoMap {
		maxSize += len(n.Pods())
	}
	pods := make([]*v1.Pod, 0, maxSize)
	for _, n := range p.snapshot.NodeInfoMap {
		for _, pod := range n.Pods() {
			if podFilter(pod) && selector.Matches(labels.Set(pod.Labels)) {
				pods = append(pods, pod)
			}
		}
	}
	return pods, nil
}

type nodeInfoLister struct {
	snapshot *Snapshot
}

// List returns the list of nodes in the snapshot.
func (n *nodeInfoLister) List() ([]*nodeinfo.NodeInfo, error) {
	return nil, nil
}

// Returns the NodeInfo of the given node name.
func (n *nodeInfoLister) Get(nodeName string) (*nodeinfo.NodeInfo, error) {
	if v, ok := n.snapshot.NodeInfoMap[nodeName]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("nodeinfo not found for node name %q", nodeName)
}
