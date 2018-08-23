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

package cache

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilnode "k8s.io/kubernetes/pkg/util/node"

	"github.com/golang/glog"
)

// NodeTree is a tree-like data structure that holds node names in each zone. Zone names are
// keys to "NodeTree.tree" and values of "NodeTree.tree" are arrays of node names.
type NodeTree struct {
	tree           map[string]*nodeArray // a map from zone (region-zone) to an array of nodes in the zone.
	zones          []string              // a list of all the zones in the tree (keys)
	zoneIndex      int
	exhaustedZones sets.String // set of zones that all of their nodes are returned by next()
	NumNodes       int
	mu             sync.RWMutex
}

// nodeArray is a struct that has nodes that are in a zone.
// We use a slice (as opposed to a set/map) to store the nodes because iterating over the nodes is
// a lot more frequent than searching them by name.
type nodeArray struct {
	nodes     []string
	lastIndex int
}

func (na *nodeArray) next() (nodeName string, exhausted bool) {
	if len(na.nodes) == 0 {
		glog.Error("The nodeArray is empty. It should have been deleted from NodeTree.")
		return "", false
	}
	if na.lastIndex >= len(na.nodes) {
		return "", true
	}
	nodeName = na.nodes[na.lastIndex]
	na.lastIndex++
	return nodeName, false
}

func newNodeTree(nodes []*v1.Node) *NodeTree {
	nt := &NodeTree{
		tree:           make(map[string]*nodeArray),
		exhaustedZones: sets.NewString(),
	}
	for _, n := range nodes {
		nt.AddNode(n)
	}
	return nt
}

// AddNode adds a node and its corresponding zone to the tree. If the zone already exists, the node
// is added to the array of nodes in that zone.
func (nt *NodeTree) AddNode(n *v1.Node) {
	nt.mu.Lock()
	defer nt.mu.Unlock()
	nt.addNode(n)
}

func (nt *NodeTree) addNode(n *v1.Node) {
	zone := utilnode.GetZoneKey(n)
	if na, ok := nt.tree[zone]; ok {
		for _, nodeName := range na.nodes {
			if nodeName == n.Name {
				glog.Warningf("node %v already exist in the NodeTree", n.Name)
				return
			}
		}
		na.nodes = append(na.nodes, n.Name)
	} else {
		nt.zones = append(nt.zones, zone)
		nt.tree[zone] = &nodeArray{nodes: []string{n.Name}, lastIndex: 0}
	}
	glog.V(5).Infof("Added node %v in group %v to NodeTree", n.Name, zone)
	nt.NumNodes++
}

// RemoveNode removes a node from the NodeTree.
func (nt *NodeTree) RemoveNode(n *v1.Node) error {
	nt.mu.Lock()
	defer nt.mu.Unlock()
	return nt.removeNode(n)
}

func (nt *NodeTree) removeNode(n *v1.Node) error {
	zone := utilnode.GetZoneKey(n)
	if na, ok := nt.tree[zone]; ok {
		for i, nodeName := range na.nodes {
			if nodeName == n.Name {
				na.nodes = append(na.nodes[:i], na.nodes[i+1:]...)
				if len(na.nodes) == 0 {
					nt.removeZone(zone)
				}
				glog.V(5).Infof("Removed node %v in group %v from NodeTree", n.Name, zone)
				nt.NumNodes--
				return nil
			}
		}
	}
	glog.Errorf("Node %v in group %v was not found", n.Name, zone)
	return fmt.Errorf("node %v in group %v was not found", n.Name, zone)
}

// removeZone removes a zone from tree.
// This function must be called while writer locks are hold.
func (nt *NodeTree) removeZone(zone string) {
	delete(nt.tree, zone)
	for i, z := range nt.zones {
		if z == zone {
			nt.zones = append(nt.zones[:i], nt.zones[i+1:]...)
		}
	}
}

// UpdateNode updates a node in the NodeTree.
func (nt *NodeTree) UpdateNode(old, new *v1.Node) {
	var oldZone string
	if old != nil {
		oldZone = utilnode.GetZoneKey(old)
	}
	newZone := utilnode.GetZoneKey(new)
	// If the zone ID of the node has not changed, we don't need to do anything. Name of the node
	// cannot be changed in an update.
	if oldZone == newZone {
		return
	}
	nt.mu.Lock()
	defer nt.mu.Unlock()
	nt.removeNode(old) // No error checking. We ignore whether the old node exists or not.
	nt.addNode(new)
}

func (nt *NodeTree) resetExhausted() {
	for _, na := range nt.tree {
		na.lastIndex = 0
	}
	nt.exhaustedZones = sets.NewString()
}

// Next returns the name of the next node. NodeTree iterates over zones and in each zone iterates
// over nodes in a round robin fashion.
func (nt *NodeTree) Next() string {
	nt.mu.Lock()
	defer nt.mu.Unlock()
	if len(nt.zones) == 0 {
		return ""
	}
	for {
		if nt.zoneIndex >= len(nt.zones) {
			nt.zoneIndex = 0
		}
		zone := nt.zones[nt.zoneIndex]
		nt.zoneIndex++
		// We do not check the set of exhausted zones before calling next() on the zone. This ensures
		// that if more nodes are added to a zone after it is exhausted, we iterate over the new nodes.
		nodeName, exhausted := nt.tree[zone].next()
		if exhausted {
			nt.exhaustedZones.Insert(zone)
			if len(nt.exhaustedZones) == len(nt.zones) { // all zones are exhausted. we should reset.
				nt.resetExhausted()
			}
		} else {
			return nodeName
		}
	}
}
