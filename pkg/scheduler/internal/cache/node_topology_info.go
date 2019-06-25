package cache

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// TopologyPair is a struct that holds node label
type TopologyPair struct {
	Key   string
	Value string
}

// NodeTopologyInfo is a struct that used to  map TopologyPair node name set.
type NodeTopologyInfo map[TopologyPair]sets.String

// AddNode adds a node and its corresponding labels to the NodeTopologyInfo when a node is added
func (t NodeTopologyInfo) AddNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.addTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// RemoveNode removes a node and corresponding labels to the NodeTopologyInfo when a node is removed
func (t NodeTopologyInfo) RemoveNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.removeTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// UpdateNode updates a node and corresponding labels to the  NodeTopologyInfo when a node is updated
func (t NodeTopologyInfo) UpdateNode(oldNode, newNode *v1.Node) {
	if oldNode == nil && newNode == nil {
		return
	}
	if oldNode == nil {
		t.AddNode(newNode)
		return
	}
	if newNode == nil {
		t.RemoveNode(oldNode)
		return
	}
	oldLabels, newLabels := oldNode.Labels, newNode.Labels
	if helper.Semantic.DeepEqual(oldLabels, newLabels) {
		return
	}

	// labels in new node but not in old node, just add new node label
	for newKey, newVal := range newLabels {
		if _, ok := oldLabels[newKey]; !ok {
			t.addTopologyPair(TopologyPair{newKey, newVal}, newNode.Name)
		}
	}


	for oldKey, oldVal := range oldLabels {
		if newVal, ok := newLabels[oldKey]; ok {
			// labels in old and new node, but value different, then remove old and add new
			if oldVal != newVal {
				t.removeTopologyPair(TopologyPair{oldKey, oldVal}, oldNode.Name)
				t.addTopologyPair(TopologyPair{oldKey, newVal}, newNode.Name)
			}
		} else {
			// labels in old node but not in new old, just remove old
			t.removeTopologyPair(TopologyPair{oldKey, oldVal}, oldNode.Name)
		}
	}
}

func (t NodeTopologyInfo) addTopologyPair(pair TopologyPair, nodeName string) {
	if _, ok := t[pair]; !ok {
		t[pair] = sets.String{}
	}
	t[pair][nodeName] = sets.Empty{}
}

func (t NodeTopologyInfo) removeTopologyPair(pair TopologyPair, nodeName string) {
	if nodeNameSet, ok := t[pair]; ok {
		delete(nodeNameSet, nodeName)
		if len(nodeNameSet) == 0 {
			delete(t, pair)
		}
	}
}

func CreateNodeTopologyInfo(nodeInfoMap map[string]*schedulernodeinfo.NodeInfo) NodeTopologyInfo {
	if nodeInfoMap == nil {
		return nil
	}
	nodeTopologyInfo := make(NodeTopologyInfo)
	for name, info := range nodeInfoMap {
		for k, v := range info.Node().Labels {
			nodeTopologyInfo.addTopologyPair(TopologyPair{
				Key: k,
				Value: v,
			}, name)
		}
	}
	return nodeTopologyInfo
}