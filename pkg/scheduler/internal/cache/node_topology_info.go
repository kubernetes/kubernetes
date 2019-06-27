package cache

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

// TopologyPair is a key/value pair, used to describe labels of a node
type TopologyPair struct {
	Key   string
	Value string
}

// NodeTopologyInfo denotes a mapping from TopologyPair to a set of nodes.
type NodeTopologyInfo map[TopologyPair]sets.String

// AddNode updates NodeTopologyInfo when a node is added
func (t NodeTopologyInfo) AddNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.addTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// RemoveNode updates NodeTopologyInfo when a node is removed
func (t NodeTopologyInfo) RemoveNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.removeTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// UpdateNode updates NodeTopologyInfo when a node is updated
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

	for oldKey, oldVal := range oldLabels {
		if newVal, ok := newLabels[oldKey]; ok {
			if oldVal != newVal {
				t.addTopologyPair(TopologyPair{oldKey, newVal}, newNode.Name)
				t.removeTopologyPair(TopologyPair{oldKey, oldVal}, oldNode.Name)
			}
		} else {
			t.removeTopologyPair(TopologyPair{oldKey, oldVal}, oldNode.Name)
		}
	}
	for newKey, newVal := range newLabels {
		if _, ok := oldLabels[newKey]; !ok {
			t.addTopologyPair(TopologyPair{newKey, newVal}, newNode.Name)
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
