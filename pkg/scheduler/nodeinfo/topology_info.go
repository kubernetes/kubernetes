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

package nodeinfo

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

// TopologyPair is a key/value pair, and now used to describe labels of a node
type TopologyPair struct {
	Key   string
	Value string
}

// TopologyInfo denotes a mapping from TopologyPair to a string set
type TopologyInfo map[TopologyPair]sets.String

// AddNode updates TopologyInfo when a node is added
func (t TopologyInfo) AddNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.addTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// RemoveNode updates TopologyInfo when a node is removed
func (t TopologyInfo) RemoveNode(node *v1.Node) {
	if node == nil {
		return
	}
	for k, v := range node.Labels {
		t.removeTopologyPair(TopologyPair{k, v}, node.Name)
	}
}

// UpdateNode updates TopologyInfo when a node is updated
func (t TopologyInfo) UpdateNode(oldNode, newNode *v1.Node) {
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

	toRemovePairs, toAddPairs := make([]TopologyPair, 0), make([]TopologyPair, 0)
	for oldKey, oldVal := range oldLabels {
		if newVal, ok := newLabels[oldKey]; ok {
			if oldVal != newVal {
				toAddPairs = append(toAddPairs, TopologyPair{oldKey, newVal})
				toRemovePairs = append(toRemovePairs, TopologyPair{oldKey, oldVal})
			}
		} else {
			toRemovePairs = append(toRemovePairs, TopologyPair{oldKey, oldVal})
		}
	}
	for newKey, newVal := range newLabels {
		if _, ok := oldLabels[newKey]; !ok {
			toAddPairs = append(toAddPairs, TopologyPair{newKey, newVal})
		}
	}

	for _, pair := range toAddPairs {
		t.addTopologyPair(pair, newNode.Name)
	}
	for _, pair := range toRemovePairs {
		t.removeTopologyPair(pair, oldNode.Name)
	}
}

func (t TopologyInfo) addTopologyPair(pair TopologyPair, nodeName string) {
	if _, ok := t[pair]; !ok {
		t[pair] = sets.String{}
	}
	t[pair][nodeName] = sets.Empty{}
}

func (t TopologyInfo) removeTopologyPair(pair TopologyPair, nodeName string) {
	if nodeNameSet, ok := t[pair]; ok {
		delete(nodeNameSet, nodeName)
		if len(nodeNameSet) == 0 {
			delete(t, pair)
		}
	}
}
