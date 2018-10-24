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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

var allNodes = []*v1.Node{
	// Node 0: a node without any region-zone label
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-0",
		},
	},
	// Node 1: a node with region label only
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion: "region-1",
			},
		},
	},
	// Node 2: a node with zone label only
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-2",
			Labels: map[string]string{
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 3: a node with proper region and zone labels
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-3",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-1",
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 4: a node with proper region and zone labels
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-4",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-1",
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 5: a node with proper region and zone labels in a different zone, same region as above
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-5",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-1",
				kubeletapis.LabelZoneFailureDomain: "zone-3",
			},
		},
	},
	// Node 6: a node with proper region and zone labels in a new region and zone
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-6",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-2",
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 7: a node with proper region and zone labels in a region and zone as node-6
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-7",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-2",
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 8: a node with proper region and zone labels in a region and zone as node-6
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-8",
			Labels: map[string]string{
				kubeletapis.LabelZoneRegion:        "region-2",
				kubeletapis.LabelZoneFailureDomain: "zone-2",
			},
		},
	}}

func verifyNodeTree(t *testing.T, nt *NodeTree, expectedTree map[string]*nodeArray) {
	expectedNumNodes := int(0)
	for _, na := range expectedTree {
		expectedNumNodes += len(na.nodes)
	}
	if nt.NumNodes != expectedNumNodes {
		t.Errorf("unexpected NodeTree.numNodes. Expected: %v, Got: %v", expectedNumNodes, nt.NumNodes)
	}
	if !reflect.DeepEqual(nt.tree, expectedTree) {
		t.Errorf("The node tree is not the same as expected. Expected: %v, Got: %v", expectedTree, nt.tree)
	}
	if len(nt.zones) != len(expectedTree) {
		t.Errorf("Number of zones in NodeTree.zones is not expected. Expected: %v, Got: %v", len(expectedTree), len(nt.zones))
	}
	for _, z := range nt.zones {
		if _, ok := expectedTree[z]; !ok {
			t.Errorf("zone %v is not expected to exist in NodeTree.zones", z)
		}
	}
}

func TestNodeTree_AddNode(t *testing.T) {
	tests := []struct {
		name         string
		nodesToAdd   []*v1.Node
		expectedTree map[string]*nodeArray
	}{
		{
			name:         "single node no labels",
			nodesToAdd:   allNodes[:1],
			expectedTree: map[string]*nodeArray{"": {[]string{"node-0"}, 0}},
		},
		{
			name:       "mix of nodes with and without proper labels",
			nodesToAdd: allNodes[:4],
			expectedTree: map[string]*nodeArray{
				"":                     {[]string{"node-0"}, 0},
				"region-1:\x00:":       {[]string{"node-1"}, 0},
				":\x00:zone-2":         {[]string{"node-2"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-3"}, 0},
			},
		},
		{
			name:       "mix of nodes with and without proper labels and some zones with multiple nodes",
			nodesToAdd: allNodes[:7],
			expectedTree: map[string]*nodeArray{
				"":                     {[]string{"node-0"}, 0},
				"region-1:\x00:":       {[]string{"node-1"}, 0},
				":\x00:zone-2":         {[]string{"node-2"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-3", "node-4"}, 0},
				"region-1:\x00:zone-3": {[]string{"node-5"}, 0},
				"region-2:\x00:zone-2": {[]string{"node-6"}, 0},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(nil)
			for _, n := range test.nodesToAdd {
				nt.AddNode(n)
			}
			verifyNodeTree(t, nt, test.expectedTree)
		})
	}
}

func TestNodeTree_RemoveNode(t *testing.T) {
	tests := []struct {
		name          string
		existingNodes []*v1.Node
		nodesToRemove []*v1.Node
		expectedTree  map[string]*nodeArray
		expectError   bool
	}{
		{
			name:          "remove a single node with no labels",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[:1],
			expectedTree: map[string]*nodeArray{
				"region-1:\x00:":       {[]string{"node-1"}, 0},
				":\x00:zone-2":         {[]string{"node-2"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-3", "node-4"}, 0},
				"region-1:\x00:zone-3": {[]string{"node-5"}, 0},
				"region-2:\x00:zone-2": {[]string{"node-6"}, 0},
			},
		},
		{
			name:          "remove a few nodes including one from a zone with multiple nodes",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[1:4],
			expectedTree: map[string]*nodeArray{
				"":                     {[]string{"node-0"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-4"}, 0},
				"region-1:\x00:zone-3": {[]string{"node-5"}, 0},
				"region-2:\x00:zone-2": {[]string{"node-6"}, 0},
			},
		},
		{
			name:          "remove all nodes",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[:7],
			expectedTree:  map[string]*nodeArray{},
		},
		{
			name:          "remove non-existing node",
			existingNodes: nil,
			nodesToRemove: allNodes[:5],
			expectedTree:  map[string]*nodeArray{},
			expectError:   true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(test.existingNodes)
			for _, n := range test.nodesToRemove {
				err := nt.RemoveNode(n)
				if test.expectError == (err == nil) {
					t.Errorf("unexpected returned error value: %v", err)
				}
			}
			verifyNodeTree(t, nt, test.expectedTree)
		})
	}
}

func TestNodeTree_UpdateNode(t *testing.T) {
	tests := []struct {
		name          string
		existingNodes []*v1.Node
		nodeToUpdate  *v1.Node
		expectedTree  map[string]*nodeArray
	}{
		{
			name:          "update a node without label",
			existingNodes: allNodes[:7],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-0",
					Labels: map[string]string{
						kubeletapis.LabelZoneRegion:        "region-1",
						kubeletapis.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string]*nodeArray{
				"region-1:\x00:":       {[]string{"node-1"}, 0},
				":\x00:zone-2":         {[]string{"node-2"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-3", "node-4", "node-0"}, 0},
				"region-1:\x00:zone-3": {[]string{"node-5"}, 0},
				"region-2:\x00:zone-2": {[]string{"node-6"}, 0},
			},
		},
		{
			name:          "update the only existing node",
			existingNodes: allNodes[:1],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-0",
					Labels: map[string]string{
						kubeletapis.LabelZoneRegion:        "region-1",
						kubeletapis.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string]*nodeArray{
				"region-1:\x00:zone-2": {[]string{"node-0"}, 0},
			},
		},
		{
			name:          "update non-existing node",
			existingNodes: allNodes[:1],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-new",
					Labels: map[string]string{
						kubeletapis.LabelZoneRegion:        "region-1",
						kubeletapis.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string]*nodeArray{
				"":                     {[]string{"node-0"}, 0},
				"region-1:\x00:zone-2": {[]string{"node-new"}, 0},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(test.existingNodes)
			var oldNode *v1.Node
			for _, n := range allNodes {
				if n.Name == test.nodeToUpdate.Name {
					oldNode = n
					break
				}
			}
			if oldNode == nil {
				oldNode = &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "nonexisting-node"}}
			}
			nt.UpdateNode(oldNode, test.nodeToUpdate)
			verifyNodeTree(t, nt, test.expectedTree)
		})
	}
}

func TestNodeTree_Next(t *testing.T) {
	tests := []struct {
		name           string
		nodesToAdd     []*v1.Node
		numRuns        int // number of times to run Next()
		expectedOutput []string
	}{
		{
			name:           "empty tree",
			nodesToAdd:     nil,
			numRuns:        2,
			expectedOutput: []string{"", ""},
		},
		{
			name:           "should go back to the first node after finishing a round",
			nodesToAdd:     allNodes[:1],
			numRuns:        2,
			expectedOutput: []string{"node-0", "node-0"},
		},
		{
			name:           "should go back to the first node after going over all nodes",
			nodesToAdd:     allNodes[:4],
			numRuns:        5,
			expectedOutput: []string{"node-0", "node-1", "node-2", "node-3", "node-0"},
		},
		{
			name:           "should go to all zones before going to the second nodes in the same zone",
			nodesToAdd:     allNodes[:9],
			numRuns:        11,
			expectedOutput: []string{"node-0", "node-1", "node-2", "node-3", "node-5", "node-6", "node-4", "node-7", "node-8", "node-0", "node-1"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(test.nodesToAdd)

			var output []string
			for i := 0; i < test.numRuns; i++ {
				output = append(output, nt.Next())
			}
			if !reflect.DeepEqual(output, test.expectedOutput) {
				t.Errorf("unexpected output. Expected: %v, Got: %v", test.expectedOutput, output)
			}
		})
	}
}

func TestNodeTreeMultiOperations(t *testing.T) {
	tests := []struct {
		name           string
		nodesToAdd     []*v1.Node
		nodesToRemove  []*v1.Node
		operations     []string
		expectedOutput []string
	}{
		{
			name:           "add and remove all nodes between two Next operations",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "next", "add", "remove", "remove", "remove", "next"},
			expectedOutput: []string{"node-2", ""},
		},
		{
			name:           "add and remove some nodes between two Next operations",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "next", "add", "remove", "remove", "next"},
			expectedOutput: []string{"node-2", "node-4"},
		},
		{
			name:           "remove nodes already iterated on and add new nodes",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "next", "next", "add", "remove", "remove", "next"},
			expectedOutput: []string{"node-2", "node-3", "node-4"},
		},
		{
			name:           "add more nodes to an exhausted zone",
			nodesToAdd:     append(allNodes[4:9], allNodes[3]),
			nodesToRemove:  nil,
			operations:     []string{"add", "add", "add", "add", "add", "next", "next", "next", "next", "add", "next", "next", "next"},
			expectedOutput: []string{"node-4", "node-5", "node-6", "node-7", "node-3", "node-8", "node-4"},
		},
		{
			name:           "remove zone and add new to ensure exhausted is reset correctly",
			nodesToAdd:     append(allNodes[3:5], allNodes[6:8]...),
			nodesToRemove:  allNodes[3:5],
			operations:     []string{"add", "add", "next", "next", "remove", "add", "add", "next", "next", "remove", "next", "next"},
			expectedOutput: []string{"node-3", "node-4", "node-6", "node-7", "node-6", "node-7"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(nil)
			addIndex := 0
			removeIndex := 0
			var output []string
			for _, op := range test.operations {
				switch op {
				case "add":
					if addIndex >= len(test.nodesToAdd) {
						t.Error("more add operations than nodesToAdd")
					} else {
						nt.AddNode(test.nodesToAdd[addIndex])
						addIndex++
					}
				case "remove":
					if removeIndex >= len(test.nodesToRemove) {
						t.Error("more remove operations than nodesToRemove")
					} else {
						nt.RemoveNode(test.nodesToRemove[removeIndex])
						removeIndex++
					}
				case "next":
					output = append(output, nt.Next())
				default:
					t.Errorf("unknow operation: %v", op)
				}
			}
			if !reflect.DeepEqual(output, test.expectedOutput) {
				t.Errorf("unexpected output. Expected: %v, Got: %v", test.expectedOutput, output)
			}
		})
	}
}
