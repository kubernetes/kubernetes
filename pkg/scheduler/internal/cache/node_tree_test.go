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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
				v1.LabelZoneRegion: "region-1",
			},
		},
	},
	// Node 2: a node with zone label only
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-2",
			Labels: map[string]string{
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 3: a node with proper region and zone labels
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-3",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-1",
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 4: a node with proper region and zone labels
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-4",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-1",
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 5: a node with proper region and zone labels in a different zone, same region as above
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-5",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-1",
				v1.LabelZoneFailureDomain: "zone-3",
			},
		},
	},
	// Node 6: a node with proper region and zone labels in a new region and zone
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-6",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-2",
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 7: a node with proper region and zone labels in a region and zone as node-6
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-7",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-2",
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 8: a node with proper region and zone labels in a region and zone as node-6
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-8",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-2",
				v1.LabelZoneFailureDomain: "zone-2",
			},
		},
	},
	// Node 9: a node with zone + region label and the deprecated zone + region label
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-9",
			Labels: map[string]string{
				v1.LabelZoneRegionStable:        "region-2",
				v1.LabelZoneFailureDomainStable: "zone-2",
				v1.LabelZoneRegion:              "region-2",
				v1.LabelZoneFailureDomain:       "zone-2",
			},
		},
	},
	// Node 10: a node with only the deprecated zone + region labels
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-10",
			Labels: map[string]string{
				v1.LabelZoneRegion:        "region-2",
				v1.LabelZoneFailureDomain: "zone-3",
			},
		},
	},
}

func verifyNodeTree(t *testing.T, nt *nodeTree, expectedTree map[string][]string) {
	expectedNumNodes := int(0)
	for _, na := range expectedTree {
		expectedNumNodes += len(na)
	}
	if numNodes := nt.numNodes; numNodes != expectedNumNodes {
		t.Errorf("unexpected nodeTree.numNodes. Expected: %v, Got: %v", expectedNumNodes, numNodes)
	}
	if !reflect.DeepEqual(nt.tree, expectedTree) {
		t.Errorf("The node tree is not the same as expected. Expected: %v, Got: %v", expectedTree, nt.tree)
	}
	if len(nt.zones) != len(expectedTree) {
		t.Errorf("Number of zones in nodeTree.zones is not expected. Expected: %v, Got: %v", len(expectedTree), len(nt.zones))
	}
	for _, z := range nt.zones {
		if _, ok := expectedTree[z]; !ok {
			t.Errorf("zone %v is not expected to exist in nodeTree.zones", z)
		}
	}
}

func TestNodeTree_AddNode(t *testing.T) {
	tests := []struct {
		name         string
		nodesToAdd   []*v1.Node
		expectedTree map[string][]string
	}{
		{
			name:         "single node no labels",
			nodesToAdd:   allNodes[:1],
			expectedTree: map[string][]string{"": {"node-0"}},
		},
		{
			name:       "mix of nodes with and without proper labels",
			nodesToAdd: allNodes[:4],
			expectedTree: map[string][]string{
				"":                     {"node-0"},
				"region-1:\x00:":       {"node-1"},
				":\x00:zone-2":         {"node-2"},
				"region-1:\x00:zone-2": {"node-3"},
			},
		},
		{
			name:       "mix of nodes with and without proper labels and some zones with multiple nodes",
			nodesToAdd: allNodes[:7],
			expectedTree: map[string][]string{
				"":                     {"node-0"},
				"region-1:\x00:":       {"node-1"},
				":\x00:zone-2":         {"node-2"},
				"region-1:\x00:zone-2": {"node-3", "node-4"},
				"region-1:\x00:zone-3": {"node-5"},
				"region-2:\x00:zone-2": {"node-6"},
			},
		},
		{
			name:       "nodes also using deprecated zone/region label",
			nodesToAdd: allNodes[9:],
			expectedTree: map[string][]string{
				"region-2:\x00:zone-2": {"node-9"},
				"region-2:\x00:zone-3": {"node-10"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(nil)
			for _, n := range test.nodesToAdd {
				nt.addNode(n)
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
		expectedTree  map[string][]string
		expectError   bool
	}{
		{
			name:          "remove a single node with no labels",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[:1],
			expectedTree: map[string][]string{
				"region-1:\x00:":       {"node-1"},
				":\x00:zone-2":         {"node-2"},
				"region-1:\x00:zone-2": {"node-3", "node-4"},
				"region-1:\x00:zone-3": {"node-5"},
				"region-2:\x00:zone-2": {"node-6"},
			},
		},
		{
			name:          "remove a few nodes including one from a zone with multiple nodes",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[1:4],
			expectedTree: map[string][]string{
				"":                     {"node-0"},
				"region-1:\x00:zone-2": {"node-4"},
				"region-1:\x00:zone-3": {"node-5"},
				"region-2:\x00:zone-2": {"node-6"},
			},
		},
		{
			name:          "remove all nodes",
			existingNodes: allNodes[:7],
			nodesToRemove: allNodes[:7],
			expectedTree:  map[string][]string{},
		},
		{
			name:          "remove non-existing node",
			existingNodes: nil,
			nodesToRemove: allNodes[:5],
			expectedTree:  map[string][]string{},
			expectError:   true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(test.existingNodes)
			for _, n := range test.nodesToRemove {
				err := nt.removeNode(n)
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
		expectedTree  map[string][]string
	}{
		{
			name:          "update a node without label",
			existingNodes: allNodes[:7],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-0",
					Labels: map[string]string{
						v1.LabelZoneRegion:        "region-1",
						v1.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string][]string{
				"region-1:\x00:":       {"node-1"},
				":\x00:zone-2":         {"node-2"},
				"region-1:\x00:zone-2": {"node-3", "node-4", "node-0"},
				"region-1:\x00:zone-3": {"node-5"},
				"region-2:\x00:zone-2": {"node-6"},
			},
		},
		{
			name:          "update the only existing node",
			existingNodes: allNodes[:1],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-0",
					Labels: map[string]string{
						v1.LabelZoneRegion:        "region-1",
						v1.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string][]string{
				"region-1:\x00:zone-2": {"node-0"},
			},
		},
		{
			name:          "update non-existing node",
			existingNodes: allNodes[:1],
			nodeToUpdate: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-new",
					Labels: map[string]string{
						v1.LabelZoneRegion:        "region-1",
						v1.LabelZoneFailureDomain: "zone-2",
					},
				},
			},
			expectedTree: map[string][]string{
				"":                     {"node-0"},
				"region-1:\x00:zone-2": {"node-new"},
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
			nt.updateNode(oldNode, test.nodeToUpdate)
			verifyNodeTree(t, nt, test.expectedTree)
		})
	}
}

func TestNodeTree_List(t *testing.T) {
	tests := []struct {
		name           string
		nodesToAdd     []*v1.Node
		expectedOutput []string
	}{
		{
			name:           "empty tree",
			nodesToAdd:     nil,
			expectedOutput: nil,
		},
		{
			name:           "one node",
			nodesToAdd:     allNodes[:1],
			expectedOutput: []string{"node-0"},
		},
		{
			name:           "four nodes",
			nodesToAdd:     allNodes[:4],
			expectedOutput: []string{"node-0", "node-1", "node-2", "node-3"},
		},
		{
			name:           "all nodes",
			nodesToAdd:     allNodes[:9],
			expectedOutput: []string{"node-0", "node-1", "node-2", "node-3", "node-5", "node-6", "node-4", "node-7", "node-8"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(test.nodesToAdd)

			output, err := nt.list()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(output, test.expectedOutput) {
				t.Errorf("unexpected output. Expected: %v, Got: %v", test.expectedOutput, output)
			}
		})
	}
}

func TestNodeTree_List_Exhausted(t *testing.T) {
	nt := newNodeTree(allNodes[:9])
	nt.numNodes++
	_, err := nt.list()
	if err == nil {
		t.Fatal("Expected an error from zone exhaustion")
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
			name:           "add and remove all nodes",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "add", "remove", "remove", "remove"},
			expectedOutput: nil,
		},
		{
			name:           "add and remove some nodes",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "add", "remove"},
			expectedOutput: []string{"node-3", "node-4"},
		},
		{
			name:           "remove three nodes",
			nodesToAdd:     allNodes[2:9],
			nodesToRemove:  allNodes[2:9],
			operations:     []string{"add", "add", "add", "remove", "remove", "remove", "add"},
			expectedOutput: []string{"node-5"},
		},
		{
			name:           "add more nodes to an exhausted zone",
			nodesToAdd:     append(allNodes[4:9:9], allNodes[3]),
			nodesToRemove:  nil,
			operations:     []string{"add", "add", "add", "add", "add", "add"},
			expectedOutput: []string{"node-4", "node-5", "node-6", "node-3", "node-7", "node-8"},
		},
		{
			name:           "remove zone and add new",
			nodesToAdd:     append(allNodes[3:5:5], allNodes[6:8]...),
			nodesToRemove:  allNodes[3:5],
			operations:     []string{"add", "add", "remove", "add", "add", "remove"},
			expectedOutput: []string{"node-6", "node-7"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nt := newNodeTree(nil)
			addIndex := 0
			removeIndex := 0
			for _, op := range test.operations {
				switch op {
				case "add":
					if addIndex >= len(test.nodesToAdd) {
						t.Error("more add operations than nodesToAdd")
					} else {
						nt.addNode(test.nodesToAdd[addIndex])
						addIndex++
					}
				case "remove":
					if removeIndex >= len(test.nodesToRemove) {
						t.Error("more remove operations than nodesToRemove")
					} else {
						nt.removeNode(test.nodesToRemove[removeIndex])
						removeIndex++
					}
				default:
					t.Errorf("unknow operation: %v", op)
				}
			}
			output, err := nt.list()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(output, test.expectedOutput) {
				t.Errorf("unexpected output. Expected: %v, Got: %v", test.expectedOutput, output)
			}
		})
	}
}
