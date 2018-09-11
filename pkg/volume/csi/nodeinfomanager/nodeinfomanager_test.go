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

package nodeinfomanager

import (
	"encoding/json"
	"github.com/container-storage-interface/spec/lib/go/csi/v0"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/kubernetes/fake"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csifake "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
	"testing"
)

type testcase struct {
	name                string
	driverName          string
	existingNode        *v1.Node
	existingNodeInfo    *csiv1alpha1.CSINodeInfo
	inputNodeID         string
	inputTopology       *csi.Topology
	expectedNodeIDMap   map[string]string
	expectedTopologyMap map[string]sets.String
	expectedLabels      map[string]string
	expectNoNodeInfo    bool
	expectFail          bool
}

type nodeIDMap map[string]string
type topologyKeyMap map[string][]string
type labelMap map[string]string

// TestAddNodeInfo tests AddNodeInfo with various existing Node and/or CSINodeInfo objects.
// The node IDs in all test cases below are the same between the Node annotation and CSINodeInfo.
func TestAddNodeInfo(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node",
			driverName:   "com.example.csi/driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */),
			inputNodeID:  "com.example.csi/csi-node1",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/zone": "zoneA",
				},
			},
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": sets.NewString("com.example.csi/zone"),
			},
			expectedLabels: map[string]string{"com.example.csi/zone": "zoneA"},
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/zone": "zoneA",
				},
			},
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": sets.NewString("com.example.csi/zone"),
			},
			expectedLabels: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
		},
		{
			name:       "pre-existing node info from the same driver, but without topology info",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil, /* topologyKeys */
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/zone": "zoneA",
				},
			},
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": sets.NewString("com.example.csi/zone"),
			},
			expectedLabels: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				topologyKeyMap{
					"net.example.storage/other-driver": {"net.example.storage/rack"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/zone": "zoneA",
				},
			},
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1":          "com.example.csi/csi-node1",
				"net.example.storage/other-driver": "net.example.storage/test-node",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1":          sets.NewString("com.example.csi/zone"),
				"net.example.storage/other-driver": sets.NewString("net.example.storage/rack"),
			},
			expectedLabels: map[string]string{
				"com.example.csi/zone":     "zoneA",
				"net.example.storage/rack": "rack1",
			},
		},
		{
			name:       "pre-existing node info from the same driver, but different node ID and topology values; labels should conflict",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/zone": "other-zone",
				},
			},
			expectFail: true,
		},
		{
			name:       "pre-existing node info from the same driver, but different node ID and topology keys; new labels should be added",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/other-node",
			inputTopology: &csi.Topology{
				Segments: map[string]string{
					"com.example.csi/rack": "rack1",
				},
			},
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/other-node",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": sets.NewString("com.example.csi/rack"),
			},
			expectedLabels: map[string]string{
				"com.example.csi/zone": "zoneA",
				"com.example.csi/rack": "rack1",
			},
		},
		{
			name:          "nil topology, empty node",
			driverName:    "com.example.csi/driver1",
			existingNode:  generateNode(nil /* nodeIDs */, nil /* labels */),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": nil,
			},
			expectedLabels: nil,
		},
		{
			name:       "nil topology, pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": nil,
			},
			expectedLabels: map[string]string{
				"com.example.csi/zone": "zoneA", // old labels are not removed
			},
		},
		{
			name:       "nil topology, pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				topologyKeyMap{
					"net.example.storage/other-driver": {"net.example.storage/rack"},
				},
			),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1":          "com.example.csi/csi-node1",
				"net.example.storage/other-driver": "net.example.storage/test-node",
			},
			expectedTopologyMap: map[string]sets.String{
				"net.example.storage/other-driver": sets.NewString("net.example.storage/rack"),
				"com.example.csi/driver1":          nil,
			},
			expectedLabels: map[string]string{
				"net.example.storage/rack": "rack1",
			},
		},
		{
			name:         "empty node ID",
			driverName:   "com.example.csi/driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */),
			inputNodeID:  "",
			expectFail:   true,
		},
	}

	test(t, true /* addNodeInfo */, true /* csiNodeInfoEnabled */, testcases)
}

// TestAddNodeInfo_CSINodeInfoDisabled tests AddNodeInfo with various existing Node annotations
// and CSINodeInfo feature gate disabled.
func TestAddNodeInfo_CSINodeInfoDisabled(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node",
			driverName:   "com.example.csi/driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */),
			inputNodeID:  "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */),
			inputNodeID: "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				nil /* labels */),
			inputNodeID: "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1":          "com.example.csi/csi-node1",
				"net.example.storage/other-driver": "net.example.storage/test-node",
			},
		},
	}

	test(t, true /* addNodeInfo */, false /* csiNodeInfoEnabled */, testcases)
}

// TestRemoveNodeInfo tests RemoveNodeInfo with various existing Node and/or CSINodeInfo objects.
func TestRemoveNodeInfo(t *testing.T) {
	testcases := []testcase{
		{
			name:              "empty node and no CSINodeInfo",
			driverName:        "com.example.csi/driver1",
			existingNode:      generateNode(nil /* nodeIDs */, nil /* labels */),
			expectedNodeIDMap: nil,
			expectedLabels:    nil,
			expectNoNodeInfo:  true,
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			expectedNodeIDMap: nil,
			expectedLabels:    map[string]string{"com.example.csi/zone": "zoneA"},
			expectNoNodeInfo:  true,
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				labelMap{
					"net.example.storage/zone": "zoneA",
				}),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				topologyKeyMap{
					"net.example.storage/other-driver": {"net.example.storage/zone"},
				},
			),
			expectedNodeIDMap: map[string]string{
				"net.example.storage/other-driver": "net.example.storage/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"net.example.storage/other-driver": sets.NewString("net.example.storage/zone"),
			},
			expectedLabels: map[string]string{"net.example.storage/zone": "zoneA"},
		},
		{
			name:       "pre-existing info about the same driver in node, but no CSINodeInfo",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */),
			expectedNodeIDMap: nil,
			expectedLabels:    nil,
			expectNoNodeInfo:  true,
		},
		{
			name: "pre-existing info about a different driver in node, but no CSINodeInfo",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				nil /* labels */),
			expectedNodeIDMap: map[string]string{
				"net.example.storage/other-driver": "net.example.storage/csi-node1",
			},
			expectedLabels:   nil,
			expectNoNodeInfo: true,
		},
	}

	test(t, false /* addNodeInfo */, true /* csiNodeInfoEnabled */, testcases)
}

// TestRemoveNodeInfo tests RemoveNodeInfo with various existing Node objects and CSINodeInfo
// feature disabled.
func TestRemoveNodeInfo_CSINodeInfoDisabled(t *testing.T) {
	testcases := []testcase{
		{
			name:              "empty node",
			driverName:        "com.example.csi/driver1",
			existingNode:      generateNode(nil /* nodeIDs */, nil /* labels */),
			expectedNodeIDMap: nil,
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */),
			expectedNodeIDMap: nil,
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				nil /* labels */),
			expectedNodeIDMap: map[string]string{
				"net.example.storage/other-driver": "net.example.storage/csi-node1",
			},
		},
	}

	test(t, false /* addNodeInfo */, false /* csiNodeInfoEnabled */, testcases)
}

func TestAddNodeInfoExistingAnnotation(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, true)()

	driverName := "com.example.csi/driver1"
	nodeID := "com.example.csi/some-node"

	testcases := []struct {
		name         string
		existingNode *v1.Node
	}{
		{
			name: "pre-existing info about the same driver in node, but no CSINodeInfo",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */),
		},
		{
			name: "pre-existing info about a different driver in node, but no CSINodeInfo",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				nil /* labels */),
		},
	}

	for _, tc := range testcases {
		t.Logf("test case: %q", tc.name)

		// Arrange
		nodeName := tc.existingNode.Name
		client := fake.NewSimpleClientset(tc.existingNode)
		csiClient := csifake.NewSimpleClientset()
		nim := NewNodeInfoManager(types.NodeName(nodeName), client, csiClient)

		// Act
		err := nim.AddNodeInfo(driverName, nodeID, 0 /* maxVolumeLimit */, nil) // TODO test maxVolumeLimit
		if err != nil {
			t.Errorf("expected no error from AddNodeInfo call but got: %v", err)
			continue
		}

		// Assert
		nodeInfo, err := csiClient.Csi().CSINodeInfos().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("error getting CSINodeInfo: %v", err)
			continue
		}

		if len(nodeInfo.CSIDrivers) != 1 {
			t.Errorf("expected 1 CSIDriverInfo entry but got: %d", len(nodeInfo.CSIDrivers))
			continue
		}

		driver := nodeInfo.CSIDrivers[0]
		if driver.Driver != driverName || driver.NodeID != nodeID {
			t.Errorf("expected Driver to be %q and NodeID to be %q, but got: %q:%q", driverName, nodeID, driver.Driver, driver.NodeID)
		}
	}
}

func test(t *testing.T, addNodeInfo bool, csiNodeInfoEnabled bool, testcases []testcase) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, csiNodeInfoEnabled)()

	for _, tc := range testcases {
		t.Logf("test case: %q", tc.name)

		//// Arrange
		nodeName := tc.existingNode.Name
		client := fake.NewSimpleClientset(tc.existingNode)
		var csiClient *csifake.Clientset
		if tc.existingNodeInfo == nil {
			csiClient = csifake.NewSimpleClientset()
		} else {
			csiClient = csifake.NewSimpleClientset(tc.existingNodeInfo)
		}
		nim := NewNodeInfoManager(types.NodeName(nodeName), client, csiClient)

		//// Act
		var err error
		if addNodeInfo {
			err = nim.AddNodeInfo(tc.driverName, tc.inputNodeID, 0 /* maxVolumeLimit */, tc.inputTopology) // TODO test maxVolumeLimit
		} else {
			err = nim.RemoveNodeInfo(tc.driverName)
		}

		//// Assert
		if tc.expectFail {
			if err == nil {
				t.Errorf("expected an error from AddNodeInfo call but got none")
			}
			continue
		} else if err != nil {
			t.Errorf("expected no error from AddNodeInfo call but got: %v", err)
			continue
		}

		/* Node Validation */
		node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("error getting node: %v", err)
			continue
		}

		// Node ID annotation
		annNodeID, ok := node.Annotations[annotationKeyNodeID]
		if ok {
			if tc.expectedNodeIDMap == nil {
				t.Errorf("expected annotation %q to not exist, but got: %q", annotationKeyNodeID, annNodeID)
			} else {
				var actualNodeIDs map[string]string
				err = json.Unmarshal([]byte(annNodeID), &actualNodeIDs)
				if err != nil {
					t.Errorf("expected no error when parsing annotation %q, but got error: %v", annotationKeyNodeID, err)
				}

				if !helper.Semantic.DeepEqual(actualNodeIDs, tc.expectedNodeIDMap) {
					t.Errorf("expected annotation %v; got: %v", tc.expectedNodeIDMap, actualNodeIDs)
				}
			}
		} else {
			if tc.expectedNodeIDMap != nil {
				t.Errorf("expected annotation %q, but got none", annotationKeyNodeID)
			}
		}

		if csiNodeInfoEnabled {
			// Topology labels
			if !helper.Semantic.DeepEqual(node.Labels, tc.expectedLabels) {
				t.Errorf("expected topology labels to be %v; got: %v", tc.expectedLabels, node.Labels)
			}

			/* CSINodeInfo validation */
			nodeInfo, err := csiClient.Csi().CSINodeInfos().Get(nodeName, metav1.GetOptions{})
			if tc.expectNoNodeInfo && errors.IsNotFound(err) {
				continue
			} else if err != nil {
				t.Errorf("error getting CSINodeInfo: %v", err)
				continue
			}

			// Extract node IDs and topology keys
			actualNodeIDs := make(map[string]string)
			actualTopologyKeys := make(map[string]sets.String)
			for _, driver := range nodeInfo.CSIDrivers {
				actualNodeIDs[driver.Driver] = driver.NodeID
				actualTopologyKeys[driver.Driver] = sets.NewString(driver.TopologyKeys...)
			}

			// Node IDs
			if !helper.Semantic.DeepEqual(actualNodeIDs, tc.expectedNodeIDMap) {
				t.Errorf("expected node IDs %v from CSINodeInfo; got: %v", tc.expectedNodeIDMap, actualNodeIDs)
			}

			// Topology keys
			if !helper.Semantic.DeepEqual(actualTopologyKeys, tc.expectedTopologyMap) {
				t.Errorf("expected topology keys %v from CSINodeInfo; got: %v", tc.expectedTopologyMap, actualTopologyKeys)
			}
		}
	}
}

func generateNode(nodeIDs, labels map[string]string) *v1.Node {
	var annotations map[string]string
	if len(nodeIDs) > 0 {
		b, _ := json.Marshal(nodeIDs)
		annotations = map[string]string{annotationKeyNodeID: string(b)}
	}
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "node1",
			Annotations: annotations,
			Labels:      labels,
		},
	}
}

func generateNodeInfo(nodeIDs map[string]string, topologyKeys map[string][]string) *csiv1alpha1.CSINodeInfo {
	var drivers []csiv1alpha1.CSIDriverInfo
	for k, nodeID := range nodeIDs {
		d := csiv1alpha1.CSIDriverInfo{
			Driver: k,
			NodeID: nodeID,
		}
		if top, exists := topologyKeys[k]; exists {
			d.TopologyKeys = top
		}
		drivers = append(drivers, d)
	}
	return &csiv1alpha1.CSINodeInfo{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		CSIDrivers: drivers,
	}
}
