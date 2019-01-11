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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	utiltesting "k8s.io/client-go/util/testing"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csifake "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

type testcase struct {
	name                string
	driverName          string
	existingNode        *v1.Node
	existingNodeInfo    *csiv1alpha1.CSINodeInfo
	inputNodeID         string
	inputTopology       map[string]string
	inputVolumeLimit    int64
	expectedNodeIDMap   map[string]string
	expectedTopologyMap map[string]sets.String
	expectedLabels      map[string]string
	expectedVolumeLimit int64
	expectFail          bool
}

type nodeIDMap map[string]string
type topologyKeyMap map[string][]string
type labelMap map[string]string

// TestInstallCSIDriver tests InstallCSIDriver with various existing Node and/or CSINodeInfo objects.
// The node IDs in all test cases below are the same between the Node annotation and CSINodeInfo.
func TestInstallCSIDriver(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node",
			driverName:   "com.example.csi/driver1",
			existingNode: generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:  "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				},
				nil /*capacity*/),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil, /* topologyKeys */
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				topologyKeyMap{
					"net.example.storage/other-driver": {"net.example.storage/rack"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "other-zone",
			},
			expectFail: true,
		},
		{
			name:       "pre-existing node info from the same driver, but different node ID and topology keys; new labels should be added",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingNodeInfo: generateNodeInfo(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				topologyKeyMap{
					"com.example.csi/driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/other-node",
			inputTopology: map[string]string{
				"com.example.csi/rack": "rack1",
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
			existingNode:  generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
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
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
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
			existingNode: generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:  "",
			expectFail:   true,
		},
		{
			name:                "new node with valid max limit",
			driverName:          "com.example.csi/driver1",
			existingNode:        generateNode("node1", nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/),
			inputVolumeLimit:    10,
			inputTopology:       nil,
			inputNodeID:         "com.example.csi/csi-node1",
			expectedVolumeLimit: 10,
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": nil,
			},
			expectedLabels: nil,
		},
		{
			name:       "node with existing valid max limit",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nil, /*nodeIDs*/
				nil, /*labels*/
				map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: *resource.NewScaledQuantity(4, -3),
					v1.ResourceName(util.GetCSIAttachLimitKey("com.example.csi/driver1")): *resource.NewQuantity(10, resource.DecimalSI),
				}),
			inputVolumeLimit:    20,
			inputTopology:       nil,
			inputNodeID:         "com.example.csi/csi-node1",
			expectedVolumeLimit: 20,
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
			expectedTopologyMap: map[string]sets.String{
				"com.example.csi/driver1": nil,
			},
			expectedLabels: nil,
		},
	}

	test(t, true /* addNodeInfo */, true /* csiNodeInfoEnabled */, testcases)
}

// TestInstallCSIDriver_CSINodeInfoDisabled tests InstallCSIDriver with various existing Node annotations
// and CSINodeInfo feature gate disabled.
func TestInstallCSIDriver_CSINodeInfoDisabled(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node",
			driverName:   "com.example.csi/driver1",
			existingNode: generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:  "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			inputNodeID: "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1": "com.example.csi/csi-node1",
			},
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				nil /* labels */, nil /*capacity*/),
			inputNodeID: "com.example.csi/csi-node1",
			expectedNodeIDMap: map[string]string{
				"com.example.csi/driver1":          "com.example.csi/csi-node1",
				"net.example.storage/other-driver": "net.example.storage/test-node",
			},
		},
	}

	test(t, true /* addNodeInfo */, false /* csiNodeInfoEnabled */, testcases)
}

// TestUninstallCSIDriver tests UninstallCSIDriver with various existing Node and/or CSINodeInfo objects.
func TestUninstallCSIDriver(t *testing.T) {
	testcases := []testcase{
		{
			name:              "empty node and empty CSINodeInfo",
			driverName:        "com.example.csi/driver1",
			existingNode:      generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: nil,
			expectedLabels:    nil,
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
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
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				labelMap{
					"net.example.storage/zone": "zoneA",
				}, nil /*capacity*/),
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
			name:       "pre-existing info about the same driver in node, but empty CSINodeInfo",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: nil,
			expectedLabels:    nil,
		},
		{
			name: "pre-existing info about a different driver in node, but empty CSINodeInfo",
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: map[string]string{
				"net.example.storage/other-driver": "net.example.storage/csi-node1",
			},
			expectedLabels: nil,
		},
		{
			name:       "new node with valid max limit",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nil, /*nodeIDs*/
				nil, /*labels*/
				map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: *resource.NewScaledQuantity(4, -3),
					v1.ResourceName(util.GetCSIAttachLimitKey("com.example.csi/driver1")): *resource.NewQuantity(10, resource.DecimalSI),
				},
			),
			inputTopology:       nil,
			inputNodeID:         "com.example.csi/csi-node1",
			expectedVolumeLimit: 0,
		},
	}

	test(t, false /* addNodeInfo */, true /* csiNodeInfoEnabled */, testcases)
}

// TestUninstallCSIDriver tests UninstallCSIDriver with various existing Node objects and CSINodeInfo
// feature disabled.
func TestUninstallCSIDriver_CSINodeInfoDisabled(t *testing.T) {
	testcases := []testcase{
		{
			name:              "empty node",
			driverName:        "com.example.csi/driver1",
			existingNode:      generateNode("node1", nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: nil,
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: nil,
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi/driver1",
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNodeIDMap: map[string]string{
				"net.example.storage/other-driver": "net.example.storage/csi-node1",
			},
		},
	}

	test(t, false /* addNodeInfo */, false /* csiNodeInfoEnabled */, testcases)
}

func TestInstallCSIDriverExistingAnnotation(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, true)()

	driverName := "com.example.csi/driver1"
	nodeID := "com.example.csi/some-node"

	testcases := []struct {
		name         string
		existingNode *v1.Node
	}{
		{
			name: "pre-existing info about the same driver in node, but empty CSINodeInfo",
			existingNode: generateNode("node1",
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
		},
		{
			name: "pre-existing info about a different driver in node, but empty CSINodeInfo",
			existingNode: generateNode("node1",
				nodeIDMap{
					"net.example.storage/other-driver": "net.example.storage/test-node",
				},
				nil /* labels */, nil /*capacity*/),
		},
	}

	for _, tc := range testcases {
		t.Logf("test case: %q", tc.name)

		// Arrange
		nodeName := tc.existingNode.Name
		client := fake.NewSimpleClientset(tc.existingNode)
		csiClient := csifake.NewSimpleClientset()

		tmpDir, err := utiltesting.MkTmpdir("nodeinfomanager-test")
		if err != nil {
			t.Fatalf("can't create temp dir: %v", err)
		}
		host := volumetest.NewFakeVolumeHostWithCSINodeName(
			tmpDir,
			client,
			csiClient,
			nil,
			nodeName,
		)

		nim := NewNodeInfoManager(types.NodeName(nodeName), host, map[string](func() bool){})

		// Act
		_, err = nim.CreateCSINodeInfo()
		if err != nil {
			t.Errorf("expected no error from creating CSINodeinfo but got: %v", err)
			continue
		}
		err = nim.InstallCSIDriver(driverName, nodeID, 0 /* maxVolumeLimit */, nil) // TODO test maxVolumeLimit
		if err != nil {
			t.Errorf("expected no error from InstallCSIDriver call but got: %v", err)
			continue
		}

		// Assert
		nodeInfo, err := csiClient.Csi().CSINodeInfos().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("error getting CSINodeInfo: %v", err)
			continue
		}

		if len(nodeInfo.Spec.Drivers) != 1 || len(nodeInfo.Status.Drivers) != 1 {
			t.Errorf("expected 1 CSIDriverInfoSpec and 1 CSIDriverInfoStatus entry but got: %d, %d",
				len(nodeInfo.Spec.Drivers), len(nodeInfo.Status.Drivers))
			continue
		}

		driver := nodeInfo.Spec.Drivers[0]
		if driver.Name != driverName || driver.NodeID != nodeID {
			t.Errorf("expected Driver to be %q and NodeID to be %q, but got: %q:%q", driverName, nodeID, driver.Name, driver.NodeID)
		}
	}
}

func TestValidateCSINodeInfo(t *testing.T) {
	testcases := []struct {
		name      string
		nodeInfo  *csiv1alpha1.CSINodeInfo
		expectErr bool
	}{
		{
			name: "multiple drivers with different node IDs, topology keys and status",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1, key2"},
						},
						{
							Name:         "driverB",
							NodeID:       "nodeA",
							TopologyKeys: []string{"keyA", "keyB"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "in-tree",
						},
						{
							Name:                  "driverB",
							Available:             false,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: false,
		},
		{
			name: "multiple drivers with same node IDs, topology keys and status",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
						{
							Name:         "driver2",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
						{
							Name:                  "driver2",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: false,
		},
		{
			name: "duplicate drivers in driver specs",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1", "key2"},
						},
						{
							Name:         "driver1",
							NodeID:       "nodeX",
							TopologyKeys: []string{"keyA", "keyB"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "duplicate drivers in driver statuses",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1", "key2"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "in-tree",
						},
						{
							Name:                  "driver1",
							Available:             false,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "single driver with duplicate topology keys in driver specs",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1", "key1"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "multiple drivers with one set of duplicate topology keys in driver specs",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
						{
							Name:         "driver2",
							NodeID:       "nodeX",
							TopologyKeys: []string{"keyA", "keyA"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
						{
							Name:                  "driver2",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "mismatch between drivers in specs and status (null intersection)",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
						{
							Name:         "driver2",
							NodeID:       "nodeX",
							TopologyKeys: []string{"keyA", "keyA"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver3",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "mismatch between drivers in specs and status (specs superset of status)",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
						{
							Name:         "driver2",
							NodeID:       "nodeX",
							TopologyKeys: []string{"keyA", "keyA"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
		{
			name: "mismatch between drivers in specs and status (specs subset of status)",
			nodeInfo: &csiv1alpha1.CSINodeInfo{
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "driver1",
							NodeID:       "node1",
							TopologyKeys: []string{"key1"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "driver1",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
						{
							Name:                  "driver2",
							Available:             true,
							VolumePluginMechanism: "csi",
						},
					},
				},
			},
			expectErr: true,
		},
	}
	for _, tc := range testcases {
		t.Logf("test case: %q", tc.name)
		err := validateCSINodeInfo(tc.nodeInfo)
		if err != nil && !tc.expectErr {
			t.Errorf("expected no errors from validateCSINodeInfo but got error %v", err)
		}
		if err == nil && tc.expectErr {
			t.Errorf("expected error from validateCSINodeInfo but got no errors")
		}
	}
}

func TestInitializeMigratedDrivers(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, true)()
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()

	nodeName := "test-node"

	testcases := []struct {
		name             string
		existingNodeInfo *csiv1alpha1.CSINodeInfo
		migratedDrivers  map[string](func() bool)
		expectedNodeInfo *csiv1alpha1.CSINodeInfo
	}{
		{
			name: "existing driver with existing fields and in-tree plugin mechanism",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing-driver",
					nodeID:                "existing-nodeid",
					topologyKeys:          []string{"existing-topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
			migratedDrivers: map[string](func() bool){
				"existing-driver": func() bool { return true },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing-driver",
					nodeID:                "existing-nodeid",
					topologyKeys:          []string{"existing-topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
		},
		{
			name:             "bunch of migrated drivers & no existing drivers",
			existingNodeInfo: nil,
			migratedDrivers: map[string](func() bool){
				"new-driver":    func() bool { return true },
				"second-driver": func() bool { return false },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "new-driver",
					nodeID:                "",
					topologyKeys:          []string{},
					available:             false,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
				{
					name:                  "second-driver",
					nodeID:                "",
					topologyKeys:          []string{},
					available:             false,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
		},
		{
			name: "Existing drivers that has the same volumePluginMechanism value",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "in-tree",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
				{
					name:                  "csi",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
			migratedDrivers: map[string](func() bool){
				"in-tree": func() bool { return false },
				"csi":     func() bool { return true },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "in-tree",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
				{
					name:                  "csi",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
		},
		{
			name: "Existing drivers that have opposite volumePluginMechanism value",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "in-tree->csi",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
				{
					name:                  "csi->in-tree",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
			migratedDrivers: map[string](func() bool){
				"in-tree->csi": func() bool { return true },
				"csi->in-tree": func() bool { return false },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "in-tree->csi",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
				{
					name:                  "csi->in-tree",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
		},
		{
			name: "Combination some existing drivers some non existing",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
			migratedDrivers: map[string](func() bool){
				"existing":  func() bool { return true },
				"newdriver": func() bool { return true },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
				{
					name:                  "newdriver",
					nodeID:                "",
					topologyKeys:          []string{},
					available:             false,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
		},
		{
			name: "Existing drivers not affected if not in migrated drivers map",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
			migratedDrivers: map[string](func() bool){},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             true,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
		},
		{
			name: "Existing drivers not available continue to be so",
			existingNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             false,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}),
			migratedDrivers: map[string](func() bool){
				"existing": func() bool { return true },
			},
			expectedNodeInfo: generateNodeInfoAdvanced(nodeName, []unifiedDriver{
				{
					name:                  "existing",
					nodeID:                "nodeid",
					topologyKeys:          []string{"topkey"},
					available:             false,
					volumePluginMechanism: csiv1alpha1.VolumePluginMechanismCSI,
				},
			}),
		},
	}

	for _, tc := range testcases {
		t.Logf("Running test case %s", tc.name)
		//// Arrange
		existingNode := generateNode(nodeName, nil, nil, nil)
		client := fake.NewSimpleClientset(existingNode)
		var csiClient *csifake.Clientset
		if tc.existingNodeInfo == nil {
			csiClient = csifake.NewSimpleClientset()
		} else {
			csiClient = csifake.NewSimpleClientset(tc.existingNodeInfo)
		}

		tmpDir, err := utiltesting.MkTmpdir("nodeinfomanager-test")
		if err != nil {
			t.Fatalf("can't create temp dir: %v", err)
		}
		host := volumetest.NewFakeVolumeHostWithCSINodeName(
			tmpDir,
			client,
			csiClient,
			nil,
			nodeName,
		)
		nim := NewNodeInfoManager(types.NodeName(nodeName), host, tc.migratedDrivers)

		//// Act
		if tc.existingNodeInfo == nil {
			// Only create new CSINodeInfo if there isn't already one
			_, err = nim.CreateCSINodeInfo()
			if err != nil {
				t.Fatalf("Failed to create CSINodeInfo: %v", err)
			}
		}
		err = nim.InitializeMigratedDrivers()
		if err != nil {
			t.Fatalf("Failed to initialize migrated drivers: %v", err)
		}

		// Check
		nodeInfo, err := csiClient.Csi().CSINodeInfos().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("Error while getting node info from csi client: %v", err)
		}
		checkNodeInfoEquivilant(t, nodeInfo, tc.expectedNodeInfo)
	}

}

type unifiedDriver struct {
	name                  string
	nodeID                string
	topologyKeys          []string
	available             bool
	volumePluginMechanism csiv1alpha1.VolumePluginMechanism
}

func checkNodeInfoEquivilant(t *testing.T, gotNodeInfo *csiv1alpha1.CSINodeInfo, expectedNodeInfo *csiv1alpha1.CSINodeInfo) {
	if gotNodeInfo == nil {
		t.Fatalf("gotNodeInfo was nil")
	}
	if expectedNodeInfo == nil {
		t.Fatalf("expectedNodeInfo was nil")
	}

	expectedDrivers := map[string]unifiedDriver{}
	// Populate expected drivers
	for _, dspec := range expectedNodeInfo.Spec.Drivers {
		driver := expectedDrivers[dspec.Name]
		driver.name = dspec.Name
		driver.nodeID = dspec.NodeID
		driver.topologyKeys = dspec.TopologyKeys
		expectedDrivers[dspec.Name] = driver
	}
	for _, dstatus := range expectedNodeInfo.Status.Drivers {
		driver := expectedDrivers[dstatus.Name]
		driver.available = dstatus.Available
		driver.volumePluginMechanism = dstatus.VolumePluginMechanism
		expectedDrivers[dstatus.Name] = driver
	}

	gotSet := sets.String{}
	// Check if got drivers are the same and checks that we didn't get one we didn't expect
	for _, dspec := range gotNodeInfo.Spec.Drivers {
		if driver, ok := expectedDrivers[dspec.Name]; ok {
			if driver.nodeID != dspec.NodeID {
				t.Fatalf("for driver %s got nodeID %s expected %s", dspec.Name, dspec.NodeID, driver.nodeID)
			}
			if !helper.Semantic.DeepEqual(dspec.TopologyKeys, driver.topologyKeys) {
				t.Fatalf("for driver %s expected topology labels to be %v; got: %v", dspec.Name, dspec.TopologyKeys, driver.topologyKeys)
			}
		} else {
			t.Fatalf("got driver %s that was not expected", dspec.Name)
		}
		gotSet.Insert(dspec.Name)
	}

	for _, dstatus := range gotNodeInfo.Status.Drivers {
		if driver, ok := expectedDrivers[dstatus.Name]; ok {
			if dstatus.Available != driver.available {
				t.Fatalf("for driver %s got available %v expected %v", dstatus.Name, dstatus.Available, driver.available)
			}
			if dstatus.VolumePluginMechanism != driver.volumePluginMechanism {
				t.Fatalf("for driver %s got volumePluginMechanism %v expected %v", dstatus.Name, dstatus.VolumePluginMechanism, driver.volumePluginMechanism)
			}
		} else {
			t.Fatalf("got driver %s that was not expected", dstatus.Name)
		}
	}

	if len(expectedDrivers) != gotSet.Len() {
		expectedSet := sets.String{}
		for k := range expectedDrivers {
			expectedSet.Insert(k)
		}
		t.Fatalf("did not get all drivers that we expected, expected %v, got %v", expectedSet, gotSet)
	}
}

func test(t *testing.T, addNodeInfo bool, csiNodeInfoEnabled bool, testcases []testcase) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, csiNodeInfoEnabled)()
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()

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

		tmpDir, err := utiltesting.MkTmpdir("nodeinfomanager-test")
		if err != nil {
			t.Fatalf("can't create temp dir: %v", err)
		}
		host := volumetest.NewFakeVolumeHostWithCSINodeName(
			tmpDir,
			client,
			csiClient,
			nil,
			nodeName,
		)
		nim := NewNodeInfoManager(types.NodeName(nodeName), host, map[string](func() bool){})

		//// Act
		nim.CreateCSINodeInfo()
		if addNodeInfo {
			err = nim.InstallCSIDriver(tc.driverName, tc.inputNodeID, tc.inputVolumeLimit, tc.inputTopology)
		} else {
			err = nim.UninstallCSIDriver(tc.driverName)
		}

		//// Assert
		if tc.expectFail {
			if err == nil {
				t.Errorf("expected an error from InstallCSIDriver call but got none")
			}
			continue
		} else if err != nil {
			t.Errorf("expected no error from InstallCSIDriver call but got: %v", err)
			continue
		}

		actions := client.Actions()

		var node *v1.Node
		if action := hasPatchAction(actions); action != nil {
			node, err = applyNodeStatusPatch(tc.existingNode, action.(clienttesting.PatchActionImpl).GetPatch())
			assert.NoError(t, err)
		} else {
			node, err = client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
			assert.NoError(t, err)
		}

		if node == nil {
			t.Errorf("error getting node: %v", err)
			continue
		}

		// We are testing max volume limits
		attachLimit := getVolumeLimit(node, tc.driverName)
		if attachLimit != tc.expectedVolumeLimit {
			t.Errorf("expected volume limit to be %d got %d", tc.expectedVolumeLimit, attachLimit)
			continue
		}

		// Node ID annotation
		foundInNode := false
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
				} else {
					foundInNode = true
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
			t.Logf("csiNodeInfo: %#v", nodeInfo)
			if err != nil {
				t.Errorf("error getting CSINodeInfo: %v", err)
				continue
			}

			// Extract node IDs and topology keys

			availableDrivers := sets.String{}
			actualNodeIDs := make(map[string]string)
			actualTopologyKeys := make(map[string]sets.String)
			for _, driver := range nodeInfo.Status.Drivers {
				if driver.Available {
					availableDrivers.Insert(driver.Name)
				}

			}
			for _, driver := range nodeInfo.Spec.Drivers {
				if availableDrivers.Has(driver.Name) {
					actualNodeIDs[driver.Name] = driver.NodeID
					actualTopologyKeys[driver.Name] = sets.NewString(driver.TopologyKeys...)
				}
			}

			// Node IDs
			// No need to check if Node ID found in NodeInfo if it was present in the NodeID
			if !foundInNode {
				if !helper.Semantic.DeepEqual(actualNodeIDs, tc.expectedNodeIDMap) {
					t.Errorf("expected node IDs %v from CSINodeInfo; got: %v", tc.expectedNodeIDMap, actualNodeIDs)
				}
			}

			// Topology keys
			if !helper.Semantic.DeepEqual(actualTopologyKeys, tc.expectedTopologyMap) {
				t.Errorf("expected topology keys %v from CSINodeInfo; got: %v", tc.expectedTopologyMap, actualTopologyKeys)
			}
		}
	}
}

func getVolumeLimit(node *v1.Node, driverName string) int64 {
	volumeLimits := map[v1.ResourceName]int64{}
	nodeAllocatables := node.Status.Allocatable
	for k, v := range nodeAllocatables {
		if v1helper.IsAttachableVolumeResourceName(k) {
			volumeLimits[k] = v.Value()
		}
	}
	attachKey := v1.ResourceName(util.GetCSIAttachLimitKey(driverName))
	attachLimit := volumeLimits[attachKey]
	return attachLimit
}

func generateNode(nodeName string, nodeIDs, labels map[string]string, capacity map[v1.ResourceName]resource.Quantity) *v1.Node {
	var annotations map[string]string
	if len(nodeIDs) > 0 {
		b, _ := json.Marshal(nodeIDs)
		annotations = map[string]string{annotationKeyNodeID: string(b)}
	}
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        nodeName,
			Annotations: annotations,
			Labels:      labels,
		},
	}

	if len(capacity) > 0 {
		node.Status.Capacity = v1.ResourceList(capacity)
		node.Status.Allocatable = v1.ResourceList(capacity)
	}
	return node
}

func generateNodeInfoAdvanced(nodeName string, drivers []unifiedDriver) *csiv1alpha1.CSINodeInfo {
	driverInfoSpecs := []csiv1alpha1.CSIDriverInfoSpec{}
	driverInfoStatuses := []csiv1alpha1.CSIDriverInfoStatus{}
	for _, driver := range drivers {
		dspec := csiv1alpha1.CSIDriverInfoSpec{
			Name:         driver.name,
			NodeID:       driver.nodeID,
			TopologyKeys: driver.topologyKeys,
		}
		dstatus := csiv1alpha1.CSIDriverInfoStatus{
			Name:                  driver.name,
			Available:             driver.available,
			VolumePluginMechanism: driver.volumePluginMechanism,
		}
		driverInfoSpecs = append(driverInfoSpecs, dspec)
		driverInfoStatuses = append(driverInfoStatuses, dstatus)

	}

	return &csiv1alpha1.CSINodeInfo{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
		Spec: csiv1alpha1.CSINodeInfoSpec{
			Drivers: driverInfoSpecs,
		},
		Status: csiv1alpha1.CSINodeInfoStatus{
			Drivers: driverInfoStatuses,
		},
	}
}
func generateNodeInfo(nodeIDs map[string]string, topologyKeys map[string][]string) *csiv1alpha1.CSINodeInfo {
	driverInfoSpecs := []csiv1alpha1.CSIDriverInfoSpec{}
	driverInfoStatuses := []csiv1alpha1.CSIDriverInfoStatus{}
	for k, nodeID := range nodeIDs {
		dspec := csiv1alpha1.CSIDriverInfoSpec{
			Name:   k,
			NodeID: nodeID,
		}
		dstatus := csiv1alpha1.CSIDriverInfoStatus{
			Name:                  k,
			Available:             true,
			VolumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
		}
		if top, exists := topologyKeys[k]; exists {
			dspec.TopologyKeys = top
		}
		driverInfoSpecs = append(driverInfoSpecs, dspec)
		driverInfoStatuses = append(driverInfoStatuses, dstatus)
	}
	return &csiv1alpha1.CSINodeInfo{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Spec: csiv1alpha1.CSINodeInfoSpec{
			Drivers: driverInfoSpecs,
		},
		Status: csiv1alpha1.CSINodeInfoStatus{
			Drivers: driverInfoStatuses,
		},
	}
}

func applyNodeStatusPatch(originalNode *v1.Node, patch []byte) (*v1.Node, error) {
	original, err := json.Marshal(originalNode)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal original node %#v: %v", originalNode, err)
	}
	updated, err := strategicpatch.StrategicMergePatch(original, patch, v1.Node{})
	if err != nil {
		return nil, fmt.Errorf("failed to apply strategic merge patch %q on node %#v: %v",
			patch, originalNode, err)
	}
	updatedNode := &v1.Node{}
	if err := json.Unmarshal(updated, updatedNode); err != nil {
		return nil, fmt.Errorf("failed to unmarshal updated node %q: %v", updated, err)
	}
	return updatedNode, nil
}

func hasPatchAction(actions []clienttesting.Action) clienttesting.Action {
	for _, action := range actions {
		if action.GetVerb() == "patch" {
			return action
		}
	}
	return nil
}
