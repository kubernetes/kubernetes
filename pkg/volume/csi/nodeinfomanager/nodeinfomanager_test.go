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
	"context"
	"encoding/json"
	goerrors "errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/ptr"
)

type testcase struct {
	name             string
	driverName       string
	existingNode     *v1.Node
	existingCSINode  *storage.CSINode
	inputNodeID      string
	inputTopology    map[string]string
	inputVolumeLimit int64
	expectedNode     *v1.Node
	expectedCSINode  *storage.CSINode
	expectFail       bool
	hasModified      bool
	migratedPlugins  map[string](func() bool)
	// specUpdates, if set, is the expected number of CSINode spec updates.
	specUpdates *int
}

type nodeIDMap map[string]string
type topologyKeyMap map[string][]string
type labelMap map[string]string

// TestInstallCSIDriver tests InstallCSIDriver with various existing Node and/or CSINode objects.
// The node IDs in all test cases below are the same between the Node annotation and CSINode.
func TestInstallCSIDriver(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node",
			driverName:   "com.example.csi.driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:  "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels:      labelMap{"com.example.csi/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
						},
					},
				},
			},
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				},
				nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"com.example.csi.driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels:      labelMap{"com.example.csi/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:       "pre-existing node info from the same driver, but without topology info",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				nil, /* topologyKeys */
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels:      labelMap{"com.example.csi/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/test-node",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"net.example.storage.other-driver": {"net.example.storage/rack"},
				},
			),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{
						"com.example.csi.driver1":          "com.example.csi/csi-node1",
						"net.example.storage.other-driver": "net.example.storage/test-node",
					})},
					Labels: labelMap{
						"com.example.csi/zone":     "zoneA",
						"net.example.storage/rack": "rack1",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "net.example.storage.other-driver",
							NodeID:       "net.example.storage/test-node",
							TopologyKeys: []string{"net.example.storage/rack"},
							Allocatable:  nil,
						},
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:       "pre-existing node info from the same driver, but different node ID and topology values; labels should conflict",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"com.example.csi.driver1": {"com.example.csi/zone"},
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
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"com.example.csi.driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID: "com.example.csi/other-node",
			inputTopology: map[string]string{
				"com.example.csi/rack": "rack1",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/other-node"})},
					Labels: labelMap{
						"com.example.csi/zone": "zoneA",
						"com.example.csi/rack": "rack1",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/other-node",
							TopologyKeys: []string{"com.example.csi/rack"},
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name: "pre-existing node info, but owned by previous node",
			existingNode: func() *v1.Node {
				node := generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/)
				node.UID = types.UID("node1")
				return node
			}(),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(nil /*nodeIDs*/, nil /*volumeLimits*/, nil /*topologyKeys*/)
				csiNode.OwnerReferences[0].UID = types.UID("node2")
				return csiNode
			}(),
			migratedPlugins: map[string](func() bool){
				"com.example.csi.driver1": func() bool { return true },
			},
			inputNodeID: "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					UID:         types.UID("node1"),
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: func() *storage.CSINode {
				csiNode := &storage.CSINode{
					ObjectMeta: getCSINodeObjectMetaWithUID("node1"),
					Spec: storage.CSINodeSpec{
						Drivers: []storage.CSINodeDriver{
							{
								NodeID: "com.example.csi/csi-node1",
							},
						},
					},
				}
				csiNode.Annotations = map[string]string{v1.MigratedPluginsAnnotationKey: "com.example.csi.driver1"}
				return csiNode
			}(),
		},
		{
			name: "pre-existing node info with driver, but owned by previous node",
			existingNode: func() *v1.Node {
				node := generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/)
				node.UID = types.UID("node1")
				return node
			}(),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(
					nodeIDMap{
						"com.example.csi.old-driver": "com.example.csi/csi-node2",
					},
					nil /*volumeLimits*/, nil, /*topologyKeys*/
				)
				csiNode.OwnerReferences[0].UID = types.UID("node2")
				return csiNode
			}(),
			driverName:  "com.example.csi.driver1",
			inputNodeID: "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					UID:         types.UID("node1"),
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMetaWithUID("node1"),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							// Only the new driver should be present because the old CSINode represented a previous node.
							Name:   "com.example.csi.driver1",
							NodeID: "com.example.csi/csi-node1",
						},
					},
				},
			},
		},
		{
			name:          "nil topology, empty node",
			driverName:    "com.example.csi.driver1",
			existingNode:  generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:       "nil topology, pre-existing node info from the same driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"com.example.csi.driver1": {"com.example.csi/zone"},
				},
			),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels: labelMap{
						"com.example.csi/zone": "zoneA",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:       "nil topology, pre-existing node info from different driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/test-node",
				},
				labelMap{
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/test-node",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"net.example.storage.other-driver": {"net.example.storage/rack"},
				},
			),
			inputNodeID:   "com.example.csi/csi-node1",
			inputTopology: nil,
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{
						"com.example.csi.driver1":          "com.example.csi/csi-node1",
						"net.example.storage.other-driver": "net.example.storage/test-node",
					})},
					Labels: labelMap{
						"net.example.storage/rack": "rack1",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "net.example.storage.other-driver",
							NodeID:       "net.example.storage/test-node",
							TopologyKeys: []string{"net.example.storage/rack"},
							Allocatable:  nil,
						},
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable:  nil,
						},
					},
				},
			},
		},
		{
			name:         "empty node ID",
			driverName:   "com.example.csi.driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			inputNodeID:  "",
			expectFail:   true,
		},
		{
			name:             "new node with valid max limit of volumes",
			driverName:       "com.example.csi.driver1",
			existingNode:     generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/),
			inputVolumeLimit: 10,
			inputTopology:    nil,
			inputNodeID:      "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable: &storage.VolumeNodeResources{
								Count: ptr.To[int32](10),
							},
						},
					},
				},
			},
		},
		{
			name:             "new node with max limit of volumes",
			driverName:       "com.example.csi.driver1",
			existingNode:     generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/),
			inputVolumeLimit: math.MaxInt32,
			inputTopology:    nil,
			inputNodeID:      "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable: &storage.VolumeNodeResources{
								Count: ptr.To[int32](math.MaxInt32),
							},
						},
					},
				},
			},
		},
		{
			name:             "new node with overflown max limit of volumes",
			driverName:       "com.example.csi.driver1",
			existingNode:     generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/),
			inputVolumeLimit: math.MaxInt32 + 1,
			inputTopology:    nil,
			inputNodeID:      "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable: &storage.VolumeNodeResources{
								Count: ptr.To[int32](math.MaxInt32),
							},
						},
					},
				},
			},
		},
		{
			name:             "new node without max limit of volumes",
			driverName:       "com.example.csi.driver1",
			existingNode:     generateNode(nil /*nodeIDs*/, nil /*labels*/, nil /*capacity*/),
			inputVolumeLimit: 0,
			inputTopology:    nil,
			inputNodeID:      "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
						},
					},
				},
			},
		},
		{
			name:       "node with existing valid max limit of volumes",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nil, /*nodeIDs*/
				nil, /*labels*/
				map[v1.ResourceName]resource.Quantity{
					v1.ResourceCPU: *resource.NewScaledQuantity(4, -3),
				}),

			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				generateVolumeLimits(10),
				nil, /* topologyKeys */
			),

			inputVolumeLimit: 20,
			inputTopology:    nil,
			inputNodeID:      "com.example.csi/csi-node1",
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
				},
				Status: v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourceCPU: *resource.NewScaledQuantity(4, -3),
					},
					Allocatable: v1.ResourceList{
						v1.ResourceCPU: *resource.NewScaledQuantity(4, -3),
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: nil,
							Allocatable:  generateVolumeLimits(20),
						},
					},
				},
			},
		},
		{
			name:       "pre-existing node info and driver registration from the same driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(
					nodeIDMap{
						"com.example.csi.driver1": "com.example.csi/csi-node1",
					},
					generateVolumeLimits(10),
					topologyKeyMap{
						"com.example.csi.driver1": {"com.example.csi/zone"},
					},
				)
				csiNode.Spec.DriverRegistrations = []storage.CSINodeDriverRegistration{
					{Name: "com.example.csi.driver1", NodeID: "com.example.csi/csi-node1"},
					{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
				}
				return csiNode
			}(),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			inputVolumeLimit: 10,
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels:      labelMap{"com.example.csi/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
							Allocatable:  generateVolumeLimits(10),
						},
					},
					// The drivers entry is unchanged, but the driver registration is removed.
					DriverRegistrations: []storage.CSINodeDriverRegistration{
						{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
					},
				},
			},
			specUpdates: new(1),
		},
		{
			name:       "pending driver registration from the same driver, and node info and driver registration from different driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nil, /* nodeIDs */
				labelMap{
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(
					nodeIDMap{
						"net.example.storage.other-driver": "net.example.storage/test-node",
					},
					nil, /* volumeLimits */
					topologyKeyMap{
						"net.example.storage.other-driver": {"net.example.storage/rack"},
					},
				)
				csiNode.Spec.DriverRegistrations = []storage.CSINodeDriverRegistration{
					{Name: "com.example.csi.driver1", NodeID: "com.example.csi/csi-node1"},
					{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
				}
				return csiNode
			}(),
			inputNodeID: "com.example.csi/csi-node1",
			inputTopology: map[string]string{
				"com.example.csi/zone": "zoneA",
			},
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"com.example.csi.driver1": "com.example.csi/csi-node1"})},
					Labels: labelMap{
						"com.example.csi/zone":     "zoneA",
						"net.example.storage/rack": "rack1",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "net.example.storage.other-driver",
							NodeID:       "net.example.storage/test-node",
							TopologyKeys: []string{"net.example.storage/rack"},
						},
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/csi-node1",
							TopologyKeys: []string{"com.example.csi/zone"},
						},
					},
					DriverRegistrations: []storage.CSINodeDriverRegistration{
						{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
					},
				},
			},
			specUpdates: new(1),
		},
	}

	test(t, true /* addNodeInfo */, testcases)
}

func generateVolumeLimits(i int32) *storage.VolumeNodeResources {
	return &storage.VolumeNodeResources{
		Count: ptr.To[int32](i),
	}
}

// TestUninstallCSIDriver tests UninstallCSIDriver with various existing Node and/or CSINode objects.
func TestUninstallCSIDriver(t *testing.T) {
	testcases := []testcase{
		{
			name:         "empty node and empty CSINode",
			driverName:   "com.example.csi.driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       storage.CSINodeSpec{},
			},
		},
		{
			name:       "pre-existing node info from the same driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				labelMap{
					"com.example.csi/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"com.example.csi.driver1": {"com.example.csi/zone"},
				},
			),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "node1",
					Labels: labelMap{"com.example.csi/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       storage.CSINodeSpec{},
			},
			hasModified: true,
		},
		{
			name:       "pre-existing node info from different driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/csi-node1",
				},
				labelMap{
					"net.example.storage/zone": "zoneA",
				}, nil /*capacity*/),
			existingCSINode: generateCSINode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/csi-node1",
				},
				nil, /* volumeLimits */
				topologyKeyMap{
					"net.example.storage.other-driver": {"net.example.storage/zone"},
				},
			),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"net.example.storage.other-driver": "net.example.storage/csi-node1"})},
					Labels:      labelMap{"net.example.storage/zone": "zoneA"},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "net.example.storage.other-driver",
							NodeID:       "net.example.storage/csi-node1",
							TopologyKeys: []string{"net.example.storage/zone"},
						},
					},
				},
			},
			hasModified: false,
		},
		{
			name:       "pre-existing info about the same driver in node, but empty CSINode",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi.driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       storage.CSINodeSpec{},
			},
		},
		{
			name: "pre-existing info about a different driver in node, but empty CSINode",
			existingNode: generateNode(
				nodeIDMap{
					"net.example.storage.other-driver": "net.example.storage/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{annotationKeyNodeID: marshall(nodeIDMap{"net.example.storage.other-driver": "net.example.storage/csi-node1"})},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       storage.CSINodeSpec{},
			},
		},
		{
			name:         "pending driver registration from the same driver",
			driverName:   "com.example.csi.driver1",
			existingNode: generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(nil /* nodeIDs */, nil /* volumeLimits */, nil /* topologyKeys */)
				csiNode.Spec.DriverRegistrations = []storage.CSINodeDriverRegistration{
					{Name: "com.example.csi.driver1", NodeID: "com.example.csi/csi-node1"},
				}
				return csiNode
			}(),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       storage.CSINodeSpec{},
			},
			hasModified: true,
			specUpdates: new(1),
		},
		{
			name:       "pre-existing node info and driver registrations from the same driver and different driver",
			driverName: "com.example.csi.driver1",
			existingNode: generateNode(
				nil, /* nodeIDs */
				labelMap{
					"com.example.csi/zone":     "zoneA",
					"net.example.storage/rack": "rack1",
				}, nil /*capacity*/),
			existingCSINode: func() *storage.CSINode {
				csiNode := generateCSINode(
					nodeIDMap{
						"com.example.csi.driver1":          "com.example.csi/csi-node1",
						"net.example.storage.other-driver": "net.example.storage/test-node",
					},
					nil, /* volumeLimits */
					topologyKeyMap{
						"com.example.csi.driver1":          {"com.example.csi/zone"},
						"net.example.storage.other-driver": {"net.example.storage/rack"},
					},
				)
				csiNode.Spec.DriverRegistrations = []storage.CSINodeDriverRegistration{
					{Name: "com.example.csi.driver1", NodeID: "com.example.csi/csi-node1"},
					{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
				}
				return csiNode
			}(),
			expectedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
					Labels: labelMap{
						"com.example.csi/zone":     "zoneA",
						"net.example.storage/rack": "rack1",
					},
				},
			},
			expectedCSINode: &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "net.example.storage.other-driver",
							NodeID:       "net.example.storage/test-node",
							TopologyKeys: []string{"net.example.storage/rack"},
						},
					},
					DriverRegistrations: []storage.CSINodeDriverRegistration{
						{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"},
					},
				},
			},
			hasModified: true,
			specUpdates: new(1),
		},
	}

	test(t, false /* addNodeInfo */, testcases)
}

func TestSetMigrationAnnotation(t *testing.T) {
	testcases := []struct {
		name            string
		migratedPlugins map[string](func() bool)
		existingNode    *storage.CSINode
		expectedNode    *storage.CSINode
		expectModified  bool
	}{
		{
			name: "nil migrated plugins",
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
		},
		{
			name: "one modified plugin",
			migratedPlugins: map[string](func() bool){
				"test": func() bool { return true },
			},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test"},
				},
			},
			expectModified: true,
		},
		{
			name: "existing plugin",
			migratedPlugins: map[string](func() bool){
				"test": func() bool { return true },
			},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test"},
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test"},
				},
			},
			expectModified: false,
		},
		{
			name:            "remove plugin",
			migratedPlugins: map[string](func() bool){},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test"},
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{},
				},
			},
			expectModified: true,
		},
		{
			name: "one modified plugin, other annotations stable",
			migratedPlugins: map[string](func() bool){
				"test": func() bool { return true },
			},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{"other": "annotation"},
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test", "other": "annotation"},
				},
			},
			expectModified: true,
		},
		{
			name: "multiple plugins modified, other annotations stable",
			migratedPlugins: map[string](func() bool){
				"test": func() bool { return true },
				"foo":  func() bool { return false },
			},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{"other": "annotation", v1.MigratedPluginsAnnotationKey: "foo"},
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "test", "other": "annotation"},
				},
			},
			expectModified: true,
		},
		{
			name: "multiple plugins added, other annotations stable",
			migratedPlugins: map[string](func() bool){
				"test": func() bool { return true },
				"foo":  func() bool { return true },
			},
			existingNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{"other": "annotation"},
				},
			},
			expectedNode: &storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node1",
					Annotations: map[string]string{v1.MigratedPluginsAnnotationKey: "foo,test", "other": "annotation"},
				},
			},
			expectModified: true,
		},
	}

	for _, tc := range testcases {
		t.Logf("test case: %s", tc.name)

		modified := setMigrationAnnotation(tc.migratedPlugins, tc.existingNode)
		if modified != tc.expectModified {
			t.Errorf("Expected modified to be %v but got %v instead", tc.expectModified, modified)
		}

		if !reflect.DeepEqual(tc.expectedNode, tc.existingNode) {
			t.Errorf("Expected CSINode: %v, but got: %v", tc.expectedNode, tc.existingNode)
		}
	}
}

func TestInstallCSIDriverExistingAnnotation(t *testing.T) {
	driverName := "com.example.csi/driver1"
	nodeID := "com.example.csi/some-node"

	testcases := []struct {
		name         string
		existingNode *v1.Node
	}{
		{
			name: "pre-existing info about the same driver in node, but empty CSINode",
			existingNode: generateNode(
				nodeIDMap{
					"com.example.csi/driver1": "com.example.csi/csi-node1",
				},
				nil /* labels */, nil /*capacity*/),
		},
		{
			name: "pre-existing info about a different driver in node, but empty CSINode",
			existingNode: generateNode(
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

		tmpDir, err := utiltesting.MkTmpdir("nodeinfomanager-test")
		if err != nil {
			t.Fatalf("can't create temp dir: %v", err)
		}
		defer os.RemoveAll(tmpDir)
		host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
			tmpDir,
			client,
			nil,
			nodeName,
			nil,
			nil,
		)

		nim := NewNodeInfoManager(types.NodeName(nodeName), host, nil)

		// Act
		_, err = nim.CreateCSINode()
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
		nodeInfo, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			t.Errorf("error getting CSINode: %v", err)
			continue
		}

		driver := nodeInfo.Spec.Drivers[0]
		if driver.Name != driverName || driver.NodeID != nodeID {
			t.Errorf("expected Driver to be %q and NodeID to be %q, but got: %q:%q", driverName, nodeID, driver.Name, driver.NodeID)
		}
	}
}

func getClientSet(existingNode *v1.Node, existingCSINode *storage.CSINode) *fake.Clientset {
	objects := []runtime.Object{}
	if existingNode != nil {
		objects = append(objects, existingNode)
	}
	if existingCSINode != nil {
		objects = append(objects, existingCSINode)
	}
	return fake.NewSimpleClientset(objects...)
}

func test(t *testing.T, addNodeInfo bool, testcases []testcase) {
	for _, tc := range testcases {
		t.Logf("test case: %q", tc.name)

		//// Arrange
		nodeName := tc.existingNode.Name
		client := getClientSet(tc.existingNode, tc.existingCSINode)

		tmpDir, err := utiltesting.MkTmpdir("nodeinfomanager-test")
		if err != nil {
			t.Fatalf("can't create temp dir: %v", err)
		}
		defer os.RemoveAll(tmpDir)
		host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
			tmpDir,
			client,
			nil,
			nodeName,
			nil,
			nil,
		)
		nim := NewNodeInfoManager(types.NodeName(nodeName), host, tc.migratedPlugins)

		//// Act
		nim.CreateCSINode()
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
		if tc.specUpdates != nil {
			assert.Equal(t, *tc.specUpdates, countSpecUpdates(actions), "CSINode spec updates")
		}

		var node *v1.Node
		if action := hasPatchAction(actions); action != nil {
			node, err = applyNodeStatusPatch(tc.existingNode, action.(clienttesting.PatchActionImpl).GetPatch())
			assert.NoError(t, err)
		} else {
			node, err = client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			assert.NoError(t, err)
		}

		if node == nil {
			t.Errorf("error getting node: %v", err)
			continue
		}

		if !helper.Semantic.DeepEqual(node, tc.expectedNode) {
			t.Errorf("expected Node %v; got: %v", tc.expectedNode, node)
		}

		// CSINode validation
		nodeInfo, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				t.Errorf("error getting CSINode: %v", err)
			}
			continue
		}
		if !helper.Semantic.DeepEqual(nodeInfo, tc.expectedCSINode) {
			t.Errorf("expected CSINode %v; got: %v", tc.expectedCSINode, nodeInfo)
		}

		if !addNodeInfo && tc.existingCSINode != nil && tc.existingNode != nil {
			if tc.hasModified && helper.Semantic.DeepEqual(nodeInfo, tc.existingCSINode) {
				t.Errorf("existing CSINode %v; got: %v", tc.existingCSINode, nodeInfo)
			}
			if !tc.hasModified && !helper.Semantic.DeepEqual(nodeInfo, tc.existingCSINode) {
				t.Errorf("existing CSINode %v; got: %v", tc.existingCSINode, nodeInfo)
			}
		}
	}
}

func generateNode(nodeIDs, labels map[string]string, capacity map[v1.ResourceName]resource.Quantity) *v1.Node {
	var annotations map[string]string
	if len(nodeIDs) > 0 {
		b, _ := json.Marshal(nodeIDs)
		annotations = map[string]string{annotationKeyNodeID: string(b)}
	}
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "node1",
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

func marshall(nodeIDs nodeIDMap) string {
	b, _ := json.Marshal(nodeIDs)
	return string(b)
}

func generateCSINode(nodeIDs nodeIDMap, volumeLimits *storage.VolumeNodeResources, topologyKeys topologyKeyMap) *storage.CSINode {
	nodeDrivers := []storage.CSINodeDriver{}
	for k, nodeID := range nodeIDs {
		dspec := storage.CSINodeDriver{
			Name:        k,
			NodeID:      nodeID,
			Allocatable: volumeLimits,
		}
		if top, exists := topologyKeys[k]; exists {
			dspec.TopologyKeys = top
		}
		nodeDrivers = append(nodeDrivers, dspec)
	}

	return &storage.CSINode{
		ObjectMeta: getCSINodeObjectMeta(),
		Spec: storage.CSINodeSpec{
			Drivers: nodeDrivers,
		},
	}
}

func getCSINodeObjectMeta() metav1.ObjectMeta {
	return metav1.ObjectMeta{
		Name: "node1",
		OwnerReferences: []metav1.OwnerReference{
			{
				APIVersion: nodeKind.Version,
				Kind:       nodeKind.Kind,
				Name:       "node1",
			},
		},
	}
}

func getCSINodeObjectMetaWithUID(uid types.UID) metav1.ObjectMeta {
	meta := getCSINodeObjectMeta()
	meta.OwnerReferences[0].UID = uid
	return meta
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

func TestUpdateCSINodeStorageHealth(t *testing.T) {
	const (
		driver1 = "com.example.csi.driver1"
		driver2 = "com.example.csi.driver2"
	)
	oldTime := metav1.NewTime(time.Unix(100, 0))
	otherTime := metav1.NewTime(time.Unix(200, 0))

	// Conditions already stored in CSINode, with transition times.
	storedDegraded := storage.StorageHealthCondition{
		Status:             storage.StorageDegraded,
		Reason:             "DiskSlow",
		Message:            "disk is slow",
		LastTransitionTime: oldTime,
	}
	storedUnreachable := storage.StorageHealthCondition{
		Status:             storage.StorageUnreachable,
		Reason:             "NetworkDown",
		LastTransitionTime: otherTime,
	}
	storedHealthy := storage.StorageHealthCondition{
		Status:             storage.StorageHealthy,
		Reason:             "StorageHealthy",
		LastTransitionTime: oldTime,
	}
	storedDegradedBlock := storage.StorageHealthCondition{
		Status:             storage.StorageDegraded,
		Reason:             "DiskSlow",
		AccessMode:         ptr.To(v1.ReadWriteOnce),
		VolumeMode:         ptr.To(v1.PersistentVolumeBlock),
		LastTransitionTime: oldTime,
	}
	storedOtherDriver := storage.StorageHealthCondition{
		Status:             storage.StorageDegraded,
		Reason:             "OtherIssue",
		LastTransitionTime: otherTime,
	}

	// Conditions reported by the driver, without transition times. When used in
	// want, a zero transition time means kubelet must assign the current time.
	degraded := storage.StorageHealthCondition{
		Status:  storage.StorageDegraded,
		Reason:  "DiskSlow",
		Message: "disk is slow",
	}
	degradedNewMessage := storage.StorageHealthCondition{
		Status:  storage.StorageDegraded,
		Reason:  "DiskSlow",
		Message: "disk is still slow",
	}
	degradedNewReason := storage.StorageHealthCondition{
		Status: storage.StorageDegraded,
		Reason: "PoolFull",
	}
	unknownSameReason := storage.StorageHealthCondition{
		Status: storage.StorageHealthUnknown,
		Reason: "DiskSlow",
	}
	degradedRWO := storage.StorageHealthCondition{
		Status:     storage.StorageDegraded,
		Reason:     "DiskSlow",
		AccessMode: ptr.To(v1.ReadWriteOnce),
	}
	degradedFilesystem := storage.StorageHealthCondition{
		Status:     storage.StorageDegraded,
		Reason:     "DiskSlow",
		AccessMode: ptr.To(v1.ReadWriteOnce),
		VolumeMode: ptr.To(v1.PersistentVolumeFilesystem),
	}
	unreachable := storage.StorageHealthCondition{
		Status: storage.StorageUnreachable,
		Reason: "NetworkDown",
	}
	healthy := storage.StorageHealthCondition{
		Status: storage.StorageHealthy,
		Reason: "StorageHealthy",
	}

	health := func(driverName string, conditions ...storage.StorageHealthCondition) storage.StorageHealth {
		return storage.StorageHealth{
			Name:             driverName,
			HealthConditions: conditions,
		}
	}

	tests := []struct {
		name            string
		featureDisabled bool
		stored          []storage.StorageHealth
		// reported is what driver1 reports.
		reported   []storage.StorageHealthCondition
		want       []storage.StorageHealth
		wantUpdate bool
	}{
		{
			name:            "feature gate disabled",
			featureDisabled: true,
			reported:        []storage.StorageHealthCondition{degraded},
		},
		{
			name:       "initial report",
			reported:   []storage.StorageHealthCondition{degraded},
			want:       []storage.StorageHealth{health(driver1, degraded)},
			wantUpdate: true,
		},
		{
			name:       "ignore supplied time for new condition",
			reported:   []storage.StorageHealthCondition{storedDegraded},
			want:       []storage.StorageHealth{health(driver1, degraded)},
			wantUpdate: true,
		},
		{
			name:       "initial report preserves other drivers",
			stored:     []storage.StorageHealth{health(driver2, storedOtherDriver)},
			reported:   []storage.StorageHealthCondition{degraded},
			want:       []storage.StorageHealth{health(driver2, storedOtherDriver), health(driver1, degraded)},
			wantUpdate: true,
		},
		{
			name:     "unchanged",
			stored:   []storage.StorageHealth{health(driver1, storedDegraded)},
			reported: []storage.StorageHealthCondition{degraded},
			want:     []storage.StorageHealth{health(driver1, storedDegraded)},
		},
		{
			name:     "message changed",
			stored:   []storage.StorageHealth{health(driver1, storedDegraded)},
			reported: []storage.StorageHealthCondition{degradedNewMessage},
			want:     []storage.StorageHealth{health(driver1, storedDegraded)},
		},
		{
			name:     "duplicate reports",
			stored:   []storage.StorageHealth{health(driver1, storedDegraded)},
			reported: []storage.StorageHealthCondition{degraded, degraded},
			want:     []storage.StorageHealth{health(driver1, storedDegraded)},
		},
		{
			name:     "reordered",
			stored:   []storage.StorageHealth{health(driver1, storedDegraded, storedUnreachable)},
			reported: []storage.StorageHealthCondition{unreachable, degraded},
			want:     []storage.StorageHealth{health(driver1, storedDegraded, storedUnreachable)},
		},
		{
			name:       "condition added",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded)},
			reported:   []storage.StorageHealthCondition{degraded, unreachable},
			want:       []storage.StorageHealth{health(driver1, storedDegraded, unreachable)},
			wantUpdate: true,
		},
		{
			name:       "condition removed",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded, storedUnreachable)},
			reported:   []storage.StorageHealthCondition{degraded},
			want:       []storage.StorageHealth{health(driver1, storedDegraded)},
			wantUpdate: true,
		},
		{
			name:       "reason changed",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded)},
			reported:   []storage.StorageHealthCondition{degradedNewReason},
			want:       []storage.StorageHealth{health(driver1, degradedNewReason)},
			wantUpdate: true,
		},
		{
			name:       "status changed",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded)},
			reported:   []storage.StorageHealthCondition{unknownSameReason},
			want:       []storage.StorageHealth{health(driver1, unknownSameReason)},
			wantUpdate: true,
		},
		{
			name:       "access mode changed",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded)},
			reported:   []storage.StorageHealthCondition{degradedRWO},
			want:       []storage.StorageHealth{health(driver1, degradedRWO)},
			wantUpdate: true,
		},
		{
			name:       "volume mode changed",
			stored:     []storage.StorageHealth{health(driver1, storedDegradedBlock)},
			reported:   []storage.StorageHealthCondition{degradedFilesystem},
			want:       []storage.StorageHealth{health(driver1, degradedFilesystem)},
			wantUpdate: true,
		},
		{
			name:       "replace conditions preserves other drivers",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded), health(driver2, storedOtherDriver)},
			reported:   []storage.StorageHealthCondition{unreachable},
			want:       []storage.StorageHealth{health(driver2, storedOtherDriver), health(driver1, unreachable)},
			wantUpdate: true,
		},
		{
			name:       "healthy replaces degraded",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded)},
			reported:   []storage.StorageHealthCondition{healthy},
			want:       []storage.StorageHealth{health(driver1, healthy)},
			wantUpdate: true,
		},
		{
			name:       "degraded replaces healthy",
			stored:     []storage.StorageHealth{health(driver1, storedHealthy)},
			reported:   []storage.StorageHealthCondition{degraded},
			want:       []storage.StorageHealth{health(driver1, degraded)},
			wantUpdate: true,
		},
		{
			name:     "healthy unchanged",
			stored:   []storage.StorageHealth{health(driver1, storedHealthy)},
			reported: []storage.StorageHealthCondition{healthy},
			want:     []storage.StorageHealth{health(driver1, storedHealthy)},
		},
		{
			name:       "clear conditions preserves other drivers",
			stored:     []storage.StorageHealth{health(driver1, storedDegraded), health(driver2, storedOtherDriver)},
			reported:   nil,
			want:       []storage.StorageHealth{health(driver2, storedOtherDriver)},
			wantUpdate: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, !tc.featureDisabled)
			nim, client, nodeName := newStorageHealthTestManager(t, []string{driver1, driver2}, tc.stored)
			reportedCopy := (&storage.StorageHealth{HealthConditions: tc.reported}).DeepCopy().HealthConditions

			before := metav1.Now().Rfc3339Copy()
			if err := nim.UpdateCSINodeStorageHealth(t.Context(), driver1, tc.reported); err != nil {
				t.Fatalf("UpdateCSINodeStorageHealth returned error: %v", err)
			}
			after := metav1.Now()

			assert.Equal(t, reportedCopy, tc.reported, "reported conditions must not be mutated")
			assert.Equal(t, tc.wantUpdate, countStatusUpdates(client.Actions()) > 0, "status update")

			got, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting CSINode: %v", err)
			}
			checkFreshTransitionTimes(t, got.Status.StorageHealth, tc.want, before, after)
			assert.Equal(t, tc.want, got.Status.StorageHealth)

			actionsBefore := len(client.Actions())
			if err := nim.UpdateCSINodeStorageHealth(t.Context(), driver1, tc.reported); err != nil {
				t.Fatalf("UpdateCSINodeStorageHealth returned error: %v", err)
			}
			assert.Zero(t, countStatusUpdates(client.Actions()[actionsBefore:]), "repeated report must not cause another status update")
		})
	}
}

func TestNodeRecreatedSameNameRefreshesCSINodeOwnerUID(t *testing.T) {
	const (
		nodeName   = "node1"
		oldNodeUID = types.UID("old-node-uid")
		newNodeUID = types.UID("new-node-uid")
		driverName = "com.example.csi.driver1"
		driverID   = "com.example.csi/csi-node1"
	)

	healthConds := []storage.StorageHealthCondition{{
		Status:  storage.StorageDegraded,
		Reason:  "DiskSlow",
		Message: "high latency",
	}}

	testcases := []struct {
		name               string
		gcDeletedOldCSI    bool
		preCreateNewCSI    bool
		enableVolumeHealth bool
		act                func(nim Interface) error
		verify             func(t *testing.T, csiNode *storage.CSINode, actions []clienttesting.Action)
	}{
		{
			name:            "InstallCSIDriver after GC deleted predecessor CSINode",
			gcDeletedOldCSI: true,
			act: func(nim Interface) error {
				return nim.InstallCSIDriver(driverName, driverID, 0, nil)
			},
			verify: func(t *testing.T, csiNode *storage.CSINode, _ []clienttesting.Action) {
				assert.Len(t, csiNode.Spec.Drivers, 1)
				assert.Equal(t, driverName, csiNode.Spec.Drivers[0].Name)
			},
		},
		{
			name:            "InstallCSIDriver before GC deleted predecessor CSINode",
			gcDeletedOldCSI: false,
			act: func(nim Interface) error {
				return nim.InstallCSIDriver(driverName, driverID, 0, nil)
			},
			verify: func(t *testing.T, csiNode *storage.CSINode, _ []clienttesting.Action) {
				// Predecessor driver entry must be gone after replacing the old CSINode.
				assert.Len(t, csiNode.Spec.Drivers, 1)
				assert.Equal(t, driverName, csiNode.Spec.Drivers[0].Name)
			},
		},
		{
			name:            "InstallCSIDriverRegistration before GC deleted predecessor CSINode",
			gcDeletedOldCSI: false,
			act: func(nim Interface) error {
				return nim.InstallCSIDriverRegistration(driverName, driverID)
			},
			verify: func(t *testing.T, csiNode *storage.CSINode, _ []clienttesting.Action) {
				// Predecessor driver entry must be gone after replacing the old CSINode.
				assert.Empty(t, csiNode.Spec.Drivers)
				assert.Equal(t, []storage.CSINodeDriverRegistration{{Name: driverName, NodeID: driverID}}, csiNode.Spec.DriverRegistrations)
			},
		},
		{
			name:            "UpdateCSIDriver after GC deleted predecessor CSINode",
			gcDeletedOldCSI: true,
			act: func(nim Interface) error {
				return nim.UpdateCSIDriver(driverName, driverID, 16, nil)
			},
		},
		{
			name:               "UpdateCSINodeStorageHealth does not delete valid CSINode",
			gcDeletedOldCSI:    true,
			preCreateNewCSI:    true,
			enableVolumeHealth: true,
			act: func(nim Interface) error {
				return nim.UpdateCSINodeStorageHealth(t.Context(), driverName, healthConds)
			},
			verify: func(t *testing.T, csiNode *storage.CSINode, actions []clienttesting.Action) {
				for _, action := range actions {
					if action.GetVerb() == "delete" && action.GetResource().Resource == "csinodes" {
						t.Fatalf("unexpectedly deleted valid CSINode")
					}
				}
				assert.Len(t, csiNode.Status.StorageHealth, 1)
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableVolumeHealth {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
			}

			// 1. Early startup caches oldNodeUID while predecessor Node and CSINode still exist.
			oldNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, UID: oldNodeUID}}
			oldCSINode := generateCSINode(nodeIDMap{"com.example.csi.old": "old-id"}, nil, nil)
			oldCSINode.OwnerReferences[0].UID = oldNodeUID

			client := getClientSet(oldNode, oldCSINode)
			host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, nodeName, nil, nil)
			nim := NewNodeInfoManager(nodeName, host, nil).(*nodeInfoManager)

			if err := nim.InitializeCSINodeWithAnnotation(); err != nil {
				t.Fatalf("InitializeCSINodeWithAnnotation failed: %v", err)
			}
			assert.Equal(t, oldNodeUID, nim.nodeID)

			// 2. Predecessor Node is replaced by a new Node with the same name and newNodeUID.
			if err := client.CoreV1().Nodes().Delete(context.TODO(), nodeName, metav1.DeleteOptions{}); err != nil {
				t.Fatalf("deleting predecessor Node failed: %v", err)
			}
			if tc.gcDeletedOldCSI {
				if err := client.StorageV1().CSINodes().Delete(context.TODO(), nodeName, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("deleting predecessor CSINode failed: %v", err)
				}
			}
			newNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, UID: newNodeUID}}
			if _, err := client.CoreV1().Nodes().Create(context.TODO(), newNode, metav1.CreateOptions{}); err != nil {
				t.Fatalf("creating replacement Node failed: %v", err)
			}
			if tc.preCreateNewCSI {
				validCSINode := generateCSINode(nodeIDMap{driverName: driverID}, nil, nil)
				validCSINode.OwnerReferences[0].UID = newNodeUID
				if _, err := client.StorageV1().CSINodes().Create(context.TODO(), validCSINode, metav1.CreateOptions{}); err != nil {
					t.Fatalf("creating replacement CSINode failed: %v", err)
				}
			}

			// 3. Execute operation with stale nim.nodeID and verify resulting CSINode owns newNodeUID.
			actionsBefore := len(client.Actions())
			if err := tc.act(nim); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			csiNode, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("getting CSINode failed: %v", err)
			}
			assert.Equal(t, newNodeUID, csiNode.OwnerReferences[0].UID)
			assert.Equal(t, newNodeUID, nim.nodeID)
			if tc.verify != nil {
				tc.verify(t, csiNode, client.Actions()[actionsBefore:])
			}
		})
	}
}

func TestUpdateCSINodeStorageHealthCancelled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	nim, client, _ := newStorageHealthTestManager(t, []string{"com.example.csi.driver1"}, nil)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	conditions := []storage.StorageHealthCondition{
		{
			Status: storage.StorageDegraded,
			Reason: "DiskSlow",
		},
	}
	err := nim.UpdateCSINodeStorageHealth(ctx, "com.example.csi.driver1", conditions)
	if !goerrors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	assert.Zero(t, countStatusUpdates(client.Actions()), "cancelled context must not update status")
}

func TestUpdateCSINodeStorageHealthSkipsUninstalledDriver(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	nim, client, nodeName := newStorageHealthTestManager(t, nil, nil)

	conditions := []storage.StorageHealthCondition{
		{
			Status: storage.StorageDegraded,
			Reason: "DiskSlow",
		},
	}
	if err := nim.UpdateCSINodeStorageHealth(t.Context(), "com.example.csi.driver1", conditions); err != nil {
		t.Fatalf("UpdateCSINodeStorageHealth returned error: %v", err)
	}

	assert.Zero(t, countStatusUpdates(client.Actions()), "health for an uninstalled driver must not be written")
	got, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting CSINode: %v", err)
	}
	assert.Empty(t, got.Status.StorageHealth)
}

func TestUninstallCSIDriverRemovesStorageHealth(t *testing.T) {
	const (
		driver1 = "com.example.csi.driver1"
		driver2 = "com.example.csi.driver2"
	)
	health1 := storage.StorageHealth{
		Name: driver1,
		HealthConditions: []storage.StorageHealthCondition{
			{
				Status: storage.StorageDegraded,
				Reason: "DiskSlow",
			},
		},
	}
	health2 := storage.StorageHealth{
		Name: driver2,
		HealthConditions: []storage.StorageHealthCondition{
			{
				Status: storage.StorageHealthy,
				Reason: "StorageHealthy",
			},
		},
	}

	tests := []struct {
		name             string
		installed        []string
		stored           []storage.StorageHealth
		wantHealth       []storage.StorageHealth
		wantStatusUpdate bool
	}{
		{
			name:             "removes health and preserves other drivers",
			installed:        []string{driver1, driver2},
			stored:           []storage.StorageHealth{health1, health2},
			wantHealth:       []storage.StorageHealth{health2},
			wantStatusUpdate: true,
		},
		{
			// A previous attempt removed the spec entry but failed to update status.
			name:             "removes leftover health when spec is already clean",
			installed:        []string{driver2},
			stored:           []storage.StorageHealth{health1, health2},
			wantHealth:       []storage.StorageHealth{health2},
			wantStatusUpdate: true,
		},
		{
			name:       "no status update without health entry",
			installed:  []string{driver1, driver2},
			stored:     []storage.StorageHealth{health2},
			wantHealth: []storage.StorageHealth{health2},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nim, client, nodeName := newStorageHealthTestManager(t, tc.installed, tc.stored)

			if err := nim.uninstallDriverFromCSINode(driver1); err != nil {
				t.Fatalf("uninstallDriverFromCSINode returned error: %v", err)
			}

			// The fake client does not drop status on Update(), so check that the
			// status subresource was written explicitly.
			assert.Equal(t, tc.wantStatusUpdate, countStatusUpdates(client.Actions()) > 0, "status update")
			got, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting CSINode: %v", err)
			}
			assert.Equal(t, tc.wantHealth, got.Status.StorageHealth)
			for _, driver := range got.Spec.Drivers {
				assert.NotEqual(t, driver1, driver.Name, "driver must be removed from spec")
			}
		})
	}
}

func TestUninstallCSIDriverRemovesDriverRegistrationAndStorageHealth(t *testing.T) {
	const (
		driver1 = "com.example.csi.driver1"
		driver2 = "com.example.csi.driver2"
	)
	health := func(driverName string) storage.StorageHealth {
		return storage.StorageHealth{
			Name: driverName,
			HealthConditions: []storage.StorageHealthCondition{
				{
					Status: storage.StorageDegraded,
					Reason: "DiskSlow",
				},
			},
		}
	}
	csiNode := &storage.CSINode{
		ObjectMeta: getCSINodeObjectMeta(),
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{Name: driver1, NodeID: "node1"},
				{Name: driver2, NodeID: "node1"},
			},
			DriverRegistrations: []storage.CSINodeDriverRegistration{
				{Name: driver1, NodeID: "node1"},
				{Name: driver2, NodeID: "node1"},
			},
		},
		Status: storage.CSINodeStatus{
			StorageHealth: []storage.StorageHealth{health(driver1), health(driver2)},
		},
	}
	client := getClientSet(generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/), csiNode)
	host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, csiNode.Name, nil, nil)
	nim := NewNodeInfoManager(types.NodeName(csiNode.Name), host, nil)

	if err := nim.UninstallCSIDriver(driver1); err != nil {
		t.Fatalf("UninstallCSIDriver returned error: %v", err)
	}

	// The drivers and driverRegistrations entries are removed in one spec update, and the
	// storage health in a separate status update.
	actions := client.Actions()
	assert.Equal(t, 1, countSpecUpdates(actions), "CSINode spec updates")
	assert.Equal(t, 1, countStatusUpdates(actions), "CSINode status updates")
	got, err := client.StorageV1().CSINodes().Get(context.TODO(), csiNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting CSINode: %v", err)
	}
	assert.Equal(t, []storage.CSINodeDriver{{Name: driver2, NodeID: "node1"}}, got.Spec.Drivers)
	assert.Equal(t, []storage.CSINodeDriverRegistration{{Name: driver2, NodeID: "node1"}}, got.Spec.DriverRegistrations)
	assert.Equal(t, []storage.StorageHealth{health(driver2)}, got.Status.StorageHealth)
}

// TestInstallCSIDriverRegistration tests InstallCSIDriverRegistration with various existing CSINode
// objects and kube-apiserver responses.
func TestInstallCSIDriverRegistration(t *testing.T) {
	const (
		nodeName   = "node1"
		driverName = "com.example.csi.driver1"
		driverID   = "com.example.csi/csi-node1"
	)
	registration := storage.CSINodeDriverRegistration{Name: driverName, NodeID: driverID}
	// The topology keys are not sorted, unlike when kubelet writes them, to show that the
	// drivers entry is kept as is.
	driver := storage.CSINodeDriver{
		Name:         driverName,
		NodeID:       driverID,
		TopologyKeys: []string{"com.example.csi/zone", "com.example.csi/rack"},
		Allocatable:  generateVolumeLimits(10),
	}
	otherRegistration := storage.CSINodeDriverRegistration{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"}
	otherDriver := storage.CSINodeDriver{
		Name:         "net.example.storage.other-driver",
		NodeID:       "net.example.storage/test-node",
		TopologyKeys: []string{"net.example.storage/rack"},
	}
	csiNode := func(spec storage.CSINodeSpec) *storage.CSINode {
		return &storage.CSINode{
			ObjectMeta: getCSINodeObjectMeta(),
			Spec:       spec,
		}
	}

	testcases := []struct {
		name         string
		stored       *storage.CSINode
		driverNodeID string
		// reactor, if set, handles CSINode updates before the fake client stores them.
		reactor func(tracker clienttesting.ObjectTracker) clienttesting.ReactionFunc
		// want is the stored CSINode after the call. If it is nil, no API calls are expected.
		want            *storage.CSINode
		wantErr         string
		wantSpecUpdates int
	}{
		{
			name:            "no CSINode",
			driverNodeID:    driverID,
			want:            csiNode(storage.CSINodeSpec{DriverRegistrations: []storage.CSINodeDriverRegistration{registration}}),
			wantSpecUpdates: 1,
		},
		{
			name:         "pre-existing driver from the same driver",
			stored:       csiNode(storage.CSINodeSpec{Drivers: []storage.CSINodeDriver{driver}}),
			driverNodeID: driverID,
			want: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			}),
			wantSpecUpdates: 1,
		},
		{
			name: "pre-existing driver and driver registration from different driver",
			stored: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{otherDriver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration},
			}),
			driverNodeID: driverID,
			want: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{otherDriver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration, registration},
			}),
			wantSpecUpdates: 1,
		},
		{
			name: "pre-existing driver and driver registration from the same driver",
			stored: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			}),
			driverNodeID: driverID,
			want: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			}),
			wantSpecUpdates: 0,
		},
		{
			name: "pending driver registration from the same driver, but different node ID",
			stored: csiNode(storage.CSINodeSpec{
				DriverRegistrations: []storage.CSINodeDriverRegistration{
					{Name: driverName, NodeID: "com.example.csi/other-node"},
					otherRegistration,
				},
			}),
			driverNodeID: driverID,
			want: csiNode(storage.CSINodeSpec{
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration, registration},
			}),
			wantSpecUpdates: 1,
		},
		{
			name:         "empty node ID",
			driverNodeID: "",
			wantErr:      "driverNodeID must not be empty",
		},
		{
			// kube-apiserver has the CSIControllerGetNodeInfo feature gate disabled.
			name:            "driver registrations dropped by kube-apiserver",
			driverNodeID:    driverID,
			reactor:         dropDriverRegistrationsReactor,
			want:            csiNode(storage.CSINodeSpec{}),
			wantErr:         string(features.CSIControllerGetNodeInfo),
			wantSpecUpdates: 1,
		},
		{
			// kube-apiserver rejects a driver registration whose node ID differs from the
			// drivers entry of the same driver.
			name:         "driver registration rejected by kube-apiserver",
			stored:       csiNode(storage.CSINodeSpec{Drivers: []storage.CSINodeDriver{{Name: driverName, NodeID: "com.example.csi/other-node"}}}),
			driverNodeID: driverID,
			reactor:      invalidCSINodeReactor,
			want:         csiNode(storage.CSINodeSpec{Drivers: []storage.CSINodeDriver{{Name: driverName, NodeID: "com.example.csi/other-node"}}}),
			wantErr:      "is invalid",
			// Every error is retried, like in InstallCSIDriver.
			wantSpecUpdates: updateBackoff.Steps,
		},
		{
			name:         "update conflict with external-attacher",
			stored:       csiNode(storage.CSINodeSpec{DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration}}),
			driverNodeID: driverID,
			// external-attacher adds the drivers entry of the other driver between the Get and
			// the Update of kubelet, which must keep the entry when it retries.
			reactor: func(tracker clienttesting.ObjectTracker) clienttesting.ReactionFunc {
				return conflictOnceReactor(tracker, func(csiNode *storage.CSINode) {
					csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, otherDriver)
				})
			},
			want: csiNode(storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{otherDriver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration, registration},
			}),
			wantSpecUpdates: 2,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := getClientSet(generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/), tc.stored)
			if tc.reactor != nil {
				client.PrependReactor("update", "csinodes", tc.reactor(client.Tracker()))
			}
			host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, nodeName, nil, nil)
			nim := NewNodeInfoManager(nodeName, host, nil)

			err := nim.InstallCSIDriverRegistration(driverName, tc.driverNodeID)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			actions := client.Actions()
			if tc.want == nil {
				assert.Empty(t, actions, "API calls")
				return
			}
			assert.Nil(t, hasPatchAction(actions), "Node must not be patched")
			assert.Equal(t, tc.wantSpecUpdates, countSpecUpdates(actions), "CSINode spec updates")

			got, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting CSINode: %v", err)
			}
			if !helper.Semantic.DeepEqual(got, tc.want) {
				t.Errorf("expected CSINode %+v; got: %+v", tc.want, got)
			}
		})
	}
}

// TestUpdateCSIDriver tests UpdateCSIDriver with and without a driver registration of the driver
// in the CSINode object.
func TestUpdateCSIDriver(t *testing.T) {
	const (
		nodeName   = "node1"
		driverName = "com.example.csi.driver1"
		driverID   = "com.example.csi/csi-node1"
	)
	topology := map[string]string{"com.example.csi/zone": "zoneA"}
	registration := storage.CSINodeDriverRegistration{Name: driverName, NodeID: driverID}
	otherRegistration := storage.CSINodeDriverRegistration{Name: "net.example.storage.other-driver", NodeID: "net.example.storage/test-node"}
	driver := storage.CSINodeDriver{
		Name:         driverName,
		NodeID:       driverID,
		TopologyKeys: []string{"com.example.csi/zone"},
		Allocatable:  generateVolumeLimits(10),
	}
	updatedDriver := storage.CSINodeDriver{
		Name:         driverName,
		NodeID:       driverID,
		TopologyKeys: []string{"com.example.csi/zone"},
		Allocatable:  generateVolumeLimits(20),
	}

	testcases := []struct {
		name            string
		stored          storage.CSINodeSpec
		want            storage.CSINodeSpec
		wantSpecUpdates int
	}{
		{
			name: "no driver registration",
			stored: storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration},
			},
			want: storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{updatedDriver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{otherRegistration},
			},
			wantSpecUpdates: 1,
		},
		{
			// external-attacher hasn't added the drivers entry yet.
			name: "pending driver registration",
			stored: storage.CSINodeSpec{
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			},
			want: storage.CSINodeSpec{
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			},
			wantSpecUpdates: 0,
		},
		{
			// external-attacher updates the drivers entry.
			name: "completed driver registration",
			stored: storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			},
			want: storage.CSINodeSpec{
				Drivers:             []storage.CSINodeDriver{driver},
				DriverRegistrations: []storage.CSINodeDriverRegistration{registration},
			},
			wantSpecUpdates: 0,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			stored := &storage.CSINode{
				ObjectMeta: getCSINodeObjectMeta(),
				Spec:       tc.stored,
			}
			client := getClientSet(generateNode(nil /* nodeIDs */, nil /* labels */, nil /*capacity*/), stored)
			host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, nodeName, nil, nil)
			nim := NewNodeInfoManager(nodeName, host, nil)

			if err := nim.UpdateCSIDriver(driverName, driverID, 20, topology); err != nil {
				t.Fatalf("UpdateCSIDriver returned error: %v", err)
			}

			assert.Equal(t, tc.wantSpecUpdates, countSpecUpdates(client.Actions()), "CSINode spec updates")
			got, err := client.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting CSINode: %v", err)
			}
			if !helper.Semantic.DeepEqual(got.Spec, tc.want) {
				t.Errorf("expected CSINode spec %+v; got: %+v", tc.want, got.Spec)
			}
		})
	}
}

// checkFreshTransitionTimes verifies that every condition whose expected
// transition time is zero got a time within [before, after], then zeroes it in
// got so the caller can compare the rest of the status exactly.
func checkFreshTransitionTimes(t *testing.T, got, want []storage.StorageHealth, before, after metav1.Time) {
	t.Helper()
	for i := range min(len(got), len(want)) {
		for j := range min(len(got[i].HealthConditions), len(want[i].HealthConditions)) {
			if !want[i].HealthConditions[j].LastTransitionTime.IsZero() {
				continue
			}
			transition := &got[i].HealthConditions[j].LastTransitionTime
			if transition.Before(&before) || transition.After(after.Time) {
				t.Errorf("%s condition %d: transition time %v not within [%v, %v]", got[i].Name, j, transition, before, after)
			}
			*transition = metav1.Time{}
		}
	}
}

// newStorageHealthTestManager returns a nodeInfoManager backed by a fake client
// holding a CSINode with the given drivers installed and the given storage health.
func newStorageHealthTestManager(t *testing.T, installed []string, stored []storage.StorageHealth) (*nodeInfoManager, *fake.Clientset, string) {
	t.Helper()
	node := generateNode(nil, nil, nil)
	drivers := make([]storage.CSINodeDriver, 0, len(installed))
	for _, name := range installed {
		drivers = append(drivers, storage.CSINodeDriver{
			Name:   name,
			NodeID: "node1",
		})
	}
	csiNode := &storage.CSINode{
		ObjectMeta: getCSINodeObjectMeta(),
		Spec: storage.CSINodeSpec{
			Drivers: drivers,
		},
		Status: storage.CSINodeStatus{
			StorageHealth: stored,
		},
	}
	client := getClientSet(node, csiNode)
	host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, node.Name, nil, nil)
	nim := NewNodeInfoManager(types.NodeName(node.Name), host, nil).(*nodeInfoManager)
	nim.nodeID = node.UID
	return nim, client, node.Name
}

func countStatusUpdates(actions []clienttesting.Action) int {
	count := 0
	for _, action := range actions {
		if action.GetVerb() == "update" && action.GetSubresource() == "status" {
			count++
		}
	}
	return count
}

func countSpecUpdates(actions []clienttesting.Action) int {
	count := 0
	for _, action := range actions {
		if action.GetVerb() == "update" && action.GetResource().Resource == "csinodes" && action.GetSubresource() == "" {
			count++
		}
	}
	return count
}

// dropDriverRegistrationsReactor stores CSINode spec updates without spec.driverRegistrations, like
// kube-apiserver with the CSIControllerGetNodeInfo feature gate disabled does for a CSINode that
// has no driver registrations yet.
// Reactors run with the fake client locked, so they use the tracker instead of the client.
func dropDriverRegistrationsReactor(tracker clienttesting.ObjectTracker) clienttesting.ReactionFunc {
	return func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "" {
			return false, nil, nil
		}
		csiNode := action.(clienttesting.UpdateAction).GetObject().(*storage.CSINode).DeepCopy()
		csiNode.Spec.DriverRegistrations = nil
		if err := tracker.Update(storage.SchemeGroupVersion.WithResource("csinodes"), csiNode, ""); err != nil {
			return true, nil, err
		}
		return true, csiNode, nil
	}
}

// invalidCSINodeReactor fails CSINode spec updates like kube-apiserver does when the CSINode
// object is invalid.
func invalidCSINodeReactor(clienttesting.ObjectTracker) clienttesting.ReactionFunc {
	return func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "" {
			return false, nil, nil
		}
		csiNode := action.(clienttesting.UpdateAction).GetObject().(*storage.CSINode)
		return true, nil, errors.NewInvalid(storage.SchemeGroupVersion.WithKind("CSINode").GroupKind(), csiNode.Name, nil)
	}
}

// conflictOnceReactor emulates another writer, such as external-attacher, that changes the stored
// CSINode with update between the Get and the first spec update of the caller. That spec update
// fails with a conflict, like it does in kube-apiserver because of the outdated resourceVersion.
// Reactors run with the fake client locked, so they use the tracker instead of the client.
func conflictOnceReactor(tracker clienttesting.ObjectTracker, update func(*storage.CSINode)) clienttesting.ReactionFunc {
	conflicted := false
	return func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "" {
			return false, nil, nil
		}
		if conflicted {
			return false, nil, nil
		}
		conflicted = true

		resource := storage.SchemeGroupVersion.WithResource("csinodes")
		name := action.(clienttesting.UpdateAction).GetObject().(*storage.CSINode).Name
		obj, err := tracker.Get(resource, "", name)
		if err != nil {
			return true, nil, err
		}
		csiNode := obj.(*storage.CSINode)
		update(csiNode)
		if err := tracker.Update(resource, csiNode, ""); err != nil {
			return true, nil, err
		}
		return true, nil, errors.NewConflict(storage.Resource("csinodes"), name, goerrors.New("the object has been modified"))
	}
}
