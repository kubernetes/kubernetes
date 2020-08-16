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

package node

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestDeleteEdges_locked(t *testing.T) {
	cases := []struct {
		desc        string
		fromType    vertexType
		toType      vertexType
		toNamespace string
		toName      string
		start       *Graph
		expect      *Graph
	}{
		{
			// single edge from a configmap to a node, will delete edge and orphaned configmap
			desc:        "edges and source orphans are deleted, destination orphans are preserved",
			fromType:    configMapVertexType,
			toType:      nodeVertexType,
			toNamespace: "",
			toName:      "node1",
			start: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap2")
				nodeVertex := g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				configmapVertex := g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				g.graph.SetEdge(newDestinationEdge(configmapVertex, nodeVertex, nodeVertex))
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap2")
				g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				return g
			}(),
		},
		{
			// two edges from the same configmap to distinct nodes, will delete one of the edges
			desc:        "edges are deleted, non-orphans and destination orphans are preserved",
			fromType:    configMapVertexType,
			toType:      nodeVertexType,
			toNamespace: "",
			toName:      "node2",
			start: func() *Graph {
				g := NewGraph()
				nodeVertex1 := g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				nodeVertex2 := g.getOrCreateVertex_locked(nodeVertexType, "", "node2")
				configmapVertex := g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				g.graph.SetEdge(newDestinationEdge(configmapVertex, nodeVertex1, nodeVertex1))
				g.graph.SetEdge(newDestinationEdge(configmapVertex, nodeVertex2, nodeVertex2))
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				nodeVertex1 := g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				g.getOrCreateVertex_locked(nodeVertexType, "", "node2")
				configmapVertex := g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				g.graph.SetEdge(newDestinationEdge(configmapVertex, nodeVertex1, nodeVertex1))
				return g
			}(),
		},
		{
			desc:        "no edges to delete",
			fromType:    configMapVertexType,
			toType:      nodeVertexType,
			toNamespace: "",
			toName:      "node1",
			start: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
		},
		{
			desc:        "destination vertex does not exist",
			fromType:    configMapVertexType,
			toType:      nodeVertexType,
			toNamespace: "",
			toName:      "node1",
			start: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
		},
		{
			desc:        "source vertex type doesn't exist",
			fromType:    configMapVertexType,
			toType:      nodeVertexType,
			toNamespace: "",
			toName:      "node1",
			start: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertex_locked(nodeVertexType, "", "node1")
				return g
			}(),
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.start.deleteEdges_locked(c.fromType, c.toType, c.toNamespace, c.toName)

			// Note: We assert on substructures (graph.Nodes(), graph.Edges()) because the graph tracks
			// freed IDs for reuse, which results in an irrelevant inequality between start and expect.

			// sort the nodes by ID
			// (the slices we get back are from map iteration, where order is not guaranteed)
			expectNodes := c.expect.graph.Nodes()
			sort.Slice(expectNodes, func(i, j int) bool {
				return expectNodes[i].ID() < expectNodes[j].ID()
			})
			startNodes := c.start.graph.Nodes()
			sort.Slice(startNodes, func(i, j int) bool {
				return startNodes[i].ID() < startNodes[j].ID()
			})
			assert.Equal(t, expectNodes, startNodes)

			// sort the edges by from ID, then to ID
			// (the slices we get back are from map iteration, where order is not guaranteed)
			expectEdges := c.expect.graph.Edges()
			sort.Slice(expectEdges, func(i, j int) bool {
				if expectEdges[i].From().ID() == expectEdges[j].From().ID() {
					return expectEdges[i].To().ID() < expectEdges[j].To().ID()
				}
				return expectEdges[i].From().ID() < expectEdges[j].From().ID()
			})
			startEdges := c.start.graph.Edges()
			sort.Slice(startEdges, func(i, j int) bool {
				if startEdges[i].From().ID() == startEdges[j].From().ID() {
					return startEdges[i].To().ID() < startEdges[j].To().ID()
				}
				return startEdges[i].From().ID() < startEdges[j].From().ID()
			})
			assert.Equal(t, expectEdges, startEdges)

			// vertices is a recursive map, no need to sort
			assert.Equal(t, c.expect.vertices, c.start.vertices)
		})
	}
}

func TestIndex(t *testing.T) {
	g := NewGraph()
	g.destinationEdgeThreshold = 3

	a := NewAuthorizer(g, nil, nil)

	addPod := func(podNumber, nodeNumber int) {
		t.Helper()
		nodeName := fmt.Sprintf("node%d", nodeNumber)
		podName := fmt.Sprintf("pod%d", podNumber)
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "ns", UID: types.UID(fmt.Sprintf("pod%duid1", podNumber))},
			Spec: corev1.PodSpec{
				NodeName:                 nodeName,
				ServiceAccountName:       "sa1",
				DeprecatedServiceAccount: "sa1",
				Volumes: []corev1.Volume{
					{Name: "volume1", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "cm1"}}}},
					{Name: "volume2", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "cm2"}}}},
					{Name: "volume3", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "cm3"}}}},
				},
			},
		}
		g.AddPod(pod)
		if ok, err := a.hasPathFrom(nodeName, configMapVertexType, "ns", "cm1"); err != nil || !ok {
			t.Errorf("expected path from %s to cm1, got %v, %v", nodeName, ok, err)
		}
	}

	toString := func(id int) string {
		for _, namespaceName := range g.vertices {
			for _, nameVertex := range namespaceName {
				for _, vertex := range nameVertex {
					if vertex.id == id {
						return vertex.String()
					}
				}
			}
		}
		return ""
	}
	expectGraph := func(expect map[string][]string) {
		t.Helper()
		actual := map[string][]string{}
		for _, node := range g.graph.Nodes() {
			sortedTo := []string{}
			for _, to := range g.graph.From(node) {
				sortedTo = append(sortedTo, toString(to.ID()))
			}
			sort.Strings(sortedTo)
			actual[toString(node.ID())] = sortedTo
		}
		if !reflect.DeepEqual(expect, actual) {
			e, _ := json.MarshalIndent(expect, "", "  ")
			a, _ := json.MarshalIndent(actual, "", "  ")
			t.Errorf("expected graph:\n%s\ngot:\n%s", string(e), string(a))
		}
	}
	expectIndex := func(expect map[string][]string) {
		t.Helper()
		actual := map[string][]string{}
		for from, to := range g.destinationEdgeIndex {
			sortedValues := []string{}
			for member, count := range to.members {
				sortedValues = append(sortedValues, fmt.Sprintf("%s=%d", toString(member), count))
			}
			sort.Strings(sortedValues)
			actual[toString(from)] = sortedValues
		}
		if !reflect.DeepEqual(expect, actual) {
			e, _ := json.MarshalIndent(expect, "", "  ")
			a, _ := json.MarshalIndent(actual, "", "  ")
			t.Errorf("expected index:\n%s\ngot:\n%s", string(e), string(a))
		}
	}

	for i := 1; i <= g.destinationEdgeThreshold; i++ {
		addPod(i, i)
		if i < g.destinationEdgeThreshold {
			// if we're under the threshold, no index expected
			expectIndex(map[string][]string{})
		}
	}
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"pod:ns/pod1":           {"node:node1"},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"configmap:ns/cm1":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
		"configmap:ns/cm2":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
		"configmap:ns/cm3":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
		"serviceAccount:ns/sa1": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
	})

	// delete one to drop below the threshold
	g.DeletePod("pod1", "ns")
	expectGraph(map[string][]string{
		"node:node2":            {},
		"node:node3":            {},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"configmap:ns/cm1":      {"pod:ns/pod2", "pod:ns/pod3"},
		"configmap:ns/cm2":      {"pod:ns/pod2", "pod:ns/pod3"},
		"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3"},
		"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3"},
	})
	expectIndex(map[string][]string{})

	// add two to get above the threshold
	addPod(1, 1)
	addPod(4, 1)
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"pod:ns/pod1":           {"node:node1"},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"pod:ns/pod4":           {"node:node1"},
		"configmap:ns/cm1":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm2":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm3":      {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"serviceAccount:ns/sa1": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=2", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm2":      {"node:node1=2", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=2", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=2", "node:node2=1", "node:node3=1"},
	})

	// delete one to remain above the threshold
	g.DeletePod("pod1", "ns")
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"pod:ns/pod4":           {"node:node1"},
		"configmap:ns/cm1":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm2":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
	})

	// Set node->configmap references
	g.SetNodeConfigMap("node1", "cm1", "ns")
	g.SetNodeConfigMap("node2", "cm1", "ns")
	g.SetNodeConfigMap("node3", "cm1", "ns")
	g.SetNodeConfigMap("node4", "cm1", "ns")
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"node:node4":            {},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"pod:ns/pod4":           {"node:node1"},
		"configmap:ns/cm1":      {"node:node1", "node:node2", "node:node3", "node:node4", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm2":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=2", "node:node2=2", "node:node3=2", "node:node4=1"},
		"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
	})

	// Update node->configmap reference
	g.SetNodeConfigMap("node1", "cm2", "ns")
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"node:node4":            {},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"pod:ns/pod4":           {"node:node1"},
		"configmap:ns/cm1":      {"node:node2", "node:node3", "node:node4", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm2":      {"node:node1", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=1", "node:node2=2", "node:node3=2", "node:node4=1"},
		"configmap:ns/cm2":      {"node:node1=2", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
	})

	// Remove node->configmap reference
	g.SetNodeConfigMap("node1", "", "")
	g.SetNodeConfigMap("node4", "", "")
	expectGraph(map[string][]string{
		"node:node1":            {},
		"node:node2":            {},
		"node:node3":            {},
		"node:node4":            {},
		"pod:ns/pod2":           {"node:node2"},
		"pod:ns/pod3":           {"node:node3"},
		"pod:ns/pod4":           {"node:node1"},
		"configmap:ns/cm1":      {"node:node2", "node:node3", "pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm2":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
		"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3", "pod:ns/pod4"},
	})
	expectIndex(map[string][]string{
		"configmap:ns/cm1":      {"node:node1=1", "node:node2=2", "node:node3=2"},
		"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
		"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
	})
}
