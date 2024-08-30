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

func TestDeleteEdgesLocked(t *testing.T) {
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
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap2")
				nodeVertex := g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				configmapVertex := g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
				g.addEdgeLocked(configmapVertex, nodeVertex, nodeVertex)
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap2")
				g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
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
				nodeVertex1 := g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				nodeVertex2 := g.getOrCreateVertexLocked(nodeVertexType, "", "node2")
				configmapVertex := g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
				g.addEdgeLocked(configmapVertex, nodeVertex1, nodeVertex1)
				g.addEdgeLocked(configmapVertex, nodeVertex2, nodeVertex2)
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				nodeVertex1 := g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				g.getOrCreateVertexLocked(nodeVertexType, "", "node2")
				configmapVertex := g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
				g.addEdgeLocked(configmapVertex, nodeVertex1, nodeVertex1)
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
				g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
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
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertexLocked(configMapVertexType, "namespace1", "configmap1")
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
				g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				return g
			}(),
			expect: func() *Graph {
				g := NewGraph()
				g.getOrCreateVertexLocked(nodeVertexType, "", "node1")
				return g
			}(),
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.start.deleteEdgesLocked(c.fromType, c.toType, c.toNamespace, c.toName)

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
}

func TestIndex2(t *testing.T) {
	NewTestGraph := func() *Graph {
		g := NewGraph()
		g.destinationEdgeThreshold = 3
		return g
	}

	pod := func(podName, nodeName, saName string, volumes []corev1.Volume, resourceClaims []corev1.PodResourceClaim) *corev1.Pod {
		p := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "ns", UID: types.UID(fmt.Sprintf("pod%suid", podName))},
			Spec: corev1.PodSpec{
				NodeName: nodeName,
			},
		}
		if saName != "" {
			p.Spec.ServiceAccountName = saName
		}
		if volumes != nil {
			p.Spec.Volumes = volumes
		}
		if resourceClaims != nil {
			p.Spec.ResourceClaims = resourceClaims
		}
		return p
	}

	podWithSAAndCMs := func(podName, nodeName string) *corev1.Pod {
		cm := func(name string) corev1.Volume {
			return corev1.Volume{Name: name, VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: name}}}}
		}
		return pod(podName, nodeName, "sa1", []corev1.Volume{
			cm("cm1"),
			cm("cm2"),
			cm("cm3"),
		}, nil)
	}

	podWithSecrets := func(podName, nodeName string) *corev1.Pod {
		secret := func(name string) corev1.Volume {
			return corev1.Volume{Name: name, VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: name}}}
		}
		return pod(podName, nodeName, "", []corev1.Volume{
			secret("s1"),
			secret("s2"),
			secret("s3"),
		}, nil)
	}

	podWithPVCs := func(podName, nodeName string) *corev1.Pod {
		pvc := func(name string) corev1.Volume {
			return corev1.Volume{Name: name, VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: name}}}
		}
		return pod(podName, nodeName, "", []corev1.Volume{
			pvc("pvc1"),
			pvc("pvc2"),
			pvc("pvc3"),
		}, nil)
	}

	podWithResourceClaims := func(podName, nodeName string) *corev1.Pod {
		rc := func(name string) corev1.PodResourceClaim {
			return corev1.PodResourceClaim{ResourceClaimName: &name}
		}
		return pod(podName, nodeName, "", nil, []corev1.PodResourceClaim{
			rc("rc1"),
			rc("rc2"),
			rc("rc3"),
		})
	}

	pv := func(pvName, pvcName, secretName string) *corev1.PersistentVolume {
		pv := &corev1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{Name: pvName, UID: types.UID(fmt.Sprintf("pv%suid", pvName))},
			Spec: corev1.PersistentVolumeSpec{
				ClaimRef: &corev1.ObjectReference{
					Name:      pvcName,
					Namespace: "ns",
				},
			},
		}
		if secretName != "" {
			pv.Spec.PersistentVolumeSource = corev1.PersistentVolumeSource{
				CSI: &corev1.CSIPersistentVolumeSource{
					NodePublishSecretRef: &corev1.SecretReference{
						Name:      secretName,
						Namespace: "ns",
					},
				},
			}
		}
		return pv
	}

	toString := func(g *Graph, id int) string {
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
	expectGraph := func(t *testing.T, g *Graph, expect map[string][]string) {
		t.Helper()
		actual := map[string][]string{}
		for _, node := range g.graph.Nodes() {
			sortedTo := []string{}
			for _, to := range g.graph.From(node) {
				sortedTo = append(sortedTo, toString(g, to.ID()))
			}
			sort.Strings(sortedTo)
			actual[toString(g, node.ID())] = sortedTo
		}
		if !reflect.DeepEqual(expect, actual) {
			e, _ := json.MarshalIndent(expect, "", "  ")
			a, _ := json.MarshalIndent(actual, "", "  ")
			t.Errorf("expected graph:\n%s\ngot:\n%s", string(e), string(a))
		}
	}
	expectIndex := func(t *testing.T, g *Graph, expect map[string][]string) {
		t.Helper()
		actual := map[string][]string{}
		for from, to := range g.destinationEdgeIndex {
			sortedValues := []string{}
			for member, count := range to.members {
				sortedValues = append(sortedValues, fmt.Sprintf("%s=%d", toString(g, member), count))
			}
			sort.Strings(sortedValues)
			actual[toString(g, from)] = sortedValues
		}
		if !reflect.DeepEqual(expect, actual) {
			e, _ := json.MarshalIndent(expect, "", "  ")
			a, _ := json.MarshalIndent(actual, "", "  ")
			t.Errorf("expected index:\n%s\ngot:\n%s", string(e), string(a))
		}
	}

	cases := []struct {
		desc             string
		startingGraph    *Graph
		graphTransformer func(*Graph)
		expectedGraph    map[string][]string
		expectedIndex    map[string][]string
	}{
		{
			desc:             "empty graph",
			startingGraph:    NewTestGraph(),
			graphTransformer: func(_ *Graph) {},
			expectedGraph:    map[string][]string{},
			expectedIndex:    map[string][]string{},
		},
		{
			desc:          "outdeg below destination edge index threshold",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddPod(podWithSAAndCMs("pod1", "node1"))
				g.AddPod(podWithSAAndCMs("pod2", "node2"))
			},
			expectedGraph: map[string][]string{
				"node:node1":            {},
				"node:node2":            {},
				"pod:ns/pod1":           {"node:node1"},
				"pod:ns/pod2":           {"node:node2"},
				"configmap:ns/cm1":      {"pod:ns/pod1", "pod:ns/pod2"},
				"configmap:ns/cm2":      {"pod:ns/pod1", "pod:ns/pod2"},
				"configmap:ns/cm3":      {"pod:ns/pod1", "pod:ns/pod2"},
				"serviceAccount:ns/sa1": {"pod:ns/pod1", "pod:ns/pod2"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "index built for configmaps and serviceaccounts",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithSAAndCMs("pod1", "node1"))
				g.AddPod(podWithSAAndCMs("pod2", "node2"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.AddPod(podWithSAAndCMs("pod3", "node3"))
			},
			expectedGraph: map[string][]string{
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
			},
			expectedIndex: map[string][]string{
				"configmap:ns/cm1":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
			},
		},
		{
			desc: "no index for configmaps and serviceaccounts - dropping below threshold",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithSAAndCMs("pod1", "node1"))
				g.AddPod(podWithSAAndCMs("pod2", "node2"))
				g.AddPod(podWithSAAndCMs("pod3", "node3"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePod("pod1", "ns")
			},
			expectedGraph: map[string][]string{
				"node:node2":            {},
				"node:node3":            {},
				"pod:ns/pod2":           {"node:node2"},
				"pod:ns/pod3":           {"node:node3"},
				"configmap:ns/cm1":      {"pod:ns/pod2", "pod:ns/pod3"},
				"configmap:ns/cm2":      {"pod:ns/pod2", "pod:ns/pod3"},
				"configmap:ns/cm3":      {"pod:ns/pod2", "pod:ns/pod3"},
				"serviceAccount:ns/sa1": {"pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "removing pod but staying above threshold",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithSAAndCMs("pod1", "node1"))
				g.AddPod(podWithSAAndCMs("pod2", "node2"))
				g.AddPod(podWithSAAndCMs("pod3", "node3"))
				g.AddPod(podWithSAAndCMs("pod4", "node1"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePod("pod1", "ns")
			},
			expectedGraph: map[string][]string{
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
			},
			expectedIndex: map[string][]string{
				"configmap:ns/cm1":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"configmap:ns/cm2":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"configmap:ns/cm3":      {"node:node1=1", "node:node2=1", "node:node3=1"},
				"serviceAccount:ns/sa1": {"node:node1=1", "node:node2=1", "node:node3=1"},
			},
		},
		{
			desc: "index built for secrets",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithSecrets("pod1", "node1"))
				g.AddPod(podWithSecrets("pod2", "node2"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.AddPod(podWithSecrets("pod3", "node3"))
			},
			expectedGraph: map[string][]string{
				"node:node1":   {},
				"node:node2":   {},
				"node:node3":   {},
				"pod:ns/pod1":  {"node:node1"},
				"pod:ns/pod2":  {"node:node2"},
				"pod:ns/pod3":  {"node:node3"},
				"secret:ns/s1": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"secret:ns/s2": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"secret:ns/s3": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{
				"secret:ns/s1": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"secret:ns/s2": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"secret:ns/s3": {"node:node1=1", "node:node2=1", "node:node3=1"},
			},
		},
		{
			desc: "no index for secrets - dropping below threshold",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithSecrets("pod1", "node1"))
				g.AddPod(podWithSecrets("pod2", "node2"))
				g.AddPod(podWithSecrets("pod3", "node3"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePod("pod1", "ns")
			},
			expectedGraph: map[string][]string{
				"node:node2":   {},
				"node:node3":   {},
				"pod:ns/pod2":  {"node:node2"},
				"pod:ns/pod3":  {"node:node3"},
				"secret:ns/s1": {"pod:ns/pod2", "pod:ns/pod3"},
				"secret:ns/s2": {"pod:ns/pod2", "pod:ns/pod3"},
				"secret:ns/s3": {"pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "index built for pvcs",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithPVCs("pod1", "node1"))
				g.AddPod(podWithPVCs("pod2", "node2"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.AddPod(podWithPVCs("pod3", "node3"))
			},
			expectedGraph: map[string][]string{
				"node:node1":  {},
				"node:node2":  {},
				"node:node3":  {},
				"pod:ns/pod1": {"node:node1"},
				"pod:ns/pod2": {"node:node2"},
				"pod:ns/pod3": {"node:node3"},
				"pvc:ns/pvc1": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"pvc:ns/pvc2": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"pvc:ns/pvc3": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{
				"pvc:ns/pvc1": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"pvc:ns/pvc2": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"pvc:ns/pvc3": {"node:node1=1", "node:node2=1", "node:node3=1"},
			},
		},
		{
			desc: "no index for pvcs - dropping below threshold",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithPVCs("pod1", "node1"))
				g.AddPod(podWithPVCs("pod2", "node2"))
				g.AddPod(podWithPVCs("pod3", "node3"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePod("pod1", "ns")
			},
			expectedGraph: map[string][]string{
				"node:node2":  {},
				"node:node3":  {},
				"pod:ns/pod2": {"node:node2"},
				"pod:ns/pod3": {"node:node3"},
				"pvc:ns/pvc1": {"pod:ns/pod2", "pod:ns/pod3"},
				"pvc:ns/pvc2": {"pod:ns/pod2", "pod:ns/pod3"},
				"pvc:ns/pvc3": {"pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc:          "index built for resourceclaims",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddPod(podWithResourceClaims("pod1", "node1"))
				g.AddPod(podWithResourceClaims("pod2", "node2"))
				g.AddPod(podWithResourceClaims("pod3", "node3"))
			},
			expectedGraph: map[string][]string{
				"node:node1":           {},
				"node:node2":           {},
				"node:node3":           {},
				"pod:ns/pod1":          {"node:node1"},
				"pod:ns/pod2":          {"node:node2"},
				"pod:ns/pod3":          {"node:node3"},
				"resourceclaim:ns/rc1": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"resourceclaim:ns/rc2": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
				"resourceclaim:ns/rc3": {"pod:ns/pod1", "pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{
				"resourceclaim:ns/rc1": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"resourceclaim:ns/rc2": {"node:node1=1", "node:node2=1", "node:node3=1"},
				"resourceclaim:ns/rc3": {"node:node1=1", "node:node2=1", "node:node3=1"},
			},
		},
		{
			desc: "no index for resourceclaims - dropping below threshold",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPod(podWithResourceClaims("pod1", "node1"))
				g.AddPod(podWithResourceClaims("pod2", "node2"))
				g.AddPod(podWithResourceClaims("pod3", "node3"))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePod("pod1", "ns")
			},
			expectedGraph: map[string][]string{
				"node:node2":           {},
				"node:node3":           {},
				"pod:ns/pod2":          {"node:node2"},
				"pod:ns/pod3":          {"node:node3"},
				"resourceclaim:ns/rc1": {"pod:ns/pod2", "pod:ns/pod3"},
				"resourceclaim:ns/rc2": {"pod:ns/pod2", "pod:ns/pod3"},
				"resourceclaim:ns/rc3": {"pod:ns/pod2", "pod:ns/pod3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc:          "resourceslices adding",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddResourceSlice("s1", "node1")
				g.AddResourceSlice("s2", "node2")
				g.AddResourceSlice("s3", "node3")
			},
			expectedGraph: map[string][]string{
				"node:node1":       {},
				"node:node2":       {},
				"node:node3":       {},
				"resourceslice:s1": {"node:node1"},
				"resourceslice:s2": {"node:node2"},
				"resourceslice:s3": {"node:node3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "resourceslices deleting",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddResourceSlice("s1", "node1")
				g.AddResourceSlice("s2", "node2")
				g.AddResourceSlice("s3", "node3")
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeleteResourceSlice("s1")
			},
			expectedGraph: map[string][]string{
				"node:node2":       {},
				"node:node3":       {},
				"resourceslice:s2": {"node:node2"},
				"resourceslice:s3": {"node:node3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc:          "volumeattachments adding",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddVolumeAttachment("va1", "node1")
				g.AddVolumeAttachment("va2", "node2")
				g.AddVolumeAttachment("va3", "node3")
			},
			expectedGraph: map[string][]string{
				"node:node1":           {},
				"node:node2":           {},
				"node:node3":           {},
				"volumeattachment:va1": {"node:node1"},
				"volumeattachment:va2": {"node:node2"},
				"volumeattachment:va3": {"node:node3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "volumeattachments deleting",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddVolumeAttachment("va1", "node1")
				g.AddVolumeAttachment("va2", "node2")
				g.AddVolumeAttachment("va3", "node3")
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeleteVolumeAttachment("va1")
			},
			expectedGraph: map[string][]string{
				"node:node2":           {},
				"node:node3":           {},
				"volumeattachment:va2": {"node:node2"},
				"volumeattachment:va3": {"node:node3"},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc:          "persistentvolumes adding",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddPV(pv("pv1", "pvc1", ""))
				g.AddPV(pv("pv2", "pvc2", ""))
				g.AddPV(pv("pv3", "pvc3", ""))
			},
			expectedGraph: map[string][]string{
				"pv:pv1":      {"pvc:ns/pvc1"},
				"pv:pv2":      {"pvc:ns/pvc2"},
				"pv:pv3":      {"pvc:ns/pvc3"},
				"pvc:ns/pvc1": {},
				"pvc:ns/pvc2": {},
				"pvc:ns/pvc3": {},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc: "persistentvolumes deleting",
			startingGraph: func() *Graph {
				g := NewTestGraph()
				g.AddPV(pv("pv1", "pvc1", ""))
				g.AddPV(pv("pv2", "pvc2", ""))
				g.AddPV(pv("pv3", "pvc3", ""))
				return g
			}(),
			graphTransformer: func(g *Graph) {
				g.DeletePV("pv1")
			},
			expectedGraph: map[string][]string{
				"pv:pv2":      {"pvc:ns/pvc2"},
				"pv:pv3":      {"pvc:ns/pvc3"},
				"pvc:ns/pvc2": {},
				"pvc:ns/pvc3": {},
			},
			expectedIndex: map[string][]string{},
		},
		{
			desc:          "persistentvolumes with secrets",
			startingGraph: NewTestGraph(),
			graphTransformer: func(g *Graph) {
				g.AddPV(pv("pv1", "pvc1", "s1"))
				g.AddPV(pv("pv2", "pvc2", "s2"))
				g.AddPV(pv("pv3", "pvc3", "s3"))
			},
			expectedGraph: map[string][]string{
				"pv:pv1":       {"pvc:ns/pvc1"},
				"pv:pv2":       {"pvc:ns/pvc2"},
				"pv:pv3":       {"pvc:ns/pvc3"},
				"pvc:ns/pvc1":  {},
				"pvc:ns/pvc2":  {},
				"pvc:ns/pvc3":  {},
				"secret:ns/s1": {"pv:pv1"},
				"secret:ns/s2": {"pv:pv2"},
				"secret:ns/s3": {"pv:pv3"},
			},
			expectedIndex: map[string][]string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			tc.graphTransformer(tc.startingGraph)
			expectGraph(t, tc.startingGraph, tc.expectedGraph)
			expectIndex(t, tc.startingGraph, tc.expectedIndex)
		})
	}
}
