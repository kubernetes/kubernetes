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
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
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
