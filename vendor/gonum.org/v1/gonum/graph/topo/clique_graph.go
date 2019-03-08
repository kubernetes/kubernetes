// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package topo

import (
	"sort"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/internal/ordered"
	"gonum.org/v1/gonum/graph/internal/set"
)

// Builder is a pure topological graph construction type.
type Builder interface {
	AddNode(graph.Node)
	SetEdge(graph.Edge)
}

// CliqueGraph builds the clique graph of g in dst using Clique and CliqueGraphEdge
// nodes and edges. The nodes returned by calls to Nodes on the nodes and edges of
// the constructed graph are the cliques and the common nodes between cliques
// respectively. The dst graph is not cleared.
func CliqueGraph(dst Builder, g graph.Undirected) {
	cliques := BronKerbosch(g)

	// Construct a consistent view of cliques in g. Sorting costs
	// us a little, but not as much as the cliques themselves.
	for _, c := range cliques {
		sort.Sort(ordered.ByID(c))
	}
	sort.Sort(ordered.BySliceIDs(cliques))

	cliqueNodes := make(cliqueNodeSets, len(cliques))
	for id, c := range cliques {
		s := make(set.Nodes, len(c))
		for _, n := range c {
			s.Add(n)
		}
		ns := &nodeSet{Clique: Clique{id: int64(id), nodes: c}, nodes: s}
		dst.AddNode(ns.Clique)
		for _, n := range c {
			nid := n.ID()
			cliqueNodes[nid] = append(cliqueNodes[nid], ns)
		}
	}

	for _, cliques := range cliqueNodes {
		for i, uc := range cliques {
			for _, vc := range cliques[i+1:] {
				// Retain the nodes that contribute to the
				// edge between the cliques.
				var edgeNodes []graph.Node
				switch 1 {
				case len(uc.Clique.nodes):
					edgeNodes = []graph.Node{uc.Clique.nodes[0]}
				case len(vc.Clique.nodes):
					edgeNodes = []graph.Node{vc.Clique.nodes[0]}
				default:
					for _, n := range make(set.Nodes).Intersect(uc.nodes, vc.nodes) {
						edgeNodes = append(edgeNodes, n)
					}
					sort.Sort(ordered.ByID(edgeNodes))
				}

				dst.SetEdge(CliqueGraphEdge{from: uc.Clique, to: vc.Clique, nodes: edgeNodes})
			}
		}
	}
}

type cliqueNodeSets map[int64][]*nodeSet

type nodeSet struct {
	Clique
	nodes set.Nodes
}

// Clique is a node in a clique graph.
type Clique struct {
	id    int64
	nodes []graph.Node
}

// ID returns the node ID.
func (n Clique) ID() int64 { return n.id }

// Nodes returns the nodes in the clique.
func (n Clique) Nodes() []graph.Node { return n.nodes }

// CliqueGraphEdge is an edge in a clique graph.
type CliqueGraphEdge struct {
	from, to Clique
	nodes    []graph.Node
}

// From returns the from node of the edge.
func (e CliqueGraphEdge) From() graph.Node { return e.from }

// To returns the to node of the edge.
func (e CliqueGraphEdge) To() graph.Node { return e.to }

// Nodes returns the common nodes in the cliques of the underlying graph
// corresponding to the from and to nodes in the clique graph.
func (e CliqueGraphEdge) Nodes() []graph.Node { return e.nodes }
