// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package topo

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/internal/linear"
	"gonum.org/v1/gonum/graph/internal/set"
)

// UndirectedCyclesIn returns a set of cycles that forms a cycle basis in the graph g.
// Any cycle in g can be constructed as a symmetric difference of its elements.
func UndirectedCyclesIn(g graph.Undirected) [][]graph.Node {
	// From "An algorithm for finding a fundamental set of cycles of a graph"
	// https://doi.org/10.1145/363219.363232

	var cycles [][]graph.Node
	done := make(set.Int64s)
	var tree linear.NodeStack
	nodes := g.Nodes()
	for nodes.Next() {
		n := nodes.Node()
		id := n.ID()
		if done.Has(id) {
			continue
		}
		done.Add(id)

		tree = tree[:0]
		tree.Push(n)
		from := sets{id: set.Int64s{}}
		to := map[int64]graph.Node{id: n}

		for tree.Len() != 0 {
			u := tree.Pop()
			uid := u.ID()
			adj := from[uid]
			for _, v := range graph.NodesOf(g.From(uid)) {
				vid := v.ID()
				switch {
				case uid == vid:
					cycles = append(cycles, []graph.Node{u})
				case !from.has(vid):
					done.Add(vid)
					to[vid] = u
					tree.Push(v)
					from.add(uid, vid)
				case !adj.Has(vid):
					c := []graph.Node{v, u}
					adj := from[vid]
					p := to[uid]
					for !adj.Has(p.ID()) {
						c = append(c, p)
						p = to[p.ID()]
					}
					c = append(c, p, c[0])
					cycles = append(cycles, c)
					adj.Add(uid)
				}
			}
		}
	}

	return cycles
}

type sets map[int64]set.Int64s

func (s sets) add(uid, vid int64) {
	e, ok := s[vid]
	if !ok {
		e = make(set.Int64s)
		s[vid] = e
	}
	e.Add(uid)
}

func (s sets) has(uid int64) bool {
	_, ok := s[uid]
	return ok
}
