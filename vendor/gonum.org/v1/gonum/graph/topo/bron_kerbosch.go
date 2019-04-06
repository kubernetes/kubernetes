// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package topo

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/internal/ordered"
	"gonum.org/v1/gonum/graph/internal/set"
)

// DegeneracyOrdering returns the degeneracy ordering and the k-cores of
// the undirected graph g.
func DegeneracyOrdering(g graph.Undirected) (order []graph.Node, cores [][]graph.Node) {
	order, offsets := degeneracyOrdering(g)

	ordered.Reverse(order)
	cores = make([][]graph.Node, len(offsets))
	offset := len(order)
	for i, n := range offsets {
		cores[i] = order[offset-n : offset]
		offset -= n
	}
	return order, cores
}

// KCore returns the k-core of the undirected graph g with nodes in an
// optimal ordering for the coloring number.
func KCore(k int, g graph.Undirected) []graph.Node {
	order, offsets := degeneracyOrdering(g)

	var offset int
	for _, n := range offsets[:k] {
		offset += n
	}
	core := make([]graph.Node, len(order)-offset)
	copy(core, order[offset:])
	return core
}

// degeneracyOrdering is the common code for DegeneracyOrdering and KCore. It
// returns l, the nodes of g in optimal ordering for coloring number and
// s, a set of relative offsets into l for each k-core, where k is an index
// into s.
func degeneracyOrdering(g graph.Undirected) (l []graph.Node, s []int) {
	nodes := graph.NodesOf(g.Nodes())

	// The algorithm used here is essentially as described at
	// http://en.wikipedia.org/w/index.php?title=Degeneracy_%28graph_theory%29&oldid=640308710

	// Initialize an output list L in return parameters.

	// Compute a number d_v for each vertex v in G,
	// the number of neighbors of v that are not already in L.
	// Initially, these numbers are just the degrees of the vertices.
	dv := make(map[int64]int, len(nodes))
	var (
		maxDegree  int
		neighbours = make(map[int64][]graph.Node)
	)
	for _, n := range nodes {
		id := n.ID()
		adj := graph.NodesOf(g.From(id))
		neighbours[id] = adj
		dv[id] = len(adj)
		if len(adj) > maxDegree {
			maxDegree = len(adj)
		}
	}

	// Initialize an array D such that D[i] contains a list of the
	// vertices v that are not already in L for which d_v = i.
	d := make([][]graph.Node, maxDegree+1)
	for _, n := range nodes {
		deg := dv[n.ID()]
		d[deg] = append(d[deg], n)
	}

	// Initialize k to 0.
	k := 0
	// Repeat n times:
	s = []int{0}
	for range nodes {
		// Scan the array cells D[0], D[1], ... until
		// finding an i for which D[i] is nonempty.
		var (
			i  int
			di []graph.Node
		)
		for i, di = range d {
			if len(di) != 0 {
				break
			}
		}

		// Set k to max(k,i).
		if i > k {
			k = i
			s = append(s, make([]int, k-len(s)+1)...)
		}

		// Select a vertex v from D[i]. Add v to the
		// beginning of L and remove it from D[i].
		var v graph.Node
		v, d[i] = di[len(di)-1], di[:len(di)-1]
		l = append(l, v)
		s[k]++
		delete(dv, v.ID())

		// For each neighbor w of v not already in L,
		// subtract one from d_w and move w to the
		// cell of D corresponding to the new value of d_w.
		for _, w := range neighbours[v.ID()] {
			dw, ok := dv[w.ID()]
			if !ok {
				continue
			}
			for i, n := range d[dw] {
				if n.ID() == w.ID() {
					d[dw][i], d[dw] = d[dw][len(d[dw])-1], d[dw][:len(d[dw])-1]
					dw--
					d[dw] = append(d[dw], w)
					break
				}
			}
			dv[w.ID()] = dw
		}
	}

	return l, s
}

// BronKerbosch returns the set of maximal cliques of the undirected graph g.
func BronKerbosch(g graph.Undirected) [][]graph.Node {
	nodes := graph.NodesOf(g.Nodes())

	// The algorithm used here is essentially BronKerbosch3 as described at
	// http://en.wikipedia.org/w/index.php?title=Bron%E2%80%93Kerbosch_algorithm&oldid=656805858

	p := set.NewNodesSize(len(nodes))
	for _, n := range nodes {
		p.Add(n)
	}
	x := set.NewNodes()
	var bk bronKerbosch
	order, _ := degeneracyOrdering(g)
	ordered.Reverse(order)
	for _, v := range order {
		neighbours := graph.NodesOf(g.From(v.ID()))
		nv := set.NewNodesSize(len(neighbours))
		for _, n := range neighbours {
			nv.Add(n)
		}
		bk.maximalCliquePivot(g, []graph.Node{v}, set.IntersectionOfNodes(p, nv), set.IntersectionOfNodes(x, nv))
		p.Remove(v)
		x.Add(v)
	}
	return bk
}

type bronKerbosch [][]graph.Node

func (bk *bronKerbosch) maximalCliquePivot(g graph.Undirected, r []graph.Node, p, x set.Nodes) {
	if len(p) == 0 && len(x) == 0 {
		*bk = append(*bk, r)
		return
	}

	neighbours := bk.choosePivotFrom(g, p, x)
	nu := set.NewNodesSize(len(neighbours))
	for _, n := range neighbours {
		nu.Add(n)
	}
	for _, v := range p {
		if nu.Has(v) {
			continue
		}
		vid := v.ID()
		neighbours := graph.NodesOf(g.From(vid))
		nv := set.NewNodesSize(len(neighbours))
		for _, n := range neighbours {
			nv.Add(n)
		}

		var found bool
		for _, n := range r {
			if n.ID() == vid {
				found = true
				break
			}
		}
		var sr []graph.Node
		if !found {
			sr = append(r[:len(r):len(r)], v)
		}

		bk.maximalCliquePivot(g, sr, set.IntersectionOfNodes(p, nv), set.IntersectionOfNodes(x, nv))
		p.Remove(v)
		x.Add(v)
	}
}

func (*bronKerbosch) choosePivotFrom(g graph.Undirected, p, x set.Nodes) (neighbors []graph.Node) {
	// TODO(kortschak): Investigate the impact of pivot choice that maximises
	// |p ⋂ neighbours(u)| as a function of input size. Until then, leave as
	// compile time option.
	if !tomitaTanakaTakahashi {
		for _, n := range p {
			return graph.NodesOf(g.From(n.ID()))
		}
		for _, n := range x {
			return graph.NodesOf(g.From(n.ID()))
		}
		panic("bronKerbosch: empty set")
	}

	var (
		max   = -1
		pivot graph.Node
	)
	maxNeighbors := func(s set.Nodes) {
	outer:
		for _, u := range s {
			nb := graph.NodesOf(g.From(u.ID()))
			c := len(nb)
			if c <= max {
				continue
			}
			for n := range nb {
				if _, ok := p[int64(n)]; ok {
					continue
				}
				c--
				if c <= max {
					continue outer
				}
			}
			max = c
			pivot = u
			neighbors = nb
		}
	}
	maxNeighbors(p)
	maxNeighbors(x)
	if pivot == nil {
		panic("bronKerbosch: empty set")
	}
	return neighbors
}
