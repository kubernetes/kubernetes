// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package topo

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/internal/ordered"
	"gonum.org/v1/gonum/graph/internal/set"
)

// Unorderable is an error containing sets of unorderable graph.Nodes.
type Unorderable [][]graph.Node

// Error satisfies the error interface.
func (e Unorderable) Error() string {
	const maxNodes = 10
	var n int
	for _, c := range e {
		n += len(c)
	}
	if n > maxNodes {
		// Don't return errors that are too long.
		return fmt.Sprintf("topo: no topological ordering: %d nodes in %d cyclic components", n, len(e))
	}
	return fmt.Sprintf("topo: no topological ordering: cyclic components: %v", [][]graph.Node(e))
}

func lexical(nodes []graph.Node) { sort.Sort(ordered.ByID(nodes)) }

// Sort performs a topological sort of the directed graph g returning the 'from' to 'to'
// sort order. If a topological ordering is not possible, an Unorderable error is returned
// listing cyclic components in g with each cyclic component's members sorted by ID. When
// an Unorderable error is returned, each cyclic component's topological position within
// the sorted nodes is marked with a nil graph.Node.
func Sort(g graph.Directed) (sorted []graph.Node, err error) {
	sccs := TarjanSCC(g)
	return sortedFrom(sccs, lexical)
}

// SortStabilized performs a topological sort of the directed graph g returning the 'from'
// to 'to' sort order, or the order defined by the in place order sort function where there
// is no unambiguous topological ordering. If a topological ordering is not possible, an
// Unorderable error is returned listing cyclic components in g with each cyclic component's
// members sorted by the provided order function. If order is nil, nodes are ordered lexically
// by node ID. When an Unorderable error is returned, each cyclic component's topological
// position within the sorted nodes is marked with a nil graph.Node.
func SortStabilized(g graph.Directed, order func([]graph.Node)) (sorted []graph.Node, err error) {
	if order == nil {
		order = lexical
	}
	sccs := tarjanSCCstabilized(g, order)
	return sortedFrom(sccs, order)
}

func sortedFrom(sccs [][]graph.Node, order func([]graph.Node)) ([]graph.Node, error) {
	sorted := make([]graph.Node, 0, len(sccs))
	var sc Unorderable
	for _, s := range sccs {
		if len(s) != 1 {
			order(s)
			sc = append(sc, s)
			sorted = append(sorted, nil)
			continue
		}
		sorted = append(sorted, s[0])
	}
	var err error
	if sc != nil {
		for i, j := 0, len(sc)-1; i < j; i, j = i+1, j-1 {
			sc[i], sc[j] = sc[j], sc[i]
		}
		err = sc
	}
	ordered.Reverse(sorted)
	return sorted, err
}

// TarjanSCC returns the strongly connected components of the graph g using Tarjan's algorithm.
//
// A strongly connected component of a graph is a set of vertices where it's possible to reach any
// vertex in the set from any other (meaning there's a cycle between them.)
//
// Generally speaking, a directed graph where the number of strongly connected components is equal
// to the number of nodes is acyclic, unless you count reflexive edges as a cycle (which requires
// only a little extra testing.)
//
func TarjanSCC(g graph.Directed) [][]graph.Node {
	return tarjanSCCstabilized(g, nil)
}

func tarjanSCCstabilized(g graph.Directed, order func([]graph.Node)) [][]graph.Node {
	nodes := graph.NodesOf(g.Nodes())
	var succ func(id int64) []graph.Node
	if order == nil {
		succ = func(id int64) []graph.Node {
			return graph.NodesOf(g.From(id))
		}
	} else {
		order(nodes)
		ordered.Reverse(nodes)

		succ = func(id int64) []graph.Node {
			to := graph.NodesOf(g.From(id))
			order(to)
			ordered.Reverse(to)
			return to
		}
	}

	t := tarjan{
		succ: succ,

		indexTable: make(map[int64]int, len(nodes)),
		lowLink:    make(map[int64]int, len(nodes)),
		onStack:    make(set.Int64s),
	}
	for _, v := range nodes {
		if t.indexTable[v.ID()] == 0 {
			t.strongconnect(v)
		}
	}
	return t.sccs
}

// tarjan implements Tarjan's strongly connected component finding
// algorithm. The implementation is from the pseudocode at
//
// http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm?oldid=642744644
//
type tarjan struct {
	succ func(id int64) []graph.Node

	index      int
	indexTable map[int64]int
	lowLink    map[int64]int
	onStack    set.Int64s

	stack []graph.Node

	sccs [][]graph.Node
}

// strongconnect is the strongconnect function described in the
// wikipedia article.
func (t *tarjan) strongconnect(v graph.Node) {
	vID := v.ID()

	// Set the depth index for v to the smallest unused index.
	t.index++
	t.indexTable[vID] = t.index
	t.lowLink[vID] = t.index
	t.stack = append(t.stack, v)
	t.onStack.Add(vID)

	// Consider successors of v.
	for _, w := range t.succ(vID) {
		wID := w.ID()
		if t.indexTable[wID] == 0 {
			// Successor w has not yet been visited; recur on it.
			t.strongconnect(w)
			t.lowLink[vID] = min(t.lowLink[vID], t.lowLink[wID])
		} else if t.onStack.Has(wID) {
			// Successor w is in stack s and hence in the current SCC.
			t.lowLink[vID] = min(t.lowLink[vID], t.indexTable[wID])
		}
	}

	// If v is a root node, pop the stack and generate an SCC.
	if t.lowLink[vID] == t.indexTable[vID] {
		// Start a new strongly connected component.
		var (
			scc []graph.Node
			w   graph.Node
		)
		for {
			w, t.stack = t.stack[len(t.stack)-1], t.stack[:len(t.stack)-1]
			t.onStack.Remove(w.ID())
			// Add w to current strongly connected component.
			scc = append(scc, w)
			if w.ID() == vID {
				break
			}
		}
		// Output the current strongly connected component.
		t.sccs = append(t.sccs, scc)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
