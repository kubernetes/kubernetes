package simple

import "k8s.io/kubernetes/third_party/forked/gonum/graph"

// This code is in a frequently-hit code path (node authorizer), and
// interface calls were looming large in the profile.  But we also
// want to keep memory usage sensible.  The majority of the interface
// calls were to establish the direction of edges, because we only
// want to visit "out" edges in the node authorizer (grant, not
// grantee).  So the sliceEdgeHolder and mapEdgeHolder memoize their
// neighbor, which saves most of the calls.  But the hot code path
// still needs to walk and check direction, and so
// splitDirectionEdgeHolder keeps the from & to edges in two separate
// lists.  Then we can convert the directional VisitEdgesFromSelf call
// into a non-directional Visit call on the edgsFromSelf list. This
// avoids most of the interface calls, and is still reasonably memory
// efficient.  For leaf splitDirectionEdgeHolders, one of those lists
// is going to be nil anyway.

// edgeHolder represents a set of edges, with no more than one edge to or from a particular neighbor node
type edgeHolder interface {
	// VisitEdges invokes visitor for each edge, pasing the edge and the id of the neighbor node in the edge
	VisitEdges(visitor func(neighbor int, edge graph.Edge))
	// VisitEdgesFromSelf invokes visitor for each edge that is directed from self
	VisitEdgesFromSelf(visitor func(toNodeID int, edge graph.Edge))
	// VisitEdgesToSelf invokes visitor for each edge that is directed to self
	VisitEdgesToSelf(visitor func(fromNodeID int, edge graph.Edge))
	// Delete removes edges to or from the specified neighbor
	Delete(neighbor int) edgeHolder
	// Set stores the edge to or from the specified neighbor
	Set(neighbor int, edge graph.Edge) edgeHolder
	// Get returns the edge to or from the specified neighbor
	Get(neighbor int) (graph.Edge, bool)
	// Len returns the number of edges
	Len() int
}

// edgeInfo caches the neighbor, avoid interface calls From()/To()/ID()
type edgeInfo struct {
	neighbor int
	edge     graph.Edge
}

// sliceEdgeHolder holds a list of edges to or from self
type sliceEdgeHolder struct {
	self  int
	edges []edgeInfo
}

func (e *sliceEdgeHolder) VisitEdges(visitor func(neighbor int, edge graph.Edge)) {
	for _, edge := range e.edges {
		visitor(edge.neighbor, edge.edge)
	}
}

func (e *sliceEdgeHolder) VisitEdgesFromSelf(visitor func(toNodeID int, edge graph.Edge)) {
	// Note this call is not cheap, hence we wrap this in a splitDirectionEdgeHolder
	for _, ei := range e.edges {
		if ei.edge.From().ID() == e.self {
			visitor(ei.neighbor, ei.edge)
		}
	}
}

func (e *sliceEdgeHolder) VisitEdgesToSelf(visitor func(toNodeID int, edge graph.Edge)) {
	// Note this call is not cheap, hence we wrap this in a splitDirectionEdgeHolder
	for _, ei := range e.edges {
		if ei.edge.To().ID() == e.self {
			visitor(ei.neighbor, ei.edge)
		}
	}
}

func (e *sliceEdgeHolder) Delete(neighbor int) edgeHolder {
	edges := e.edges[:0]
	for i, edge := range e.edges {
		if edge.neighbor == neighbor {
			continue
		}
		edges = append(edges, e.edges[i])
	}
	e.edges = edges
	return e
}
func (e *sliceEdgeHolder) Set(neighbor int, newEdge graph.Edge) edgeHolder {
	for i := range e.edges {
		if e.edges[i].neighbor == neighbor {
			e.edges[i].edge = newEdge
			return e
		}
	}

	if len(e.edges) < 4 {
		e.edges = append(e.edges, edgeInfo{edge: newEdge, neighbor: neighbor})
		return e
	}

	h := &mapEdgeHolder{
		self:  e.self,
		edges: make(map[int]graph.Edge, len(e.edges)+1),
	}
	for _, ei := range e.edges {
		h.edges[ei.neighbor] = ei.edge
	}
	h.edges[neighbor] = newEdge
	return h
}
func (e *sliceEdgeHolder) Get(neighbor int) (graph.Edge, bool) {
	for i := range e.edges {
		if e.edges[i].neighbor == neighbor {
			return e.edges[i].edge, true
		}
	}
	return nil, false
}
func (e *sliceEdgeHolder) Len() int {
	return len(e.edges)
}

// mapEdgeHolder holds a map of neighbors to edges
type mapEdgeHolder struct {
	self  int
	edges map[int]graph.Edge
}

func (e *mapEdgeHolder) VisitEdges(visitor func(neighbor int, edge graph.Edge)) {
	for neighbor, edge := range e.edges {
		visitor(neighbor, edge)
	}
}

func (e *mapEdgeHolder) VisitEdgesFromSelf(visitor func(toNodeID int, edge graph.Edge)) {
	// Note this call is not cheap, hence we wrap this in a splitDirectionEdgeHolder
	for neighbor, edge := range e.edges {
		if edge.From().ID() == e.self {
			visitor(neighbor, edge)
		}
	}
}

func (e *mapEdgeHolder) VisitEdgesToSelf(visitor func(toNodeID int, edge graph.Edge)) {
	// Note this call is not cheap, hence we wrap this in a splitDirectionEdgeHolder
	for neighbor, edge := range e.edges {
		if edge.To().ID() == e.self {
			visitor(neighbor, edge)
		}
	}
}
func (e *mapEdgeHolder) Delete(neighbor int) edgeHolder {
	delete(e.edges, neighbor)
	return e
}
func (e *mapEdgeHolder) Set(neighbor int, edge graph.Edge) edgeHolder {
	e.edges[neighbor] = edge
	return e
}
func (e *mapEdgeHolder) Get(neighbor int) (graph.Edge, bool) {
	edge, ok := e.edges[neighbor]
	return edge, ok
}
func (e *mapEdgeHolder) Len() int {
	return len(e.edges)
}

type splitDirectionEdgeHolder struct {
	self          int
	edgesFromSelf edgeHolder
	edgesToSelf   edgeHolder
}

func (h *splitDirectionEdgeHolder) VisitEdges(visitor func(neighbor int, edge graph.Edge)) {
	if h.edgesFromSelf != nil {
		h.edgesFromSelf.VisitEdges(visitor)
	}
	if h.edgesToSelf != nil {
		h.edgesToSelf.VisitEdges(visitor)
	}
}

func (h *splitDirectionEdgeHolder) VisitEdgesFromSelf(visitor func(toNodeID int, edge graph.Edge)) {
	// Note the big win here - we switch to a Visit, which does not check direction
	// (we also don't visit edgesToSelf, but that is a smaller win)
	if h.edgesFromSelf != nil {
		h.edgesFromSelf.VisitEdges(visitor)
	}
}

func (h *splitDirectionEdgeHolder) VisitEdgesToSelf(visitor func(toNodeID int, edge graph.Edge)) {
	if h.edgesToSelf != nil {
		h.edgesToSelf.VisitEdges(visitor)
	}
}

func (h *splitDirectionEdgeHolder) Delete(neighbor int) edgeHolder {
	if h.edgesFromSelf != nil {
		h.edgesFromSelf.Delete(neighbor)
	}
	if h.edgesToSelf != nil {
		h.edgesToSelf.Delete(neighbor)
	}
	return h
}

func (h *splitDirectionEdgeHolder) Set(neighbor int, edge graph.Edge) edgeHolder {
	if edge.From().ID() == h.self {
		if h.edgesFromSelf == nil {
			h.edgesFromSelf = &sliceEdgeHolder{self: h.self}
		}
		h.edgesFromSelf = h.edgesFromSelf.Set(neighbor, edge)

		// Mostly to keep the tests happy - this behaviour is pretty unexpected
		if h.edgesToSelf != nil {
			h.edgesToSelf = h.edgesToSelf.Delete(neighbor)
		}
		return h
	}

	if edge.To().ID() == h.self {
		if h.edgesToSelf == nil {
			h.edgesToSelf = &sliceEdgeHolder{self: h.self}
		}
		h.edgesToSelf = h.edgesToSelf.Set(neighbor, edge)

		// Mostly to keep the tests happy - this behaviour is pretty unexpected
		if h.edgesFromSelf != nil {
			h.edgesFromSelf = h.edgesFromSelf.Delete(neighbor)
		}

		return h
	}

	panic("constraint violated: wrong edge list")
}

func (h *splitDirectionEdgeHolder) Get(neighbor int) (graph.Edge, bool) {
	if h.edgesFromSelf != nil {
		e, ok := h.edgesFromSelf.Get(neighbor)
		if ok {
			return e, true
		}
	}
	if h.edgesToSelf != nil {
		e, ok := h.edgesToSelf.Get(neighbor)
		if ok {
			return e, true
		}
	}
	return nil, false
}

func (h *splitDirectionEdgeHolder) Len() int {
	n := 0
	if h.edgesFromSelf != nil {
		n += h.edgesFromSelf.Len()
	}
	if h.edgesToSelf != nil {
		n += h.edgesToSelf.Len()
	}
	return n
}
