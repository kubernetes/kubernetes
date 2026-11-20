package simple

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/kubernetes/third_party/forked/gonum/graph"
)

func TestEdgeHolder(t *testing.T) {
	holder := edgeHolder(&sliceEdgeHolder{self: 1})

	// Empty tests
	if len := holder.Len(); len != 0 {
		t.Errorf("expected 0")
	}
	if n, ok := holder.Get(2); ok || n != nil {
		t.Errorf("expected nil,false")
	}
	holder.Visit(func(_ int, _ graph.Edge) (shouldContinue bool) { t.Errorf("unexpected call to visitor"); return true })
	holder = holder.Delete(2)

	// Insert an edge to ourselves
	holder = holder.Set(1, Edge{F: Node(1), T: Node(1)})
	if len := holder.Len(); len != 1 {
		t.Errorf("expected 1")
	}
	if n, ok := holder.Get(1); !ok || n == nil || n.From().ID() != 1 || n.To().ID() != 1 {
		t.Errorf("expected edge to ourselves, got %#v", n)
	}
	neighbors := []int{}
	holder.Visit(func(neighbor int, _ graph.Edge) (shouldContinue bool) {
		neighbors = append(neighbors, neighbor)
		return true
	})
	if !reflect.DeepEqual(neighbors, []int{1}) {
		t.Errorf("expected a single visit to ourselves, got %v", neighbors)
	}

	// Insert edges from us to other nodes
	holder = holder.Set(2, Edge{F: Node(1), T: Node(2)})
	holder = holder.Set(3, Edge{F: Node(1), T: Node(3)})
	holder = holder.Set(4, Edge{F: Node(1), T: Node(4)})
	if len := holder.Len(); len != 4 {
		t.Errorf("expected 4")
	}
	if n, ok := holder.Get(2); !ok || n == nil || n.From().ID() != 1 || n.To().ID() != 2 {
		t.Errorf("expected edge from us to another node, got %#v", n)
	}
	neighbors = []int{}
	holder.Visit(func(neighbor int, _ graph.Edge) (shouldContinue bool) {
		neighbors = append(neighbors, neighbor)
		return true
	})
	if !reflect.DeepEqual(neighbors, []int{1, 2, 3, 4}) {
		t.Errorf("expected a single visit to ourselves, got %v", neighbors)
	}

	// Insert edges to us to other nodes
	holder = holder.Set(2, Edge{F: Node(2), T: Node(1)})
	holder = holder.Set(3, Edge{F: Node(3), T: Node(1)})
	holder = holder.Set(4, Edge{F: Node(4), T: Node(1)})
	if len := holder.Len(); len != 4 {
		t.Errorf("expected 4")
	}
	if n, ok := holder.Get(2); !ok || n == nil || n.From().ID() != 2 || n.To().ID() != 1 {
		t.Errorf("expected reversed edge, got %#v", n)
	}
	neighbors = []int{}
	holder.Visit(func(neighbor int, _ graph.Edge) (shouldContinue bool) {
		neighbors = append(neighbors, neighbor)
		return true
	})
	if !reflect.DeepEqual(neighbors, []int{1, 2, 3, 4}) {
		t.Errorf("expected a single visit to ourselves, got %v", neighbors)
	}

	if _, ok := holder.(*sliceEdgeHolder); !ok {
		t.Errorf("expected slice edge holder")
	}

	// Make the transition to a map
	holder = holder.Set(5, Edge{F: Node(5), T: Node(1)})

	if _, ok := holder.(mapEdgeHolder); !ok {
		t.Errorf("expected map edge holder")
	}
	if len := holder.Len(); len != 5 {
		t.Errorf("expected 5")
	}
	if n, ok := holder.Get(2); !ok || n == nil || n.From().ID() != 2 || n.To().ID() != 1 {
		t.Errorf("expected old edges, got %#v", n)
	}
	if n, ok := holder.Get(5); !ok || n == nil || n.From().ID() != 5 || n.To().ID() != 1 {
		t.Errorf("expected new edge, got %#v", n)
	}
	neighbors = []int{}
	holder.Visit(func(neighbor int, _ graph.Edge) (shouldContinue bool) {
		neighbors = append(neighbors, neighbor)
		return true
	})
	sort.Ints(neighbors) // sort, map order is random
	if !reflect.DeepEqual(neighbors, []int{1, 2, 3, 4, 5}) {
		t.Errorf("expected 1,2,3,4,5, got %v", neighbors)
	}
	holder = holder.Delete(1)
	holder = holder.Delete(2)
	holder = holder.Delete(3)
	holder = holder.Delete(4)
	holder = holder.Delete(5)
	holder = holder.Delete(6)
	if len := holder.Len(); len != 0 {
		t.Errorf("expected 0")
	}
}

func TestEdgeHolderVisitStopsEarly(t *testing.T) {
	// 3 edges stays a sliceEdgeHolder; 6 edges promotes to a mapEdgeHolder.
	for _, numEdges := range []int{3, 6} {
		holder := edgeHolder(&sliceEdgeHolder{self: 1})
		for i := 2; i < 2+numEdges; i++ {
			holder = holder.Set(i, Edge{F: Node(1), T: Node(i)})
		}

		calls := 0
		holder.Visit(func(_ int, _ graph.Edge) (shouldContinue bool) {
			calls++
			return false
		})
		if calls != 1 {
			t.Errorf("%T with %d edges: expected visitor to be called once, got %d", holder, numEdges, calls)
		}
	}
}
