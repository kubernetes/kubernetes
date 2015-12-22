package cluster_test

import (
	"fmt"
	"testing"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
)

func NewNodes() []meta.NodeInfo {
	var nodes []meta.NodeInfo
	for i := 1; i <= 2; i++ {
		nodes = append(nodes, meta.NodeInfo{
			ID:   uint64(i),
			Host: fmt.Sprintf("localhost:999%d", i),
		})
	}
	return nodes
}

func TestBalancerEmptyNodes(t *testing.T) {
	b := cluster.NewNodeBalancer([]meta.NodeInfo{})
	got := b.Next()
	if got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestBalancerUp(t *testing.T) {
	nodes := NewNodes()
	b := cluster.NewNodeBalancer(nodes)

	// First node in randomized round-robin order
	first := b.Next()
	if first == nil {
		t.Errorf("expected datanode, got %v", first)
	}

	// Second node in randomized round-robin order
	second := b.Next()
	if second == nil {
		t.Errorf("expected datanode, got %v", second)
	}

	// Should never get the same node in order twice
	if first.ID == second.ID {
		t.Errorf("expected first != second. got %v = %v", first.ID, second.ID)
	}
}

/*
func TestBalancerDown(t *testing.T) {
	nodes := NewNodes()
	b := cluster.NewNodeBalancer(nodes)

	nodes[0].Down()

	// First node in randomized round-robin order
	first := b.Next()
	if first == nil {
		t.Errorf("expected datanode, got %v", first)
	}

	// Second node should rollover to the first up node
	second := b.Next()
	if second == nil {
		t.Errorf("expected datanode, got %v", second)
	}

	// Health node should be returned each time
	if first.ID != 2 && first.ID != second.ID {
		t.Errorf("expected first != second. got %v = %v", first.ID, second.ID)
	}
}
*/

/*
func TestBalancerBackUp(t *testing.T) {
	nodes := newDataNodes()
	b := cluster.NewNodeBalancer(nodes)

	nodes[0].Down()

	for i := 0; i < 3; i++ {
		got := b.Next()
		if got == nil {
			t.Errorf("expected datanode, got %v", got)
		}

		if exp := uint64(2); got.ID != exp {
			t.Errorf("wrong node id: exp %v, got %v", exp, got.ID)
		}
	}

	nodes[0].Up()

	// First node in randomized round-robin order
	first := b.Next()
	if first == nil {
		t.Errorf("expected datanode, got %v", first)
	}

	// Second node should rollover to the first up node
	second := b.Next()
	if second == nil {
		t.Errorf("expected datanode, got %v", second)
	}

	// Should get both nodes returned
	if first.ID == second.ID {
		t.Errorf("expected first != second. got %v = %v", first.ID, second.ID)
	}
}
*/
