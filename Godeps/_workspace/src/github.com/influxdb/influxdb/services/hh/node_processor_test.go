package hh

import (
	"io"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
)

type fakeShardWriter struct {
	ShardWriteFn func(shardID, nodeID uint64, points []models.Point) error
}

func (f *fakeShardWriter) WriteShard(shardID, nodeID uint64, points []models.Point) error {
	return f.ShardWriteFn(shardID, nodeID, points)
}

type fakeMetaStore struct {
	NodeFn func(nodeID uint64) (*meta.NodeInfo, error)
}

func (f *fakeMetaStore) Node(nodeID uint64) (*meta.NodeInfo, error) {
	return f.NodeFn(nodeID)
}

func TestNodeProcessorSendBlock(t *testing.T) {
	dir, err := ioutil.TempDir("", "node_processor_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	// expected data to be queue and sent to the shardWriter
	var expShardID, expNodeID, count = uint64(100), uint64(200), 0
	pt := models.MustNewPoint("cpu", models.Tags{"foo": "bar"}, models.Fields{"value": 1.0}, time.Unix(0, 0))

	sh := &fakeShardWriter{
		ShardWriteFn: func(shardID, nodeID uint64, points []models.Point) error {
			count++
			if shardID != expShardID {
				t.Errorf("SendWrite() shardID mismatch: got %v, exp %v", shardID, expShardID)
			}
			if nodeID != expNodeID {
				t.Errorf("SendWrite() nodeID mismatch: got %v, exp %v", nodeID, expNodeID)
			}

			if exp := 1; len(points) != exp {
				t.Fatalf("SendWrite() points mismatch: got %v, exp %v", len(points), exp)
			}

			if points[0].String() != pt.String() {
				t.Fatalf("SendWrite() points mismatch:\n got %v\n exp %v", points[0].String(), pt.String())
			}

			return nil
		},
	}
	metastore := &fakeMetaStore{
		NodeFn: func(nodeID uint64) (*meta.NodeInfo, error) {
			if nodeID == expNodeID {
				return &meta.NodeInfo{}, nil
			}
			return nil, nil
		},
	}

	n := NewNodeProcessor(expNodeID, dir, sh, metastore)
	if n == nil {
		t.Fatalf("Failed to create node processor: %v", err)
	}

	if err := n.Open(); err != nil {
		t.Fatalf("Failed to open node processor: %v", err)
	}

	// Check the active state.
	active, err := n.Active()
	if err != nil {
		t.Fatalf("Failed to check node processor state: %v", err)
	}
	if !active {
		t.Fatalf("Node processor state is unexpected value of: %v", active)
	}

	// This should queue a write for the active node.
	if err := n.WriteShard(expShardID, []models.Point{pt}); err != nil {
		t.Fatalf("SendWrite() failed to write points: %v", err)
	}

	// This should send the write to the shard writer
	if _, err := n.SendWrite(); err != nil {
		t.Fatalf("SendWrite() failed to write points: %v", err)
	}

	if exp := 1; count != exp {
		t.Fatalf("SendWrite() write count mismatch: got %v, exp %v", count, exp)
	}

	// All data should have been handled so no writes should be sent again
	if _, err := n.SendWrite(); err != nil && err != io.EOF {
		t.Fatalf("SendWrite() failed to write points: %v", err)
	}

	// Count should stay the same
	if exp := 1; count != exp {
		t.Fatalf("SendWrite() write count mismatch: got %v, exp %v", count, exp)
	}

	// Make the node inactive.
	sh.ShardWriteFn = func(shardID, nodeID uint64, points []models.Point) error {
		t.Fatalf("write sent to inactive node")
		return nil
	}
	metastore.NodeFn = func(nodeID uint64) (*meta.NodeInfo, error) {
		return nil, nil
	}

	// Check the active state.
	active, err = n.Active()
	if err != nil {
		t.Fatalf("Failed to check node processor state: %v", err)
	}
	if active {
		t.Fatalf("Node processor state is unexpected value of: %v", active)
	}

	// This should queue a write for the node.
	if err := n.WriteShard(expShardID, []models.Point{pt}); err != nil {
		t.Fatalf("SendWrite() failed to write points: %v", err)
	}

	// This should not send the write to the shard writer since the node is inactive.
	if _, err := n.SendWrite(); err != nil && err != io.EOF {
		t.Fatalf("SendWrite() failed to write points: %v", err)
	}

	if exp := 1; count != exp {
		t.Fatalf("SendWrite() write count mismatch: got %v, exp %v", count, exp)
	}

	if err := n.Close(); err != nil {
		t.Fatalf("Failed to close node processor: %v", err)
	}

	// Confirm that purging works ok.
	if err := n.Purge(); err != nil {
		t.Fatalf("Failed to purge node processor: %v", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("Node processor directory still present after purge")
	}
}
