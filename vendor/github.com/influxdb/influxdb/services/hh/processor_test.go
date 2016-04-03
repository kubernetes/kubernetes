package hh

import (
	"io/ioutil"
	"testing"
	"time"

	"github.com/influxdb/influxdb/tsdb"
)

type fakeShardWriter struct {
	ShardWriteFn func(shardID, nodeID uint64, points []tsdb.Point) error
}

func (f *fakeShardWriter) WriteShard(shardID, nodeID uint64, points []tsdb.Point) error {
	return f.ShardWriteFn(shardID, nodeID, points)
}

func TestProcessorProcess(t *testing.T) {
	dir, err := ioutil.TempDir("", "processor_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	// expected data to be queue and sent to the shardWriter
	var expShardID, expNodeID, count = uint64(100), uint64(200), 0
	pt := tsdb.NewPoint("cpu", tsdb.Tags{"foo": "bar"}, tsdb.Fields{"value": 1.0}, time.Unix(0, 0))

	sh := &fakeShardWriter{
		ShardWriteFn: func(shardID, nodeID uint64, points []tsdb.Point) error {
			count += 1
			if shardID != expShardID {
				t.Errorf("Process() shardID mismatch: got %v, exp %v", shardID, expShardID)
			}
			if nodeID != expNodeID {
				t.Errorf("Process() nodeID mismatch: got %v, exp %v", nodeID, expNodeID)
			}

			if exp := 1; len(points) != exp {
				t.Fatalf("Process() points mismatch: got %v, exp %v", len(points), exp)
			}

			if points[0].String() != pt.String() {
				t.Fatalf("Process() points mismatch:\n got %v\n exp %v", points[0].String(), pt.String())
			}

			return nil
		},
	}

	p, err := NewProcessor(dir, sh, ProcessorOptions{MaxSize: 1024})
	if err != nil {
		t.Fatalf("Process() failed to create processor: %v", err)
	}

	// This should queue the writes
	if err := p.WriteShard(expShardID, expNodeID, []tsdb.Point{pt}); err != nil {
		t.Fatalf("Process() failed to write points: %v", err)
	}

	// This should send the write to the shard writer
	if err := p.Process(); err != nil {
		t.Fatalf("Process() failed to write points: %v", err)
	}

	if exp := 1; count != exp {
		t.Fatalf("Process() write count mismatch: got %v, exp %v", count, exp)
	}

	// Queue should be empty so no writes should be send again
	if err := p.Process(); err != nil {
		t.Fatalf("Process() failed to write points: %v", err)
	}

	// Count should stay the same
	if exp := 1; count != exp {
		t.Fatalf("Process() write count mismatch: got %v, exp %v", count, exp)
	}

}
