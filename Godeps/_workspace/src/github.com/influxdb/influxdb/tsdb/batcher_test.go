package tsdb_test

import (
	"testing"
	"time"

	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

// TestBatch_Size ensures that a batcher generates a batch when the size threshold is reached.
func TestBatch_Size(t *testing.T) {
	batchSize := 5
	batcher := tsdb.NewPointBatcher(batchSize, 0, time.Hour)
	if batcher == nil {
		t.Fatal("failed to create batcher for size test")
	}

	batcher.Start()

	var p models.Point
	go func() {
		for i := 0; i < batchSize; i++ {
			batcher.In() <- p
		}
	}()
	batch := <-batcher.Out()
	if len(batch) != batchSize {
		t.Errorf("received batch has incorrect length exp %d, got %d", batchSize, len(batch))
	}
	checkPointBatcherStats(t, batcher, -1, batchSize, 1, 0)
}

// TestBatch_Size ensures that a buffered batcher generates a batch when the size threshold is reached.
func TestBatch_SizeBuffered(t *testing.T) {
	batchSize := 5
	batcher := tsdb.NewPointBatcher(batchSize, 5, time.Hour)
	if batcher == nil {
		t.Fatal("failed to create batcher for size test")
	}

	batcher.Start()

	var p models.Point
	go func() {
		for i := 0; i < batchSize; i++ {
			batcher.In() <- p
		}
	}()
	batch := <-batcher.Out()
	if len(batch) != batchSize {
		t.Errorf("received batch has incorrect length exp %d, got %d", batchSize, len(batch))
	}
	checkPointBatcherStats(t, batcher, -1, batchSize, 1, 0)
}

// TestBatch_Size ensures that a batcher generates a batch when the timeout triggers.
func TestBatch_Timeout(t *testing.T) {
	batchSize := 5
	batcher := tsdb.NewPointBatcher(batchSize+1, 0, 100*time.Millisecond)
	if batcher == nil {
		t.Fatal("failed to create batcher for timeout test")
	}

	batcher.Start()

	var p models.Point
	go func() {
		for i := 0; i < batchSize; i++ {
			batcher.In() <- p
		}
	}()
	batch := <-batcher.Out()
	if len(batch) != batchSize {
		t.Errorf("received batch has incorrect length exp %d, got %d", batchSize, len(batch))
	}
	checkPointBatcherStats(t, batcher, -1, batchSize, 0, 1)
}

// TestBatch_Flush ensures that a batcher generates a batch when flushed
func TestBatch_Flush(t *testing.T) {
	batchSize := 2
	batcher := tsdb.NewPointBatcher(batchSize, 0, time.Hour)
	if batcher == nil {
		t.Fatal("failed to create batcher for flush test")
	}

	batcher.Start()

	var p models.Point
	go func() {
		batcher.In() <- p
		batcher.Flush()
	}()
	batch := <-batcher.Out()
	if len(batch) != 1 {
		t.Errorf("received batch has incorrect length exp %d, got %d", 1, len(batch))
	}
	checkPointBatcherStats(t, batcher, -1, 1, 0, 0)
}

// TestBatch_MultipleBatches ensures that a batcher correctly processes multiple batches.
func TestBatch_MultipleBatches(t *testing.T) {
	batchSize := 2
	batcher := tsdb.NewPointBatcher(batchSize, 0, 100*time.Millisecond)
	if batcher == nil {
		t.Fatal("failed to create batcher for size test")
	}

	batcher.Start()

	var p models.Point
	var b []models.Point

	batcher.In() <- p
	batcher.In() <- p
	b = <-batcher.Out() // Batch threshold reached.
	if len(b) != batchSize {
		t.Errorf("received batch (size) has incorrect length exp %d, got %d", batchSize, len(b))
	}

	batcher.In() <- p
	b = <-batcher.Out() // Timeout triggered.
	if len(b) != 1 {
		t.Errorf("received batch (timeout) has incorrect length exp %d, got %d", 1, len(b))
	}

	checkPointBatcherStats(t, batcher, -1, 3, 1, 1)
}

func checkPointBatcherStats(t *testing.T, b *tsdb.PointBatcher, batchTotal, pointTotal, sizeTotal, timeoutTotal int) {
	stats := b.Stats()

	if batchTotal != -1 && stats.BatchTotal != uint64(batchTotal) {
		t.Errorf("batch total stat is incorrect: %d", stats.BatchTotal)
	}
	if pointTotal != -1 && stats.PointTotal != uint64(pointTotal) {
		t.Errorf("point total stat is incorrect: %d", stats.PointTotal)
	}
	if sizeTotal != -1 && stats.SizeTotal != uint64(sizeTotal) {
		t.Errorf("size total stat is incorrect: %d", stats.SizeTotal)
	}
	if timeoutTotal != -1 && stats.TimeoutTotal != uint64(timeoutTotal) {
		t.Errorf("timeout total stat is incorrect: %d", stats.TimeoutTotal)
	}
}
