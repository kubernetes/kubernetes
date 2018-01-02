package tsdb

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/models"
)

// PointBatcher accepts Points and will emit a batch of those points when either
// a) the batch reaches a certain size, or b) a certain time passes.
type PointBatcher struct {
	stats PointBatcherStats

	size     int
	duration time.Duration

	stop  chan struct{}
	in    chan models.Point
	out   chan []models.Point
	flush chan struct{}

	wg *sync.WaitGroup
}

// NewPointBatcher returns a new PointBatcher. sz is the batching size,
// bp is the maximum number of batches that may be pending. d is the time
// after which a batch will be emitted after the first point is received
// for the batch, regardless of its size.
func NewPointBatcher(sz int, bp int, d time.Duration) *PointBatcher {
	return &PointBatcher{
		size:     sz,
		duration: d,
		stop:     make(chan struct{}),
		in:       make(chan models.Point, bp*sz),
		out:      make(chan []models.Point),
		flush:    make(chan struct{}),
	}
}

// PointBatcherStats are the statistics each batcher tracks.
type PointBatcherStats struct {
	BatchTotal   uint64 // Total count of batches transmitted.
	PointTotal   uint64 // Total count of points processed.
	SizeTotal    uint64 // Number of batches that reached size threshold.
	TimeoutTotal uint64 // Number of timeouts that occurred.
}

// Start starts the batching process. Returns the in and out channels for points
// and point-batches respectively.
func (b *PointBatcher) Start() {
	// Already running?
	if b.wg != nil {
		return
	}

	var timer *time.Timer
	var batch []models.Point
	var timerCh <-chan time.Time

	emit := func() {
		b.out <- batch
		atomic.AddUint64(&b.stats.BatchTotal, 1)
		batch = nil
	}

	b.wg = &sync.WaitGroup{}
	b.wg.Add(1)

	go func() {
		defer b.wg.Done()
		for {
			select {
			case <-b.stop:
				if len(batch) > 0 {
					emit()
					timerCh = nil
				}
				return
			case p := <-b.in:
				atomic.AddUint64(&b.stats.PointTotal, 1)
				if batch == nil {
					batch = make([]models.Point, 0, b.size)
					if b.duration > 0 {
						timer = time.NewTimer(b.duration)
						timerCh = timer.C
					}
				}

				batch = append(batch, p)
				if len(batch) >= b.size { // 0 means send immediately.
					atomic.AddUint64(&b.stats.SizeTotal, 1)
					emit()
					timerCh = nil
				}

			case <-b.flush:
				if len(batch) > 0 {
					emit()
					timerCh = nil
				}

			case <-timerCh:
				atomic.AddUint64(&b.stats.TimeoutTotal, 1)
				emit()
			}
		}
	}()
}

// Stop stops the batching process. Stop waits for the batching routine
// to stop before returning.
func (b *PointBatcher) Stop() {
	// If not running, nothing to stop.
	if b.wg == nil {
		return
	}

	close(b.stop)
	b.wg.Wait()
}

// In returns the channel to which points should be written.
func (b *PointBatcher) In() chan<- models.Point {
	return b.in
}

// Out returns the channel from which batches should be read.
func (b *PointBatcher) Out() <-chan []models.Point {
	return b.out
}

// Flush instructs the batcher to emit any pending points in a batch, regardless of batch size.
// If there are no pending points, no batch is emitted.
func (b *PointBatcher) Flush() {
	b.flush <- struct{}{}
}

// Stats returns a PointBatcherStats object for the PointBatcher. While the each statistic should be
// closely correlated with each other statistic, it is not guaranteed.
func (b *PointBatcher) Stats() *PointBatcherStats {
	stats := PointBatcherStats{}
	stats.BatchTotal = atomic.LoadUint64(&b.stats.BatchTotal)
	stats.PointTotal = atomic.LoadUint64(&b.stats.PointTotal)
	stats.SizeTotal = atomic.LoadUint64(&b.stats.SizeTotal)
	stats.TimeoutTotal = atomic.LoadUint64(&b.stats.TimeoutTotal)
	return &stats
}
