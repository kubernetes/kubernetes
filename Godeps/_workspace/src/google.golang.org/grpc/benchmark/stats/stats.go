package stats

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"time"
)

// Stats is a simple helper for gathering additional statistics like histogram
// during benchmarks. This is not thread safe.
type Stats struct {
	numBuckets int
	unit       time.Duration
	min, max   int64
	histogram  *Histogram

	durations durationSlice
	dirty     bool
}

type durationSlice []time.Duration

// NewStats creates a new Stats instance. If numBuckets is not positive,
// the default value (16) will be used.
func NewStats(numBuckets int) *Stats {
	if numBuckets <= 0 {
		numBuckets = 16
	}
	return &Stats{
		// Use one more bucket for the last unbounded bucket.
		numBuckets: numBuckets + 1,
		durations:  make(durationSlice, 0, 100000),
	}
}

// Add adds an elapsed time per operation to the stats.
func (stats *Stats) Add(d time.Duration) {
	stats.durations = append(stats.durations, d)
	stats.dirty = true
}

// Clear resets the stats, removing all values.
func (stats *Stats) Clear() {
	stats.durations = stats.durations[:0]
	stats.histogram = nil
	stats.dirty = false
}

// maybeUpdate updates internal stat data if there was any newly added
// stats since this was updated.
func (stats *Stats) maybeUpdate() {
	if !stats.dirty {
		return
	}

	stats.min = math.MaxInt64
	stats.max = 0
	for _, d := range stats.durations {
		if stats.min > int64(d) {
			stats.min = int64(d)
		}
		if stats.max < int64(d) {
			stats.max = int64(d)
		}
	}

	// Use the largest unit that can represent the minimum time duration.
	stats.unit = time.Nanosecond
	for _, u := range []time.Duration{time.Microsecond, time.Millisecond, time.Second} {
		if stats.min <= int64(u) {
			break
		}
		stats.unit = u
	}

	// Adjust the min/max according to the new unit.
	stats.min /= int64(stats.unit)
	stats.max /= int64(stats.unit)
	numBuckets := stats.numBuckets
	if n := int(stats.max - stats.min + 1); n < numBuckets {
		numBuckets = n
	}
	stats.histogram = NewHistogram(HistogramOptions{
		NumBuckets: numBuckets,
		// max(i.e., Nth lower bound) = min + (1 + growthFactor)^(numBuckets-2).
		GrowthFactor:       math.Pow(float64(stats.max-stats.min), 1/float64(stats.numBuckets-2)) - 1,
		SmallestBucketSize: 1.0,
		MinValue:           stats.min})

	for _, d := range stats.durations {
		stats.histogram.Add(int64(d / stats.unit))
	}

	stats.dirty = false
}

// Print writes textual output of the Stats.
func (stats *Stats) Print(w io.Writer) {
	stats.maybeUpdate()

	if stats.histogram == nil {
		fmt.Fprint(w, "Histogram (empty)\n")
	} else {
		fmt.Fprintf(w, "Histogram (unit: %s)\n", fmt.Sprintf("%v", stats.unit)[1:])
		stats.histogram.Value().Print(w)
	}
}

// String returns the textual output of the Stats as string.
func (stats *Stats) String() string {
	var b bytes.Buffer
	stats.Print(&b)
	return b.String()
}
