package stats

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
)

// HistogramValue is the value of Histogram objects.
type HistogramValue struct {
	// Count is the total number of values added to the histogram.
	Count int64
	// Sum is the sum of all the values added to the histogram.
	Sum int64
	// Min is the minimum of all the values added to the histogram.
	Min int64
	// Max is the maximum of all the values added to the histogram.
	Max int64
	// Buckets contains all the buckets of the histogram.
	Buckets []HistogramBucket
}

// HistogramBucket is one histogram bucket.
type HistogramBucket struct {
	// LowBound is the lower bound of the bucket.
	LowBound int64
	// Count is the number of values in the bucket.
	Count int64
}

// Print writes textual output of the histogram values.
func (v HistogramValue) Print(w io.Writer) {
	avg := float64(v.Sum) / float64(v.Count)
	fmt.Fprintf(w, "Count: %d  Min: %d  Max: %d  Avg: %.2f\n", v.Count, v.Min, v.Max, avg)
	fmt.Fprintf(w, "%s\n", strings.Repeat("-", 60))
	if v.Count <= 0 {
		return
	}

	maxBucketDigitLen := len(strconv.FormatInt(v.Buckets[len(v.Buckets)-1].LowBound, 10))
	if maxBucketDigitLen < 3 {
		// For "inf".
		maxBucketDigitLen = 3
	}
	maxCountDigitLen := len(strconv.FormatInt(v.Count, 10))
	percentMulti := 100 / float64(v.Count)

	accCount := int64(0)
	for i, b := range v.Buckets {
		fmt.Fprintf(w, "[%*d, ", maxBucketDigitLen, b.LowBound)
		if i+1 < len(v.Buckets) {
			fmt.Fprintf(w, "%*d)", maxBucketDigitLen, v.Buckets[i+1].LowBound)
		} else {
			fmt.Fprintf(w, "%*s)", maxBucketDigitLen, "inf")
		}

		accCount += b.Count
		fmt.Fprintf(w, "  %*d  %5.1f%%  %5.1f%%", maxCountDigitLen, b.Count, float64(b.Count)*percentMulti, float64(accCount)*percentMulti)

		const barScale = 0.1
		barLength := int(float64(b.Count)*percentMulti*barScale + 0.5)
		fmt.Fprintf(w, "  %s\n", strings.Repeat("#", barLength))
	}
}

// String returns the textual output of the histogram values as string.
func (v HistogramValue) String() string {
	var b bytes.Buffer
	v.Print(&b)
	return b.String()
}

// A Histogram accumulates values in the form of a histogram. The type of the
// values is int64, which is suitable for keeping track of things like RPC
// latency in milliseconds. New histogram objects should be obtained via the
// New() function.
type Histogram struct {
	opts    HistogramOptions
	buckets []bucketInternal
	count   *Counter
	sum     *Counter
	tracker *Tracker
}

// HistogramOptions contains the parameters that define the histogram's buckets.
type HistogramOptions struct {
	// NumBuckets is the number of buckets.
	NumBuckets int
	// GrowthFactor is the growth factor of the buckets. A value of 0.1
	// indicates that bucket N+1 will be 10% larger than bucket N.
	GrowthFactor float64
	// SmallestBucketSize is the size of the first bucket. Bucket sizes are
	// rounded down to the nearest integer.
	SmallestBucketSize float64
	// MinValue is the lower bound of the first bucket.
	MinValue int64
}

// bucketInternal is the internal representation of a bucket, which includes a
// rate counter.
type bucketInternal struct {
	lowBound int64
	count    *Counter
}

// NewHistogram returns a pointer to a new Histogram object that was created
// with the provided options.
func NewHistogram(opts HistogramOptions) *Histogram {
	if opts.NumBuckets == 0 {
		opts.NumBuckets = 32
	}
	if opts.SmallestBucketSize == 0.0 {
		opts.SmallestBucketSize = 1.0
	}
	h := Histogram{
		opts:    opts,
		buckets: make([]bucketInternal, opts.NumBuckets),
		count:   newCounter(),
		sum:     newCounter(),
		tracker: newTracker(),
	}
	low := opts.MinValue
	delta := opts.SmallestBucketSize
	for i := 0; i < opts.NumBuckets; i++ {
		h.buckets[i].lowBound = low
		h.buckets[i].count = newCounter()
		low = low + int64(delta)
		delta = delta * (1.0 + opts.GrowthFactor)
	}
	return &h
}

// Opts returns a copy of the options used to create the Histogram.
func (h *Histogram) Opts() HistogramOptions {
	return h.opts
}

// Add adds a value to the histogram.
func (h *Histogram) Add(value int64) error {
	bucket, err := h.findBucket(value)
	if err != nil {
		return err
	}
	h.buckets[bucket].count.Incr(1)
	h.count.Incr(1)
	h.sum.Incr(value)
	h.tracker.Push(value)
	return nil
}

// LastUpdate returns the time at which the object was last updated.
func (h *Histogram) LastUpdate() time.Time {
	return h.count.LastUpdate()
}

// Value returns the accumulated state of the histogram since it was created.
func (h *Histogram) Value() HistogramValue {
	b := make([]HistogramBucket, len(h.buckets))
	for i, v := range h.buckets {
		b[i] = HistogramBucket{
			LowBound: v.lowBound,
			Count:    v.count.Value(),
		}
	}

	v := HistogramValue{
		Count:   h.count.Value(),
		Sum:     h.sum.Value(),
		Min:     h.tracker.Min(),
		Max:     h.tracker.Max(),
		Buckets: b,
	}
	return v
}

// Delta1h returns the change in the last hour.
func (h *Histogram) Delta1h() HistogramValue {
	b := make([]HistogramBucket, len(h.buckets))
	for i, v := range h.buckets {
		b[i] = HistogramBucket{
			LowBound: v.lowBound,
			Count:    v.count.Delta1h(),
		}
	}

	v := HistogramValue{
		Count:   h.count.Delta1h(),
		Sum:     h.sum.Delta1h(),
		Min:     h.tracker.Min1h(),
		Max:     h.tracker.Max1h(),
		Buckets: b,
	}
	return v
}

// Delta10m returns the change in the last 10 minutes.
func (h *Histogram) Delta10m() HistogramValue {
	b := make([]HistogramBucket, len(h.buckets))
	for i, v := range h.buckets {
		b[i] = HistogramBucket{
			LowBound: v.lowBound,
			Count:    v.count.Delta10m(),
		}
	}

	v := HistogramValue{
		Count:   h.count.Delta10m(),
		Sum:     h.sum.Delta10m(),
		Min:     h.tracker.Min10m(),
		Max:     h.tracker.Max10m(),
		Buckets: b,
	}
	return v
}

// Delta1m returns the change in the last 10 minutes.
func (h *Histogram) Delta1m() HistogramValue {
	b := make([]HistogramBucket, len(h.buckets))
	for i, v := range h.buckets {
		b[i] = HistogramBucket{
			LowBound: v.lowBound,
			Count:    v.count.Delta1m(),
		}
	}

	v := HistogramValue{
		Count:   h.count.Delta1m(),
		Sum:     h.sum.Delta1m(),
		Min:     h.tracker.Min1m(),
		Max:     h.tracker.Max1m(),
		Buckets: b,
	}
	return v
}

// findBucket does a binary search to find in which bucket the value goes.
func (h *Histogram) findBucket(value int64) (int, error) {
	lastBucket := len(h.buckets) - 1
	min, max := 0, lastBucket
	for max >= min {
		b := (min + max) / 2
		if value >= h.buckets[b].lowBound && (b == lastBucket || value < h.buckets[b+1].lowBound) {
			return b, nil
		}
		if value < h.buckets[b].lowBound {
			max = b - 1
			continue
		}
		min = b + 1
	}
	return 0, fmt.Errorf("no bucket for value: %d", value)
}
