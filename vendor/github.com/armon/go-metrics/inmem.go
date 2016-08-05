package metrics

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// InmemSink provides a MetricSink that does in-memory aggregation
// without sending metrics over a network. It can be embedded within
// an application to provide profiling information.
type InmemSink struct {
	// How long is each aggregation interval
	interval time.Duration

	// Retain controls how many metrics interval we keep
	retain time.Duration

	// maxIntervals is the maximum length of intervals.
	// It is retain / interval.
	maxIntervals int

	// intervals is a slice of the retained intervals
	intervals    []*IntervalMetrics
	intervalLock sync.RWMutex
}

// IntervalMetrics stores the aggregated metrics
// for a specific interval
type IntervalMetrics struct {
	sync.RWMutex

	// The start time of the interval
	Interval time.Time

	// Gauges maps the key to the last set value
	Gauges map[string]float32

	// Points maps the string to the list of emitted values
	// from EmitKey
	Points map[string][]float32

	// Counters maps the string key to a sum of the counter
	// values
	Counters map[string]*AggregateSample

	// Samples maps the key to an AggregateSample,
	// which has the rolled up view of a sample
	Samples map[string]*AggregateSample
}

// NewIntervalMetrics creates a new IntervalMetrics for a given interval
func NewIntervalMetrics(intv time.Time) *IntervalMetrics {
	return &IntervalMetrics{
		Interval: intv,
		Gauges:   make(map[string]float32),
		Points:   make(map[string][]float32),
		Counters: make(map[string]*AggregateSample),
		Samples:  make(map[string]*AggregateSample),
	}
}

// AggregateSample is used to hold aggregate metrics
// about a sample
type AggregateSample struct {
	Count       int       // The count of emitted pairs
	Sum         float64   // The sum of values
	SumSq       float64   // The sum of squared values
	Min         float64   // Minimum value
	Max         float64   // Maximum value
	LastUpdated time.Time // When value was last updated
}

// Computes a Stddev of the values
func (a *AggregateSample) Stddev() float64 {
	num := (float64(a.Count) * a.SumSq) - math.Pow(a.Sum, 2)
	div := float64(a.Count * (a.Count - 1))
	if div == 0 {
		return 0
	}
	return math.Sqrt(num / div)
}

// Computes a mean of the values
func (a *AggregateSample) Mean() float64 {
	if a.Count == 0 {
		return 0
	}
	return a.Sum / float64(a.Count)
}

// Ingest is used to update a sample
func (a *AggregateSample) Ingest(v float64) {
	a.Count++
	a.Sum += v
	a.SumSq += (v * v)
	if v < a.Min || a.Count == 1 {
		a.Min = v
	}
	if v > a.Max || a.Count == 1 {
		a.Max = v
	}
	a.LastUpdated = time.Now()
}

func (a *AggregateSample) String() string {
	if a.Count == 0 {
		return "Count: 0"
	} else if a.Stddev() == 0 {
		return fmt.Sprintf("Count: %d Sum: %0.3f LastUpdated: %s", a.Count, a.Sum, a.LastUpdated)
	} else {
		return fmt.Sprintf("Count: %d Min: %0.3f Mean: %0.3f Max: %0.3f Stddev: %0.3f Sum: %0.3f LastUpdated: %s",
			a.Count, a.Min, a.Mean(), a.Max, a.Stddev(), a.Sum, a.LastUpdated)
	}
}

// NewInmemSink is used to construct a new in-memory sink.
// Uses an aggregation interval and maximum retention period.
func NewInmemSink(interval, retain time.Duration) *InmemSink {
	i := &InmemSink{
		interval:     interval,
		retain:       retain,
		maxIntervals: int(retain / interval),
	}
	i.intervals = make([]*IntervalMetrics, 0, i.maxIntervals)
	return i
}

func (i *InmemSink) SetGauge(key []string, val float32) {
	k := i.flattenKey(key)
	intv := i.getInterval()

	intv.Lock()
	defer intv.Unlock()
	intv.Gauges[k] = val
}

func (i *InmemSink) EmitKey(key []string, val float32) {
	k := i.flattenKey(key)
	intv := i.getInterval()

	intv.Lock()
	defer intv.Unlock()
	vals := intv.Points[k]
	intv.Points[k] = append(vals, val)
}

func (i *InmemSink) IncrCounter(key []string, val float32) {
	k := i.flattenKey(key)
	intv := i.getInterval()

	intv.Lock()
	defer intv.Unlock()

	agg := intv.Counters[k]
	if agg == nil {
		agg = &AggregateSample{}
		intv.Counters[k] = agg
	}
	agg.Ingest(float64(val))
}

func (i *InmemSink) AddSample(key []string, val float32) {
	k := i.flattenKey(key)
	intv := i.getInterval()

	intv.Lock()
	defer intv.Unlock()

	agg := intv.Samples[k]
	if agg == nil {
		agg = &AggregateSample{}
		intv.Samples[k] = agg
	}
	agg.Ingest(float64(val))
}

// Data is used to retrieve all the aggregated metrics
// Intervals may be in use, and a read lock should be acquired
func (i *InmemSink) Data() []*IntervalMetrics {
	// Get the current interval, forces creation
	i.getInterval()

	i.intervalLock.RLock()
	defer i.intervalLock.RUnlock()

	intervals := make([]*IntervalMetrics, len(i.intervals))
	copy(intervals, i.intervals)
	return intervals
}

func (i *InmemSink) getExistingInterval(intv time.Time) *IntervalMetrics {
	i.intervalLock.RLock()
	defer i.intervalLock.RUnlock()

	n := len(i.intervals)
	if n > 0 && i.intervals[n-1].Interval == intv {
		return i.intervals[n-1]
	}
	return nil
}

func (i *InmemSink) createInterval(intv time.Time) *IntervalMetrics {
	i.intervalLock.Lock()
	defer i.intervalLock.Unlock()

	// Check for an existing interval
	n := len(i.intervals)
	if n > 0 && i.intervals[n-1].Interval == intv {
		return i.intervals[n-1]
	}

	// Add the current interval
	current := NewIntervalMetrics(intv)
	i.intervals = append(i.intervals, current)
	n++

	// Truncate the intervals if they are too long
	if n >= i.maxIntervals {
		copy(i.intervals[0:], i.intervals[n-i.maxIntervals:])
		i.intervals = i.intervals[:i.maxIntervals]
	}
	return current
}

// getInterval returns the current interval to write to
func (i *InmemSink) getInterval() *IntervalMetrics {
	intv := time.Now().Truncate(i.interval)
	if m := i.getExistingInterval(intv); m != nil {
		return m
	}
	return i.createInterval(intv)
}

// Flattens the key for formatting, removes spaces
func (i *InmemSink) flattenKey(parts []string) string {
	joined := strings.Join(parts, ".")
	return strings.Replace(joined, " ", "_", -1)
}
