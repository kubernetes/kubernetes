// Copyright 2021 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.17
// +build go1.17

package prometheus

import (
	"math"
	"runtime"
	"runtime/metrics"
	"strings"
	"sync"

	//nolint:staticcheck // Ignore SA1019. Need to keep deprecated package for compatibility.
	"github.com/golang/protobuf/proto"
	"github.com/prometheus/client_golang/prometheus/internal"
	dto "github.com/prometheus/client_model/go"
)

type goCollector struct {
	base baseGoCollector

	// mu protects updates to all fields ensuring a consistent
	// snapshot is always produced by Collect.
	mu sync.Mutex

	// rm... fields all pertain to the runtime/metrics package.
	rmSampleBuf []metrics.Sample
	rmSampleMap map[string]*metrics.Sample
	rmMetrics   []collectorMetric

	// With Go 1.17, the runtime/metrics package was introduced.
	// From that point on, metric names produced by the runtime/metrics
	// package could be generated from runtime/metrics names. However,
	// these differ from the old names for the same values.
	//
	// This field exist to export the same values under the old names
	// as well.
	msMetrics memStatsMetrics
}

// NewGoCollector is the obsolete version of collectors.NewGoCollector.
// See there for documentation.
//
// Deprecated: Use collectors.NewGoCollector instead.
func NewGoCollector() Collector {
	descriptions := metrics.All()

	// Collect all histogram samples so that we can get their buckets.
	// The API guarantees that the buckets are always fixed for the lifetime
	// of the process.
	var histograms []metrics.Sample
	for _, d := range descriptions {
		if d.Kind == metrics.KindFloat64Histogram {
			histograms = append(histograms, metrics.Sample{Name: d.Name})
		}
	}
	metrics.Read(histograms)
	bucketsMap := make(map[string][]float64)
	for i := range histograms {
		bucketsMap[histograms[i].Name] = histograms[i].Value.Float64Histogram().Buckets
	}

	// Generate a Desc and ValueType for each runtime/metrics metric.
	metricSet := make([]collectorMetric, 0, len(descriptions))
	sampleBuf := make([]metrics.Sample, 0, len(descriptions))
	sampleMap := make(map[string]*metrics.Sample, len(descriptions))
	for i := range descriptions {
		d := &descriptions[i]
		namespace, subsystem, name, ok := internal.RuntimeMetricsToProm(d)
		if !ok {
			// Just ignore this metric; we can't do anything with it here.
			// If a user decides to use the latest version of Go, we don't want
			// to fail here. This condition is tested elsewhere.
			continue
		}

		// Set up sample buffer for reading, and a map
		// for quick lookup of sample values.
		sampleBuf = append(sampleBuf, metrics.Sample{Name: d.Name})
		sampleMap[d.Name] = &sampleBuf[len(sampleBuf)-1]

		var m collectorMetric
		if d.Kind == metrics.KindFloat64Histogram {
			_, hasSum := rmExactSumMap[d.Name]
			unit := d.Name[strings.IndexRune(d.Name, ':')+1:]
			m = newBatchHistogram(
				NewDesc(
					BuildFQName(namespace, subsystem, name),
					d.Description,
					nil,
					nil,
				),
				internal.RuntimeMetricsBucketsForUnit(bucketsMap[d.Name], unit),
				hasSum,
			)
		} else if d.Cumulative {
			m = NewCounter(CounterOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      name,
				Help:      d.Description,
			})
		} else {
			m = NewGauge(GaugeOpts{
				Namespace: namespace,
				Subsystem: subsystem,
				Name:      name,
				Help:      d.Description,
			})
		}
		metricSet = append(metricSet, m)
	}
	return &goCollector{
		base:        newBaseGoCollector(),
		rmSampleBuf: sampleBuf,
		rmSampleMap: sampleMap,
		rmMetrics:   metricSet,
		msMetrics:   goRuntimeMemStats(),
	}
}

// Describe returns all descriptions of the collector.
func (c *goCollector) Describe(ch chan<- *Desc) {
	c.base.Describe(ch)
	for _, i := range c.msMetrics {
		ch <- i.desc
	}
	for _, m := range c.rmMetrics {
		ch <- m.Desc()
	}
}

// Collect returns the current state of all metrics of the collector.
func (c *goCollector) Collect(ch chan<- Metric) {
	// Collect base non-memory metrics.
	c.base.Collect(ch)

	// Collect must be thread-safe, so prevent concurrent use of
	// rmSampleBuf. Just read into rmSampleBuf but write all the data
	// we get into our Metrics or MemStats.
	//
	// This lock also ensures that the Metrics we send out are all from
	// the same updates, ensuring their mutual consistency insofar as
	// is guaranteed by the runtime/metrics package.
	//
	// N.B. This locking is heavy-handed, but Collect is expected to be called
	// relatively infrequently. Also the core operation here, metrics.Read,
	// is fast (O(tens of microseconds)) so contention should certainly be
	// low, though channel operations and any allocations may add to that.
	c.mu.Lock()
	defer c.mu.Unlock()

	// Populate runtime/metrics sample buffer.
	metrics.Read(c.rmSampleBuf)

	// Update all our metrics from rmSampleBuf.
	for i, sample := range c.rmSampleBuf {
		// N.B. switch on concrete type because it's significantly more efficient
		// than checking for the Counter and Gauge interface implementations. In
		// this case, we control all the types here.
		switch m := c.rmMetrics[i].(type) {
		case *counter:
			// Guard against decreases. This should never happen, but a failure
			// to do so will result in a panic, which is a harsh consequence for
			// a metrics collection bug.
			v0, v1 := m.get(), unwrapScalarRMValue(sample.Value)
			if v1 > v0 {
				m.Add(unwrapScalarRMValue(sample.Value) - m.get())
			}
			m.Collect(ch)
		case *gauge:
			m.Set(unwrapScalarRMValue(sample.Value))
			m.Collect(ch)
		case *batchHistogram:
			m.update(sample.Value.Float64Histogram(), c.exactSumFor(sample.Name))
			m.Collect(ch)
		default:
			panic("unexpected metric type")
		}
	}
	// ms is a dummy MemStats that we populate ourselves so that we can
	// populate the old metrics from it.
	var ms runtime.MemStats
	memStatsFromRM(&ms, c.rmSampleMap)
	for _, i := range c.msMetrics {
		ch <- MustNewConstMetric(i.desc, i.valType, i.eval(&ms))
	}
}

// unwrapScalarRMValue unwraps a runtime/metrics value that is assumed
// to be scalar and returns the equivalent float64 value. Panics if the
// value is not scalar.
func unwrapScalarRMValue(v metrics.Value) float64 {
	switch v.Kind() {
	case metrics.KindUint64:
		return float64(v.Uint64())
	case metrics.KindFloat64:
		return v.Float64()
	case metrics.KindBad:
		// Unsupported metric.
		//
		// This should never happen because we always populate our metric
		// set from the runtime/metrics package.
		panic("unexpected unsupported metric")
	default:
		// Unsupported metric kind.
		//
		// This should never happen because we check for this during initialization
		// and flag and filter metrics whose kinds we don't understand.
		panic("unexpected unsupported metric kind")
	}
}

var rmExactSumMap = map[string]string{
	"/gc/heap/allocs-by-size:bytes": "/gc/heap/allocs:bytes",
	"/gc/heap/frees-by-size:bytes":  "/gc/heap/frees:bytes",
}

// exactSumFor takes a runtime/metrics metric name (that is assumed to
// be of kind KindFloat64Histogram) and returns its exact sum and whether
// its exact sum exists.
//
// The runtime/metrics API for histograms doesn't currently expose exact
// sums, but some of the other metrics are in fact exact sums of histograms.
func (c *goCollector) exactSumFor(rmName string) float64 {
	sumName, ok := rmExactSumMap[rmName]
	if !ok {
		return 0
	}
	s, ok := c.rmSampleMap[sumName]
	if !ok {
		return 0
	}
	return unwrapScalarRMValue(s.Value)
}

func memStatsFromRM(ms *runtime.MemStats, rm map[string]*metrics.Sample) {
	lookupOrZero := func(name string) uint64 {
		if s, ok := rm[name]; ok {
			return s.Value.Uint64()
		}
		return 0
	}

	// Currently, MemStats adds tiny alloc count to both Mallocs AND Frees.
	// The reason for this is because MemStats couldn't be extended at the time
	// but there was a desire to have Mallocs at least be a little more representative,
	// while having Mallocs - Frees still represent a live object count.
	// Unfortunately, MemStats doesn't actually export a large allocation count,
	// so it's impossible to pull this number out directly.
	tinyAllocs := lookupOrZero("/gc/heap/tiny/allocs:objects")
	ms.Mallocs = lookupOrZero("/gc/heap/allocs:objects") + tinyAllocs
	ms.Frees = lookupOrZero("/gc/heap/frees:objects") + tinyAllocs

	ms.TotalAlloc = lookupOrZero("/gc/heap/allocs:bytes")
	ms.Sys = lookupOrZero("/memory/classes/total:bytes")
	ms.Lookups = 0 // Already always zero.
	ms.HeapAlloc = lookupOrZero("/memory/classes/heap/objects:bytes")
	ms.Alloc = ms.HeapAlloc
	ms.HeapInuse = ms.HeapAlloc + lookupOrZero("/memory/classes/heap/unused:bytes")
	ms.HeapReleased = lookupOrZero("/memory/classes/heap/released:bytes")
	ms.HeapIdle = ms.HeapReleased + lookupOrZero("/memory/classes/heap/free:bytes")
	ms.HeapSys = ms.HeapInuse + ms.HeapIdle
	ms.HeapObjects = lookupOrZero("/gc/heap/objects:objects")
	ms.StackInuse = lookupOrZero("/memory/classes/heap/stacks:bytes")
	ms.StackSys = ms.StackInuse + lookupOrZero("/memory/classes/os-stacks:bytes")
	ms.MSpanInuse = lookupOrZero("/memory/classes/metadata/mspan/inuse:bytes")
	ms.MSpanSys = ms.MSpanInuse + lookupOrZero("/memory/classes/metadata/mspan/free:bytes")
	ms.MCacheInuse = lookupOrZero("/memory/classes/metadata/mcache/inuse:bytes")
	ms.MCacheSys = ms.MCacheInuse + lookupOrZero("/memory/classes/metadata/mcache/free:bytes")
	ms.BuckHashSys = lookupOrZero("/memory/classes/profiling/buckets:bytes")
	ms.GCSys = lookupOrZero("/memory/classes/metadata/other:bytes")
	ms.OtherSys = lookupOrZero("/memory/classes/other:bytes")
	ms.NextGC = lookupOrZero("/gc/heap/goal:bytes")

	// N.B. LastGC is omitted because runtime.GCStats already has this.
	// See https://github.com/prometheus/client_golang/issues/842#issuecomment-861812034
	// for more details.
	ms.LastGC = 0

	// N.B. GCCPUFraction is intentionally omitted. This metric is not useful,
	// and often misleading due to the fact that it's an average over the lifetime
	// of the process.
	// See https://github.com/prometheus/client_golang/issues/842#issuecomment-861812034
	// for more details.
	ms.GCCPUFraction = 0
}

// batchHistogram is a mutable histogram that is updated
// in batches.
type batchHistogram struct {
	selfCollector

	// Static fields updated only once.
	desc   *Desc
	hasSum bool

	// Because this histogram operates in batches, it just uses a
	// single mutex for everything. updates are always serialized
	// but Write calls may operate concurrently with updates.
	// Contention between these two sources should be rare.
	mu      sync.Mutex
	buckets []float64 // Inclusive lower bounds, like runtime/metrics.
	counts  []uint64
	sum     float64 // Used if hasSum is true.
}

// newBatchHistogram creates a new batch histogram value with the given
// Desc, buckets, and whether or not it has an exact sum available.
//
// buckets must always be from the runtime/metrics package, following
// the same conventions.
func newBatchHistogram(desc *Desc, buckets []float64, hasSum bool) *batchHistogram {
	h := &batchHistogram{
		desc:    desc,
		buckets: buckets,
		// Because buckets follows runtime/metrics conventions, there's
		// 1 more value in the buckets list than there are buckets represented,
		// because in runtime/metrics, the bucket values represent *boundaries*,
		// and non-Inf boundaries are inclusive lower bounds for that bucket.
		counts: make([]uint64, len(buckets)-1),
		hasSum: hasSum,
	}
	h.init(h)
	return h
}

// update updates the batchHistogram from a runtime/metrics histogram.
//
// sum must be provided if the batchHistogram was created to have an exact sum.
// h.buckets must be a strict subset of his.Buckets.
func (h *batchHistogram) update(his *metrics.Float64Histogram, sum float64) {
	counts, buckets := his.Counts, his.Buckets

	h.mu.Lock()
	defer h.mu.Unlock()

	// Clear buckets.
	for i := range h.counts {
		h.counts[i] = 0
	}
	// Copy and reduce buckets.
	var j int
	for i, count := range counts {
		h.counts[j] += count
		if buckets[i+1] == h.buckets[j+1] {
			j++
		}
	}
	if h.hasSum {
		h.sum = sum
	}
}

func (h *batchHistogram) Desc() *Desc {
	return h.desc
}

func (h *batchHistogram) Write(out *dto.Metric) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	sum := float64(0)
	if h.hasSum {
		sum = h.sum
	}
	dtoBuckets := make([]*dto.Bucket, 0, len(h.counts))
	totalCount := uint64(0)
	for i, count := range h.counts {
		totalCount += count
		if !h.hasSum {
			// N.B. This computed sum is an underestimate.
			sum += h.buckets[i] * float64(count)
		}

		// Skip the +Inf bucket, but only for the bucket list.
		// It must still count for sum and totalCount.
		if math.IsInf(h.buckets[i+1], 1) {
			break
		}
		// Float64Histogram's upper bound is exclusive, so make it inclusive
		// by obtaining the next float64 value down, in order.
		upperBound := math.Nextafter(h.buckets[i+1], h.buckets[i])
		dtoBuckets = append(dtoBuckets, &dto.Bucket{
			CumulativeCount: proto.Uint64(totalCount),
			UpperBound:      proto.Float64(upperBound),
		})
	}
	out.Histogram = &dto.Histogram{
		Bucket:      dtoBuckets,
		SampleCount: proto.Uint64(totalCount),
		SampleSum:   proto.Float64(sum),
	}
	return nil
}
