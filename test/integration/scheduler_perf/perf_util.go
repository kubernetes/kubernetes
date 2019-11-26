package benchmark

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
)

// LatencyMetric represent 50th, 90th and 99th duration quantiles.
type LatencyMetric struct {
	Perc50 time.Duration `json:"Perc50"`
	Perc90 time.Duration `json:"Perc90"`
	Perc99 time.Duration `json:"Perc99"`
}

// DataItem is the data point.
type DataItem struct {
	// Data is a map from bucket to real data point (e.g. "Perc90" -> 23.5). Notice
	// that all data items with the same label combination should have the same buckets.
	Data map[string]float64 `json:"data"`
	// Unit is the data unit. Notice that all data items with the same label combination
	// should have the same unit.
	Unit string `json:"unit"`
	// Labels is the labels of the data item.
	Labels map[string]string `json:"labels,omitempty"`
}

// DataItems is the data point set.
type DataItems struct {
	Version   string     `json:"version"`
	DataItems []DataItem `json:"dataItems"`
}

// ToPerfData converts latency metric to PerfData.
func (metric *LatencyMetric) ToPerfData(name string) DataItem {
	return DataItem{
		Data: map[string]float64{
			"Perc50": float64(metric.Perc50) / float64(time.Millisecond),
			"Perc90": float64(metric.Perc90) / float64(time.Millisecond),
			"Perc99": float64(metric.Perc99) / float64(time.Millisecond),
		},
		Unit: "ms",
		Labels: map[string]string{
			"Metric": name,
		},
	}
}

type bucket struct {
	upperBound float64
	count      float64
}

// buckets implements sort.Interface.
type buckets []bucket

func (b buckets) Len() int           { return len(b) }
func (b buckets) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b buckets) Less(i, j int) bool { return b[i].upperBound < b[j].upperBound }

// bucketQuantile calculates the quantile 'q' based on the given buckets. The
// buckets will be sorted by upperBound by this function (i.e. no sorting
// needed before calling this function). The quantile value is interpolated
// assuming a linear distribution within a bucket. However, if the quantile
// falls into the highest bucket, the upper bound of the 2nd highest bucket is
// returned. A natural lower bound of 0 is assumed if the upper bound of the
// lowest bucket is greater 0. In that case, interpolation in the lowest bucket
// happens linearly between 0 and the upper bound of the lowest bucket.
// However, if the lowest bucket has an upper bound less or equal 0, this upper
// bound is returned if the quantile falls into the lowest bucket.
//
// There are a number of special cases (once we have a way to report errors
// happening during evaluations of AST functions, we should report those
// explicitly):
//
// If 'buckets' has fewer than 2 elements, NaN is returned.
//
// If the highest bucket is not +Inf, NaN is returned.
//
// If q<0, -Inf is returned.
//
// If q>1, +Inf is returned.
func bucketQuantile(q float64, buckets buckets) float64 {
	if q < 0 {
		return math.Inf(-1)
	}
	if q > 1 {
		return math.Inf(+1)
	}
	sort.Sort(buckets)
	buckets[len(buckets)-1].upperBound = math.Inf(+1)
	if !math.IsInf(buckets[len(buckets)-1].upperBound, +1) {
		return math.NaN()
	}

	buckets = coalesceBuckets(buckets)
	ensureMonotonic(buckets)

	if len(buckets) < 2 {
		return math.NaN()
	}

	rank := q * buckets[len(buckets)-1].count
	b := sort.Search(len(buckets)-1, func(i int) bool { return buckets[i].count >= rank })

	if b == len(buckets)-1 {
		return buckets[len(buckets)-2].upperBound
	}
	if b == 0 && buckets[0].upperBound <= 0 {
		return buckets[0].upperBound
	}
	var (
		bucketStart float64
		bucketEnd   = buckets[b].upperBound
		count       = buckets[b].count
	)
	if b > 0 {
		bucketStart = buckets[b-1].upperBound
		count -= buckets[b-1].count
		rank -= buckets[b-1].count
	}
	return bucketStart + (bucketEnd-bucketStart)*(rank/count)
}

// coalesceBuckets merges buckets with the same upper bound.
//
// The input buckets must be sorted.
func coalesceBuckets(buckets buckets) buckets {
	last := buckets[0]
	i := 0
	for _, b := range buckets[1:] {
		if b.upperBound == last.upperBound {
			last.count += b.count
		} else {
			buckets[i] = last
			last = b
			i++
		}
	}
	buckets[i] = last
	return buckets[:i+1]
}

// The assumption that bucket counts increase monotonically with increasing
// upperBound may be violated during:
//
//   * Recording rule evaluation of histogram_quantile, especially when rate()
//      has been applied to the underlying bucket timeseries.
//   * Evaluation of histogram_quantile computed over federated bucket
//      timeseries, especially when rate() has been applied.
//
// This is because scraped data is not made available to rule evaluation or
// federation atomically, so some buckets are computed with data from the
// most recent scrapes, but the other buckets are missing data from the most
// recent scrape.
//
// Monotonicity is usually guaranteed because if a bucket with upper bound
// u1 has count c1, then any bucket with a higher upper bound u > u1 must
// have counted all c1 observations and perhaps more, so that c  >= c1.
//
// Randomly interspersed partial sampling breaks that guarantee, and rate()
// exacerbates it. Specifically, suppose bucket le=1000 has a count of 10 from
// 4 samples but the bucket with le=2000 has a count of 7 from 3 samples. The
// monotonicity is broken. It is exacerbated by rate() because under normal
// operation, cumulative counting of buckets will cause the bucket counts to
// diverge such that small differences from missing samples are not a problem.
// rate() removes this divergence.)
//
// bucketQuantile depends on that monotonicity to do a binary search for the
// bucket with the φ-quantile count, so breaking the monotonicity
// guarantee causes bucketQuantile() to return undefined (nonsense) results.
//
// As a somewhat hacky solution until ingestion is atomic per scrape, we
// calculate the "envelope" of the histogram buckets, essentially removing
// any decreases in the count between successive buckets.

func ensureMonotonic(buckets buckets) {
	max := buckets[0].count
	for i := range buckets[1:] {
		switch {
		case buckets[i].count > max:
			max = buckets[i].count
		case buckets[i].count < max:
			buckets[i].count = max
		}
	}
}

type histogram struct {
	sampleCount uint64
	sampleSum   float64
	buckets     []bucket
}

func (h *histogram) string() string {
	var lines []string
	var last float64
	for _, b := range h.buckets {
		lines = append(lines, fmt.Sprintf("%v %v", b.upperBound, b.count-last))
		last = b.count
	}
	return strings.Join(lines, "\n")
}

func (h *histogram) subtract(y *histogram) {
	h.sampleCount -= y.sampleCount
	h.sampleSum -= y.sampleSum
	for i := range h.buckets {
		h.buckets[i].count -= y.buckets[i].count
	}
}

func (h *histogram) deepCopy() *histogram {
	var buckets []bucket
	for _, b := range h.buckets {
		buckets = append(buckets, b)
	}
	return &histogram{
		sampleCount: h.sampleCount,
		sampleSum:   h.sampleSum,
		buckets:     buckets,
	}
}

type histogramSet map[string]map[string]*histogram

func (hs histogramSet) string(metricName string) string {
	sets := make(map[string][]float64)
	names := []string{}
	labels := []string{}
	hLen := 0
	for name, h := range hs {
		sets[name] = []float64{}
		names = append(names, name)
		var last float64
		for _, b := range h[metricName].buckets {
			sets[name] = append(sets[name], b.count-last)
			last = b.count
			if hLen == 0 {
				labels = append(labels, fmt.Sprintf("%v", b.upperBound))
			}
		}

		hLen = len(h[metricName].buckets)
	}

	sort.Strings(names)

	lines := []string{"# " + strings.Join(names, " ")}
	for i := 0; i < hLen; i++ {
		counts := []string{}
		for _, name := range names {
			counts = append(counts, fmt.Sprintf("%v", sets[name][i]))
		}
		lines = append(lines, fmt.Sprintf("%v %v", labels[i], strings.Join(counts, " ")))
	}

	return strings.Join(lines, "\n")
}

var prevHistogram = make(map[string]*histogram)

func collectRelativeMetrics(metrics []string) map[string]*histogram {
	rh := make(map[string]*histogram)
	m, _ := legacyregistry.DefaultGatherer.Gather()
	for _, mFamily := range m {
		if mFamily.Name == nil {
			continue
		}
		if !strings.HasPrefix(*mFamily.Name, "scheduler") {
			continue
		}

		metricFound := false
		for _, metricsName := range metrics {
			if *mFamily.Name == metricsName {
				metricFound = true
				break
			}
		}

		if !metricFound {
			continue
		}

		hist := mFamily.GetMetric()[0].GetHistogram()
		h := &histogram{
			sampleCount: *hist.SampleCount,
			sampleSum:   *hist.SampleSum,
		}

		for _, bckt := range hist.Bucket {
			b := bucket{
				count:      float64(*bckt.CumulativeCount),
				upperBound: *bckt.UpperBound,
			}
			h.buckets = append(h.buckets, b)
		}

		rh[*mFamily.Name] = h.deepCopy()
		if prevHistogram[*mFamily.Name] != nil {
			rh[*mFamily.Name].subtract(prevHistogram[*mFamily.Name])
		}
		prevHistogram[*mFamily.Name] = h

	}

	return rh
}
