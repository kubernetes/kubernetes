// Copyright 2015 The Prometheus Authors
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

package prometheus

import (
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync/atomic"

	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"
)

// A Histogram counts individual observations from an event or sample stream in
// configurable buckets. Similar to a summary, it also provides a sum of
// observations and an observation count.
//
// On the Prometheus server, quantiles can be calculated from a Histogram using
// the histogram_quantile function in the query language.
//
// Note that Histograms, in contrast to Summaries, can be aggregated with the
// Prometheus query language (see the documentation for detailed
// procedures). However, Histograms require the user to pre-define suitable
// buckets, and they are in general less accurate. The Observe method of a
// Histogram has a very low performance overhead in comparison with the Observe
// method of a Summary.
//
// To create Histogram instances, use NewHistogram.
type Histogram interface {
	Metric
	Collector

	// Observe adds a single observation to the histogram.
	Observe(float64)
}

// bucketLabel is used for the label that defines the upper bound of a
// bucket of a histogram ("le" -> "less or equal").
const bucketLabel = "le"

var (
	// DefBuckets are the default Histogram buckets. The default buckets are
	// tailored to broadly measure the response time (in seconds) of a
	// network service. Most likely, however, you will be required to define
	// buckets customized to your use case.
	DefBuckets = []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10}

	errBucketLabelNotAllowed = fmt.Errorf(
		"%q is not allowed as label name in histograms", bucketLabel,
	)
)

// LinearBuckets creates 'count' buckets, each 'width' wide, where the lowest
// bucket has an upper bound of 'start'. The final +Inf bucket is not counted
// and not included in the returned slice. The returned slice is meant to be
// used for the Buckets field of HistogramOpts.
//
// The function panics if 'count' is zero or negative.
func LinearBuckets(start, width float64, count int) []float64 {
	if count < 1 {
		panic("LinearBuckets needs a positive count")
	}
	buckets := make([]float64, count)
	for i := range buckets {
		buckets[i] = start
		start += width
	}
	return buckets
}

// ExponentialBuckets creates 'count' buckets, where the lowest bucket has an
// upper bound of 'start' and each following bucket's upper bound is 'factor'
// times the previous bucket's upper bound. The final +Inf bucket is not counted
// and not included in the returned slice. The returned slice is meant to be
// used for the Buckets field of HistogramOpts.
//
// The function panics if 'count' is 0 or negative, if 'start' is 0 or negative,
// or if 'factor' is less than or equal 1.
func ExponentialBuckets(start, factor float64, count int) []float64 {
	if count < 1 {
		panic("ExponentialBuckets needs a positive count")
	}
	if start <= 0 {
		panic("ExponentialBuckets needs a positive start value")
	}
	if factor <= 1 {
		panic("ExponentialBuckets needs a factor greater than 1")
	}
	buckets := make([]float64, count)
	for i := range buckets {
		buckets[i] = start
		start *= factor
	}
	return buckets
}

// HistogramOpts bundles the options for creating a Histogram metric. It is
// mandatory to set Name and Help to a non-empty string. All other fields are
// optional and can safely be left at their zero value.
type HistogramOpts struct {
	// Namespace, Subsystem, and Name are components of the fully-qualified
	// name of the Histogram (created by joining these components with
	// "_"). Only Name is mandatory, the others merely help structuring the
	// name. Note that the fully-qualified name of the Histogram must be a
	// valid Prometheus metric name.
	Namespace string
	Subsystem string
	Name      string

	// Help provides information about this Histogram. Mandatory!
	//
	// Metrics with the same fully-qualified name must have the same Help
	// string.
	Help string

	// ConstLabels are used to attach fixed labels to this
	// Histogram. Histograms with the same fully-qualified name must have the
	// same label names in their ConstLabels.
	//
	// Note that in most cases, labels have a value that varies during the
	// lifetime of a process. Those labels are usually managed with a
	// HistogramVec. ConstLabels serve only special purposes. One is for the
	// special case where the value of a label does not change during the
	// lifetime of a process, e.g. if the revision of the running binary is
	// put into a label. Another, more advanced purpose is if more than one
	// Collector needs to collect Histograms with the same fully-qualified
	// name. In that case, those Summaries must differ in the values of
	// their ConstLabels. See the Collector examples.
	//
	// If the value of a label never changes (not even between binaries),
	// that label most likely should not be a label at all (but part of the
	// metric name).
	ConstLabels Labels

	// Buckets defines the buckets into which observations are counted. Each
	// element in the slice is the upper inclusive bound of a bucket. The
	// values must be sorted in strictly increasing order. There is no need
	// to add a highest bucket with +Inf bound, it will be added
	// implicitly. The default value is DefBuckets.
	Buckets []float64
}

// NewHistogram creates a new Histogram based on the provided HistogramOpts. It
// panics if the buckets in HistogramOpts are not in strictly increasing order.
func NewHistogram(opts HistogramOpts) Histogram {
	return newHistogram(
		NewDesc(
			BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
			opts.Help,
			nil,
			opts.ConstLabels,
		),
		opts,
	)
}

func newHistogram(desc *Desc, opts HistogramOpts, labelValues ...string) Histogram {
	if len(desc.variableLabels) != len(labelValues) {
		panic(errInconsistentCardinality)
	}

	for _, n := range desc.variableLabels {
		if n == bucketLabel {
			panic(errBucketLabelNotAllowed)
		}
	}
	for _, lp := range desc.constLabelPairs {
		if lp.GetName() == bucketLabel {
			panic(errBucketLabelNotAllowed)
		}
	}

	if len(opts.Buckets) == 0 {
		opts.Buckets = DefBuckets
	}

	h := &histogram{
		desc:        desc,
		upperBounds: opts.Buckets,
		labelPairs:  makeLabelPairs(desc, labelValues),
	}
	for i, upperBound := range h.upperBounds {
		if i < len(h.upperBounds)-1 {
			if upperBound >= h.upperBounds[i+1] {
				panic(fmt.Errorf(
					"histogram buckets must be in increasing order: %f >= %f",
					upperBound, h.upperBounds[i+1],
				))
			}
		} else {
			if math.IsInf(upperBound, +1) {
				// The +Inf bucket is implicit. Remove it here.
				h.upperBounds = h.upperBounds[:i]
			}
		}
	}
	// Finally we know the final length of h.upperBounds and can make counts.
	h.counts = make([]uint64, len(h.upperBounds))

	h.Init(h) // Init self-collection.
	return h
}

type histogram struct {
	// sumBits contains the bits of the float64 representing the sum of all
	// observations. sumBits and count have to go first in the struct to
	// guarantee alignment for atomic operations.
	// http://golang.org/pkg/sync/atomic/#pkg-note-BUG
	sumBits uint64
	count   uint64

	SelfCollector
	// Note that there is no mutex required.

	desc *Desc

	upperBounds []float64
	counts      []uint64

	labelPairs []*dto.LabelPair
}

func (h *histogram) Desc() *Desc {
	return h.desc
}

func (h *histogram) Observe(v float64) {
	// TODO(beorn7): For small numbers of buckets (<30), a linear search is
	// slightly faster than the binary search. If we really care, we could
	// switch from one search strategy to the other depending on the number
	// of buckets.
	//
	// Microbenchmarks (BenchmarkHistogramNoLabels):
	// 11 buckets: 38.3 ns/op linear - binary 48.7 ns/op
	// 100 buckets: 78.1 ns/op linear - binary 54.9 ns/op
	// 300 buckets: 154 ns/op linear - binary 61.6 ns/op
	i := sort.SearchFloat64s(h.upperBounds, v)
	if i < len(h.counts) {
		atomic.AddUint64(&h.counts[i], 1)
	}
	atomic.AddUint64(&h.count, 1)
	for {
		oldBits := atomic.LoadUint64(&h.sumBits)
		newBits := math.Float64bits(math.Float64frombits(oldBits) + v)
		if atomic.CompareAndSwapUint64(&h.sumBits, oldBits, newBits) {
			break
		}
	}
}

func (h *histogram) Write(out *dto.Metric) error {
	his := &dto.Histogram{}
	buckets := make([]*dto.Bucket, len(h.upperBounds))

	his.SampleSum = proto.Float64(math.Float64frombits(atomic.LoadUint64(&h.sumBits)))
	his.SampleCount = proto.Uint64(atomic.LoadUint64(&h.count))
	var count uint64
	for i, upperBound := range h.upperBounds {
		count += atomic.LoadUint64(&h.counts[i])
		buckets[i] = &dto.Bucket{
			CumulativeCount: proto.Uint64(count),
			UpperBound:      proto.Float64(upperBound),
		}
	}
	his.Bucket = buckets
	out.Histogram = his
	out.Label = h.labelPairs
	return nil
}

// HistogramVec is a Collector that bundles a set of Histograms that all share the
// same Desc, but have different values for their variable labels. This is used
// if you want to count the same thing partitioned by various dimensions
// (e.g. HTTP request latencies, partitioned by status code and method). Create
// instances with NewHistogramVec.
type HistogramVec struct {
	MetricVec
}

// NewHistogramVec creates a new HistogramVec based on the provided HistogramOpts and
// partitioned by the given label names. At least one label name must be
// provided.
func NewHistogramVec(opts HistogramOpts, labelNames []string) *HistogramVec {
	desc := NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		labelNames,
		opts.ConstLabels,
	)
	return &HistogramVec{
		MetricVec: MetricVec{
			children: map[uint64]Metric{},
			desc:     desc,
			hash:     fnv.New64a(),
			newMetric: func(lvs ...string) Metric {
				return newHistogram(desc, opts, lvs...)
			},
		},
	}
}

// GetMetricWithLabelValues replaces the method of the same name in
// MetricVec. The difference is that this method returns a Histogram and not a
// Metric so that no type conversion is required.
func (m *HistogramVec) GetMetricWithLabelValues(lvs ...string) (Histogram, error) {
	metric, err := m.MetricVec.GetMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(Histogram), err
	}
	return nil, err
}

// GetMetricWith replaces the method of the same name in MetricVec. The
// difference is that this method returns a Histogram and not a Metric so that no
// type conversion is required.
func (m *HistogramVec) GetMetricWith(labels Labels) (Histogram, error) {
	metric, err := m.MetricVec.GetMetricWith(labels)
	if metric != nil {
		return metric.(Histogram), err
	}
	return nil, err
}

// WithLabelValues works as GetMetricWithLabelValues, but panics where
// GetMetricWithLabelValues would have returned an error. By not returning an
// error, WithLabelValues allows shortcuts like
//     myVec.WithLabelValues("404", "GET").Observe(42.21)
func (m *HistogramVec) WithLabelValues(lvs ...string) Histogram {
	return m.MetricVec.WithLabelValues(lvs...).(Histogram)
}

// With works as GetMetricWith, but panics where GetMetricWithLabels would have
// returned an error. By not returning an error, With allows shortcuts like
//     myVec.With(Labels{"code": "404", "method": "GET"}).Observe(42.21)
func (m *HistogramVec) With(labels Labels) Histogram {
	return m.MetricVec.With(labels).(Histogram)
}

type constHistogram struct {
	desc       *Desc
	count      uint64
	sum        float64
	buckets    map[float64]uint64
	labelPairs []*dto.LabelPair
}

func (h *constHistogram) Desc() *Desc {
	return h.desc
}

func (h *constHistogram) Write(out *dto.Metric) error {
	his := &dto.Histogram{}
	buckets := make([]*dto.Bucket, 0, len(h.buckets))

	his.SampleCount = proto.Uint64(h.count)
	his.SampleSum = proto.Float64(h.sum)

	for upperBound, count := range h.buckets {
		buckets = append(buckets, &dto.Bucket{
			CumulativeCount: proto.Uint64(count),
			UpperBound:      proto.Float64(upperBound),
		})
	}

	if len(buckets) > 0 {
		sort.Sort(buckSort(buckets))
	}
	his.Bucket = buckets

	out.Histogram = his
	out.Label = h.labelPairs

	return nil
}

// NewConstHistogram returns a metric representing a Prometheus histogram with
// fixed values for the count, sum, and bucket counts. As those parameters
// cannot be changed, the returned value does not implement the Histogram
// interface (but only the Metric interface). Users of this package will not
// have much use for it in regular operations. However, when implementing custom
// Collectors, it is useful as a throw-away metric that is generated on the fly
// to send it to Prometheus in the Collect method.
//
// buckets is a map of upper bounds to cumulative counts, excluding the +Inf
// bucket.
//
// NewConstHistogram returns an error if the length of labelValues is not
// consistent with the variable labels in Desc.
func NewConstHistogram(
	desc *Desc,
	count uint64,
	sum float64,
	buckets map[float64]uint64,
	labelValues ...string,
) (Metric, error) {
	if len(desc.variableLabels) != len(labelValues) {
		return nil, errInconsistentCardinality
	}
	return &constHistogram{
		desc:       desc,
		count:      count,
		sum:        sum,
		buckets:    buckets,
		labelPairs: makeLabelPairs(desc, labelValues),
	}, nil
}

// MustNewConstHistogram is a version of NewConstHistogram that panics where
// NewConstMetric would have returned an error.
func MustNewConstHistogram(
	desc *Desc,
	count uint64,
	sum float64,
	buckets map[float64]uint64,
	labelValues ...string,
) Metric {
	m, err := NewConstHistogram(desc, count, sum, buckets, labelValues...)
	if err != nil {
		panic(err)
	}
	return m
}

type buckSort []*dto.Bucket

func (s buckSort) Len() int {
	return len(s)
}

func (s buckSort) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s buckSort) Less(i, j int) bool {
	return s[i].GetUpperBound() < s[j].GetUpperBound()
}
