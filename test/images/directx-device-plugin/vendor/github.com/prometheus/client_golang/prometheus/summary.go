// Copyright 2014 The Prometheus Authors
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
	"math"
	"sort"
	"sync"
	"time"

	"github.com/beorn7/perks/quantile"
	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"
)

// quantileLabel is used for the label that defines the quantile in a
// summary.
const quantileLabel = "quantile"

// A Summary captures individual observations from an event or sample stream and
// summarizes them in a manner similar to traditional summary statistics: 1. sum
// of observations, 2. observation count, 3. rank estimations.
//
// A typical use-case is the observation of request latencies. By default, a
// Summary provides the median, the 90th and the 99th percentile of the latency
// as rank estimations. However, the default behavior will change in the
// upcoming v0.10 of the library. There will be no rank estimations at all by
// default. For a sane transition, it is recommended to set the desired rank
// estimations explicitly.
//
// Note that the rank estimations cannot be aggregated in a meaningful way with
// the Prometheus query language (i.e. you cannot average or add them). If you
// need aggregatable quantiles (e.g. you want the 99th percentile latency of all
// queries served across all instances of a service), consider the Histogram
// metric type. See the Prometheus documentation for more details.
//
// To create Summary instances, use NewSummary.
type Summary interface {
	Metric
	Collector

	// Observe adds a single observation to the summary.
	Observe(float64)
}

// DefObjectives are the default Summary quantile values.
//
// Deprecated: DefObjectives will not be used as the default objectives in
// v0.10 of the library. The default Summary will have no quantiles then.
var (
	DefObjectives = map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001}

	errQuantileLabelNotAllowed = fmt.Errorf(
		"%q is not allowed as label name in summaries", quantileLabel,
	)
)

// Default values for SummaryOpts.
const (
	// DefMaxAge is the default duration for which observations stay
	// relevant.
	DefMaxAge time.Duration = 10 * time.Minute
	// DefAgeBuckets is the default number of buckets used to calculate the
	// age of observations.
	DefAgeBuckets = 5
	// DefBufCap is the standard buffer size for collecting Summary observations.
	DefBufCap = 500
)

// SummaryOpts bundles the options for creating a Summary metric. It is
// mandatory to set Name to a non-empty string. While all other fields are
// optional and can safely be left at their zero value, it is recommended to set
// a help string and to explicitly set the Objectives field to the desired value
// as the default value will change in the upcoming v0.10 of the library.
type SummaryOpts struct {
	// Namespace, Subsystem, and Name are components of the fully-qualified
	// name of the Summary (created by joining these components with
	// "_"). Only Name is mandatory, the others merely help structuring the
	// name. Note that the fully-qualified name of the Summary must be a
	// valid Prometheus metric name.
	Namespace string
	Subsystem string
	Name      string

	// Help provides information about this Summary.
	//
	// Metrics with the same fully-qualified name must have the same Help
	// string.
	Help string

	// ConstLabels are used to attach fixed labels to this metric. Metrics
	// with the same fully-qualified name must have the same label names in
	// their ConstLabels.
	//
	// Due to the way a Summary is represented in the Prometheus text format
	// and how it is handled by the Prometheus server internally, “quantile”
	// is an illegal label name. Construction of a Summary or SummaryVec
	// will panic if this label name is used in ConstLabels.
	//
	// ConstLabels are only used rarely. In particular, do not use them to
	// attach the same labels to all your metrics. Those use cases are
	// better covered by target labels set by the scraping Prometheus
	// server, or by one specific metric (e.g. a build_info or a
	// machine_role metric). See also
	// https://prometheus.io/docs/instrumenting/writing_exporters/#target-labels,-not-static-scraped-labels
	ConstLabels Labels

	// Objectives defines the quantile rank estimates with their respective
	// absolute error. If Objectives[q] = e, then the value reported for q
	// will be the φ-quantile value for some φ between q-e and q+e.  The
	// default value is DefObjectives. It is used if Objectives is left at
	// its zero value (i.e. nil). To create a Summary without Objectives,
	// set it to an empty map (i.e. map[float64]float64{}).
	//
	// Deprecated: Note that the current value of DefObjectives is
	// deprecated. It will be replaced by an empty map in v0.10 of the
	// library. Please explicitly set Objectives to the desired value.
	Objectives map[float64]float64

	// MaxAge defines the duration for which an observation stays relevant
	// for the summary. Must be positive. The default value is DefMaxAge.
	MaxAge time.Duration

	// AgeBuckets is the number of buckets used to exclude observations that
	// are older than MaxAge from the summary. A higher number has a
	// resource penalty, so only increase it if the higher resolution is
	// really required. For very high observation rates, you might want to
	// reduce the number of age buckets. With only one age bucket, you will
	// effectively see a complete reset of the summary each time MaxAge has
	// passed. The default value is DefAgeBuckets.
	AgeBuckets uint32

	// BufCap defines the default sample stream buffer size.  The default
	// value of DefBufCap should suffice for most uses. If there is a need
	// to increase the value, a multiple of 500 is recommended (because that
	// is the internal buffer size of the underlying package
	// "github.com/bmizerany/perks/quantile").
	BufCap uint32
}

// Great fuck-up with the sliding-window decay algorithm... The Merge method of
// perk/quantile is actually not working as advertised - and it might be
// unfixable, as the underlying algorithm is apparently not capable of merging
// summaries in the first place. To avoid using Merge, we are currently adding
// observations to _each_ age bucket, i.e. the effort to add a sample is
// essentially multiplied by the number of age buckets. When rotating age
// buckets, we empty the previous head stream. On scrape time, we simply take
// the quantiles from the head stream (no merging required). Result: More effort
// on observation time, less effort on scrape time, which is exactly the
// opposite of what we try to accomplish, but at least the results are correct.
//
// The quite elegant previous contraption to merge the age buckets efficiently
// on scrape time (see code up commit 6b9530d72ea715f0ba612c0120e6e09fbf1d49d0)
// can't be used anymore.

// NewSummary creates a new Summary based on the provided SummaryOpts.
func NewSummary(opts SummaryOpts) Summary {
	return newSummary(
		NewDesc(
			BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
			opts.Help,
			nil,
			opts.ConstLabels,
		),
		opts,
	)
}

func newSummary(desc *Desc, opts SummaryOpts, labelValues ...string) Summary {
	if len(desc.variableLabels) != len(labelValues) {
		panic(makeInconsistentCardinalityError(desc.fqName, desc.variableLabels, labelValues))
	}

	for _, n := range desc.variableLabels {
		if n == quantileLabel {
			panic(errQuantileLabelNotAllowed)
		}
	}
	for _, lp := range desc.constLabelPairs {
		if lp.GetName() == quantileLabel {
			panic(errQuantileLabelNotAllowed)
		}
	}

	if opts.Objectives == nil {
		opts.Objectives = DefObjectives
	}

	if opts.MaxAge < 0 {
		panic(fmt.Errorf("illegal max age MaxAge=%v", opts.MaxAge))
	}
	if opts.MaxAge == 0 {
		opts.MaxAge = DefMaxAge
	}

	if opts.AgeBuckets == 0 {
		opts.AgeBuckets = DefAgeBuckets
	}

	if opts.BufCap == 0 {
		opts.BufCap = DefBufCap
	}

	s := &summary{
		desc: desc,

		objectives:       opts.Objectives,
		sortedObjectives: make([]float64, 0, len(opts.Objectives)),

		labelPairs: makeLabelPairs(desc, labelValues),

		hotBuf:         make([]float64, 0, opts.BufCap),
		coldBuf:        make([]float64, 0, opts.BufCap),
		streamDuration: opts.MaxAge / time.Duration(opts.AgeBuckets),
	}
	s.headStreamExpTime = time.Now().Add(s.streamDuration)
	s.hotBufExpTime = s.headStreamExpTime

	for i := uint32(0); i < opts.AgeBuckets; i++ {
		s.streams = append(s.streams, s.newStream())
	}
	s.headStream = s.streams[0]

	for qu := range s.objectives {
		s.sortedObjectives = append(s.sortedObjectives, qu)
	}
	sort.Float64s(s.sortedObjectives)

	s.init(s) // Init self-collection.
	return s
}

type summary struct {
	selfCollector

	bufMtx sync.Mutex // Protects hotBuf and hotBufExpTime.
	mtx    sync.Mutex // Protects every other moving part.
	// Lock bufMtx before mtx if both are needed.

	desc *Desc

	objectives       map[float64]float64
	sortedObjectives []float64

	labelPairs []*dto.LabelPair

	sum float64
	cnt uint64

	hotBuf, coldBuf []float64

	streams                          []*quantile.Stream
	streamDuration                   time.Duration
	headStream                       *quantile.Stream
	headStreamIdx                    int
	headStreamExpTime, hotBufExpTime time.Time
}

func (s *summary) Desc() *Desc {
	return s.desc
}

func (s *summary) Observe(v float64) {
	s.bufMtx.Lock()
	defer s.bufMtx.Unlock()

	now := time.Now()
	if now.After(s.hotBufExpTime) {
		s.asyncFlush(now)
	}
	s.hotBuf = append(s.hotBuf, v)
	if len(s.hotBuf) == cap(s.hotBuf) {
		s.asyncFlush(now)
	}
}

func (s *summary) Write(out *dto.Metric) error {
	sum := &dto.Summary{}
	qs := make([]*dto.Quantile, 0, len(s.objectives))

	s.bufMtx.Lock()
	s.mtx.Lock()
	// Swap bufs even if hotBuf is empty to set new hotBufExpTime.
	s.swapBufs(time.Now())
	s.bufMtx.Unlock()

	s.flushColdBuf()
	sum.SampleCount = proto.Uint64(s.cnt)
	sum.SampleSum = proto.Float64(s.sum)

	for _, rank := range s.sortedObjectives {
		var q float64
		if s.headStream.Count() == 0 {
			q = math.NaN()
		} else {
			q = s.headStream.Query(rank)
		}
		qs = append(qs, &dto.Quantile{
			Quantile: proto.Float64(rank),
			Value:    proto.Float64(q),
		})
	}

	s.mtx.Unlock()

	if len(qs) > 0 {
		sort.Sort(quantSort(qs))
	}
	sum.Quantile = qs

	out.Summary = sum
	out.Label = s.labelPairs
	return nil
}

func (s *summary) newStream() *quantile.Stream {
	return quantile.NewTargeted(s.objectives)
}

// asyncFlush needs bufMtx locked.
func (s *summary) asyncFlush(now time.Time) {
	s.mtx.Lock()
	s.swapBufs(now)

	// Unblock the original goroutine that was responsible for the mutation
	// that triggered the compaction.  But hold onto the global non-buffer
	// state mutex until the operation finishes.
	go func() {
		s.flushColdBuf()
		s.mtx.Unlock()
	}()
}

// rotateStreams needs mtx AND bufMtx locked.
func (s *summary) maybeRotateStreams() {
	for !s.hotBufExpTime.Equal(s.headStreamExpTime) {
		s.headStream.Reset()
		s.headStreamIdx++
		if s.headStreamIdx >= len(s.streams) {
			s.headStreamIdx = 0
		}
		s.headStream = s.streams[s.headStreamIdx]
		s.headStreamExpTime = s.headStreamExpTime.Add(s.streamDuration)
	}
}

// flushColdBuf needs mtx locked.
func (s *summary) flushColdBuf() {
	for _, v := range s.coldBuf {
		for _, stream := range s.streams {
			stream.Insert(v)
		}
		s.cnt++
		s.sum += v
	}
	s.coldBuf = s.coldBuf[0:0]
	s.maybeRotateStreams()
}

// swapBufs needs mtx AND bufMtx locked, coldBuf must be empty.
func (s *summary) swapBufs(now time.Time) {
	if len(s.coldBuf) != 0 {
		panic("coldBuf is not empty")
	}
	s.hotBuf, s.coldBuf = s.coldBuf, s.hotBuf
	// hotBuf is now empty and gets new expiration set.
	for now.After(s.hotBufExpTime) {
		s.hotBufExpTime = s.hotBufExpTime.Add(s.streamDuration)
	}
}

type quantSort []*dto.Quantile

func (s quantSort) Len() int {
	return len(s)
}

func (s quantSort) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s quantSort) Less(i, j int) bool {
	return s[i].GetQuantile() < s[j].GetQuantile()
}

// SummaryVec is a Collector that bundles a set of Summaries that all share the
// same Desc, but have different values for their variable labels. This is used
// if you want to count the same thing partitioned by various dimensions
// (e.g. HTTP request latencies, partitioned by status code and method). Create
// instances with NewSummaryVec.
type SummaryVec struct {
	*metricVec
}

// NewSummaryVec creates a new SummaryVec based on the provided SummaryOpts and
// partitioned by the given label names.
//
// Due to the way a Summary is represented in the Prometheus text format and how
// it is handled by the Prometheus server internally, “quantile” is an illegal
// label name. NewSummaryVec will panic if this label name is used.
func NewSummaryVec(opts SummaryOpts, labelNames []string) *SummaryVec {
	for _, ln := range labelNames {
		if ln == quantileLabel {
			panic(errQuantileLabelNotAllowed)
		}
	}
	desc := NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		labelNames,
		opts.ConstLabels,
	)
	return &SummaryVec{
		metricVec: newMetricVec(desc, func(lvs ...string) Metric {
			return newSummary(desc, opts, lvs...)
		}),
	}
}

// GetMetricWithLabelValues returns the Summary for the given slice of label
// values (same order as the VariableLabels in Desc). If that combination of
// label values is accessed for the first time, a new Summary is created.
//
// It is possible to call this method without using the returned Summary to only
// create the new Summary but leave it at its starting value, a Summary without
// any observations.
//
// Keeping the Summary for later use is possible (and should be considered if
// performance is critical), but keep in mind that Reset, DeleteLabelValues and
// Delete can be used to delete the Summary from the SummaryVec. In that case,
// the Summary will still exist, but it will not be exported anymore, even if a
// Summary with the same label values is created later. See also the CounterVec
// example.
//
// An error is returned if the number of label values is not the same as the
// number of VariableLabels in Desc (minus any curried labels).
//
// Note that for more than one label value, this method is prone to mistakes
// caused by an incorrect order of arguments. Consider GetMetricWith(Labels) as
// an alternative to avoid that type of mistake. For higher label numbers, the
// latter has a much more readable (albeit more verbose) syntax, but it comes
// with a performance overhead (for creating and processing the Labels map).
// See also the GaugeVec example.
func (v *SummaryVec) GetMetricWithLabelValues(lvs ...string) (Observer, error) {
	metric, err := v.metricVec.getMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(Observer), err
	}
	return nil, err
}

// GetMetricWith returns the Summary for the given Labels map (the label names
// must match those of the VariableLabels in Desc). If that label map is
// accessed for the first time, a new Summary is created. Implications of
// creating a Summary without using it and keeping the Summary for later use are
// the same as for GetMetricWithLabelValues.
//
// An error is returned if the number and names of the Labels are inconsistent
// with those of the VariableLabels in Desc (minus any curried labels).
//
// This method is used for the same purpose as
// GetMetricWithLabelValues(...string). See there for pros and cons of the two
// methods.
func (v *SummaryVec) GetMetricWith(labels Labels) (Observer, error) {
	metric, err := v.metricVec.getMetricWith(labels)
	if metric != nil {
		return metric.(Observer), err
	}
	return nil, err
}

// WithLabelValues works as GetMetricWithLabelValues, but panics where
// GetMetricWithLabelValues would have returned an error. Not returning an
// error allows shortcuts like
//     myVec.WithLabelValues("404", "GET").Observe(42.21)
func (v *SummaryVec) WithLabelValues(lvs ...string) Observer {
	s, err := v.GetMetricWithLabelValues(lvs...)
	if err != nil {
		panic(err)
	}
	return s
}

// With works as GetMetricWith, but panics where GetMetricWithLabels would have
// returned an error. Not returning an error allows shortcuts like
//     myVec.With(prometheus.Labels{"code": "404", "method": "GET"}).Observe(42.21)
func (v *SummaryVec) With(labels Labels) Observer {
	s, err := v.GetMetricWith(labels)
	if err != nil {
		panic(err)
	}
	return s
}

// CurryWith returns a vector curried with the provided labels, i.e. the
// returned vector has those labels pre-set for all labeled operations performed
// on it. The cardinality of the curried vector is reduced accordingly. The
// order of the remaining labels stays the same (just with the curried labels
// taken out of the sequence – which is relevant for the
// (GetMetric)WithLabelValues methods). It is possible to curry a curried
// vector, but only with labels not yet used for currying before.
//
// The metrics contained in the SummaryVec are shared between the curried and
// uncurried vectors. They are just accessed differently. Curried and uncurried
// vectors behave identically in terms of collection. Only one must be
// registered with a given registry (usually the uncurried version). The Reset
// method deletes all metrics, even if called on a curried vector.
func (v *SummaryVec) CurryWith(labels Labels) (ObserverVec, error) {
	vec, err := v.curryWith(labels)
	if vec != nil {
		return &SummaryVec{vec}, err
	}
	return nil, err
}

// MustCurryWith works as CurryWith but panics where CurryWith would have
// returned an error.
func (v *SummaryVec) MustCurryWith(labels Labels) ObserverVec {
	vec, err := v.CurryWith(labels)
	if err != nil {
		panic(err)
	}
	return vec
}

type constSummary struct {
	desc       *Desc
	count      uint64
	sum        float64
	quantiles  map[float64]float64
	labelPairs []*dto.LabelPair
}

func (s *constSummary) Desc() *Desc {
	return s.desc
}

func (s *constSummary) Write(out *dto.Metric) error {
	sum := &dto.Summary{}
	qs := make([]*dto.Quantile, 0, len(s.quantiles))

	sum.SampleCount = proto.Uint64(s.count)
	sum.SampleSum = proto.Float64(s.sum)

	for rank, q := range s.quantiles {
		qs = append(qs, &dto.Quantile{
			Quantile: proto.Float64(rank),
			Value:    proto.Float64(q),
		})
	}

	if len(qs) > 0 {
		sort.Sort(quantSort(qs))
	}
	sum.Quantile = qs

	out.Summary = sum
	out.Label = s.labelPairs

	return nil
}

// NewConstSummary returns a metric representing a Prometheus summary with fixed
// values for the count, sum, and quantiles. As those parameters cannot be
// changed, the returned value does not implement the Summary interface (but
// only the Metric interface). Users of this package will not have much use for
// it in regular operations. However, when implementing custom Collectors, it is
// useful as a throw-away metric that is generated on the fly to send it to
// Prometheus in the Collect method.
//
// quantiles maps ranks to quantile values. For example, a median latency of
// 0.23s and a 99th percentile latency of 0.56s would be expressed as:
//     map[float64]float64{0.5: 0.23, 0.99: 0.56}
//
// NewConstSummary returns an error if the length of labelValues is not
// consistent with the variable labels in Desc or if Desc is invalid.
func NewConstSummary(
	desc *Desc,
	count uint64,
	sum float64,
	quantiles map[float64]float64,
	labelValues ...string,
) (Metric, error) {
	if desc.err != nil {
		return nil, desc.err
	}
	if err := validateLabelValues(labelValues, len(desc.variableLabels)); err != nil {
		return nil, err
	}
	return &constSummary{
		desc:       desc,
		count:      count,
		sum:        sum,
		quantiles:  quantiles,
		labelPairs: makeLabelPairs(desc, labelValues),
	}, nil
}

// MustNewConstSummary is a version of NewConstSummary that panics where
// NewConstMetric would have returned an error.
func MustNewConstSummary(
	desc *Desc,
	count uint64,
	sum float64,
	quantiles map[float64]float64,
	labelValues ...string,
) Metric {
	m, err := NewConstSummary(desc, count, sum, quantiles, labelValues...)
	if err != nil {
		panic(err)
	}
	return m
}
