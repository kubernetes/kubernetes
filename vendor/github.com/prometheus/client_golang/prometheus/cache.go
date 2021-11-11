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

package prometheus

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/golang/protobuf/proto"
	"github.com/prometheus/client_golang/prometheus/internal"
	dto "github.com/prometheus/client_model/go"
)

var _ rawCollector = &CachedCollector{}

// CachedCollector allows creating allocation friendly metrics which change less frequently than scrape time, yet
// label values can are changing over time. This collector
//
// If you happen to use NewDesc, NewConstMetric or MustNewConstMetric inside Collector.Collect routine, consider
// using CachedCollector instead.
type CachedCollector struct {
	metrics            map[uint64]*dto.Metric
	metricFamilyByName map[string]*dto.MetricFamily

	pendingSession bool
}

func NewCachedCollector() *CachedCollector {
	return &CachedCollector{
		metrics:            make(map[uint64]*dto.Metric),
		metricFamilyByName: map[string]*dto.MetricFamily{},
	}
}

func (c *CachedCollector) Collect() []*dto.MetricFamily {
	// TODO(bwplotka): Optimize potential penalty here.
	return internal.NormalizeMetricFamilies(c.metricFamilyByName)
}

// NewSession allows to collect all metrics in one go and update cache as much in-place
// as possible to save allocations.
// NOTE: Not concurrency safe and only one allowed at the time (until commit).
func (c *CachedCollector) NewSession() *CollectSession {
	c.pendingSession = true
	return &CollectSession{
		c:              c,
		currentMetrics: make(map[uint64]*dto.Metric, len(c.metrics)),
		currentByName:  make(map[string]*dto.MetricFamily, len(c.metricFamilyByName)),
	}
}

type CollectSession struct {
	closed bool

	c              *CachedCollector
	currentMetrics map[uint64]*dto.Metric
	currentByName  map[string]*dto.MetricFamily
}

func (s *CollectSession) Commit() {
	// TODO(bwplotka): Sort metrics within family.
	s.c.metricFamilyByName = s.currentByName
	s.c.metrics = s.currentMetrics

	s.closed = true
	s.c.pendingSession = false
}

func (s *CollectSession) MustAddMetric(fqName, help string, labelNames, labelValues []string, valueType ValueType, value float64, ts *time.Time) {
	if err := s.AddMetric(fqName, help, labelNames, labelValues, valueType, value, ts); err != nil {
		panic(err)
	}
}

// AddMetric ...
// TODO(bwplotka): Add validation.
func (s *CollectSession) AddMetric(fqName, help string, labelNames, labelValues []string, valueType ValueType, value float64, ts *time.Time) error {
	if s.closed {
		return errors.New("new metric: collect session is closed, but was attempted to be used")
	}

	// Label names can be unsorted, will be sorting them later. The only implication is cachability if
	// consumer provide non-deterministic order of those (unlikely since label values has to be matched.

	if len(labelNames) != len(labelValues) {
		return errors.New("new metric: label name has different len than values")
	}

	d, ok := s.currentByName[fqName]
	if !ok {
		d, ok = s.c.metricFamilyByName[fqName]
		if ok {
			d.Metric = d.Metric[:0]
		}
	}

	if !ok {
		// TODO(bwplotka): Validate?
		d = &dto.MetricFamily{}
		d.Name = proto.String(fqName)
		d.Type = valueType.ToDTO()
		d.Help = proto.String(help)
	} else {
		// TODO(bwplotka): Validate if same family?
		d.Type = valueType.ToDTO()
		d.Help = proto.String(help)
	}
	s.currentByName[fqName] = d

	h := xxhash.New()
	h.WriteString(fqName)
	h.Write(separatorByteSlice)
	for i := range labelNames {
		h.WriteString(labelNames[i])
		h.Write(separatorByteSlice)
		h.WriteString(labelValues[i])
		h.Write(separatorByteSlice)
	}
	hSum := h.Sum64()

	if _, ok := s.currentMetrics[hSum]; ok {
		return fmt.Errorf("found duplicate metric (same labels and values) to add %v", fqName)
	}
	m, ok := s.c.metrics[hSum]
	if !ok {
		m = &dto.Metric{
			Label: make([]*dto.LabelPair, 0, len(labelNames)),
		}
		for i := range labelNames {
			m.Label = append(m.Label, &dto.LabelPair{
				Name:  proto.String(labelNames[i]),
				Value: proto.String(labelValues[i]),
			})
		}
		sort.Sort(labelPairSorter(m.Label))
	}
	s.currentMetrics[hSum] = m
	switch valueType {
	case CounterValue:
		v := m.Counter
		if v == nil {
			v = &dto.Counter{}
		}
		v.Value = proto.Float64(value)
		m.Counter = v
		m.Gauge = nil
		m.Untyped = nil
	case GaugeValue:
		v := m.Gauge
		if v == nil {
			v = &dto.Gauge{}
		}
		v.Value = proto.Float64(value)
		m.Counter = nil
		m.Gauge = v
		m.Untyped = nil
	case UntypedValue:
		v := m.Untyped
		if v == nil {
			v = &dto.Untyped{}
		}
		v.Value = proto.Float64(value)
		m.Counter = nil
		m.Gauge = nil
		m.Untyped = v
	default:
		return fmt.Errorf("unsupported value type %v", valueType)
	}

	m.TimestampMs = nil
	if ts != nil {
		m.TimestampMs = proto.Int64(ts.Unix()*1000 + int64(ts.Nanosecond()/1000000))
	}

	// Will be sorted later.
	d.Metric = append(d.Metric, m)
	return nil
}

type BlockingRegistry struct {
	*Registry

	// rawCollector represents special collectors which requires blocking collect for the whole duration
	// of returned dto.MetricFamily usage.
	rawCollectors []rawCollector
	mu            sync.Mutex
}

func NewBlockingRegistry() *BlockingRegistry {
	return &BlockingRegistry{
		Registry: NewRegistry(),
	}
}

type rawCollector interface {
	Collect() []*dto.MetricFamily
}

func (b *BlockingRegistry) RegisterRaw(r rawCollector) error {
	// TODO(bwplotka): Register, I guess for dups/check purposes?
	b.rawCollectors = append(b.rawCollectors, r)
	return nil
}

func (b *BlockingRegistry) MustRegisterRaw(r rawCollector) {
	if err := b.RegisterRaw(r); err != nil {
		panic(err)
	}
}

func (b *BlockingRegistry) Gather() (_ []*dto.MetricFamily, done func(), err error) {
	b.mu.Lock()
	mfs, err := b.Registry.Gather()

	// TODO(bwplotka): Returned mfs are sorted, so sort raw ones and inject?
	// TODO(bwplotka): Implement concurrency for those?
	for _, r := range b.rawCollectors {
		// TODO(bwplotka): Check for duplicates.
		mfs = append(mfs, r.Collect()...)
	}

	// TODO(bwplotka): Consider sort in place, given metric family in gather is sorted already.
	sort.Slice(mfs, func(i, j int) bool {
		return *mfs[i].Name < *mfs[j].Name
	})
	return mfs, func() { b.mu.Unlock() }, err
}

// TransactionalGatherer ...
type TransactionalGatherer interface {
	// Gather ...
	Gather() (_ []*dto.MetricFamily, done func(), err error)
}

func ToTransactionalGatherer(g Gatherer) TransactionalGatherer {
	return &noTransactionGatherer{g: g}
}

type noTransactionGatherer struct {
	g Gatherer
}

func (g *noTransactionGatherer) Gather() (_ []*dto.MetricFamily, done func(), err error) {
	mfs, err := g.g.Gather()
	return mfs, func() {}, err
}
