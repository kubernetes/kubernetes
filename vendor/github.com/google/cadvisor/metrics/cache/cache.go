package cache

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/model"

	//nolint:staticcheck // Ignore SA1019. Need to keep deprecated package for compatibility.
	"github.com/golang/protobuf/proto"
	dto "github.com/prometheus/client_model/go"
)

var _ prometheus.TransactionalGatherer = &CachedTGatherer{}

var separatorByteSlice = []byte{model.SeparatorByte} // For convenient use with xxhash.

// CachedTGatherer is a transactional gatherer that allows maintaining a set of metrics which
// change less frequently than scrape time, yet label values and values change over time.
//
// If you happen to use NewDesc, NewConstMetric or MustNewConstMetric inside Collector.Collect routine, consider
// using CachedTGatherer instead.
//
// Use CachedTGatherer with classic Registry using NewMultiTRegistry and ToTransactionalGatherer helpers.
// NOTE(bwplotka): Experimental, API and behaviour can change.
type CachedTGatherer struct {
	metricFamiliesByName map[string]*family
	mMu                  sync.RWMutex

	locked            bool
	resetMode         bool
	desiredTouchState bool
}

func NewCachedTGatherer() *CachedTGatherer {
	return &CachedTGatherer{
		desiredTouchState:    true,
		resetMode:            true,
		metricFamiliesByName: map[string]*family{},
	}
}

func NewWatchCachedTGatherer() *CachedTGatherer {
	return &CachedTGatherer{
		desiredTouchState:    true,
		resetMode:            false,
		metricFamiliesByName: map[string]*family{},
	}
}

type family struct {
	*dto.MetricFamily

	metricsByHash map[uint64]*metric
	touchState    bool
	needsRebuild  bool
}

type metric struct {
	*dto.Metric
	touchState bool
}

// normalizeMetricFamilies returns a MetricFamily slice with empty
// MetricFamilies pruned and the remaining MetricFamilies sorted by name within
// the slice, with the contained Metrics sorted within each MetricFamily.
func normalizeMetricFamilies(metricFamiliesByName map[string]*family) []*dto.MetricFamily {
	names := make([]string, 0, len(metricFamiliesByName))
	for name, mf := range metricFamiliesByName {
		if len(mf.Metric) > 0 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	result := make([]*dto.MetricFamily, 0, len(names))
	for _, name := range names {
		result = append(result, metricFamiliesByName[name].MetricFamily)
	}
	return result
}

// Gather implements TransactionalGatherer interface.
func (c *CachedTGatherer) Gather() (_ []*dto.MetricFamily, done func(), err error) {
	c.mMu.RLock()
	return normalizeMetricFamilies(c.metricFamiliesByName), c.mMu.RUnlock, nil
}

func (c *CachedTGatherer) closeUpdateSession() {
	if c.resetMode {
		// Trading off-time instead of memory allocated for otherwise needed replacement map.
		for name, mf := range c.metricFamiliesByName {
			if mf.touchState != c.desiredTouchState {
				delete(c.metricFamiliesByName, name)
				continue
			}
			for hash, m := range mf.metricsByHash {
				if m.touchState != c.desiredTouchState {
					delete(mf.metricsByHash, hash)
					continue
				}
			}
			if len(mf.metricsByHash) == 0 {
				delete(c.metricFamiliesByName, name)
			}
		}

		// Avoid resetting state by flipping what we will expect in the next update.
		c.desiredTouchState = !c.desiredTouchState
	}

	for _, mf := range c.metricFamiliesByName {
		if !mf.needsRebuild {
			continue
		}

		mf.Metric = mf.Metric[:0]
		if cap(mf.Metric) < len(mf.metricsByHash) {
			mf.Metric = make([]*dto.Metric, 0, len(mf.metricsByHash))
		}
		for _, m := range mf.metricsByHash {
			mf.Metric = append(mf.Metric, m.Metric)
		}
		sort.Sort(metricSorter(mf.Metric))

		mf.needsRebuild = false
	}

	c.locked = false
	c.mMu.Unlock()
}

func (c *CachedTGatherer) StartUpdateSession() (done func()) {
	c.mMu.Lock()
	c.locked = true

	return c.closeUpdateSession
}

type Metric struct {
	FQName string // __name__
	// Label names can be unsorted, we will be sorting them later. The only implication is cachability if
	// consumer provide non-deterministic order of those.
	LabelNames  []string
	LabelValues []string

	Help      string
	ValueType prometheus.ValueType
	Value     float64

	// Timestamp is optional. Pass nil for no explicit timestamp.
	Timestamp *time.Time
}

// hash returns unique hash for this key.
func (m *Metric) hash() uint64 {
	h := xxhash.New()
	_, _ = h.WriteString(m.FQName)
	_, _ = h.Write(separatorByteSlice)

	for i := range m.LabelNames {
		_, _ = h.WriteString(m.LabelValues[i])
		_, _ = h.Write(separatorByteSlice)
		_, _ = h.WriteString(m.LabelValues[i])
		_, _ = h.Write(separatorByteSlice)
	}
	return h.Sum64()
}

// InsertInPlace all strings are reused, arrays are not reused.
func (c *CachedTGatherer) InsertInPlace(entry Metric) error {
	if !c.locked {
		return errors.New("can't use InsertInPlace without starting session via StartUpdateSession")
	}

	if entry.FQName == "" {
		return errors.New("fqName cannot be empty")
	}
	if len(entry.LabelNames) != len(entry.LabelValues) {
		return errors.New("new metric: label name has different length than values")
	}

	// TODO(bwplotka): Validate FQ name and if lbl names and values are same length?
	// Update metric family.
	mf, ok := c.metricFamiliesByName[entry.FQName]
	if !ok {
		mf = &family{
			MetricFamily: &dto.MetricFamily{
				Name: new(string),
				Help: new(string),
			},
			metricsByHash: map[uint64]*metric{},
		}
		mf.Type = entry.ValueType.ToDTO()
		*mf.Name = entry.FQName
		*mf.Help = entry.Help
	}
	if c.resetMode {
		// Maintain if things were touched for more efficient clean up.
		mf.touchState = c.desiredTouchState
	}

	c.metricFamiliesByName[entry.FQName] = mf

	// Update metric pointer.
	hSum := entry.hash()
	m, ok := mf.metricsByHash[hSum]
	if !ok {
		m = &metric{
			Metric: &dto.Metric{Label: make([]*dto.LabelPair, 0, len(entry.LabelNames))},
		}
		for j := range entry.LabelNames {
			p := &dto.LabelPair{
				Name:  new(string),
				Value: new(string),
			}
			*p.Name = entry.LabelNames[j]
			*p.Value = entry.LabelValues[j]

			m.Label = append(m.Label, p)
		}
		sort.Sort(labelPairSorter(m.Label))
		mf.needsRebuild = true
	}
	if c.resetMode {
		m.touchState = c.desiredTouchState
	}

	switch entry.ValueType {
	case prometheus.CounterValue:
		v := m.Counter
		if v == nil {
			v = &dto.Counter{Value: new(float64)}
		}
		*v.Value = entry.Value
		m.Counter = v
		m.Gauge = nil
		m.Untyped = nil
	case prometheus.GaugeValue:
		v := m.Gauge
		if v == nil {
			v = &dto.Gauge{Value: new(float64)}
		}
		*v.Value = entry.Value
		m.Counter = nil
		m.Gauge = v
		m.Untyped = nil
	case prometheus.UntypedValue:
		v := m.Untyped
		if v == nil {
			v = &dto.Untyped{Value: new(float64)}
		}
		*v.Value = entry.Value
		m.Counter = nil
		m.Gauge = nil
		m.Untyped = v
	default:
		return fmt.Errorf("unsupported value type %v", entry.ValueType)
	}

	m.TimestampMs = nil
	if entry.Timestamp != nil {
		m.TimestampMs = proto.Int64(entry.Timestamp.Unix()*1000 + int64(entry.Timestamp.Nanosecond()/1000000))
	}
	mf.metricsByHash[hSum] = m

	return nil
}

func (c *CachedTGatherer) Delete(entry Metric) error {
	if !c.locked {
		return errors.New("can't use Delete without start session using StartUpdateSession")
	}

	if c.resetMode {
		return errors.New("does not makes sense to delete entries in resetMode")
	}

	if entry.FQName == "" {
		return errors.New("fqName cannot be empty")
	}
	if len(entry.LabelNames) != len(entry.LabelValues) {
		return errors.New("new metric: label name has different length than values")
	}

	mf, ok := c.metricFamiliesByName[entry.FQName]
	if !ok {
		return nil
	}

	hSum := entry.hash()
	if _, ok := mf.metricsByHash[hSum]; !ok {
		return nil
	}

	if len(mf.metricsByHash) == 1 {
		delete(c.metricFamiliesByName, entry.FQName)
		return nil
	}

	mf.needsRebuild = true
	delete(mf.metricsByHash, hSum)

	return nil
}

// labelPairSorter implements sort.Interface. It is used to sort a slice of
// dto.LabelPair pointers.
type labelPairSorter []*dto.LabelPair

func (s labelPairSorter) Len() int {
	return len(s)
}

func (s labelPairSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s labelPairSorter) Less(i, j int) bool {
	return s[i].GetName() < s[j].GetName()
}

// MetricSorter is a sortable slice of *dto.Metric.
type metricSorter []*dto.Metric

func (s metricSorter) Len() int {
	return len(s)
}

func (s metricSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s metricSorter) Less(i, j int) bool {
	if len(s[i].Label) != len(s[j].Label) {
		// This should not happen. The metrics are
		// inconsistent. However, we have to deal with the fact, as
		// people might use custom collectors or metric family injection
		// to create inconsistent metrics. So let's simply compare the
		// number of labels in this case. That will still yield
		// reproducible sorting.
		return len(s[i].Label) < len(s[j].Label)
	}
	for n, lp := range s[i].Label {
		vi := lp.GetValue()
		vj := s[j].Label[n].GetValue()
		if vi != vj {
			return vi < vj
		}
	}

	// We should never arrive here. Multiple metrics with the same
	// label set in the same scrape will lead to undefined ingestion
	// behavior. However, as above, we have to provide stable sorting
	// here, even for inconsistent metrics. So sort equal metrics
	// by their timestamp, with missing timestamps (implying "now")
	// coming last.
	if s[i].TimestampMs == nil {
		return false
	}
	if s[j].TimestampMs == nil {
		return true
	}
	return s[i].GetTimestampMs() < s[j].GetTimestampMs()
}
