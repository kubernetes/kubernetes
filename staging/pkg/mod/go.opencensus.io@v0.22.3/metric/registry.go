// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metric

import (
	"sort"
	"sync"
	"time"

	"go.opencensus.io/metric/metricdata"
)

// Registry creates and manages a set of gauges and cumulative.
// External synchronization is required if you want to add gauges and cumulative to the same
// registry from multiple goroutines.
type Registry struct {
	baseMetrics sync.Map
}

type metricOptions struct {
	unit        metricdata.Unit
	labelkeys   []metricdata.LabelKey
	constLabels map[metricdata.LabelKey]metricdata.LabelValue
	desc        string
}

// Options apply changes to metricOptions.
type Options func(*metricOptions)

// WithDescription applies provided description.
func WithDescription(desc string) Options {
	return func(mo *metricOptions) {
		mo.desc = desc
	}
}

// WithUnit applies provided unit.
func WithUnit(unit metricdata.Unit) Options {
	return func(mo *metricOptions) {
		mo.unit = unit
	}
}

// WithLabelKeys applies provided label.
func WithLabelKeys(keys ...string) Options {
	return func(mo *metricOptions) {
		labelKeys := make([]metricdata.LabelKey, 0)
		for _, key := range keys {
			labelKeys = append(labelKeys, metricdata.LabelKey{Key: key})
		}
		mo.labelkeys = labelKeys
	}
}

// WithLabelKeysAndDescription applies provided label.
func WithLabelKeysAndDescription(labelKeys ...metricdata.LabelKey) Options {
	return func(mo *metricOptions) {
		mo.labelkeys = labelKeys
	}
}

// WithConstLabel applies provided constant label.
func WithConstLabel(constLabels map[metricdata.LabelKey]metricdata.LabelValue) Options {
	return func(mo *metricOptions) {
		mo.constLabels = constLabels
	}
}

// NewRegistry initializes a new Registry.
func NewRegistry() *Registry {
	return &Registry{}
}

// AddFloat64Gauge creates and adds a new float64-valued gauge to this registry.
func (r *Registry) AddFloat64Gauge(name string, mos ...Options) (*Float64Gauge, error) {
	f := &Float64Gauge{
		bm: baseMetric{
			bmType: gaugeFloat64,
		},
	}
	_, err := r.initBaseMetric(&f.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return f, nil
}

// AddInt64Gauge creates and adds a new int64-valued gauge to this registry.
func (r *Registry) AddInt64Gauge(name string, mos ...Options) (*Int64Gauge, error) {
	i := &Int64Gauge{
		bm: baseMetric{
			bmType: gaugeInt64,
		},
	}
	_, err := r.initBaseMetric(&i.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return i, nil
}

// AddInt64DerivedGauge creates and adds a new derived int64-valued gauge to this registry.
// A derived gauge is convenient form of gauge where the object associated with the gauge
// provides its value by implementing func() int64.
func (r *Registry) AddInt64DerivedGauge(name string, mos ...Options) (*Int64DerivedGauge, error) {
	i := &Int64DerivedGauge{
		bm: baseMetric{
			bmType: derivedGaugeInt64,
		},
	}
	_, err := r.initBaseMetric(&i.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return i, nil
}

// AddFloat64DerivedGauge creates and adds a new derived float64-valued gauge to this registry.
// A derived gauge is convenient form of gauge where the object associated with the gauge
// provides its value by implementing func() float64.
func (r *Registry) AddFloat64DerivedGauge(name string, mos ...Options) (*Float64DerivedGauge, error) {
	f := &Float64DerivedGauge{
		bm: baseMetric{
			bmType: derivedGaugeFloat64,
		},
	}
	_, err := r.initBaseMetric(&f.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func bmTypeToMetricType(bm *baseMetric) metricdata.Type {
	switch bm.bmType {
	case derivedGaugeFloat64:
		return metricdata.TypeGaugeFloat64
	case derivedGaugeInt64:
		return metricdata.TypeGaugeInt64
	case gaugeFloat64:
		return metricdata.TypeGaugeFloat64
	case gaugeInt64:
		return metricdata.TypeGaugeInt64
	case derivedCumulativeFloat64:
		return metricdata.TypeCumulativeFloat64
	case derivedCumulativeInt64:
		return metricdata.TypeCumulativeInt64
	case cumulativeFloat64:
		return metricdata.TypeCumulativeFloat64
	case cumulativeInt64:
		return metricdata.TypeCumulativeInt64
	default:
		panic("unsupported metric type")
	}
}

// AddFloat64Cumulative creates and adds a new float64-valued cumulative to this registry.
func (r *Registry) AddFloat64Cumulative(name string, mos ...Options) (*Float64Cumulative, error) {
	f := &Float64Cumulative{
		bm: baseMetric{
			bmType: cumulativeFloat64,
		},
	}
	_, err := r.initBaseMetric(&f.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return f, nil
}

// AddInt64Cumulative creates and adds a new int64-valued cumulative to this registry.
func (r *Registry) AddInt64Cumulative(name string, mos ...Options) (*Int64Cumulative, error) {
	i := &Int64Cumulative{
		bm: baseMetric{
			bmType: cumulativeInt64,
		},
	}
	_, err := r.initBaseMetric(&i.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return i, nil
}

// AddInt64DerivedCumulative creates and adds a new derived int64-valued cumulative to this registry.
// A derived cumulative is convenient form of cumulative where the object associated with the cumulative
// provides its value by implementing func() int64.
func (r *Registry) AddInt64DerivedCumulative(name string, mos ...Options) (*Int64DerivedCumulative, error) {
	i := &Int64DerivedCumulative{
		bm: baseMetric{
			bmType: derivedCumulativeInt64,
		},
	}
	_, err := r.initBaseMetric(&i.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return i, nil
}

// AddFloat64DerivedCumulative creates and adds a new derived float64-valued gauge to this registry.
// A derived cumulative is convenient form of cumulative where the object associated with the cumulative
// provides its value by implementing func() float64.
func (r *Registry) AddFloat64DerivedCumulative(name string, mos ...Options) (*Float64DerivedCumulative, error) {
	f := &Float64DerivedCumulative{
		bm: baseMetric{
			bmType: derivedCumulativeFloat64,
		},
	}
	_, err := r.initBaseMetric(&f.bm, name, mos...)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func createMetricOption(mos ...Options) *metricOptions {
	o := &metricOptions{}
	for _, mo := range mos {
		mo(o)
	}
	return o
}

func (r *Registry) initBaseMetric(bm *baseMetric, name string, mos ...Options) (*baseMetric, error) {
	val, ok := r.baseMetrics.Load(name)
	if ok {
		existing := val.(*baseMetric)
		if existing.bmType != bm.bmType {
			return nil, errMetricExistsWithDiffType
		}
	}
	bm.start = time.Now()
	o := createMetricOption(mos...)

	var constLabelKeys []metricdata.LabelKey
	for k := range o.constLabels {
		constLabelKeys = append(constLabelKeys, k)
	}
	sort.Slice(constLabelKeys, func(i, j int) bool {
		return constLabelKeys[i].Key < constLabelKeys[j].Key
	})

	var constLabelValues []metricdata.LabelValue
	for _, k := range constLabelKeys {
		constLabelValues = append(constLabelValues, o.constLabels[k])
	}

	bm.keys = append(constLabelKeys, o.labelkeys...)
	bm.constLabelValues = constLabelValues

	bm.desc = metricdata.Descriptor{
		Name:        name,
		Description: o.desc,
		Unit:        o.unit,
		LabelKeys:   bm.keys,
		Type:        bmTypeToMetricType(bm),
	}
	r.baseMetrics.Store(name, bm)
	return bm, nil
}

// Read reads all gauges and cumulatives in this registry and returns their values as metrics.
func (r *Registry) Read() []*metricdata.Metric {
	ms := []*metricdata.Metric{}
	r.baseMetrics.Range(func(k, v interface{}) bool {
		bm := v.(*baseMetric)
		ms = append(ms, bm.read())
		return true
	})
	return ms
}
