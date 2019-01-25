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
	"sort"

	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"
)

// ValueType is an enumeration of metric types that represent a simple value.
type ValueType int

// Possible values for the ValueType enum.
const (
	_ ValueType = iota
	CounterValue
	GaugeValue
	UntypedValue
)

// valueFunc is a generic metric for simple values retrieved on collect time
// from a function. It implements Metric and Collector. Its effective type is
// determined by ValueType. This is a low-level building block used by the
// library to back the implementations of CounterFunc, GaugeFunc, and
// UntypedFunc.
type valueFunc struct {
	selfCollector

	desc       *Desc
	valType    ValueType
	function   func() float64
	labelPairs []*dto.LabelPair
}

// newValueFunc returns a newly allocated valueFunc with the given Desc and
// ValueType. The value reported is determined by calling the given function
// from within the Write method. Take into account that metric collection may
// happen concurrently. If that results in concurrent calls to Write, like in
// the case where a valueFunc is directly registered with Prometheus, the
// provided function must be concurrency-safe.
func newValueFunc(desc *Desc, valueType ValueType, function func() float64) *valueFunc {
	result := &valueFunc{
		desc:       desc,
		valType:    valueType,
		function:   function,
		labelPairs: makeLabelPairs(desc, nil),
	}
	result.init(result)
	return result
}

func (v *valueFunc) Desc() *Desc {
	return v.desc
}

func (v *valueFunc) Write(out *dto.Metric) error {
	return populateMetric(v.valType, v.function(), v.labelPairs, out)
}

// NewConstMetric returns a metric with one fixed value that cannot be
// changed. Users of this package will not have much use for it in regular
// operations. However, when implementing custom Collectors, it is useful as a
// throw-away metric that is generated on the fly to send it to Prometheus in
// the Collect method. NewConstMetric returns an error if the length of
// labelValues is not consistent with the variable labels in Desc or if Desc is
// invalid.
func NewConstMetric(desc *Desc, valueType ValueType, value float64, labelValues ...string) (Metric, error) {
	if desc.err != nil {
		return nil, desc.err
	}
	if err := validateLabelValues(labelValues, len(desc.variableLabels)); err != nil {
		return nil, err
	}
	return &constMetric{
		desc:       desc,
		valType:    valueType,
		val:        value,
		labelPairs: makeLabelPairs(desc, labelValues),
	}, nil
}

// MustNewConstMetric is a version of NewConstMetric that panics where
// NewConstMetric would have returned an error.
func MustNewConstMetric(desc *Desc, valueType ValueType, value float64, labelValues ...string) Metric {
	m, err := NewConstMetric(desc, valueType, value, labelValues...)
	if err != nil {
		panic(err)
	}
	return m
}

type constMetric struct {
	desc       *Desc
	valType    ValueType
	val        float64
	labelPairs []*dto.LabelPair
}

func (m *constMetric) Desc() *Desc {
	return m.desc
}

func (m *constMetric) Write(out *dto.Metric) error {
	return populateMetric(m.valType, m.val, m.labelPairs, out)
}

func populateMetric(
	t ValueType,
	v float64,
	labelPairs []*dto.LabelPair,
	m *dto.Metric,
) error {
	m.Label = labelPairs
	switch t {
	case CounterValue:
		m.Counter = &dto.Counter{Value: proto.Float64(v)}
	case GaugeValue:
		m.Gauge = &dto.Gauge{Value: proto.Float64(v)}
	case UntypedValue:
		m.Untyped = &dto.Untyped{Value: proto.Float64(v)}
	default:
		return fmt.Errorf("encountered unknown type %v", t)
	}
	return nil
}

func makeLabelPairs(desc *Desc, labelValues []string) []*dto.LabelPair {
	totalLen := len(desc.variableLabels) + len(desc.constLabelPairs)
	if totalLen == 0 {
		// Super fast path.
		return nil
	}
	if len(desc.variableLabels) == 0 {
		// Moderately fast path.
		return desc.constLabelPairs
	}
	labelPairs := make([]*dto.LabelPair, 0, totalLen)
	for i, n := range desc.variableLabels {
		labelPairs = append(labelPairs, &dto.LabelPair{
			Name:  proto.String(n),
			Value: proto.String(labelValues[i]),
		})
	}
	labelPairs = append(labelPairs, desc.constLabelPairs...)
	sort.Sort(labelPairSorter(labelPairs))
	return labelPairs
}
