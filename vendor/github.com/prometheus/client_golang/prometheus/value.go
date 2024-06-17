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
	"errors"
	"fmt"
	"sort"
	"time"
	"unicode/utf8"

	"github.com/prometheus/client_golang/prometheus/internal"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ValueType is an enumeration of metric types that represent a simple value.
type ValueType int

// Possible values for the ValueType enum. Use UntypedValue to mark a metric
// with an unknown type.
const (
	_ ValueType = iota
	CounterValue
	GaugeValue
	UntypedValue
)

var (
	CounterMetricTypePtr = func() *dto.MetricType { d := dto.MetricType_COUNTER; return &d }()
	GaugeMetricTypePtr   = func() *dto.MetricType { d := dto.MetricType_GAUGE; return &d }()
	UntypedMetricTypePtr = func() *dto.MetricType { d := dto.MetricType_UNTYPED; return &d }()
)

func (v ValueType) ToDTO() *dto.MetricType {
	switch v {
	case CounterValue:
		return CounterMetricTypePtr
	case GaugeValue:
		return GaugeMetricTypePtr
	default:
		return UntypedMetricTypePtr
	}
}

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
		labelPairs: MakeLabelPairs(desc, nil),
	}
	result.init(result)
	return result
}

func (v *valueFunc) Desc() *Desc {
	return v.desc
}

func (v *valueFunc) Write(out *dto.Metric) error {
	return populateMetric(v.valType, v.function(), v.labelPairs, nil, out, nil)
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
	if err := validateLabelValues(labelValues, len(desc.variableLabels.names)); err != nil {
		return nil, err
	}

	metric := &dto.Metric{}
	if err := populateMetric(valueType, value, MakeLabelPairs(desc, labelValues), nil, metric, nil); err != nil {
		return nil, err
	}

	return &constMetric{
		desc:   desc,
		metric: metric,
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

// NewConstMetricWithCreatedTimestamp does the same thing as NewConstMetric, but generates Counters
// with created timestamp set and returns an error for other metric types.
func NewConstMetricWithCreatedTimestamp(desc *Desc, valueType ValueType, value float64, ct time.Time, labelValues ...string) (Metric, error) {
	if desc.err != nil {
		return nil, desc.err
	}
	if err := validateLabelValues(labelValues, len(desc.variableLabels.names)); err != nil {
		return nil, err
	}
	switch valueType {
	case CounterValue:
		break
	default:
		return nil, errors.New("created timestamps are only supported for counters")
	}

	metric := &dto.Metric{}
	if err := populateMetric(valueType, value, MakeLabelPairs(desc, labelValues), nil, metric, timestamppb.New(ct)); err != nil {
		return nil, err
	}

	return &constMetric{
		desc:   desc,
		metric: metric,
	}, nil
}

// MustNewConstMetricWithCreatedTimestamp is a version of NewConstMetricWithCreatedTimestamp that panics where
// NewConstMetricWithCreatedTimestamp would have returned an error.
func MustNewConstMetricWithCreatedTimestamp(desc *Desc, valueType ValueType, value float64, ct time.Time, labelValues ...string) Metric {
	m, err := NewConstMetricWithCreatedTimestamp(desc, valueType, value, ct, labelValues...)
	if err != nil {
		panic(err)
	}
	return m
}

type constMetric struct {
	desc   *Desc
	metric *dto.Metric
}

func (m *constMetric) Desc() *Desc {
	return m.desc
}

func (m *constMetric) Write(out *dto.Metric) error {
	out.Label = m.metric.Label
	out.Counter = m.metric.Counter
	out.Gauge = m.metric.Gauge
	out.Untyped = m.metric.Untyped
	return nil
}

func populateMetric(
	t ValueType,
	v float64,
	labelPairs []*dto.LabelPair,
	e *dto.Exemplar,
	m *dto.Metric,
	ct *timestamppb.Timestamp,
) error {
	m.Label = labelPairs
	switch t {
	case CounterValue:
		m.Counter = &dto.Counter{Value: proto.Float64(v), Exemplar: e, CreatedTimestamp: ct}
	case GaugeValue:
		m.Gauge = &dto.Gauge{Value: proto.Float64(v)}
	case UntypedValue:
		m.Untyped = &dto.Untyped{Value: proto.Float64(v)}
	default:
		return fmt.Errorf("encountered unknown type %v", t)
	}
	return nil
}

// MakeLabelPairs is a helper function to create protobuf LabelPairs from the
// variable and constant labels in the provided Desc. The values for the
// variable labels are defined by the labelValues slice, which must be in the
// same order as the corresponding variable labels in the Desc.
//
// This function is only needed for custom Metric implementations. See MetricVec
// example.
func MakeLabelPairs(desc *Desc, labelValues []string) []*dto.LabelPair {
	totalLen := len(desc.variableLabels.names) + len(desc.constLabelPairs)
	if totalLen == 0 {
		// Super fast path.
		return nil
	}
	if len(desc.variableLabels.names) == 0 {
		// Moderately fast path.
		return desc.constLabelPairs
	}
	labelPairs := make([]*dto.LabelPair, 0, totalLen)
	for i, l := range desc.variableLabels.names {
		labelPairs = append(labelPairs, &dto.LabelPair{
			Name:  proto.String(l),
			Value: proto.String(labelValues[i]),
		})
	}
	labelPairs = append(labelPairs, desc.constLabelPairs...)
	sort.Sort(internal.LabelPairSorter(labelPairs))
	return labelPairs
}

// ExemplarMaxRunes is the max total number of runes allowed in exemplar labels.
const ExemplarMaxRunes = 128

// newExemplar creates a new dto.Exemplar from the provided values. An error is
// returned if any of the label names or values are invalid or if the total
// number of runes in the label names and values exceeds ExemplarMaxRunes.
func newExemplar(value float64, ts time.Time, l Labels) (*dto.Exemplar, error) {
	e := &dto.Exemplar{}
	e.Value = proto.Float64(value)
	tsProto := timestamppb.New(ts)
	if err := tsProto.CheckValid(); err != nil {
		return nil, err
	}
	e.Timestamp = tsProto
	labelPairs := make([]*dto.LabelPair, 0, len(l))
	var runes int
	for name, value := range l {
		if !checkLabelName(name) {
			return nil, fmt.Errorf("exemplar label name %q is invalid", name)
		}
		runes += utf8.RuneCountInString(name)
		if !utf8.ValidString(value) {
			return nil, fmt.Errorf("exemplar label value %q is not valid UTF-8", value)
		}
		runes += utf8.RuneCountInString(value)
		labelPairs = append(labelPairs, &dto.LabelPair{
			Name:  proto.String(name),
			Value: proto.String(value),
		})
	}
	if runes > ExemplarMaxRunes {
		return nil, fmt.Errorf("exemplar labels have %d runes, exceeding the limit of %d", runes, ExemplarMaxRunes)
	}
	e.Label = labelPairs
	return e, nil
}
