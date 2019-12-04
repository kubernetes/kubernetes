// Copyright 2019, OpenCensus Authors
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
//

package view

import (
	"time"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/stats"
)

func getUnit(unit string) metricdata.Unit {
	switch unit {
	case "1":
		return metricdata.UnitDimensionless
	case "ms":
		return metricdata.UnitMilliseconds
	case "By":
		return metricdata.UnitBytes
	}
	return metricdata.UnitDimensionless
}

func getType(v *View) metricdata.Type {
	m := v.Measure
	agg := v.Aggregation

	switch agg.Type {
	case AggTypeSum:
		switch m.(type) {
		case *stats.Int64Measure:
			return metricdata.TypeCumulativeInt64
		case *stats.Float64Measure:
			return metricdata.TypeCumulativeFloat64
		default:
			panic("unexpected measure type")
		}
	case AggTypeDistribution:
		return metricdata.TypeCumulativeDistribution
	case AggTypeLastValue:
		switch m.(type) {
		case *stats.Int64Measure:
			return metricdata.TypeGaugeInt64
		case *stats.Float64Measure:
			return metricdata.TypeGaugeFloat64
		default:
			panic("unexpected measure type")
		}
	case AggTypeCount:
		switch m.(type) {
		case *stats.Int64Measure:
			return metricdata.TypeCumulativeInt64
		case *stats.Float64Measure:
			return metricdata.TypeCumulativeInt64
		default:
			panic("unexpected measure type")
		}
	default:
		panic("unexpected aggregation type")
	}
}

func getLableKeys(v *View) []metricdata.LabelKey {
	labelKeys := []metricdata.LabelKey{}
	for _, k := range v.TagKeys {
		labelKeys = append(labelKeys, metricdata.LabelKey{Key: k.Name()})
	}
	return labelKeys
}

func viewToMetricDescriptor(v *View) *metricdata.Descriptor {
	return &metricdata.Descriptor{
		Name:        v.Name,
		Description: v.Description,
		Unit:        getUnit(v.Measure.Unit()),
		Type:        getType(v),
		LabelKeys:   getLableKeys(v),
	}
}

func toLabelValues(row *Row, expectedKeys []metricdata.LabelKey) []metricdata.LabelValue {
	labelValues := []metricdata.LabelValue{}
	tagMap := make(map[string]string)
	for _, tag := range row.Tags {
		tagMap[tag.Key.Name()] = tag.Value
	}

	for _, key := range expectedKeys {
		if val, ok := tagMap[key.Key]; ok {
			labelValues = append(labelValues, metricdata.NewLabelValue(val))
		} else {
			labelValues = append(labelValues, metricdata.LabelValue{})
		}
	}
	return labelValues
}

func rowToTimeseries(v *viewInternal, row *Row, now time.Time, startTime time.Time) *metricdata.TimeSeries {
	return &metricdata.TimeSeries{
		Points:      []metricdata.Point{row.Data.toPoint(v.metricDescriptor.Type, now)},
		LabelValues: toLabelValues(row, v.metricDescriptor.LabelKeys),
		StartTime:   startTime,
	}
}

func viewToMetric(v *viewInternal, now time.Time, startTime time.Time) *metricdata.Metric {
	if v.metricDescriptor.Type == metricdata.TypeGaugeInt64 ||
		v.metricDescriptor.Type == metricdata.TypeGaugeFloat64 {
		startTime = time.Time{}
	}

	rows := v.collectedRows()
	if len(rows) == 0 {
		return nil
	}

	ts := []*metricdata.TimeSeries{}
	for _, row := range rows {
		ts = append(ts, rowToTimeseries(v, row, now, startTime))
	}

	m := &metricdata.Metric{
		Descriptor: *v.metricDescriptor,
		TimeSeries: ts,
	}
	return m
}
