// Copyright 2013 The Prometheus Authors
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

package extraction

import (
	"fmt"
	"io"
	"math"

	dto "github.com/prometheus/client_model/go"

	"github.com/matttproud/golang_protobuf_extensions/pbutil"

	"github.com/prometheus/client_golang/model"
)

type metricFamilyProcessor struct{}

// MetricFamilyProcessor decodes varint encoded record length-delimited streams
// of io.prometheus.client.MetricFamily.
//
// See http://godoc.org/github.com/matttproud/golang_protobuf_extensions/ext for
// more details.
var MetricFamilyProcessor = &metricFamilyProcessor{}

func (m *metricFamilyProcessor) ProcessSingle(i io.Reader, out Ingester, o *ProcessOptions) error {
	family := &dto.MetricFamily{}

	for {
		family.Reset()

		if _, err := pbutil.ReadDelimited(i, family); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if err := extractMetricFamily(out, o, family); err != nil {
			return err
		}
	}
}

func extractMetricFamily(out Ingester, o *ProcessOptions, family *dto.MetricFamily) error {
	switch family.GetType() {
	case dto.MetricType_COUNTER:
		if err := extractCounter(out, o, family); err != nil {
			return err
		}
	case dto.MetricType_GAUGE:
		if err := extractGauge(out, o, family); err != nil {
			return err
		}
	case dto.MetricType_SUMMARY:
		if err := extractSummary(out, o, family); err != nil {
			return err
		}
	case dto.MetricType_UNTYPED:
		if err := extractUntyped(out, o, family); err != nil {
			return err
		}
	case dto.MetricType_HISTOGRAM:
		if err := extractHistogram(out, o, family); err != nil {
			return err
		}
	}
	return nil
}

func extractCounter(out Ingester, o *ProcessOptions, f *dto.MetricFamily) error {
	samples := make(model.Samples, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Counter == nil {
			continue
		}

		sample := &model.Sample{
			Metric: model.Metric{},
			Value:  model.SampleValue(m.Counter.GetValue()),
		}
		samples = append(samples, sample)

		if m.TimestampMs != nil {
			sample.Timestamp = model.TimestampFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			sample.Timestamp = o.Timestamp
		}

		metric := sample.Metric
		for _, p := range m.Label {
			metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		metric[model.MetricNameLabel] = model.LabelValue(f.GetName())
	}

	return out.Ingest(samples)
}

func extractGauge(out Ingester, o *ProcessOptions, f *dto.MetricFamily) error {
	samples := make(model.Samples, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Gauge == nil {
			continue
		}

		sample := &model.Sample{
			Metric: model.Metric{},
			Value:  model.SampleValue(m.Gauge.GetValue()),
		}
		samples = append(samples, sample)

		if m.TimestampMs != nil {
			sample.Timestamp = model.TimestampFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			sample.Timestamp = o.Timestamp
		}

		metric := sample.Metric
		for _, p := range m.Label {
			metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		metric[model.MetricNameLabel] = model.LabelValue(f.GetName())
	}

	return out.Ingest(samples)
}

func extractSummary(out Ingester, o *ProcessOptions, f *dto.MetricFamily) error {
	samples := make(model.Samples, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Summary == nil {
			continue
		}

		timestamp := o.Timestamp
		if m.TimestampMs != nil {
			timestamp = model.TimestampFromUnixNano(*m.TimestampMs * 1000000)
		}

		for _, q := range m.Summary.Quantile {
			sample := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(q.GetValue()),
				Timestamp: timestamp,
			}
			samples = append(samples, sample)

			metric := sample.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			// BUG(matt): Update other names to "quantile".
			metric[model.LabelName(model.QuantileLabel)] = model.LabelValue(fmt.Sprint(q.GetQuantile()))
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName())
		}

		if m.Summary.SampleSum != nil {
			sum := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(m.Summary.GetSampleSum()),
				Timestamp: timestamp,
			}
			samples = append(samples, sum)

			metric := sum.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_sum")
		}

		if m.Summary.SampleCount != nil {
			count := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(m.Summary.GetSampleCount()),
				Timestamp: timestamp,
			}
			samples = append(samples, count)

			metric := count.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_count")
		}
	}

	return out.Ingest(samples)
}

func extractUntyped(out Ingester, o *ProcessOptions, f *dto.MetricFamily) error {
	samples := make(model.Samples, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Untyped == nil {
			continue
		}

		sample := &model.Sample{
			Metric: model.Metric{},
			Value:  model.SampleValue(m.Untyped.GetValue()),
		}
		samples = append(samples, sample)

		if m.TimestampMs != nil {
			sample.Timestamp = model.TimestampFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			sample.Timestamp = o.Timestamp
		}

		metric := sample.Metric
		for _, p := range m.Label {
			metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		metric[model.MetricNameLabel] = model.LabelValue(f.GetName())
	}

	return out.Ingest(samples)
}

func extractHistogram(out Ingester, o *ProcessOptions, f *dto.MetricFamily) error {
	samples := make(model.Samples, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Histogram == nil {
			continue
		}

		timestamp := o.Timestamp
		if m.TimestampMs != nil {
			timestamp = model.TimestampFromUnixNano(*m.TimestampMs * 1000000)
		}

		infSeen := false

		for _, q := range m.Histogram.Bucket {
			sample := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(q.GetCumulativeCount()),
				Timestamp: timestamp,
			}
			samples = append(samples, sample)

			metric := sample.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			metric[model.LabelName(model.BucketLabel)] = model.LabelValue(fmt.Sprint(q.GetUpperBound()))
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_bucket")

			if math.IsInf(q.GetUpperBound(), +1) {
				infSeen = true
			}
		}

		if m.Histogram.SampleSum != nil {
			sum := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(m.Histogram.GetSampleSum()),
				Timestamp: timestamp,
			}
			samples = append(samples, sum)

			metric := sum.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_sum")
		}

		if m.Histogram.SampleCount != nil {
			count := &model.Sample{
				Metric:    model.Metric{},
				Value:     model.SampleValue(m.Histogram.GetSampleCount()),
				Timestamp: timestamp,
			}
			samples = append(samples, count)

			metric := count.Metric
			for _, p := range m.Label {
				metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_count")

			if !infSeen {
				infBucket := &model.Sample{
					Metric:    model.Metric{},
					Value:     count.Value,
					Timestamp: timestamp,
				}
				samples = append(samples, infBucket)

				metric := infBucket.Metric
				for _, p := range m.Label {
					metric[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
				}
				metric[model.LabelName(model.BucketLabel)] = model.LabelValue("+Inf")
				metric[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_bucket")
			}
		}
	}

	return out.Ingest(samples)
}
