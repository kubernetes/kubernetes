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

package expfmt

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/encoding/protodelim"

	"github.com/prometheus/common/model"
)

// Decoder types decode an input stream into metric families.
type Decoder interface {
	Decode(*dto.MetricFamily) error
}

// DecodeOptions contains options used by the Decoder and in sample extraction.
type DecodeOptions struct {
	// Timestamp is added to each value from the stream that has no explicit timestamp set.
	Timestamp model.Time
}

// ResponseFormat extracts the correct format from a HTTP response header.
// If no matching format can be found FormatUnknown is returned.
func ResponseFormat(h http.Header) Format {
	ct := h.Get(hdrContentType)

	mediatype, params, err := mime.ParseMediaType(ct)
	if err != nil {
		return FmtUnknown
	}

	const textType = "text/plain"

	switch mediatype {
	case ProtoType:
		if p, ok := params["proto"]; ok && p != ProtoProtocol {
			return FmtUnknown
		}
		if e, ok := params["encoding"]; ok && e != "delimited" {
			return FmtUnknown
		}
		return FmtProtoDelim

	case textType:
		if v, ok := params["version"]; ok && v != TextVersion {
			return FmtUnknown
		}
		return FmtText
	}

	return FmtUnknown
}

// NewDecoder returns a new decoder based on the given input format. Metric
// names are validated based on the provided Format -- if the format requires
// escaping, raditional Prometheues validity checking is used. Otherwise, names
// are checked for UTF-8 validity. Supported formats include delimited protobuf
// and Prometheus text format.  For historical reasons, this decoder fallbacks
// to classic text decoding for any other format. This decoder does not fully
// support OpenMetrics although it may often succeed due to the similarities
// between the formats. This decoder may not support the latest features of
// Prometheus text format and is not intended for high-performance applications.
// See: https://github.com/prometheus/common/issues/812
func NewDecoder(r io.Reader, format Format) Decoder {
	scheme := model.LegacyValidation
	if format.ToEscapingScheme() == model.NoEscaping {
		scheme = model.UTF8Validation
	}
	switch format.FormatType() {
	case TypeProtoDelim:
		return &protoDecoder{r: bufio.NewReader(r), s: scheme}
	case TypeProtoText, TypeProtoCompact:
		return &errDecoder{err: fmt.Errorf("format %s not supported for decoding", format)}
	}
	return &textDecoder{r: r, s: scheme}
}

// protoDecoder implements the Decoder interface for protocol buffers.
type protoDecoder struct {
	r protodelim.Reader
	s model.ValidationScheme
}

// Decode implements the Decoder interface.
func (d *protoDecoder) Decode(v *dto.MetricFamily) error {
	opts := protodelim.UnmarshalOptions{
		MaxSize: -1,
	}
	if err := opts.UnmarshalFrom(d.r, v); err != nil {
		return err
	}
	if !d.s.IsValidMetricName(v.GetName()) {
		return fmt.Errorf("invalid metric name %q", v.GetName())
	}
	for _, m := range v.GetMetric() {
		if m == nil {
			continue
		}
		for _, l := range m.GetLabel() {
			if l == nil {
				continue
			}
			if !model.LabelValue(l.GetValue()).IsValid() {
				return fmt.Errorf("invalid label value %q", l.GetValue())
			}
			if !d.s.IsValidLabelName(l.GetName()) {
				return fmt.Errorf("invalid label name %q", l.GetName())
			}
		}
	}
	return nil
}

// errDecoder is an error-state decoder that always returns the same error.
type errDecoder struct {
	err error
}

func (d *errDecoder) Decode(*dto.MetricFamily) error {
	return d.err
}

// textDecoder implements the Decoder interface for the text protocol.
type textDecoder struct {
	r    io.Reader
	fams map[string]*dto.MetricFamily
	s    model.ValidationScheme
	err  error
}

// Decode implements the Decoder interface.
func (d *textDecoder) Decode(v *dto.MetricFamily) error {
	if d.err == nil {
		// Read all metrics in one shot.
		p := NewTextParser(d.s)
		d.fams, d.err = p.TextToMetricFamilies(d.r)
		// If we don't get an error, store io.EOF for the end.
		if d.err == nil {
			d.err = io.EOF
		}
	}
	// Pick off one MetricFamily per Decode until there's nothing left.
	for key, fam := range d.fams {
		v.Name = fam.Name
		v.Help = fam.Help
		v.Type = fam.Type
		v.Metric = fam.Metric
		delete(d.fams, key)
		return nil
	}
	return d.err
}

// SampleDecoder wraps a Decoder to extract samples from the metric families
// decoded by the wrapped Decoder.
type SampleDecoder struct {
	Dec  Decoder
	Opts *DecodeOptions

	f dto.MetricFamily
}

// Decode calls the Decode method of the wrapped Decoder and then extracts the
// samples from the decoded MetricFamily into the provided model.Vector.
func (sd *SampleDecoder) Decode(s *model.Vector) error {
	err := sd.Dec.Decode(&sd.f)
	if err != nil {
		return err
	}
	*s, err = extractSamples(&sd.f, sd.Opts)
	return err
}

// ExtractSamples builds a slice of samples from the provided metric
// families. If an error occurs during sample extraction, it continues to
// extract from the remaining metric families. The returned error is the last
// error that has occurred.
func ExtractSamples(o *DecodeOptions, fams ...*dto.MetricFamily) (model.Vector, error) {
	var (
		all     model.Vector
		lastErr error
	)
	for _, f := range fams {
		some, err := extractSamples(f, o)
		if err != nil {
			lastErr = err
			continue
		}
		all = append(all, some...)
	}
	return all, lastErr
}

func extractSamples(f *dto.MetricFamily, o *DecodeOptions) (model.Vector, error) {
	switch f.GetType() {
	case dto.MetricType_COUNTER:
		return extractCounter(o, f), nil
	case dto.MetricType_GAUGE:
		return extractGauge(o, f), nil
	case dto.MetricType_SUMMARY:
		return extractSummary(o, f), nil
	case dto.MetricType_UNTYPED:
		return extractUntyped(o, f), nil
	case dto.MetricType_HISTOGRAM:
		return extractHistogram(o, f), nil
	}
	return nil, fmt.Errorf("expfmt.extractSamples: unknown metric family type %v", f.GetType())
}

func extractCounter(o *DecodeOptions, f *dto.MetricFamily) model.Vector {
	samples := make(model.Vector, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Counter == nil {
			continue
		}

		lset := make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName())

		smpl := &model.Sample{
			Metric: model.Metric(lset),
			Value:  model.SampleValue(m.Counter.GetValue()),
		}

		if m.TimestampMs != nil {
			smpl.Timestamp = model.TimeFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			smpl.Timestamp = o.Timestamp
		}

		samples = append(samples, smpl)
	}

	return samples
}

func extractGauge(o *DecodeOptions, f *dto.MetricFamily) model.Vector {
	samples := make(model.Vector, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Gauge == nil {
			continue
		}

		lset := make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName())

		smpl := &model.Sample{
			Metric: model.Metric(lset),
			Value:  model.SampleValue(m.Gauge.GetValue()),
		}

		if m.TimestampMs != nil {
			smpl.Timestamp = model.TimeFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			smpl.Timestamp = o.Timestamp
		}

		samples = append(samples, smpl)
	}

	return samples
}

func extractUntyped(o *DecodeOptions, f *dto.MetricFamily) model.Vector {
	samples := make(model.Vector, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Untyped == nil {
			continue
		}

		lset := make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName())

		smpl := &model.Sample{
			Metric: model.Metric(lset),
			Value:  model.SampleValue(m.Untyped.GetValue()),
		}

		if m.TimestampMs != nil {
			smpl.Timestamp = model.TimeFromUnixNano(*m.TimestampMs * 1000000)
		} else {
			smpl.Timestamp = o.Timestamp
		}

		samples = append(samples, smpl)
	}

	return samples
}

func extractSummary(o *DecodeOptions, f *dto.MetricFamily) model.Vector {
	samples := make(model.Vector, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Summary == nil {
			continue
		}

		timestamp := o.Timestamp
		if m.TimestampMs != nil {
			timestamp = model.TimeFromUnixNano(*m.TimestampMs * 1000000)
		}

		for _, q := range m.Summary.Quantile {
			lset := make(model.LabelSet, len(m.Label)+2)
			for _, p := range m.Label {
				lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			// BUG(matt): Update other names to "quantile".
			lset[model.LabelName(model.QuantileLabel)] = model.LabelValue(fmt.Sprint(q.GetQuantile()))
			lset[model.MetricNameLabel] = model.LabelValue(f.GetName())

			samples = append(samples, &model.Sample{
				Metric:    model.Metric(lset),
				Value:     model.SampleValue(q.GetValue()),
				Timestamp: timestamp,
			})
		}

		lset := make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_sum")

		samples = append(samples, &model.Sample{
			Metric:    model.Metric(lset),
			Value:     model.SampleValue(m.Summary.GetSampleSum()),
			Timestamp: timestamp,
		})

		lset = make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_count")

		samples = append(samples, &model.Sample{
			Metric:    model.Metric(lset),
			Value:     model.SampleValue(m.Summary.GetSampleCount()),
			Timestamp: timestamp,
		})
	}

	return samples
}

func extractHistogram(o *DecodeOptions, f *dto.MetricFamily) model.Vector {
	samples := make(model.Vector, 0, len(f.Metric))

	for _, m := range f.Metric {
		if m.Histogram == nil {
			continue
		}

		timestamp := o.Timestamp
		if m.TimestampMs != nil {
			timestamp = model.TimeFromUnixNano(*m.TimestampMs * 1000000)
		}

		infSeen := false

		for _, q := range m.Histogram.Bucket {
			lset := make(model.LabelSet, len(m.Label)+2)
			for _, p := range m.Label {
				lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			lset[model.LabelName(model.BucketLabel)] = model.LabelValue(fmt.Sprint(q.GetUpperBound()))
			lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_bucket")

			if math.IsInf(q.GetUpperBound(), +1) {
				infSeen = true
			}

			samples = append(samples, &model.Sample{
				Metric:    model.Metric(lset),
				Value:     model.SampleValue(q.GetCumulativeCount()),
				Timestamp: timestamp,
			})
		}

		lset := make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_sum")

		samples = append(samples, &model.Sample{
			Metric:    model.Metric(lset),
			Value:     model.SampleValue(m.Histogram.GetSampleSum()),
			Timestamp: timestamp,
		})

		lset = make(model.LabelSet, len(m.Label)+1)
		for _, p := range m.Label {
			lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
		}
		lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_count")

		count := &model.Sample{
			Metric:    model.Metric(lset),
			Value:     model.SampleValue(m.Histogram.GetSampleCount()),
			Timestamp: timestamp,
		}
		samples = append(samples, count)

		if !infSeen {
			// Append an infinity bucket sample.
			lset := make(model.LabelSet, len(m.Label)+2)
			for _, p := range m.Label {
				lset[model.LabelName(p.GetName())] = model.LabelValue(p.GetValue())
			}
			lset[model.LabelName(model.BucketLabel)] = model.LabelValue("+Inf")
			lset[model.MetricNameLabel] = model.LabelValue(f.GetName() + "_bucket")

			samples = append(samples, &model.Sample{
				Metric:    model.Metric(lset),
				Value:     count.Value,
				Timestamp: timestamp,
			})
		}
	}

	return samples
}
