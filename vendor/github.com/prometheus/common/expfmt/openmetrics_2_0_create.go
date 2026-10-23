// Copyright The Prometheus Authors
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
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// MetricFamilyToOpenMetrics20 converts a MetricFamily proto message into the
// OpenMetrics text format version 2.0.0 and writes the resulting lines to 'out'.
// It returns the number of bytes written and any error encountered.
//
// NOTE: This method targets OpenMetrics 2.0.0 (currently aligned with 2.0-rc.0) which is experimental and
// encode-only. All metric types defined in client_model (counter, gauge, summary, untyped, histogram,
// and gaugehistogram) are supported; info and stateset are not supported as they are not present in
// client_model. Breaking changes might happen in the future.
// EncoderOptions are accepted for signature compatibility with
// MetricFamilyToOpenMetrics and are currently ignored.
//
// OpenMetrics 2.0 enforces stricter validation rules defined in the specification
// than Prometheus text or OpenMetrics 1.0 formats (such as requiring non-negative
// count and sum, and non-negative quantile values for summaries). Consequently,
// MetricFamilyToOpenMetrics20 may reject metric families that the other encoders accept.
func MetricFamilyToOpenMetrics20(out io.Writer, in *dto.MetricFamily, options ...EncoderOption) (written int, err error) {
	// Options are accepted for signature compatibility and ignored.
	_ = options
	name := in.GetName()
	if name == "" {
		return 0, fmt.Errorf("MetricFamily has no name: %s", in)
	}
	if containsRawNewline(name) {
		return 0, fmt.Errorf("MetricFamily name %q contains raw newlines", name)
	}
	if !utf8.ValidString(name) {
		return 0, fmt.Errorf("MetricFamily name %q is not valid UTF-8", name)
	}
	if in.Help != nil {
		if !utf8.ValidString(*in.Help) {
			return 0, fmt.Errorf("MetricFamily help %q is not valid UTF-8", *in.Help)
		}
		// A carriage return may appear in HELP, but not at the end, where it
		// would make the line end in "\r\n".
		if strings.HasSuffix(*in.Help, "\r") {
			return 0, fmt.Errorf("MetricFamily help %q ends with a carriage return", *in.Help)
		}
	}
	if in.Unit != nil {
		if containsRawNewline(*in.Unit) {
			return 0, fmt.Errorf("MetricFamily unit %q contains raw newlines", *in.Unit)
		}
		if !utf8.ValidString(*in.Unit) {
			return 0, fmt.Errorf("MetricFamily unit %q is not valid UTF-8", *in.Unit)
		}
	}

	// Try the interface upgrade. If it doesn't work, we'll use a
	// bufio.Writer from the sync.Pool.
	w, ok := out.(enhancedWriter)
	if !ok {
		b := bufPool.Get().(*bufio.Writer)
		b.Reset(out)
		w = b
		defer func() {
			bErr := b.Flush()
			if err == nil {
				err = bErr
			}
			bufPool.Put(b)
		}()
	}

	var (
		n          int
		metricType = in.GetType()
	)

	// Comments, first HELP, then TYPE.
	if in.Help != nil {
		n, err = w.WriteString("# HELP ")
		written += n
		if err != nil {
			return written, err
		}
		n, err = writeName(w, name)
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte(' ')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeEscapedString(w, *in.Help, true)
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte('\n')
		written++
		if err != nil {
			return written, err
		}
	}
	n, err = w.WriteString("# TYPE ")
	written += n
	if err != nil {
		return written, err
	}
	n, err = writeName(w, name)
	written += n
	if err != nil {
		return written, err
	}
	switch metricType {
	case dto.MetricType_COUNTER:
		n, err = w.WriteString(" counter\n")
	case dto.MetricType_GAUGE:
		n, err = w.WriteString(" gauge\n")
	case dto.MetricType_SUMMARY:
		n, err = w.WriteString(" summary\n")
	case dto.MetricType_UNTYPED:
		n, err = w.WriteString(" unknown\n")
	case dto.MetricType_HISTOGRAM:
		n, err = w.WriteString(" histogram\n")
	case dto.MetricType_GAUGE_HISTOGRAM:
		n, err = w.WriteString(" gaugehistogram\n")
	default:
		// TODO: Support Info and StateSet once they are supported in the
		// Prometheus protobuf format.
		return written, fmt.Errorf("unknown metric type %s", metricType.String())
	}
	written += n
	if err != nil {
		return written, err
	}
	if in.Unit != nil {
		n, err = w.WriteString("# UNIT ")
		written += n
		if err != nil {
			return written, err
		}
		n, err = writeName(w, name)
		written += n
		if err != nil {
			return written, err
		}

		err = w.WriteByte(' ')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeEscapedString(w, *in.Unit, true)
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte('\n')
		written++
		if err != nil {
			return written, err
		}
	}

	// Finally the samples, one line for each.
	for _, metric := range in.Metric {
		if metric == nil {
			return written, fmt.Errorf("expected non-nil metric in MetricFamily %s", name)
		}
		switch metricType {
		case dto.MetricType_COUNTER:
			if metric.Counter == nil {
				return written, fmt.Errorf("expected counter in metric %s %s", name, metric)
			}
			val := metric.Counter.GetValue()
			if math.IsNaN(val) {
				return written, fmt.Errorf("counter value cannot be NaN in metric %s", name)
			}
			if val < 0 {
				return written, fmt.Errorf("counter value cannot be negative (%g) in metric %s", val, name)
			}
			n, err = writeOpenMetrics20Sample(w, name, metric, val, 0, false, metric.Counter.CreatedTimestamp, metric.Counter.Exemplar)
		case dto.MetricType_GAUGE:
			if metric.Gauge == nil {
				return written, fmt.Errorf("expected gauge in metric %s %s", name, metric)
			}
			n, err = writeOpenMetrics20Sample(w, name, metric, metric.Gauge.GetValue(), 0, false, nil, nil)
		case dto.MetricType_UNTYPED:
			if metric.Untyped == nil {
				return written, fmt.Errorf("expected untyped in metric %s %s", name, metric)
			}
			n, err = writeOpenMetrics20Sample(w, name, metric, metric.Untyped.GetValue(), 0, false, nil, nil)
		case dto.MetricType_SUMMARY:
			if metric.Summary == nil {
				return written, fmt.Errorf("expected summary in metric %s %s", name, metric)
			}
			n, err = writeCompositeSummary(w, name, metric)
		case dto.MetricType_HISTOGRAM, dto.MetricType_GAUGE_HISTOGRAM:
			if metric.Histogram == nil {
				return written, fmt.Errorf("expected histogram in metric %s %s", name, metric)
			}
			n, err = writeCompositeHistogram(w, name, metric, metricType == dto.MetricType_GAUGE_HISTOGRAM)
		default:
			return written, fmt.Errorf("unexpected type in metric %s %s", name, metric)
		}
		written += n
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// writeOpenMetrics20Sample writes a single sample for simple types (Counter, Gauge, Untyped).
func writeOpenMetrics20Sample(w enhancedWriter, name string, metric *dto.Metric, floatValue float64, intValue uint64, useIntValue bool, startTimestamp *timestamppb.Timestamp, exemplar *dto.Exemplar) (int, error) {
	if err := validateLabels20(metric.Label); err != nil {
		return 0, err
	}
	written := 0
	n, err := writeOpenMetricsNameAndLabelPairs(w, name, metric.Label, "", 0)
	written += n
	if err != nil {
		return written, err
	}
	err = w.WriteByte(' ')
	written++
	if err != nil {
		return written, err
	}

	if useIntValue {
		n, err = writeUint(w, intValue)
	} else {
		n, err = writeOpenMetricsFloat(w, floatValue)
	}
	written += n
	if err != nil {
		return written, err
	}

	if metric.TimestampMs != nil {
		err = w.WriteByte(' ')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeOpenMetrics20Timestamp(w, float64(*metric.TimestampMs)/1000)
		written += n
		if err != nil {
			return written, err
		}
	}

	// Start Timestamp
	if startTimestamp != nil {
		if err := startTimestamp.CheckValid(); err != nil {
			return written, fmt.Errorf("invalid created timestamp in metric %s: %w", name, err)
		}
		n, err = w.WriteString(" st@")
		written += n
		if err != nil {
			return written, err
		}
		n, err = writeProtoTimestamp(w, startTimestamp)
		written += n
		if err != nil {
			return written, err
		}
	}

	if exemplar != nil {
		n, err = writeExemplar20(w, exemplar)
		written += n
		if err != nil {
			return written, err
		}
	}

	err = w.WriteByte('\n')
	written++
	if err != nil {
		return written, err
	}
	return written, nil
}

// writeExemplar20 writes the provided exemplar in OpenMetrics 2.0 format to w.
// In OpenMetrics 2.0, invalid exemplars or exemplars without a timestamp are dropped.
func writeExemplar20(w enhancedWriter, e *dto.Exemplar) (int, error) {
	if e == nil {
		return 0, nil
	}
	// In OpenMetrics 2.0, invalid exemplars are dropped rather than failing the entire exposition.
	if err := validateExemplar20(e); err != nil {
		return 0, nil
	}
	written := 0
	n, err := w.WriteString(" # ")
	written += n
	if err != nil {
		return written, err
	}
	if len(e.Label) == 0 {
		n, err = w.WriteString("{}")
	} else {
		n, err = writeOpenMetricsNameAndLabelPairs(w, "", e.Label, "", 0)
	}
	written += n
	if err != nil {
		return written, err
	}
	err = w.WriteByte(' ')
	written++
	if err != nil {
		return written, err
	}
	n, err = writeOpenMetricsFloat(w, e.GetValue())
	written += n
	if err != nil {
		return written, err
	}
	err = w.WriteByte(' ')
	written++
	if err != nil {
		return written, err
	}
	ts := e.Timestamp
	n, err = writeProtoTimestamp(w, ts)
	written += n
	if err != nil {
		return written, err
	}
	return written, nil
}

// writeOpenMetrics20Timestamp writes a float64 as a timestamp without scientific notation.
func writeOpenMetrics20Timestamp(w enhancedWriter, f float64) (int, error) {
	bp := numBufPool.Get().(*[]byte)
	*bp = strconv.AppendFloat((*bp)[:0], f, 'f', -1, 64)
	written, err := w.Write(*bp)
	numBufPool.Put(bp)
	return written, err
}

func writeCompositeSummary(w enhancedWriter, name string, metric *dto.Metric) (int, error) {
	s := metric.Summary
	if s == nil {
		return 0, fmt.Errorf("expected summary in metric %s", name)
	}

	if err := validateLabels20(metric.Label); err != nil {
		return 0, err
	}
	for _, lp := range metric.Label {
		if lp.GetName() == "quantile" {
			return 0, fmt.Errorf("metric %s is a summary but label set contains %q label", name, "quantile")
		}
	}

	sum := s.GetSampleSum()
	if math.IsNaN(sum) {
		return 0, fmt.Errorf("summary sum cannot be NaN in metric %s", name)
	}
	if sum < 0 {
		return 0, fmt.Errorf("summary sum cannot be negative (%g) in metric %s", sum, name)
	}

	var prevQuantile float64
	for i, q := range s.Quantile {
		if q == nil {
			return 0, fmt.Errorf("expected non-nil quantile in metric %s", name)
		}
		qv := q.GetQuantile()
		if math.IsNaN(qv) {
			return 0, fmt.Errorf("summary quantile cannot be NaN in metric %s", name)
		}
		if qv < 0 || qv > 1 {
			return 0, fmt.Errorf("summary quantile %g must be between 0 and 1 in metric %s", qv, name)
		}
		if i > 0 && qv <= prevQuantile {
			return 0, fmt.Errorf("summary quantiles must be strictly increasing: %g <= %g in metric %s", qv, prevQuantile, name)
		}
		prevQuantile = qv

		v := q.GetValue()
		if !math.IsNaN(v) && v < 0 {
			return 0, fmt.Errorf("summary quantile value cannot be negative (%g) in metric %s", v, name)
		}
	}

	if s.CreatedTimestamp != nil {
		if err := s.CreatedTimestamp.CheckValid(); err != nil {
			return 0, fmt.Errorf("invalid created timestamp in metric %s: %w", name, err)
		}
	}

	written := 0
	n, err := writeOpenMetricsNameAndLabelPairs(w, name, metric.Label, "", 0)
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(" {count:")
	written += n
	if err != nil {
		return written, err
	}

	n, err = writeUint(w, s.GetSampleCount())
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(",sum:")
	written += n
	if err != nil {
		return written, err
	}

	n, err = writeFloat(w, sum)
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(",quantile:[")
	written += n
	if err != nil {
		return written, err
	}

	for i, q := range s.Quantile {
		if i > 0 {
			err = w.WriteByte(',')
			written++
			if err != nil {
				return written, err
			}
		}
		n, err = writeFloat(w, q.GetQuantile())
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte(':')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeFloat(w, q.GetValue())
		written += n
		if err != nil {
			return written, err
		}
	}

	n, err = w.WriteString("]}")
	written += n
	if err != nil {
		return written, err
	}

	if metric.TimestampMs != nil {
		err = w.WriteByte(' ')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeOpenMetrics20Timestamp(w, float64(*metric.TimestampMs)/1000)
		written += n
		if err != nil {
			return written, err
		}
	}

	if s.CreatedTimestamp != nil {
		n, err = w.WriteString(" st@")
		written += n
		if err != nil {
			return written, err
		}
		n, err = writeProtoTimestamp(w, s.CreatedTimestamp)
		written += n
		if err != nil {
			return written, err
		}
	}

	err = w.WriteByte('\n')
	written++
	if err != nil {
		return written, err
	}

	return written, nil
}

func writeCompositeHistogram(w enhancedWriter, name string, metric *dto.Metric, isGauge bool) (int, error) {
	h := metric.Histogram
	if h == nil {
		return 0, fmt.Errorf("expected histogram in metric %s", name)
	}

	isNative := h.Schema != nil
	hasClassicBuckets := len(h.Bucket) > 0 || !isNative

	if err := validateLabels20(metric.Label); err != nil {
		return 0, err
	}
	if hasClassicBuckets {
		for _, lp := range metric.Label {
			if lp.GetName() == "le" {
				return 0, fmt.Errorf("metric %s has classic buckets but label set contains %q label", name, "le")
			}
		}
	}

	if h.SampleCountFloat != nil {
		c := *h.SampleCountFloat
		if math.IsNaN(c) {
			if isGauge {
				return 0, fmt.Errorf("gaugehistogram count cannot be NaN in metric %s", name)
			}
			return 0, fmt.Errorf("histogram count cannot be NaN in metric %s", name)
		}
		if !isGauge && c < 0 {
			return 0, fmt.Errorf("histogram count cannot be negative (%g) in metric %s", c, name)
		}
	}

	var isFloatCount bool
	var sampleCountFloat float64
	var sampleCountUint uint64
	switch {
	case h.SampleCountFloat != nil && (*h.SampleCountFloat > 0 || h.SampleCount == nil || isGauge):
		isFloatCount = true
		sampleCountFloat = *h.SampleCountFloat
	case h.SampleCount != nil:
		sampleCountUint = *h.SampleCount
		sampleCountFloat = float64(sampleCountUint)
	}

	if isNative {
		if err := validateNativeHistogram(name, h, isGauge); err != nil {
			return 0, err
		}
	}

	if hasClassicBuckets {
		if err := validateClassicBuckets(name, h, sampleCountFloat, sampleCountUint, isFloatCount, isGauge); err != nil {
			return 0, err
		}
	}

	if !isGauge && h.CreatedTimestamp != nil {
		if err := h.CreatedTimestamp.CheckValid(); err != nil {
			return 0, fmt.Errorf("invalid created timestamp in metric %s: %w", name, err)
		}
	}

	written := 0
	n, err := writeOpenMetricsNameAndLabelPairs(w, name, metric.Label, "", 0)
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(" {")
	written += n
	if err != nil {
		return written, err
	}

	if isGauge {
		n, err = w.WriteString("gcount:")
	} else {
		n, err = w.WriteString("count:")
	}
	written += n
	if err != nil {
		return written, err
	}

	if isFloatCount {
		n, err = writeFloat(w, sampleCountFloat)
	} else {
		n, err = writeUint(w, sampleCountUint)
	}
	written += n
	if err != nil {
		return written, err
	}

	if isGauge {
		n, err = w.WriteString(",gsum:")
	} else {
		n, err = w.WriteString(",sum:")
	}
	written += n
	if err != nil {
		return written, err
	}
	n, err = writeFloat(w, h.GetSampleSum())
	written += n
	if err != nil {
		return written, err
	}

	if isNative {
		n, err = writeNativeBuckets(w, h, isGauge)
		written += n
		if err != nil {
			return written, err
		}
	}

	var classicExemplars []*dto.Exemplar
	if hasClassicBuckets {
		n, err = writeClassicBuckets(w, h, sampleCountFloat, sampleCountUint, isFloatCount, isGauge, &classicExemplars)
		written += n
		if err != nil {
			return written, err
		}
	}

	err = w.WriteByte('}')
	written++
	if err != nil {
		return written, err
	}

	if metric.TimestampMs != nil {
		err = w.WriteByte(' ')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeOpenMetrics20Timestamp(w, float64(*metric.TimestampMs)/1000)
		written += n
		if err != nil {
			return written, err
		}
	}

	if !isGauge && h.CreatedTimestamp != nil {
		ts := h.CreatedTimestamp
		n, err = w.WriteString(" st@")
		written += n
		if err != nil {
			return written, err
		}
		n, err = writeProtoTimestamp(w, ts)
		written += n
		if err != nil {
			return written, err
		}
	}

	var exemplarsToEmit []*dto.Exemplar
	if len(classicExemplars) > 0 {
		// We follow the spec suggestion and prefer classic histogram
		// exemplars in dual mode.
		exemplarsToEmit = classicExemplars
	} else if len(h.Exemplars) > 0 {
		exemplarsToEmit = h.Exemplars
	}

	for _, e := range exemplarsToEmit {
		if e == nil || e.Timestamp == nil {
			continue
		}
		n, err = writeExemplar20(w, e)
		written += n
		if err != nil {
			return written, err
		}
	}

	err = w.WriteByte('\n')
	written++
	if err != nil {
		return written, err
	}

	return written, nil
}

func validateNativeHistogram(name string, h *dto.Histogram, isGauge bool) error {
	schema := *h.Schema
	if schema < -4 || schema > 8 {
		return fmt.Errorf("native histogram schema %d is out of range [-4, 8] in metric %s", schema, name)
	}

	zeroThreshold := h.GetZeroThreshold()
	if math.IsNaN(zeroThreshold) || math.IsInf(zeroThreshold, 0) || zeroThreshold < 0 {
		return fmt.Errorf("native histogram zero_threshold %g must be a non-negative, finite number in metric %s", zeroThreshold, name)
	}

	if h.ZeroCountFloat != nil {
		c := *h.ZeroCountFloat
		if math.IsNaN(c) {
			return fmt.Errorf("native histogram zero_count cannot be NaN in metric %s", name)
		}
		if !isGauge && c < 0 {
			return fmt.Errorf("native histogram zero_count cannot be negative (%g) in metric %s", c, name)
		}
	}

	if err := validateSpansAndBuckets(name, "negative", h.NegativeSpan, h.NegativeDelta, h.NegativeCount, isGauge); err != nil {
		return err
	}
	return validateSpansAndBuckets(name, "positive", h.PositiveSpan, h.PositiveDelta, h.PositiveCount, isGauge)
}

func validateSpansAndBuckets(
	name string,
	spanName string,
	spans []*dto.BucketSpan,
	deltas []int64,
	floatCounts []float64,
	isGauge bool,
) error {
	isFloatBuckets := len(floatCounts) > 0
	var numBuckets int
	if isFloatBuckets {
		numBuckets = len(floatCounts)
	} else {
		numBuckets = len(deltas)
	}

	var totalLength uint64
	for i, span := range spans {
		if span == nil {
			return errors.New("expected non-nil bucket span")
		}
		if i > 0 && span.GetOffset() < 0 {
			return fmt.Errorf("subsequent %s span offset cannot be negative: %d in metric %s", spanName, span.GetOffset(), name)
		}
		totalLength += uint64(span.GetLength())
	}

	if numBuckets == 0 {
		if totalLength == 0 {
			return nil
		}
		return fmt.Errorf("sum of %s span lengths (%d) does not match bucket count (0) in metric %s", spanName, totalLength, name)
	}

	if totalLength != uint64(numBuckets) {
		return fmt.Errorf("sum of %s span lengths (%d) does not match bucket count (%d) in metric %s", spanName, totalLength, numBuckets, name)
	}

	if isFloatBuckets {
		for _, v := range floatCounts {
			if math.IsNaN(v) {
				return fmt.Errorf("%s bucket count cannot be NaN in metric %s", spanName, name)
			}
			if !isGauge && v < 0 {
				return fmt.Errorf("%s bucket count cannot be negative (%g) in metric %s", spanName, v, name)
			}
		}
	} else {
		var current int64
		for _, d := range deltas {
			current += d
			if !isGauge && current < 0 {
				return fmt.Errorf("%s bucket count cannot be negative (%d) in metric %s", spanName, current, name)
			}
		}
	}
	return nil
}

func writeNativeBuckets(w enhancedWriter, h *dto.Histogram, isGauge bool) (int, error) {
	schema := *h.Schema
	zeroThreshold := h.GetZeroThreshold()

	var isFloatZeroCount bool
	var zeroCountFloat float64
	var zeroCountUint uint64
	switch {
	case h.ZeroCountFloat != nil && (*h.ZeroCountFloat > 0 || h.ZeroCount == nil || isGauge):
		isFloatZeroCount = true
		zeroCountFloat = *h.ZeroCountFloat
	case h.ZeroCount != nil:
		zeroCountUint = *h.ZeroCount
		zeroCountFloat = float64(zeroCountUint)
	}

	written := 0
	n, err := w.WriteString(",schema:")
	written += n
	if err != nil {
		return written, err
	}
	n, err = writeInt(w, int64(schema))
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(",zero_threshold:")
	written += n
	if err != nil {
		return written, err
	}
	n, err = writeFloat(w, zeroThreshold)
	written += n
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(",zero_count:")
	written += n
	if err != nil {
		return written, err
	}
	if isFloatZeroCount {
		n, err = writeFloat(w, zeroCountFloat)
	} else {
		n, err = writeUint(w, zeroCountUint)
	}
	written += n
	if err != nil {
		return written, err
	}

	n, err = writeSpansAndBuckets(w, ",negative_spans:[", ",negative_buckets:[", h.NegativeSpan, h.NegativeDelta, h.NegativeCount)
	written += n
	if err != nil {
		return written, err
	}

	n, err = writeSpansAndBuckets(w, ",positive_spans:[", ",positive_buckets:[", h.PositiveSpan, h.PositiveDelta, h.PositiveCount)
	written += n
	if err != nil {
		return written, err
	}

	return written, nil
}

func writeSpansAndBuckets(
	w enhancedWriter,
	spansPrefix string,
	bucketsPrefix string,
	spans []*dto.BucketSpan,
	deltas []int64,
	floatCounts []float64,
) (int, error) {
	isFloatBuckets := len(floatCounts) > 0
	var numBuckets int
	if isFloatBuckets {
		numBuckets = len(floatCounts)
	} else {
		numBuckets = len(deltas)
	}

	if numBuckets == 0 {
		return 0, nil
	}

	written := 0
	n, err := w.WriteString(spansPrefix)
	written += n
	if err != nil {
		return written, err
	}
	for i, span := range spans {
		if i > 0 {
			err = w.WriteByte(',')
			written++
			if err != nil {
				return written, err
			}
		}
		n, err = writeInt(w, int64(span.GetOffset()))
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte(':')
		written++
		if err != nil {
			return written, err
		}
		n, err = writeUint(w, uint64(span.GetLength()))
		written += n
		if err != nil {
			return written, err
		}
	}
	err = w.WriteByte(']')
	written++
	if err != nil {
		return written, err
	}

	n, err = w.WriteString(bucketsPrefix)
	written += n
	if err != nil {
		return written, err
	}
	if isFloatBuckets {
		for i, v := range floatCounts {
			if i > 0 {
				err = w.WriteByte(',')
				written++
				if err != nil {
					return written, err
				}
			}
			n, err = writeFloat(w, v)
			written += n
			if err != nil {
				return written, err
			}
		}
	} else {
		var current int64
		for i, d := range deltas {
			if i > 0 {
				err = w.WriteByte(',')
				written++
				if err != nil {
					return written, err
				}
			}
			current += d
			n, err = writeInt(w, current)
			written += n
			if err != nil {
				return written, err
			}
		}
	}
	err = w.WriteByte(']')
	written++
	if err != nil {
		return written, err
	}

	return written, nil
}

func validateClassicBuckets(
	name string,
	h *dto.Histogram,
	sampleCount float64,
	sampleCountUint uint64,
	isFloatCount bool,
	isGauge bool,
) error {
	var infSeen bool
	var prevBound float64
	var prevCount float64
	for i, b := range h.Bucket {
		if b == nil {
			return errors.New("expected non-nil bucket")
		}
		ub := b.GetUpperBound()
		if math.IsNaN(ub) {
			return fmt.Errorf("classic bucket upper bound cannot be NaN in metric %s", name)
		}
		if i > 0 && ub <= prevBound {
			return fmt.Errorf("classic bucket upper bounds must be strictly increasing: %g <= %g in metric %s", ub, prevBound, name)
		}
		prevBound = ub

		if b.CumulativeCountFloat != nil {
			c := *b.CumulativeCountFloat
			if math.IsNaN(c) {
				return fmt.Errorf("classic bucket count cannot be NaN in metric %s", name)
			}
			if !isGauge && c < 0 {
				return fmt.Errorf("classic bucket count cannot be negative (%g) in metric %s", c, name)
			}
		}

		var bCount float64
		switch {
		case b.CumulativeCountFloat != nil && (*b.CumulativeCountFloat > 0 || b.CumulativeCount == nil || isGauge):
			bCount = *b.CumulativeCountFloat
		case b.CumulativeCount != nil:
			bCount = float64(*b.CumulativeCount)
		}

		if !isGauge && i > 0 && bCount < prevCount {
			return fmt.Errorf("classic bucket counts must be monotonically increasing: %g < %g in metric %s", bCount, prevCount, name)
		}
		prevCount = bCount

		if math.IsInf(ub, +1) {
			if i != len(h.Bucket)-1 {
				return fmt.Errorf("+Inf bucket must be the last bucket in metric %s", name)
			}
			infSeen = true
			if isFloatCount {
				if bCount != sampleCount {
					return fmt.Errorf("classic bucket +Inf count (%g) does not match sample count (%g) in metric %s", bCount, sampleCount, name)
				}
			} else {
				if b.CumulativeCount != nil {
					if *b.CumulativeCount != sampleCountUint {
						return fmt.Errorf("classic bucket +Inf count (%d) does not match sample count (%d) in metric %s", *b.CumulativeCount, sampleCountUint, name)
					}
				} else {
					if bCount < 0 || bCount > math.MaxUint64 || uint64(bCount) != sampleCountUint || float64(uint64(bCount)) != bCount {
						return fmt.Errorf("classic bucket +Inf count (%g) does not match sample count (%d) in metric %s", bCount, sampleCountUint, name)
					}
				}
			}
		}
	}

	if !infSeen && len(h.Bucket) > 0 && !isGauge {
		if sampleCount < prevCount {
			return fmt.Errorf("sample count (%g) is less than highest bucket count (%g) in metric %s", sampleCount, prevCount, name)
		}
	}
	return nil
}

func writeClassicBuckets(
	w enhancedWriter,
	h *dto.Histogram,
	sampleCount float64,
	sampleCountUint uint64,
	isFloatCount bool,
	isGauge bool,
	collectedExemplars *[]*dto.Exemplar,
) (int, error) {
	var infSeen bool
	written := 0
	n, err := w.WriteString(",bucket:[")
	written += n
	if err != nil {
		return written, err
	}

	for i, b := range h.Bucket {
		if i > 0 {
			err = w.WriteByte(',')
			written++
			if err != nil {
				return written, err
			}
		}
		n, err = writeFloat(w, b.GetUpperBound())
		written += n
		if err != nil {
			return written, err
		}
		err = w.WriteByte(':')
		written++
		if err != nil {
			return written, err
		}
		switch {
		case b.CumulativeCountFloat != nil && (*b.CumulativeCountFloat > 0 || b.CumulativeCount == nil || isGauge):
			n, err = writeFloat(w, *b.CumulativeCountFloat)
		case b.CumulativeCount != nil:
			n, err = writeUint(w, *b.CumulativeCount)
		default:
			n, err = writeUint(w, 0)
		}
		written += n
		if err != nil {
			return written, err
		}
		if math.IsInf(b.GetUpperBound(), +1) {
			infSeen = true
		}
		if b.Exemplar != nil && b.Exemplar.Timestamp != nil {
			*collectedExemplars = append(*collectedExemplars, b.Exemplar)
		}
	}

	if !infSeen {
		if len(h.Bucket) > 0 {
			err = w.WriteByte(',')
			written++
			if err != nil {
				return written, err
			}
		}
		n, err = w.WriteString("+Inf:")
		written += n
		if err != nil {
			return written, err
		}
		if isFloatCount {
			n, err = writeFloat(w, sampleCount)
		} else {
			n, err = writeUint(w, sampleCountUint)
		}
		written += n
		if err != nil {
			return written, err
		}
	}

	err = w.WriteByte(']')
	written++
	if err != nil {
		return written, err
	}

	return written, nil
}

func validateLabels20(labels []*dto.LabelPair) error {
	for _, lp := range labels {
		if lp == nil {
			return errors.New("expected non-nil label pair")
		}
		lname := lp.GetName()
		if lname == "" {
			return errors.New("label name cannot be empty")
		}
		if containsRawNewline(lname) {
			return fmt.Errorf("label name %q contains raw newlines", lname)
		}
	}
	return nil
}

func containsRawNewline(s string) bool {
	return strings.IndexByte(s, '\n') >= 0 || strings.IndexByte(s, '\r') >= 0
}

func validateExemplar20(e *dto.Exemplar) error {
	if e.Timestamp == nil {
		return errors.New("exemplar timestamp is required")
	}
	if err := e.Timestamp.CheckValid(); err != nil {
		return err
	}
	return validateLabels20(e.Label)
}

func writeProtoTimestamp(w enhancedWriter, ts *timestamppb.Timestamp) (int, error) {
	if err := ts.CheckValid(); err != nil {
		return 0, err
	}
	if ts.Nanos == 0 {
		return writeInt(w, ts.Seconds)
	}
	sec := ts.Seconds
	nanos := int64(ts.Nanos)
	if sec < 0 {
		sec++
		nanos = 1_000_000_000 - nanos
		if sec == 0 {
			n, err := w.WriteString("-0")
			if err != nil {
				return n, err
			}
			n2, err := writeNanos(w, nanos)
			return n + n2, err
		}
	}

	n, err := writeInt(w, sec)
	if err != nil {
		return n, err
	}
	n2, err := writeNanos(w, nanos)
	return n + n2, err
}

func writeNanos(w enhancedWriter, nanos int64) (int, error) {
	err := w.WriteByte('.')
	if err != nil {
		return 0, err
	}
	written := 1
	bp := numBufPool.Get().(*[]byte)
	*bp = strconv.AppendInt((*bp)[:0], nanos, 10)
	pad := 9 - len(*bp)
	for range pad {
		err = w.WriteByte('0')
		written++
		if err != nil {
			numBufPool.Put(bp)
			return written, err
		}
	}
	val := *bp
	for len(val) > 0 && val[len(val)-1] == '0' {
		val = val[:len(val)-1]
	}
	n2, err := w.Write(val)
	written += n2
	numBufPool.Put(bp)
	return written, err
}
