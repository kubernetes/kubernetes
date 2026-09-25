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

	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// MetricFamilyToOpenMetrics20 converts a MetricFamily proto message into the
// OpenMetrics text format version 2.0.0 and writes the resulting lines to 'out'.
// It returns the number of bytes written and any error encountered.
//
// NOTE: This method targets OpenMetrics 2.0.0 (currently aligned with 2.0-rc.0) which is experimental and
// encode-only (currently supporting counter, gauge, and untyped metric types).
// Breaking changes might happen in the future. This implementation is still a
// work-in-progress, and does not yet support all features of the format.
// EncoderOptions are accepted for signature compatibility with
// MetricFamilyToOpenMetrics and are currently ignored.
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
	if in.Unit != nil && containsRawNewline(*in.Unit) {
		return 0, fmt.Errorf("MetricFamily unit %q contains raw newlines", *in.Unit)
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

// Stubs for Summary and Histogram

func writeCompositeSummary(w enhancedWriter, name string, metric *dto.Metric) (int, error) {
	_ = w
	_ = name
	_ = metric
	return 0, errors.New("summary not implemented yet")
}

func writeCompositeHistogram(w enhancedWriter, name string, metric *dto.Metric, isGauge bool) (int, error) {
	_ = w
	_ = name
	_ = metric
	_ = isGauge
	return 0, errors.New("histogram not implemented yet")
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
	n, err := writeInt(w, ts.Seconds)
	if err != nil {
		return n, err
	}
	if ts.Nanos == 0 {
		return n, nil
	}
	err = w.WriteByte('.')
	written := n + 1
	if err != nil {
		return written, err
	}
	bp := numBufPool.Get().(*[]byte)
	*bp = strconv.AppendInt((*bp)[:0], int64(ts.Nanos), 10)
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
