// Copyright 2018 The Prometheus Authors
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

// Package testutil provides helpers to test code using the prometheus package
// of client_golang.
//
// While writing unit tests to verify correct instrumentation of your code, it's
// a common mistake to mostly test the instrumentation library instead of your
// own code. Rather than verifying that a prometheus.Counter's value has changed
// as expected or that it shows up in the exposition after registration, it is
// in general more robust and more faithful to the concept of unit tests to use
// mock implementations of the prometheus.Counter and prometheus.Registerer
// interfaces that simply assert that the Add or Register methods have been
// called with the expected arguments. However, this might be overkill in simple
// scenarios. The ToFloat64 function is provided for simple inspection of a
// single-value metric, but it has to be used with caution.
//
// End-to-end tests to verify all or larger parts of the metrics exposition can
// be implemented with the CollectAndCompare or GatherAndCompare functions. The
// most appropriate use is not so much testing instrumentation of your code, but
// testing custom prometheus.Collector implementations and in particular whole
// exporters, i.e. programs that retrieve telemetry data from a 3rd party source
// and convert it into Prometheus metrics.
//
// In a similar pattern, CollectAndLint and GatherAndLint can be used to detect
// metrics that have issues with their name, type, or metadata without being
// necessarily invalid, e.g. a counter with a name missing the “_total” suffix.
package testutil

import (
	"bytes"
	"fmt"
	"io"

	"github.com/prometheus/common/expfmt"

	dto "github.com/prometheus/client_model/go"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/internal"
)

// ToFloat64 collects all Metrics from the provided Collector. It expects that
// this results in exactly one Metric being collected, which must be a Gauge,
// Counter, or Untyped. In all other cases, ToFloat64 panics. ToFloat64 returns
// the value of the collected Metric.
//
// The Collector provided is typically a simple instance of Gauge or Counter, or
// – less commonly – a GaugeVec or CounterVec with exactly one element. But any
// Collector fulfilling the prerequisites described above will do.
//
// Use this function with caution. It is computationally very expensive and thus
// not suited at all to read values from Metrics in regular code. This is really
// only for testing purposes, and even for testing, other approaches are often
// more appropriate (see this package's documentation).
//
// A clear anti-pattern would be to use a metric type from the prometheus
// package to track values that are also needed for something else than the
// exposition of Prometheus metrics. For example, you would like to track the
// number of items in a queue because your code should reject queuing further
// items if a certain limit is reached. It is tempting to track the number of
// items in a prometheus.Gauge, as it is then easily available as a metric for
// exposition, too. However, then you would need to call ToFloat64 in your
// regular code, potentially quite often. The recommended way is to track the
// number of items conventionally (in the way you would have done it without
// considering Prometheus metrics) and then expose the number with a
// prometheus.GaugeFunc.
func ToFloat64(c prometheus.Collector) float64 {
	var (
		m      prometheus.Metric
		mCount int
		mChan  = make(chan prometheus.Metric)
		done   = make(chan struct{})
	)

	go func() {
		for m = range mChan {
			mCount++
		}
		close(done)
	}()

	c.Collect(mChan)
	close(mChan)
	<-done

	if mCount != 1 {
		panic(fmt.Errorf("collected %d metrics instead of exactly 1", mCount))
	}

	pb := &dto.Metric{}
	m.Write(pb)
	if pb.Gauge != nil {
		return pb.Gauge.GetValue()
	}
	if pb.Counter != nil {
		return pb.Counter.GetValue()
	}
	if pb.Untyped != nil {
		return pb.Untyped.GetValue()
	}
	panic(fmt.Errorf("collected a non-gauge/counter/untyped metric: %s", pb))
}

// CollectAndCount registers the provided Collector with a newly created
// pedantic Registry. It then calls GatherAndCount with that Registry and with
// the provided metricNames. In the unlikely case that the registration or the
// gathering fails, this function panics. (This is inconsistent with the other
// CollectAnd… functions in this package and has historical reasons. Changing
// the function signature would be a breaking change and will therefore only
// happen with the next major version bump.)
func CollectAndCount(c prometheus.Collector, metricNames ...string) int {
	reg := prometheus.NewPedanticRegistry()
	if err := reg.Register(c); err != nil {
		panic(fmt.Errorf("registering collector failed: %s", err))
	}
	result, err := GatherAndCount(reg, metricNames...)
	if err != nil {
		panic(err)
	}
	return result
}

// GatherAndCount gathers all metrics from the provided Gatherer and counts
// them. It returns the number of metric children in all gathered metric
// families together. If any metricNames are provided, only metrics with those
// names are counted.
func GatherAndCount(g prometheus.Gatherer, metricNames ...string) (int, error) {
	got, err := g.Gather()
	if err != nil {
		return 0, fmt.Errorf("gathering metrics failed: %s", err)
	}
	if metricNames != nil {
		got = filterMetrics(got, metricNames)
	}

	result := 0
	for _, mf := range got {
		result += len(mf.GetMetric())
	}
	return result, nil
}

// CollectAndCompare registers the provided Collector with a newly created
// pedantic Registry. It then calls GatherAndCompare with that Registry and with
// the provided metricNames.
func CollectAndCompare(c prometheus.Collector, expected io.Reader, metricNames ...string) error {
	reg := prometheus.NewPedanticRegistry()
	if err := reg.Register(c); err != nil {
		return fmt.Errorf("registering collector failed: %s", err)
	}
	return GatherAndCompare(reg, expected, metricNames...)
}

// GatherAndCompare gathers all metrics from the provided Gatherer and compares
// it to an expected output read from the provided Reader in the Prometheus text
// exposition format. If any metricNames are provided, only metrics with those
// names are compared.
func GatherAndCompare(g prometheus.Gatherer, expected io.Reader, metricNames ...string) error {
	got, err := g.Gather()
	if err != nil {
		return fmt.Errorf("gathering metrics failed: %s", err)
	}
	if metricNames != nil {
		got = filterMetrics(got, metricNames)
	}
	var tp expfmt.TextParser
	wantRaw, err := tp.TextToMetricFamilies(expected)
	if err != nil {
		return fmt.Errorf("parsing expected metrics failed: %s", err)
	}
	want := internal.NormalizeMetricFamilies(wantRaw)

	return compare(got, want)
}

// compare encodes both provided slices of metric families into the text format,
// compares their string message, and returns an error if they do not match.
// The error contains the encoded text of both the desired and the actual
// result.
func compare(got, want []*dto.MetricFamily) error {
	var gotBuf, wantBuf bytes.Buffer
	enc := expfmt.NewEncoder(&gotBuf, expfmt.FmtText)
	for _, mf := range got {
		if err := enc.Encode(mf); err != nil {
			return fmt.Errorf("encoding gathered metrics failed: %s", err)
		}
	}
	enc = expfmt.NewEncoder(&wantBuf, expfmt.FmtText)
	for _, mf := range want {
		if err := enc.Encode(mf); err != nil {
			return fmt.Errorf("encoding expected metrics failed: %s", err)
		}
	}

	if wantBuf.String() != gotBuf.String() {
		return fmt.Errorf(`
metric output does not match expectation; want:

%s
got:

%s`, wantBuf.String(), gotBuf.String())

	}
	return nil
}

func filterMetrics(metrics []*dto.MetricFamily, names []string) []*dto.MetricFamily {
	var filtered []*dto.MetricFamily
	for _, m := range metrics {
		for _, name := range names {
			if m.GetName() == name {
				filtered = append(filtered, m)
				break
			}
		}
	}
	return filtered
}
