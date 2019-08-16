/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"reflect"
	"testing"
)

const fakeFilename = "testdata/metric.go"

func TestSkipMetrics(t *testing.T) {
	for _, test := range []struct {
		testName string
		src      string
	}{
		{
			testName: "Skip alpha metric with local variable",
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var name = "metric"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.ALPHA,
		},
	)
`},
		{
			testName: "Skip alpha metric created via function call",
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
func getName() string {
	return "metric"
}
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           getName(),
			StabilityLevel: metrics.ALPHA,
		},
	)
`},
		{
			testName: "Skip metric without stability set",
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name: "metric",
		},
	)
`},
		{
			testName: "Skip functions of similar signature (not imported from framework path) with import rename",
			src: `
package test
import metrics "k8s.io/fake/path"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Skip functions of similar signature (not imported from framework path)",
			src: `
package test
import "k8s.io/fake/path/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Skip . package import of non metric framework",
			src: `
package test
import . "k8s.io/fake/path"
var _ = NewCounter(
		&CounterOpts{
			StabilityLevel: STABLE,
		},
	)
`},
	} {
		t.Run(test.testName, func(t *testing.T) {
			metrics, errors := searchFileForStableMetrics(fakeFilename, test.src)
			if len(metrics) != 0 {
				t.Errorf("Didn't expect any stable metrics found, got: %d", len(metrics))
			}
			if len(errors) != 0 {
				t.Errorf("Didn't expect any errors found, got: %s", errors)
			}
		})
	}
}

func TestStableMetric(t *testing.T) {
	for _, test := range []struct {
		testName string
		src      string
		metric   metric
	}{
		{
			testName: "Counter",
			metric: metric{
				Name:           "metric",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: "STABLE",
				Help:           "help",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
	&metrics.CounterOpts{
		Name: "metric",
		Subsystem: "subsystem",
		Namespace: "namespace",
		Help: "help",
		StabilityLevel: metrics.STABLE,
	},
)
`},
		{
			testName: "CounterVec",
			metric: metric{
				Name:           "metric",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				Labels:         []string{"label-1"},
				StabilityLevel: "STABLE",
				Help:           "help",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name: "metric",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"label-1"},
	)
`},
		{
			testName: "Gauge",
			metric: metric{
				Name:           "gauge",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: "STABLE",
				Help:           "help",
				Type:           gaugeMetricType,
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewGauge(
		&metrics.GaugeOpts{
			Name: "gauge",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "GaugeVec",
			metric: metric{
				Name:           "gauge",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: "STABLE",
				Help:           "help",
				Type:           gaugeMetricType,
				Labels:         []string{"label-1", "label-2"},
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name: "gauge",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"label-2", "label-1"},
	)
`},
		{
			testName: "Histogram",
			metric: metric{
				Name:           "histogram",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: "STABLE",
				Buckets:        []float64{0.001, 0.01, 0.1, 1, 10, 100},
				Help:           "help",
				Type:           histogramMetricType,
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			Namespace: "namespace",
			Subsystem: "subsystem",
			StabilityLevel: metrics.STABLE,
			Help: "help",
			Buckets: []float64{0.001, 0.01, 0.1, 1, 10, 100},
		},
	)
`},
		{
			testName: "HistogramVec",
			metric: metric{
				Name:           "histogram",
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: "STABLE",
				Buckets:        []float64{0.001, 0.01, 0.1, 1, 10, 100},
				Help:           "help",
				Type:           histogramMetricType,
				Labels:         []string{"label-1", "label-2"},
			},
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name: "histogram",
			Namespace: "namespace",
			Subsystem: "subsystem",
			StabilityLevel: metrics.STABLE,
			Help: "help",
			Buckets: []float64{0.001, 0.01, 0.1, 1, 10, 100},
		},
		[]string{"label-2", "label-1"},
	)
`},
		{
			testName: "Custom import",
			metric: metric{
				Name:           "metric",
				StabilityLevel: "STABLE",
				Type:           counterMetricType,
			},
			src: `
package test
import custom "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = custom.NewCounter(
		&custom.CounterOpts{
			Name: "metric",
			StabilityLevel: custom.STABLE,
		},
	)
`},
	} {
		t.Run(test.testName, func(t *testing.T) {
			metrics, errors := searchFileForStableMetrics(fakeFilename, test.src)
			if len(errors) != 0 {
				t.Errorf("Unexpected errors: %s", errors)
			}
			if len(metrics) != 1 {
				t.Fatalf("Unexpected number of metrics: got %d, want 1", len(metrics))
			}
			if test.metric.Labels == nil {
				test.metric.Labels = []string{}
			}
			if !reflect.DeepEqual(metrics[0], test.metric) {
				t.Errorf("metric:\ngot  %v\nwant %v", metrics[0], test.metric)
			}
		})
	}
}

func TestIncorrectStableMetricDeclarations(t *testing.T) {
	for _, test := range []struct {
		testName string
		src      string
		err      error
	}{
		{
			testName: "Fail on stable summary metric (Summary is DEPRECATED)",
			err:      fmt.Errorf("testdata/metric.go:4:9: Stable summary metric is not supported"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewSummary(
		&metrics.SummaryOpts{
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on stable metric with attribute set to variable",
			err:      fmt.Errorf("testdata/metric.go:7:4: Non string attribute it not supported"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
const name = "metric"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on stable metric with attribute set to local function return",
			err:      fmt.Errorf("testdata/metric.go:9:4: Non string attribute it not supported"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
func getName() string {
	return "metric"
}
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           getName(),
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on stable metric with attribute set to imported function return",
			err:      fmt.Errorf("testdata/metric.go:7:4: Non string attribute it not supported"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
import "k8s.io/kubernetes/utils"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           utils.getMetricName(),
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on metric with stability set to function return",
			err:      fmt.Errorf("testdata/metric.go:9:20: StabilityLevel should be passed STABLE, ALPHA or removed"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
func getMetricStability() metrics.StabilityLevel {
	return metrics.STABLE
}
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: getMetricsStability(),
		},
	)
`},
		{
			testName: "error for passing stability as string",
			err:      fmt.Errorf("testdata/metric.go:6:20: StabilityLevel should be passed STABLE, ALPHA or removed"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: "stable",
		},
	)
`},
		{
			testName: "error for passing stability as unknown const",
			err:      fmt.Errorf("testdata/metric.go:6:20: StabilityLevel should be passed STABLE, ALPHA or removed"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: metrics.UNKNOWN,
		},
	)
`},
		{
			testName: "error for passing stability as variable",
			err:      fmt.Errorf("testdata/metric.go:7:20: StabilityLevel should be passed STABLE, ALPHA or removed"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var stable = metrics.STABLE
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: stable,
		},
	)
`},
		{
			testName: "error for stable metric created via function call",
			err:      fmt.Errorf("testdata/metric.go:6:10: Opts for STABLE metric was not directly passed to new metric function"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(getStableCounterOpts())
func getStableCounterOpts() *metrics.CounterOpts {
	return &metrics.CounterOpts{
		StabilityLevel: metrics.STABLE,
	}
}
`},
		{
			testName: "error . package import of metric framework",
			err:      fmt.Errorf(`testdata/metric.go:3:8: Importing through "." metrics framework is not supported`),
			src: `
package test
import . "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = NewCounter(
		&CounterOpts{
			StabilityLevel: STABLE,
		},
	)
`},
		{
			testName: "error stable metric opts passed to local function",
			err:      fmt.Errorf("testdata/metric.go:4:9: Opts for STABLE metric was not directly passed to new metric function"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = RegisterMetric(
		&metrics.CounterOpts{
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "error stable metric opts passed to imported function",
			err:      fmt.Errorf("testdata/metric.go:4:9: Opts for STABLE metric was not directly passed to new metric function"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = test.RegisterMetric(
		&metrics.CounterOpts{
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "error stable metric opts passed to imported function",
			err:      fmt.Errorf("testdata/metric.go:6:4: Positional arguments are not supported"),
			src: `
package test
import "k8s.io/kubernetes/staging/src/k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			"counter",
		},
	)
`},
	} {
		t.Run(test.testName, func(t *testing.T) {
			_, errors := searchFileForStableMetrics(fakeFilename, test.src)
			if len(errors) != 1 {
				t.Fatalf("Unexpected number of errors, got %d, want 1", len(errors))
			}
			if !reflect.DeepEqual(errors[0], test.err) {
				t.Errorf("error:\ngot  %v\nwant %v", errors[0], test.err)
			}
		})
	}
}
