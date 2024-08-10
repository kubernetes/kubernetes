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

	"github.com/google/go-cmp/cmp"

	"k8s.io/component-base/metrics"
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
import "k8s.io/component-base/metrics"
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
import "k8s.io/component-base/metrics"
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
import "k8s.io/component-base/metrics"
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
				Name:              "metric",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				StabilityLevel:    "STABLE",
				DeprecatedVersion: "1.16",
				Help:              "help",
				Type:              counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
	&metrics.CounterOpts{
		Name: "metric",
		Subsystem: "subsystem",
		Namespace: "namespace",
		Help: "help",
		DeprecatedVersion: "1.16",
		StabilityLevel: metrics.STABLE,
	},
)
`},
		{
			testName: "CounterVec",
			metric: metric{
				Name:              "metric",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Labels:            []string{"label-1"},
				StabilityLevel:    "STABLE",
				DeprecatedVersion: "1.16",
				Help:              "help",
				Type:              counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name: "metric",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			DeprecatedVersion: "1.16",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"label-1"},
	)
`},
		{
			testName: "Gauge",
			metric: metric{
				Name:              "gauge",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				StabilityLevel:    "STABLE",
				DeprecatedVersion: "1.16",
				Help:              "help",
				Type:              gaugeMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewGauge(
		&metrics.GaugeOpts{
			Name: "gauge",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			DeprecatedVersion: "1.16",
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "GaugeVec",
			metric: metric{
				Name:              "gauge",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				StabilityLevel:    "STABLE",
				DeprecatedVersion: "1.16",
				Help:              "help",
				Type:              gaugeMetricType,
				Labels:            []string{"label-1", "label-2"},
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name: "gauge",
			Namespace: "namespace",
			Subsystem: "subsystem",
			Help: "help",
			DeprecatedVersion: "1.16",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"label-2", "label-1"},
	)
`},
		{
			testName: "Histogram",
			metric: metric{
				Name:              "histogram",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				DeprecatedVersion: "1.16",
				StabilityLevel:    "STABLE",
				Buckets:           []float64{0.001, 0.01, 0.1, 1, 10, 100},
				Help:              "help",
				Type:              histogramMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			Namespace: "namespace",
			Subsystem: "subsystem",
			StabilityLevel: metrics.STABLE,
			Help: "help",
			DeprecatedVersion: "1.16",
			Buckets: []float64{0.001, 0.01, 0.1, 1, 10, 100},
		},
	)
`},
		{
			testName: "HistogramVec",
			metric: metric{
				Name:              "histogram",
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				DeprecatedVersion: "1.16",
				StabilityLevel:    "STABLE",
				Buckets:           []float64{0.001, 0.01, 0.1, 1, 10, 100},
				Help:              "help",
				Type:              histogramMetricType,
				Labels:            []string{"label-1", "label-2"},
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name: "histogram",
			Namespace: "namespace",
			Subsystem: "subsystem",
			StabilityLevel: metrics.STABLE,
			Help: "help",
			DeprecatedVersion: "1.16",
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
import custom "k8s.io/component-base/metrics"
var _ = custom.NewCounter(
		&custom.CounterOpts{
			Name: "metric",
			StabilityLevel: custom.STABLE,
		},
	)
`},
		{
			testName: "Custom import NewDesc",
			metric: metric{
				Name:           "apiserver_storage_size_bytes",
				Help:           "Size of the storage database file physically allocated in bytes.",
				Labels:         []string{"server"},
				StabilityLevel: "STABLE",
				Type:           customType,
				ConstLabels:    map[string]string{},
			},
			src: `
package test
import custom "k8s.io/component-base/metrics"
var _ = custom.NewDesc("apiserver_storage_size_bytes", "Size of the storage database file physically allocated in bytes.", []string{"server"}, nil, custom.STABLE, "")
`},
		{
			testName: "Const",
			metric: metric{
				Name:           "metric",
				StabilityLevel: "STABLE",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
const name = "metric"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Variable",
			metric: metric{
				Name:           "metric",
				StabilityLevel: "STABLE",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var name = "metric"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Multiple consts in block",
			metric: metric{
				Name:           "metric",
				StabilityLevel: "STABLE",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
const (
 unrelated1 = "unrelated1"
 name = "metric"
 unrelated2 = "unrelated2"
)
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Multiple variables in Block",
			metric: metric{
				Name:           "metric",
				StabilityLevel: "STABLE",
				Type:           counterMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var (
 unrelated1 = "unrelated1"
 name = "metric"
 _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           name,
			StabilityLevel: metrics.STABLE,
		},
	)
)
`},
		{
			testName: "Histogram with linear buckets",
			metric: metric{
				Name:           "histogram",
				StabilityLevel: "STABLE",
				Buckets:        metrics.LinearBuckets(1, 1, 3),
				Type:           histogramMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			StabilityLevel: metrics.STABLE,
			Buckets: metrics.LinearBuckets(1, 1, 3),
		},
	)
`},
		{
			testName: "Histogram with exponential buckets",
			metric: metric{
				Name:           "histogram",
				StabilityLevel: "STABLE",
				Buckets:        metrics.ExponentialBuckets(1, 2, 3),
				Type:           histogramMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			StabilityLevel: metrics.STABLE,
			Buckets: metrics.ExponentialBuckets(1, 2, 3),
		},
	)
`},
		{
			testName: "Histogram with default buckets",
			metric: metric{
				Name:           "histogram",
				StabilityLevel: "STABLE",
				Buckets:        metrics.DefBuckets,
				Type:           histogramMetricType,
			},
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			StabilityLevel: metrics.STABLE,
			Buckets: metrics.DefBuckets,
		},
	)
`},
		{
			testName: "Imported k8s.io constant",
			metric: metric{
				Name:           "importedCounter",
				StabilityLevel: "STABLE",
				Subsystem:      "kubelet",
				Type:           counterMetricType,
			},
			src: `
package test
import compbasemetrics "k8s.io/component-base/metrics"
import "k8s.io/kubernetes/pkg/kubelet/metrics"
var _ = compbasemetrics.NewCounter(
	&compbasemetrics.CounterOpts{
			Name: "importedCounter",
			StabilityLevel: compbasemetrics.STABLE,
			Subsystem: metrics.KubeletSubsystem,
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
			if diff := cmp.Diff(metrics[0], test.metric); diff != "" {
				t.Errorf("metric diff: %s", diff)
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
			testName: "Fail on stable metric with attribute set to unknown variable",
			err:      fmt.Errorf("testdata/metric.go:6:4: Metric attribute was not correctly set. Please use only global consts in same file"),
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           unknownVariable,
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on stable metric with attribute set to local function return",
			err:      fmt.Errorf("testdata/metric.go:9:4: Non string attribute is not supported"),
			src: `
package test
import "k8s.io/component-base/metrics"
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
			err:      fmt.Errorf("testdata/metric.go:7:4: Non string attribute is not supported"),
			src: `
package test
import "k8s.io/component-base/metrics"
import "os"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           os.Getenv("name"), // any imported function will do
			StabilityLevel: metrics.STABLE,
		},
	)
`},
		{
			testName: "Fail on metric with stability set to function return",
			err:      fmt.Errorf("testdata/metric.go:9:20: %s", errStabilityLevel),
			src: `
package test
import "k8s.io/component-base/metrics"
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
			err:      fmt.Errorf("testdata/metric.go:6:20: %s", errStabilityLevel),
			src: `
package test
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			StabilityLevel: "stable",
		},
	)
`},
		{
			testName: "error for passing stability as variable",
			err:      fmt.Errorf("testdata/metric.go:7:20: %s", errStabilityLevel),
			src: `
package test
import "k8s.io/component-base/metrics"
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
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounter(getStableCounterOpts())
func getStableCounterOpts() *metrics.CounterOpts {
	return &metrics.CounterOpts{
		StabilityLevel: metrics.STABLE,
	}
}
`},
		{
			testName: "error . package import of metric framework",
			err:      fmt.Errorf(`testdata/metric.go:3:8: Importing using "." is not supported`),
			src: `
package test
import . "k8s.io/component-base/metrics"
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
import "k8s.io/component-base/metrics"
var _ = RegisterMetric(
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
import "k8s.io/component-base/metrics"
var _ = metrics.NewCounter(
		&metrics.CounterOpts{
			"counter",
		},
	)
`},
		{
			testName: "error stable histogram with unknown prometheus bucket variable",
			err:      fmt.Errorf("testdata/metric.go:9:13: Buckets should be set to list of floats, result from function call of prometheus.LinearBuckets or prometheus.ExponentialBuckets"),
			src: `
package test
import "k8s.io/component-base/metrics"
import "github.com/prometheus/client_golang/prometheus"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			StabilityLevel: metrics.STABLE,
			Buckets: prometheus.FakeBuckets,
		},
	)
`},
		{
			testName: "error stable summary with unknown prometheus objective variable",
			err:      fmt.Errorf("testdata/metric.go:9:16: Objectives should be set to map of floats to floats"),
			src: `
package test
import "k8s.io/component-base/metrics"
import "github.com/prometheus/client_golang/prometheus"
var _ = metrics.NewSummary(
		&metrics.SummaryOpts{
			Name: "summary",
			StabilityLevel: metrics.STABLE,
			Objectives: prometheus.FakeObjectives,
		},
	)
`},
		{
			testName: "error stable histogram with unknown bucket variable from unknown library",
			err:      fmt.Errorf("testdata/metric.go:9:13: Buckets should be set to list of floats, result from function call of prometheus.LinearBuckets or prometheus.ExponentialBuckets"),
			src: `
package test
import "k8s.io/component-base/metrics"
import "github.com/prometheus/client_golang/prometheus"
var _ = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Name: "histogram",
			StabilityLevel: metrics.STABLE,
			Buckets: prometheus.DefBuckets,
		},
	)
`},
	} {
		t.Run(test.testName, func(t *testing.T) {
			_, errors := searchFileForStableMetrics(fakeFilename, test.src)
			if len(errors) != 1 {
				t.Errorf("Unexpected number of errors, got %d, want 1", len(errors))
			}
			if !reflect.DeepEqual(errors[0], test.err) {
				t.Errorf("error:\ngot  %v\nwant %v", errors[0], test.err)
			}
		})
	}
}
