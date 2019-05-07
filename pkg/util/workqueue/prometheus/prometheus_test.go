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

package prometheus

import (
	"fmt"
	"testing"

	client "github.com/prometheus/client_model/go"

	"github.com/prometheus/client_golang/prometheus"
)

func TestRegistration(t *testing.T) {
	type checkFunc func([]*client.MetricFamily) error

	gaugeHasValue := func(name string, expected float64) checkFunc {
		return func(mfs []*client.MetricFamily) error {
			for _, mf := range mfs {
				if mf.GetName() != name {
					continue
				}

				if got := mf.GetMetric()[0].Gauge.GetValue(); got != expected {
					return fmt.Errorf("expected %q gauge value %v, got %v", name, expected, got)
				}

				return nil
			}

			return fmt.Errorf("want metric %q, got none", name)
		}
	}

	counterHasValue := func(name string, expected float64) checkFunc {
		return func(mfs []*client.MetricFamily) error {
			for _, mf := range mfs {
				if mf.GetName() != name {
					continue
				}

				if got := mf.GetMetric()[0].Counter.GetValue(); got != expected {
					return fmt.Errorf("expected %q counter value %v, got %v", name, expected, got)
				}

				return nil
			}

			return fmt.Errorf("want metric %q, got none", name)
		}
	}

	histogramHasSum := func(name string, expected float64) checkFunc {
		return func(mfs []*client.MetricFamily) error {
			for _, mf := range mfs {
				if mf.GetName() != name {
					continue
				}

				if got := mf.GetMetric()[0].Histogram.GetSampleSum(); got != expected {
					return fmt.Errorf("expected %q histogram sample sum %v, got %v", name, expected, got)
				}

				return nil
			}

			return fmt.Errorf("want metric %q, got none", name)
		}
	}

	tests := []struct {
		name     string
		register func(*prometheusMetricsProvider)
		checks   []checkFunc
	}{
		{
			name: "depth",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewDepthMetric("foo")
				d.Inc()
			},

			checks: []checkFunc{
				gaugeHasValue("workqueue_depth", 1.0),
			},
		},
		{
			name: "adds",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewAddsMetric("foo")
				d.Inc()
			},

			checks: []checkFunc{
				counterHasValue("workqueue_adds_total", 1.0),
			},
		},
		{
			name: "latency",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewLatencyMetric("foo")
				d.Observe(10.0)
			},

			checks: []checkFunc{
				histogramHasSum("workqueue_queue_duration_seconds", 10.0),
			},
		},
		{
			name: "duration",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewWorkDurationMetric("foo")
				d.Observe(10.0)
			},

			checks: []checkFunc{
				histogramHasSum("workqueue_work_duration_seconds", 10.0),
			},
		},
		{
			name: "unfinished work",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewUnfinishedWorkSecondsMetric("foo")
				d.Set(3.0)
			},

			checks: []checkFunc{
				gaugeHasValue("workqueue_unfinished_work_seconds", 3.0),
			},
		},
		{
			name: "unfinished work",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewUnfinishedWorkSecondsMetric("foo")
				d.Set(3.0)
			},

			checks: []checkFunc{
				gaugeHasValue("workqueue_unfinished_work_seconds", 3.0),
			},
		},
		{
			name: "longest running processor",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewLongestRunningProcessorSecondsMetric("foo")
				d.Set(3.0)
			},

			checks: []checkFunc{
				gaugeHasValue("workqueue_longest_running_processor_seconds", 3.0),
			},
		},
		{
			name: "retries",

			register: func(p *prometheusMetricsProvider) {
				d := p.NewRetriesMetric("foo")
				d.Inc()
			},

			checks: []checkFunc{
				counterHasValue("workqueue_retries_total", 1.0),
			},
		},

		{
			name: "double registration",

			register: func(p *prometheusMetricsProvider) {
				d1 := p.NewDepthMetric("bar")
				d1.Inc()
				d2 := p.NewDepthMetric("bar")
				d2.Inc()

				d3 := p.NewDeprecatedDepthMetric("bar")
				d3.Inc()
				d4 := p.NewDeprecatedDepthMetric("bar")
				d4.Inc()
			},

			checks: []checkFunc{
				gaugeHasValue("workqueue_depth", 2.0),
				gaugeHasValue("bar_depth", 2.0),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// reset prometheus registry for each test
			reg := prometheus.NewRegistry()
			prometheus.DefaultRegisterer = reg
			prometheus.DefaultGatherer = reg
			registerMetrics()

			var p prometheusMetricsProvider

			tc.register(&p)
			mfs, err := prometheus.DefaultGatherer.Gather()
			if err != nil {
				t.Fatal(err)
			}

			for _, check := range tc.checks {
				if err := check(mfs); err != nil {
					t.Error(err)
				}
			}
		})
	}
}
