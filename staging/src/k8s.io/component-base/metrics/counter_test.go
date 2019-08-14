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

package metrics

import (
	"bytes"
	"github.com/blang/semver"
	"github.com/prometheus/common/expfmt"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"testing"
)

func TestCounter(t *testing.T) {
	var tests = []struct {
		desc string
		*CounterOpts
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			CounterOpts: &CounterOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				StabilityLevel: ALPHA,
				Help:           "counter help",
			},
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] counter help",
		},
		{
			desc: "Test deprecated",
			CounterOpts: &CounterOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "counter help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.15.0",
			},
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] (Deprecated since 1.15.0) counter help",
		},
		{
			desc: "Test hidden",
			CounterOpts: &CounterOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "counter help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.14.0",
			},
			expectedMetricCount: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := NewKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewCounter(test.CounterOpts)
			registry.MustRegister(c)

			ms, err := registry.Gather()
			var buf bytes.Buffer
			enc := expfmt.NewEncoder(&buf, "text/plain; version=0.0.4; charset=utf-8")

			if len(ms) != test.expectedMetricCount {
				t.Errorf("Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			}
			if err != nil {
				t.Fatalf("Gather failed %v", err)
			}
			for _, metric := range ms {
				err := enc.Encode(metric)
				if err != nil {
					t.Fatalf("Unexpected err %v in encoding the metric", err)
				}
				if metric.GetHelp() != test.expectedHelp {
					t.Errorf("Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
				}
			}

			// increment the counter N number of times and verify that the metric retains the count correctly
			numberOfTimesToIncrement := 3
			for i := 0; i < numberOfTimesToIncrement; i++ {
				c.Inc()
			}
			ms, err = registry.Gather()
			if err != nil {
				t.Fatalf("Gather failed %v", err)
			}
			for _, mf := range ms {
				for _, m := range mf.GetMetric() {
					if int(m.GetCounter().GetValue()) != numberOfTimesToIncrement {
						t.Errorf("Got %v, wanted %v as the count", m.GetCounter().GetValue(), numberOfTimesToIncrement)
					}
				}
			}
		})
	}
}

func TestCounterVec(t *testing.T) {
	var tests = []struct {
		desc string
		*CounterOpts
		labels                    []string
		registryVersion           *semver.Version
		expectedMetricFamilyCount int
		expectedHelp              string
	}{
		{
			desc: "Test non deprecated",
			CounterOpts: &CounterOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "counter help",
			},
			labels:                    []string{"label_a", "label_b"},
			expectedMetricFamilyCount: 1,
			expectedHelp:              "[ALPHA] counter help",
		},
		{
			desc: "Test deprecated",
			CounterOpts: &CounterOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "counter help",
				DeprecatedVersion: "1.15.0",
			},
			labels:                    []string{"label_a", "label_b"},
			expectedMetricFamilyCount: 1,
			expectedHelp:              "[ALPHA] (Deprecated since 1.15.0) counter help",
		},
		{
			desc: "Test hidden",
			CounterOpts: &CounterOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "counter help",
				DeprecatedVersion: "1.14.0",
			},
			labels:                    []string{"label_a", "label_b"},
			expectedMetricFamilyCount: 0,
			expectedHelp:              "counter help",
		},
		{
			desc: "Test alpha",
			CounterOpts: &CounterOpts{
				StabilityLevel: ALPHA,
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				Help:           "counter help",
			},
			labels:                    []string{"label_a", "label_b"},
			expectedMetricFamilyCount: 1,
			expectedHelp:              "[ALPHA] counter help",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := NewKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewCounterVec(test.CounterOpts, test.labels)
			registry.MustRegister(c)
			c.WithLabelValues("1", "2").Inc()
			mfs, err := registry.Gather()
			if len(mfs) != test.expectedMetricFamilyCount {
				t.Errorf("Got %v metric families, Want: %v metric families", len(mfs), test.expectedMetricFamilyCount)
			}
			if err != nil {
				t.Fatalf("Gather failed %v", err)
			}
			// this no-opts here when there are no metric families (i.e. when the metric is hidden)
			for _, mf := range mfs {
				if len(mf.GetMetric()) != 1 {
					t.Errorf("Got %v metrics, wanted 1 as the count", len(mf.GetMetric()))
				}
				if mf.GetHelp() != test.expectedHelp {
					t.Errorf("Got %s as help message, want %s", mf.GetHelp(), test.expectedHelp)
				}
			}

			// let's increment the counter and verify that the metric still works
			c.WithLabelValues("1", "3").Inc()
			c.WithLabelValues("2", "3").Inc()
			mfs, err = registry.Gather()
			if err != nil {
				t.Fatalf("Gather failed %v", err)
			}

			// this no-opts here when there are no metric families (i.e. when the metric is hidden)
			for _, mf := range mfs {
				if len(mf.GetMetric()) != 3 {
					t.Errorf("Got %v metrics, wanted 3 as the count", len(mf.GetMetric()))
				}
			}
		})
	}
}
