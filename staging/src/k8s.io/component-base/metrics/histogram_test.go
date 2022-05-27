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
	"testing"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestHistogram(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		*HistogramOpts
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			HistogramOpts: &HistogramOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "histogram help message",
				Buckets:   prometheus.DefBuckets,
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] histogram help message",
		},
		{
			desc: "Test deprecated",
			HistogramOpts: &HistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.15.0",
				Buckets:           prometheus.DefBuckets,
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] (Deprecated since 1.15.0) histogram help message",
		},
		{
			desc: "Test hidden",
			HistogramOpts: &HistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.14.0",
				Buckets:           prometheus.DefBuckets,
			},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "histogram help message",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewHistogram(test.HistogramOpts)
			registry.MustRegister(c)
			cm := c.ObserverMetric.(prometheus.Metric)

			metricChan := make(chan prometheus.Metric, 2)
			c.Collect(metricChan)
			close(metricChan)
			m1 := <-metricChan
			if m1 != cm {
				t.Error("Unexpected metric", m1, cm)
			}
			m2, ok := <-metricChan
			if ok {
				t.Error("Unexpected second metric", m2)
			}

			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)

			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's increment the counter and verify that the metric still works
			c.Observe(1)
			c.Observe(2)
			c.Observe(3)
			c.Observe(1.5)
			expected := 4
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				for _, m := range mf.GetMetric() {
					assert.Equalf(t, expected, int(m.GetHistogram().GetSampleCount()), "Got %v, want %v as the sample count", m.GetHistogram().GetSampleCount(), expected)
				}
			}
		})
	}
}

func TestHistogramVec(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		*HistogramOpts
		labels              []string
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			HistogramOpts: &HistogramOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "histogram help message",
				Buckets:   prometheus.DefBuckets,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] histogram help message",
		},
		{
			desc: "Test deprecated",
			HistogramOpts: &HistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.15.0",
				Buckets:           prometheus.DefBuckets,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] (Deprecated since 1.15.0) histogram help message",
		},
		{
			desc: "Test hidden",
			HistogramOpts: &HistogramOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "histogram help message",
				DeprecatedVersion: "1.14.0",
				Buckets:           prometheus.DefBuckets,
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "histogram help message",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewHistogramVec(test.HistogramOpts, test.labels)
			registry.MustRegister(c)
			ov12 := c.WithLabelValues("1", "2")
			cm1 := ov12.(prometheus.Metric)
			ov12.Observe(1.0)

			if test.expectedMetricCount > 0 {
				metricChan := make(chan prometheus.Metric, 2)
				c.Collect(metricChan)
				close(metricChan)
				m1 := <-metricChan
				if m1 != cm1 {
					t.Error("Unexpected metric", m1, cm1)
				}
				m2, ok := <-metricChan
				if ok {
					t.Error("Unexpected second metric", m2)
				}
			}

			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)
			for _, metric := range ms {
				if metric.GetHelp() != test.expectedHelp {
					assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
				}
			}

			// let's increment the counter and verify that the metric still works
			c.WithLabelValues("1", "3").Observe(1.0)
			c.WithLabelValues("2", "3").Observe(1.0)
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				assert.Equalf(t, 3, len(mf.GetMetric()), "Got %v metrics, wanted 3 as the count", len(mf.GetMetric()))
				for _, m := range mf.GetMetric() {
					assert.Equalf(t, uint64(1), m.GetHistogram().GetSampleCount(), "Got %v metrics, expected histogram sample count to equal 1", m.GetHistogram().GetSampleCount())
				}
			}
		})
	}
}

func TestHistogramWithLabelValueAllowList(t *testing.T) {
	labelAllowValues := map[string]string{
		"namespace_subsystem_metric_allowlist_test,label_a": "allowed",
	}
	labels := []string{"label_a", "label_b"}
	opts := &HistogramOpts{
		Namespace: "namespace",
		Name:      "metric_allowlist_test",
		Subsystem: "subsystem",
	}
	var tests = []struct {
		desc               string
		labelValues        [][]string
		expectMetricValues map[string]uint64
	}{
		{
			desc:        "Test no unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"allowed", "b2"}},
			expectMetricValues: map[string]uint64{
				"allowed b1": 1.0,
				"allowed b2": 1.0,
			},
		},
		{
			desc:        "Test unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"not_allowed", "b1"}},
			expectMetricValues: map[string]uint64{
				"allowed b1":    1.0,
				"unexpected b1": 1.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			SetLabelAllowListFromCLI(labelAllowValues)
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewHistogramVec(opts, labels)
			registry.MustRegister(c)

			for _, lv := range test.labelValues {
				c.WithLabelValues(lv...).Observe(1.0)
			}
			mfs, err := registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range mfs {
				if *mf.Name != BuildFQName(opts.Namespace, opts.Subsystem, opts.Name) {
					continue
				}
				mfMetric := mf.GetMetric()

				for _, m := range mfMetric {
					var aValue, bValue string
					for _, l := range m.Label {
						if *l.Name == "label_a" {
							aValue = *l.Value
						}
						if *l.Name == "label_b" {
							bValue = *l.Value
						}
					}
					labelValuePair := aValue + " " + bValue
					expectedValue, ok := test.expectMetricValues[labelValuePair]
					assert.True(t, ok, "Got unexpected label values, lable_a is %v, label_b is %v", aValue, bValue)
					actualValue := m.GetHistogram().GetSampleCount()
					assert.Equalf(t, expectedValue, actualValue, "Got %v, wanted %v as the count while setting label_a to %v and label b to %v", actualValue, expectedValue, aValue, bValue)
				}
			}
		})
	}
}
