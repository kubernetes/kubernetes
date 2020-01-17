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
	"strings"
	"testing"

	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestGauge(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		GaugeOpts
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			GaugeOpts: GaugeOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "gauge help",
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] gauge help",
		},
		{
			desc: "Test deprecated",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.15.0",
			},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] (Deprecated since 1.15.0) gauge help",
		},
		{
			desc: "Test hidden",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.14.0",
			},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewGauge(&test.GaugeOpts)
			registry.MustRegister(c)

			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)

			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's increment the counter and verify that the metric still works
			c.Set(100)
			c.Set(101)
			expected := 101
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				for _, m := range mf.GetMetric() {
					assert.Equalf(t, expected, int(m.GetGauge().GetValue()), "Got %v, wanted %v as the count", m.GetGauge().GetValue(), expected)
					t.Logf("%v\n", m.GetGauge().GetValue())
				}
			}
		})
	}
}

func TestGaugeVec(t *testing.T) {
	v115 := semver.MustParse("1.15.0")
	var tests = []struct {
		desc string
		GaugeOpts
		labels              []string
		registryVersion     *semver.Version
		expectedMetricCount int
		expectedHelp        string
	}{
		{
			desc: "Test non deprecated",
			GaugeOpts: GaugeOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "gauge help",
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] gauge help",
		},
		{
			desc: "Test deprecated",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.15.0",
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] (Deprecated since 1.15.0) gauge help",
		},
		{
			desc: "Test hidden",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.14.0",
			},
			labels:              []string{"label_a", "label_b"},
			registryVersion:     &v115,
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			c := NewGaugeVec(&test.GaugeOpts, test.labels)
			registry.MustRegister(c)
			c.WithLabelValues("1", "2").Set(1.0)
			ms, err := registry.Gather()
			assert.Equalf(t, test.expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			assert.Nil(t, err, "Gather failed %v", err)
			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's increment the counter and verify that the metric still works
			c.WithLabelValues("1", "3").Set(1.0)
			c.WithLabelValues("2", "3").Set(1.0)
			ms, err = registry.Gather()
			assert.Nil(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				assert.Equalf(t, 3, len(mf.GetMetric()), "Got %v metrics, wanted 3 as the count", len(mf.GetMetric()))
			}
		})
	}
}

func TestGaugeFunc(t *testing.T) {
	currentVersion := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}

	var function = func() float64 {
		return 1
	}

	var tests = []struct {
		desc string
		GaugeOpts
		expectedMetrics string
	}{
		{
			desc: "Test non deprecated",
			GaugeOpts: GaugeOpts{
				Namespace: "namespace",
				Subsystem: "subsystem",
				Name:      "metric_non_deprecated",
				Help:      "gauge help",
			},
			expectedMetrics: `
# HELP namespace_subsystem_metric_non_deprecated [ALPHA] gauge help
# TYPE namespace_subsystem_metric_non_deprecated gauge
namespace_subsystem_metric_non_deprecated 1
			`,
		},
		{
			desc: "Test deprecated",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_deprecated",
				Help:              "gauge help",
				DeprecatedVersion: "1.17.0",
			},
			expectedMetrics: `
# HELP namespace_subsystem_metric_deprecated [ALPHA] (Deprecated since 1.17.0) gauge help
# TYPE namespace_subsystem_metric_deprecated gauge
namespace_subsystem_metric_deprecated 1
`,
		},
		{
			desc: "Test hidden",
			GaugeOpts: GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_hidden",
				Help:              "gauge help",
				DeprecatedVersion: "1.16.0",
			},
			expectedMetrics: "",
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(currentVersion)
			gauge := newGaugeFunc(tc.GaugeOpts, function, parseVersion(currentVersion))
			if gauge != nil { // hidden metrics will not be initialize, register is not allowed
				registry.RawMustRegister(gauge)
			}

			metricName := BuildFQName(tc.GaugeOpts.Namespace, tc.GaugeOpts.Subsystem, tc.GaugeOpts.Name)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(tc.expectedMetrics), metricName); err != nil {
				t.Fatal(err)
			}
		})
	}
}
