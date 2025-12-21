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

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestGauge(t *testing.T) {
	version1_15Alpha1 := apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	}

	var tests = []struct {
		desc string
		*GaugeOpts
		expectedMetricCount int
		expectedHelp        string
	}{
		// Non-deprecated metrics
		{
			desc: "ALPHA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				StabilityLevel: ALPHA,
				Help:           "gauge help",
			},
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] gauge help",
		},
		{
			desc: "BETA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: BETA,
			},
			expectedMetricCount: 1,
			expectedHelp:        "[BETA] gauge help",
		},
		{
			desc: "STABLE metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: STABLE,
			},
			expectedMetricCount: 1,
			expectedHelp:        "[STABLE] gauge help",
		},
		// Deprecated metrics
		{
			desc: "ALPHA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				StabilityLevel:    ALPHA,
				Help:              "gauge help",
				DeprecatedVersion: "1.15.0",
			},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "BETA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				StabilityLevel:    BETA,
				Help:              "gauge help",
				DeprecatedVersion: "1.15.0",
			},
			expectedMetricCount: 1,
			expectedHelp:        "[BETA] (Deprecated since 1.15.0) gauge help",
		},
		{
			desc: "STABLE metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.15.0",
			},
			expectedMetricCount: 1,
			expectedHelp:        "[STABLE] (Deprecated since 1.15.0) gauge help",
		},
		// Hidden metrics
		{
			desc: "ALPHA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				StabilityLevel:    ALPHA,
				Help:              "gauge help",
				DeprecatedVersion: "1.14.0",
			},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "BETA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				StabilityLevel:    BETA,
				Help:              "gauge help",
				DeprecatedVersion: "1.14.0",
			},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "STABLE metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.12.0",
			},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(version1_15Alpha1)
			c := NewGauge(test.GaugeOpts)
			registry.MustRegister(c)

			ms, err := registry.Gather()
			assert.Lenf(t, ms, test.expectedMetricCount, "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			require.NoError(t, err, "Gather failed %v", err)

			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's increment the counter and verify that the metric still works
			c.Set(100)
			c.Set(101)
			expected := 101
			ms, err = registry.Gather()
			require.NoError(t, err, "Gather failed %v", err)

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
	version1_15Alpha1 := apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	}

	var tests = []struct {
		desc string
		*GaugeOpts
		labels              []string
		expectedMetricCount int
		expectedHelp        string
	}{
		// Non-deprecated metrics
		{
			desc: "ALPHA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace: "namespace",
				Name:      "metric_test_name",
				Subsystem: "subsystem",
				Help:      "gauge help",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 1,
			expectedHelp:        "[ALPHA] gauge help",
		},
		{
			desc: "BETA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: BETA,
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 1,
			expectedHelp:        "[BETA] gauge help",
		},
		{
			desc: "STABLE metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_test_name",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: STABLE,
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 1,
			expectedHelp:        "[STABLE] gauge help",
		},
		// Deprecated metrics
		{
			desc: "ALPHA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.15.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "BETA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.15.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 1,
			expectedHelp:        "[BETA] (Deprecated since 1.15.0) gauge help",
		},
		{
			desc: "STABLE metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.15.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 1,
			expectedHelp:        "[STABLE] (Deprecated since 1.15.0) gauge help",
		},
		// Hidden metrics
		{
			desc: "ALPHA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				DeprecatedVersion: "1.14.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "BETA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.14.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
		{
			desc: "STABLE metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_test_name",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.12.0",
			},
			labels:              []string{"label_a", "label_b"},
			expectedMetricCount: 0,
			expectedHelp:        "gauge help",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(version1_15Alpha1)
			c := NewGaugeVec(test.GaugeOpts, test.labels)
			registry.MustRegister(c)
			c.WithLabelValues("1", "2").Set(1.0)
			ms, err := registry.Gather()
			assert.Lenf(t, ms, test.expectedMetricCount, "Got %v metrics, Want: %v metrics", len(ms), test.expectedMetricCount)
			require.NoError(t, err, "Gather failed %v", err)
			for _, metric := range ms {
				assert.Equalf(t, test.expectedHelp, metric.GetHelp(), "Got %s as help message, want %s", metric.GetHelp(), test.expectedHelp)
			}

			// let's increment the counter and verify that the metric still works
			c.WithLabelValues("1", "3").Set(1.0)
			c.WithLabelValues("2", "3").Set(1.0)
			ms, err = registry.Gather()
			require.NoError(t, err, "Gather failed %v", err)

			for _, mf := range ms {
				assert.Lenf(t, mf.GetMetric(), 3, "Got %v metrics, wanted 3 as the count", len(mf.GetMetric()))
			}
		})
	}
}

func TestGaugeFunc(t *testing.T) {
	version1_15Alpha1 := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}

	var function = func() float64 {
		return 1
	}

	var tests = []struct {
		desc string
		*GaugeOpts
		expectedMetrics string
	}{
		// Non-deprecated metrics
		{
			desc: "ALPHA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: ALPHA,
				Name:           "metric_non_deprecated",
				Help:           "gauge help",
			},
			expectedMetrics: `
# HELP namespace_subsystem_metric_non_deprecated [ALPHA] gauge help
# TYPE namespace_subsystem_metric_non_deprecated gauge
namespace_subsystem_metric_non_deprecated 1
			`,
		},
		{
			desc: "BETA metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: BETA,
				Name:           "metric_non_deprecated",
				Help:           "gauge help",
			},
			expectedMetrics: `
# HELP namespace_subsystem_metric_non_deprecated [BETA] gauge help
# TYPE namespace_subsystem_metric_non_deprecated gauge
namespace_subsystem_metric_non_deprecated 1
			`,
		},
		{
			desc: "STABLE metric non deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:      "namespace",
				Subsystem:      "subsystem",
				StabilityLevel: STABLE,
				Name:           "metric_non_deprecated",
				Help:           "gauge help",
			},
			expectedMetrics: `
# HELP namespace_subsystem_metric_non_deprecated [STABLE] gauge help
# TYPE namespace_subsystem_metric_non_deprecated gauge
namespace_subsystem_metric_non_deprecated 1
			`,
		},
		// Deprecated metrics
		{
			desc: "ALPHA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_deprecated",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.17.0",
			},
			expectedMetrics: "",
		},
		{
			desc: "BETA metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_deprecated",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.17.0",
			},
			expectedMetrics: `# HELP namespace_subsystem_metric_deprecated [BETA] (Deprecated since 1.17.0) gauge help
# TYPE namespace_subsystem_metric_deprecated gauge
namespace_subsystem_metric_deprecated 1
			`,
		},
		{
			desc: "STABLE metric deprecated",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_deprecated",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.17.0",
			},
			expectedMetrics: `# HELP namespace_subsystem_metric_deprecated [STABLE] (Deprecated since 1.17.0) gauge help
# TYPE namespace_subsystem_metric_deprecated gauge
namespace_subsystem_metric_deprecated 1
			`,
		},
		// Hidden metrics
		{
			desc: "ALPHA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_hidden",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.17.0",
			},
			expectedMetrics: "",
		},
		{
			desc: "BETA metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_hidden",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.16.0",
			},
			expectedMetrics: "",
		},
		{
			desc: "STABLE metric hidden",
			GaugeOpts: &GaugeOpts{
				Namespace:         "namespace",
				Subsystem:         "subsystem",
				Name:              "metric_hidden",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.14.0",
			},
			expectedMetrics: "",
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(version1_15Alpha1)
			gauge := newGaugeFunc(tc.GaugeOpts, function, parseVersion(version1_15Alpha1))
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

func TestGaugeWithLabelValueAllowList(t *testing.T) {
	labelAllowValues := map[string]string{
		"namespace_subsystem_metric_allowlist_test,label_a": "allowed",
	}
	labels := []string{"label_a", "label_b"}
	opts := &GaugeOpts{
		Namespace: "namespace",
		Name:      "metric_allowlist_test",
		Subsystem: "subsystem",
	}
	var tests = []struct {
		desc               string
		labelValues        [][]string
		expectMetricValues map[string]float64
	}{
		{
			desc:        "Test no unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"allowed", "b2"}},
			expectMetricValues: map[string]float64{
				"allowed b1": 100.0,
				"allowed b2": 100.0,
			},
		},
		{
			desc:        "Test unexpected input",
			labelValues: [][]string{{"allowed", "b1"}, {"not_allowed", "b1"}},
			expectMetricValues: map[string]float64{
				"allowed b1":    100.0,
				"unexpected b1": 100.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			labelValueAllowLists = map[string]*MetricLabelAllowList{}

			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			g := NewGaugeVec(opts, labels)
			registry.MustRegister(g)
			SetLabelAllowListFromCLI(labelAllowValues)
			for _, lv := range test.labelValues {
				g.WithLabelValues(lv...).Set(100.0)
			}
			mfs, err := registry.Gather()
			require.NoError(t, err, "Gather failed %v", err)

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
					actualValue := m.GetGauge().GetValue()
					assert.InDeltaf(t, expectedValue, actualValue, 0.01, "Got %v, wanted %v as the gauge while setting label_a to %v and label b to %v", actualValue, expectedValue, aValue, bValue)
				}
			}
		})
	}
}

func TestGaugeVecDeleteLabelValues(t *testing.T) {
	version := apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	}
	var tests = []struct {
		desc               string
		opts               *GaugeOpts
		labels             []string
		expectMetricExists bool
		expectDelete       bool
	}{
		// Non-deprecated metrics
		{
			desc: "ALPHA metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: ALPHA,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "BETA metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: BETA,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "STABLE metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: STABLE,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		// Deprecated metrics
		{
			desc: "ALPHA metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "BETA metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "STABLE metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		// Hidden metrics
		{
			desc: "ALPHA metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.14.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "BETA metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.14.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "STABLE metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.12.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(version)
			gv := NewGaugeVec(test.opts, test.labels)
			registry.MustRegister(gv)
			gv.WithLabelValues("foo", "bar").Set(42)

			ms, err := registry.Gather()
			require.NoError(t, err)
			found := false
			for _, mf := range ms {
				if *mf.Name == BuildFQName(test.opts.Namespace, test.opts.Subsystem, test.opts.Name) {
					for _, m := range mf.GetMetric() {
						for _, l := range m.Label {
							if *l.Name == "label_a" && *l.Value == "foo" {
								found = true
							}
						}
					}
				}
			}
			assert.Equal(t, test.expectMetricExists, found, "Metric existence mismatch before deletion")

			deleted := gv.DeleteLabelValues("foo", "bar")
			assert.Equal(t, test.expectDelete, deleted, "DeleteLabelValues return mismatch")

			// Confirm it no longer exists
			ms, err = registry.Gather()
			require.NoError(t, err)
			found = false
			for _, mf := range ms {
				if *mf.Name == BuildFQName(test.opts.Namespace, test.opts.Subsystem, test.opts.Name) {
					for _, m := range mf.GetMetric() {
						for _, l := range m.Label {
							if *l.Name == "label_a" && *l.Value == "foo" {
								found = true
							}
						}
					}
				}
			}
			assert.False(t, found, "Metric with label values should not exist after deletion")
		})
	}
}

func TestGaugeVecDeleteLabelValuesChecked(t *testing.T) {
	version := apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	}
	var tests = []struct {
		desc               string
		opts               *GaugeOpts
		labels             []string
		expectMetricExists bool
		expectDelete       bool
	}{
		// Non-deprecated metrics
		{
			desc: "ALPHA metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_checked_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: ALPHA,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "BETA metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_checked_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: BETA,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "STABLE metric non deprecated",
			opts: &GaugeOpts{
				Namespace:      "namespace",
				Name:           "metric_delete_checked_table",
				Subsystem:      "subsystem",
				Help:           "gauge help",
				StabilityLevel: STABLE,
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		// Deprecated metrics
		{
			desc: "ALPHA metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "BETA metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		{
			desc: "STABLE metric deprecated",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.15.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: true,
			expectDelete:       true,
		},
		// Hidden metrics
		{
			desc: "ALPHA metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    ALPHA,
				DeprecatedVersion: "1.14.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "BETA metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    BETA,
				DeprecatedVersion: "1.14.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
		{
			desc: "STABLE metric hidden",
			opts: &GaugeOpts{
				Namespace:         "namespace",
				Name:              "metric_delete_checked_table",
				Subsystem:         "subsystem",
				Help:              "gauge help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.12.0",
			},
			labels:             []string{"label_a", "label_b"},
			expectMetricExists: false,
			expectDelete:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(version)
			gv := NewGaugeVec(test.opts, test.labels)
			registry.MustRegister(gv)
			gv.WithLabelValues("foo", "bar").Set(42)

			ms, err := registry.Gather()
			require.NoError(t, err)
			found := false
			for _, mf := range ms {
				if *mf.Name == BuildFQName(test.opts.Namespace, test.opts.Subsystem, test.opts.Name) {
					for _, m := range mf.GetMetric() {
						for _, l := range m.Label {
							if *l.Name == "label_a" && *l.Value == "foo" {
								found = true
							}
						}
					}
				}
			}
			assert.Equal(t, test.expectMetricExists, found, "Metric existence mismatch before deletion")

			deleted, err := gv.DeleteLabelValuesChecked("foo", "bar")
			assert.Equal(t, test.expectDelete, deleted, "DeleteLabelValuesChecked return mismatch")
			require.NoError(t, err, "DeleteLabelValuesChecked should not return error")

			// Confirm it no longer exists
			ms, err = registry.Gather()
			require.NoError(t, err)
			found = false
			for _, mf := range ms {
				if *mf.Name == BuildFQName(test.opts.Namespace, test.opts.Subsystem, test.opts.Name) {
					for _, m := range mf.GetMetric() {
						for _, l := range m.Label {
							if *l.Name == "label_a" && *l.Value == "foo" {
								found = true
							}
						}
					}
				}
			}
			assert.False(t, found, "Metric with label values should not exist after deletion")
		})
	}
}
