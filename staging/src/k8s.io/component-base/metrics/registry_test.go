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
	"sync"
	"testing"

	"github.com/blang/semver/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

var (
	v115         = semver.MustParse("1.15.0")
	alphaCounter = NewCounter(
		&CounterOpts{
			Namespace:      "some_namespace",
			Name:           "test_counter_name",
			Subsystem:      "subsystem",
			StabilityLevel: ALPHA,
			Help:           "counter help",
		},
	)
	alphaDeprecatedCounter = NewCounter(
		&CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_dep_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    ALPHA,
			Help:              "counter help",
			DeprecatedVersion: "1.15.0",
		},
	)
	alphaHiddenCounter = NewCounter(
		&CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_hidden_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    ALPHA,
			Help:              "counter help",
			DeprecatedVersion: "1.14.0",
		},
	)
)

func TestShouldHide(t *testing.T) {
	currentVersion := parseVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	})

	var tests = []struct {
		desc              string
		deprecatedVersion string
		shouldHide        bool
	}{
		{
			desc:              "current minor release should not be hidden",
			deprecatedVersion: "1.17.0",
			shouldHide:        false,
		},
		{
			desc:              "older minor release should be hidden",
			deprecatedVersion: "1.16.0",
			shouldHide:        true,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.desc, func(t *testing.T) {
			result := shouldHide(&currentVersion, parseSemver(tc.deprecatedVersion))
			assert.Equalf(t, tc.shouldHide, result, "expected should hide %v, but got %v", tc.shouldHide, result)
		})
	}
}

func TestRegister(t *testing.T) {
	var tests = []struct {
		desc                    string
		metrics                 []*Counter
		expectedErrors          []error
		expectedIsCreatedValues []bool
		expectedIsDeprecated    []bool
		expectedIsHidden        []bool
	}{
		{
			desc:                    "test alpha metric",
			metrics:                 []*Counter{alphaCounter},
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{false},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test registering same metric multiple times",
			metrics:                 []*Counter{alphaCounter, alphaCounter},
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{true, true},
			expectedIsDeprecated:    []bool{false, false},
			expectedIsHidden:        []bool{false, false},
		},
		{
			desc:                    "test alpha deprecated metric",
			metrics:                 []*Counter{alphaDeprecatedCounter},
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test alpha hidden metric",
			metrics:                 []*Counter{alphaHiddenCounter},
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{false},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{true},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			for i, m := range test.metrics {
				err := registry.Register(m)
				if err != nil && err.Error() != test.expectedErrors[i].Error() {
					t.Errorf("Got unexpected error %v, wanted %v", err, test.expectedErrors[i])
				}
				if m.IsCreated() != test.expectedIsCreatedValues[i] {
					t.Errorf("Got isCreated == %v, wanted isCreated to be %v", m.IsCreated(), test.expectedIsCreatedValues[i])
				}
				if m.IsDeprecated() != test.expectedIsDeprecated[i] {
					t.Errorf("Got IsDeprecated == %v, wanted IsDeprecated to be %v", m.IsDeprecated(), test.expectedIsDeprecated[i])
				}
				if m.IsHidden() != test.expectedIsHidden[i] {
					t.Errorf("Got IsHidden == %v, wanted IsHidden to be %v", m.IsHidden(), test.expectedIsDeprecated[i])
				}
			}
		})
	}
}

func TestMustRegister(t *testing.T) {
	var tests = []struct {
		desc            string
		metrics         []*Counter
		registryVersion *semver.Version
		expectedPanics  []bool
	}{
		{
			desc:            "test alpha metric",
			metrics:         []*Counter{alphaCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
		{
			desc:            "test registering same metric multiple times",
			metrics:         []*Counter{alphaCounter, alphaCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, true},
		},
		{
			desc:            "test alpha deprecated metric",
			metrics:         []*Counter{alphaDeprecatedCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
		{
			desc:            "test must registering same deprecated metric",
			metrics:         []*Counter{alphaDeprecatedCounter, alphaDeprecatedCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, true},
		},
		{
			desc:            "test alpha hidden metric",
			metrics:         []*Counter{alphaHiddenCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
			for i, m := range test.metrics {
				if test.expectedPanics[i] {
					assert.Panics(t,
						func() { registry.MustRegister(m) },
						"Did not panic even though we expected it.")
				} else {
					registry.MustRegister(m)
				}
			}
		})
	}

}
func TestShowHiddenMetric(t *testing.T) {
	registry := newKubeRegistry(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})

	expectedMetricCount := 0
	registry.MustRegister(alphaHiddenCounter)

	ms, err := registry.Gather()
	require.NoError(t, err, "Gather failed %v", err)
	assert.Lenf(t, ms, expectedMetricCount, "Got %v metrics, Want: %v metrics", len(ms), expectedMetricCount)

	showHidden.Store(true)
	defer showHidden.Store(false)
	registry.MustRegister(NewCounter(
		&CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_show_hidden_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    ALPHA,
			Help:              "counter help",
			DeprecatedVersion: "1.14.0",
		},
	))
	expectedMetricCount = 1

	ms, err = registry.Gather()
	require.NoError(t, err, "Gather failed %v", err)
	assert.Lenf(t, ms, expectedMetricCount, "Got %v metrics, Want: %v metrics", len(ms), expectedMetricCount)
}

func TestValidateShowHiddenMetricsVersion(t *testing.T) {
	currentVersion := parseVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	})

	var tests = []struct {
		desc          string
		targetVersion string
		expectedError bool
	}{
		{
			desc:          "invalid version is not allowed",
			targetVersion: "1.invalid",
			expectedError: true,
		},
		{
			desc:          "patch version is not allowed",
			targetVersion: "1.16.0",
			expectedError: true,
		},
		{
			desc:          "old version is not allowed",
			targetVersion: "1.15",
			expectedError: true,
		},
		{
			desc:          "new version is not allowed",
			targetVersion: "1.17",
			expectedError: true,
		},
		{
			desc:          "valid version is allowed",
			targetVersion: "1.16",
			expectedError: false,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.desc, func(t *testing.T) {
			err := validateShowHiddenMetricsVersion(currentVersion, tc.targetVersion)

			if tc.expectedError {
				assert.Errorf(t, err, "Failed to test: %s", tc.desc)
			} else {
				assert.NoErrorf(t, err, "Failed to test: %s", tc.desc)
			}
		})
	}
}

func TestEnableHiddenMetrics(t *testing.T) {
	currentVersion := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	}

	var tests = []struct {
		name           string
		fqName         string
		counter        *Counter
		mustRegister   bool
		expectedMetric string
	}{
		{
			name:   "hide by register",
			fqName: "hidden_metric_register",
			counter: NewCounter(&CounterOpts{
				Name:              "hidden_metric_register",
				Help:              "counter help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.16.0",
			}),
			mustRegister: false,
			expectedMetric: `
				# HELP hidden_metric_register [STABLE] (Deprecated since 1.16.0) counter help
				# TYPE hidden_metric_register counter
				hidden_metric_register 1
				`,
		},
		{
			name:   "hide by must register",
			fqName: "hidden_metric_must_register",
			counter: NewCounter(&CounterOpts{
				Name:              "hidden_metric_must_register",
				Help:              "counter help",
				StabilityLevel:    STABLE,
				DeprecatedVersion: "1.16.0",
			}),
			mustRegister: true,
			expectedMetric: `
				# HELP hidden_metric_must_register [STABLE] (Deprecated since 1.16.0) counter help
				# TYPE hidden_metric_must_register counter
				hidden_metric_must_register 1
				`,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			registry := newKubeRegistry(currentVersion)
			if tc.mustRegister {
				registry.MustRegister(tc.counter)
			} else {
				_ = registry.Register(tc.counter)
			}

			tc.counter.Inc() // no-ops, because counter hasn't been initialized
			if err := testutil.GatherAndCompare(registry, strings.NewReader(""), tc.fqName); err != nil {
				t.Fatal(err)
			}

			SetShowHidden()
			defer func() {
				showHiddenOnce = *new(sync.Once)
				showHidden.Store(false)
			}()

			tc.counter.Inc()
			if err := testutil.GatherAndCompare(registry, strings.NewReader(tc.expectedMetric), tc.fqName); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEnableHiddenStableCollector(t *testing.T) {
	var currentVersion = apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	}
	var normal = NewDesc("test_enable_hidden_custom_metric_normal", "this is a normal metric", []string{"name"}, nil, STABLE, "")
	var hiddenA = NewDesc("test_enable_hidden_custom_metric_hidden_a", "this is the hidden metric A", []string{"name"}, nil, STABLE, "1.16.0")
	var hiddenB = NewDesc("test_enable_hidden_custom_metric_hidden_b", "this is the hidden metric B", []string{"name"}, nil, STABLE, "1.16.0")

	var tests = []struct {
		name                      string
		descriptors               []*Desc
		metricNames               []string
		expectMetricsBeforeEnable string
		expectMetricsAfterEnable  string
	}{
		{
			name:        "all hidden",
			descriptors: []*Desc{hiddenA, hiddenB},
			metricNames: []string{"test_enable_hidden_custom_metric_hidden_a",
				"test_enable_hidden_custom_metric_hidden_b"},
			expectMetricsBeforeEnable: "",
			expectMetricsAfterEnable: `
        		# HELP test_enable_hidden_custom_metric_hidden_a [STABLE] (Deprecated since 1.16.0) this is the hidden metric A
        		# TYPE test_enable_hidden_custom_metric_hidden_a gauge
        		test_enable_hidden_custom_metric_hidden_a{name="value"} 1
        		# HELP test_enable_hidden_custom_metric_hidden_b [STABLE] (Deprecated since 1.16.0) this is the hidden metric B
        		# TYPE test_enable_hidden_custom_metric_hidden_b gauge
        		test_enable_hidden_custom_metric_hidden_b{name="value"} 1
			`,
		},
		{
			name:        "partial hidden",
			descriptors: []*Desc{normal, hiddenA, hiddenB},
			metricNames: []string{"test_enable_hidden_custom_metric_normal",
				"test_enable_hidden_custom_metric_hidden_a",
				"test_enable_hidden_custom_metric_hidden_b"},
			expectMetricsBeforeEnable: `
        		# HELP test_enable_hidden_custom_metric_normal [STABLE] this is a normal metric
        		# TYPE test_enable_hidden_custom_metric_normal gauge
        		test_enable_hidden_custom_metric_normal{name="value"} 1
			`,
			expectMetricsAfterEnable: `
        		# HELP test_enable_hidden_custom_metric_normal [STABLE] this is a normal metric
        		# TYPE test_enable_hidden_custom_metric_normal gauge
        		test_enable_hidden_custom_metric_normal{name="value"} 1
        		# HELP test_enable_hidden_custom_metric_hidden_a [STABLE] (Deprecated since 1.16.0) this is the hidden metric A
        		# TYPE test_enable_hidden_custom_metric_hidden_a gauge
        		test_enable_hidden_custom_metric_hidden_a{name="value"} 1
        		# HELP test_enable_hidden_custom_metric_hidden_b [STABLE] (Deprecated since 1.16.0) this is the hidden metric B
        		# TYPE test_enable_hidden_custom_metric_hidden_b gauge
        		test_enable_hidden_custom_metric_hidden_b{name="value"} 1
			`,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			registry := newKubeRegistry(currentVersion)
			customCollector := newTestCustomCollector(tc.descriptors...)
			registry.CustomMustRegister(customCollector)

			if err := testutil.GatherAndCompare(registry, strings.NewReader(tc.expectMetricsBeforeEnable), tc.metricNames...); err != nil {
				t.Fatalf("before enable test failed: %v", err)
			}

			SetShowHidden()
			defer func() {
				showHiddenOnce = *new(sync.Once)
				showHidden.Store(false)
			}()

			if err := testutil.GatherAndCompare(registry, strings.NewReader(tc.expectMetricsAfterEnable), tc.metricNames...); err != nil {
				t.Fatalf("after enable test failed: %v", err)
			}

			// refresh descriptors so as to share with cases.
			for _, d := range tc.descriptors {
				d.ClearState()
			}
		})
	}
}

func TestRegistryReset(t *testing.T) {
	currentVersion := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	}
	registry := newKubeRegistry(currentVersion)
	resettableMetric := NewCounterVec(&CounterOpts{
		Name: "reset_metric",
		Help: "this metric can be reset",
	}, []string{"label"})
	// gauges cannot be reset
	nonResettableMetric := NewGauge(&GaugeOpts{
		Name: "not_reset_metric",
		Help: "this metric cannot be reset",
	})

	registry.MustRegister(resettableMetric)
	registry.MustRegister(nonResettableMetric)
	resettableMetric.WithLabelValues("one").Inc()
	resettableMetric.WithLabelValues("two").Inc()
	resettableMetric.WithLabelValues("two").Inc()
	nonResettableMetric.Inc()

	nonResettableOutput := `
        # HELP not_reset_metric [ALPHA] this metric cannot be reset
        # TYPE not_reset_metric gauge
        not_reset_metric 1
`
	resettableOutput := `
        # HELP reset_metric [ALPHA] this metric can be reset
        # TYPE reset_metric counter
        reset_metric{label="one"} 1
        reset_metric{label="two"} 2
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(nonResettableOutput+resettableOutput), "reset_metric", "not_reset_metric"); err != nil {
		t.Fatal(err)
	}
	registry.Reset()
	if err := testutil.GatherAndCompare(registry, strings.NewReader(nonResettableOutput), "reset_metric", "not_reset_metric"); err != nil {
		t.Fatal(err)
	}
}

func TestDisabledMetrics(t *testing.T) {
	o := NewOptions()
	o.DisabledMetrics = []string{"should_be_disabled"}
	o.Apply()
	currentVersion := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	}
	registry := newKubeRegistry(currentVersion)
	disabledMetric := NewCounterVec(&CounterOpts{
		Name: "should_be_disabled",
		Help: "this metric should be disabled",
	}, []string{"label"})
	// gauges cannot be reset
	enabledMetric := NewGauge(&GaugeOpts{
		Name: "should_be_enabled",
		Help: "this metric should not be disabled",
	})

	registry.MustRegister(disabledMetric)
	registry.MustRegister(enabledMetric)
	disabledMetric.WithLabelValues("one").Inc()
	disabledMetric.WithLabelValues("two").Inc()
	disabledMetric.WithLabelValues("two").Inc()
	enabledMetric.Inc()

	enabledMetricOutput := `
        # HELP should_be_enabled [ALPHA] this metric should not be disabled
        # TYPE should_be_enabled gauge
        should_be_enabled 1
`

	if err := testutil.GatherAndCompare(registry, strings.NewReader(enabledMetricOutput), "should_be_disabled", "should_be_enabled"); err != nil {
		t.Fatal(err)
	}
}
