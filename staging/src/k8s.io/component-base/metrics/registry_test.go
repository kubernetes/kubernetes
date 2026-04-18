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
	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

// resetMetricsGlobalState resets all global state used by metrics tests.
// NOTE: This should be called at the start of tests and in cleanup to ensure test isolation.
func resetMetricsGlobalState(t *testing.T) {
	resetter := func() {
		hiddenMetricsTotal.Reset()
		disabledMetricsTotal.Reset()

		showHidden.Store(false)
		showHiddenOnce = sync.Once{}

		alphaCounter.ClearState()
		alphaDeprecatedCounter.ClearState()
		betaDeprecatedCounter.ClearState()
		alphaHiddenCounter.ClearState()

		disabledMetricsLock.Lock()
		disabledMetrics = map[string]struct{}{}
		disabledMetricsLock.Unlock()
	}

	// Reset at the start of tests
	resetter()

	// Register cleanup to reset again after test completes
	t.Cleanup(resetter)
}

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
	betaDeprecatedCounter = NewCounter(
		&CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_beta_dep_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    BETA,
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
	currentV1_17 := parseSemver("1.17.0")
	currentV1_18Alpha0 := parseSemver("1.18.0-alpha.0")
	currentV1_18Alpha1 := parseSemver("1.18.0-alpha.1")

	var tests = []struct {
		desc              string
		currentVersion    *semver.Version
		stabilityLevel    StabilityLevel
		deprecatedVersion string
		shouldHide        bool
	}{
		{
			desc:              "INTERNAL metric deprecated in the current release - should be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.17.0",
			shouldHide:        true,
		},
		{
			desc:              "INTERNAL metric deprecated 1 release ago - should be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.16.0",
			shouldHide:        true,
		},
		{
			desc:              "INTERNAL metric to be deprecated 1 release later - should not be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "BETA metric deprecated in the current release - should not be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.17.0",
			shouldHide:        false,
		},
		{
			desc:              "BETA metric deprecated 1 release ago - should be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.16.0",
			shouldHide:        true,
		},
		{
			desc:              "BETA metric to be deprecated 1 release later - should not be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated in the current release - should not be hidden",
			currentVersion:    parseSemver("1.17.0"),
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.17.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated 3 releases ago - should be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.14.0",
			shouldHide:        true,
		},
		{
			desc:              "STABLE metric deprecated 2 releases ago - should not be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.15.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric to be deprecated 1 release later - should not be hidden",
			currentVersion:    currentV1_17,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		// --- Pre-Alpha.0 Tests ---
		{
			desc:              "INTERNAL metric deprecated in the current minor - should not be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "INTERNAL metric deprecated 1 release ago - should be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.17.0",
			shouldHide:        true,
		},
		{
			desc:              "BETA metric deprecated in the current minor - should not be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "BETA metric deprecated 1 release ago - should not be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.17.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated in the current minor - should be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated 1 release ago - should not be hidden",
			currentVersion:    currentV1_18Alpha0,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.17.0",
			shouldHide:        false,
		},
		// --- Pre-Alpha.1 Tests ---
		{

			desc:              "INTERNAL metric in the current minor - should be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.18.0",
			shouldHide:        true,
		},
		{
			desc:              "INTERNAL metric deprecated in prior patch - should be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    INTERNAL,
			deprecatedVersion: "1.18.0",
			shouldHide:        true,
		},
		{
			desc:              "BETA metric deprecated in the current minor - should not be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "BETA metric deprecated in prior patch - should not be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "BETA metric deprecated 1 release ago - should be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    BETA,
			deprecatedVersion: "1.17.0",
			shouldHide:        true,
		},
		{
			desc:              "STABLE metric deprecated in the current minor - should not be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated in prior patch - should not be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.18.0",
			shouldHide:        false,
		},
		{
			desc:              "STABLE metric deprecated 3 minors ago - should be hidden",
			currentVersion:    currentV1_18Alpha1,
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.15.0",
			shouldHide:        true,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.desc, func(t *testing.T) {
			result := shouldHide(tc.stabilityLevel, tc.currentVersion, parseSemver(tc.deprecatedVersion))
			assert.Equalf(t, tc.shouldHide, result, "expected should hide %v, but got %v", tc.shouldHide, result)
		})
	}
}

func TestRegister(t *testing.T) {
	resetMetricsGlobalState(t)

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
			expectedIsCreatedValues: []bool{false},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{true},
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
	resetMetricsGlobalState(t)

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
			metrics:         []*Counter{betaDeprecatedCounter, betaDeprecatedCounter},
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

func TestDeferredMetricCreation(t *testing.T) {
	resetMetricsGlobalState(t)

	registry := newKubeRegistry(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	registry.EnableDeferredCreation()

	counter := NewCounter(&CounterOpts{
		Namespace:      "deferred",
		Name:           "test_counter",
		Help:           "counter help",
		StabilityLevel: ALPHA,
	})
	histogram := NewHistogram(&HistogramOpts{
		Namespace:      "deferred",
		Name:           "test_histogram",
		Help:           "histogram help",
		Buckets:        prometheus.DefBuckets,
		StabilityLevel: ALPHA,
	})

	// Metrics registered while in deferred mode should not be created yet.
	registry.MustRegister(counter)
	registry.MustRegister(histogram)
	if counter.IsCreated() {
		t.Error("counter should not be created while in deferred mode")
	}
	if histogram.IsCreated() {
		t.Error("histogram should not be created while in deferred mode")
	}

	// After finalization, metrics should be created.
	registry.FinalizeDeferredMetrics()
	if !counter.IsCreated() {
		t.Error("counter should be created after finalization")
	}
	if !histogram.IsCreated() {
		t.Error("histogram should be created after finalization")
	}

	// Subsequent registrations should take effect immediately.
	lateCounter := NewCounter(&CounterOpts{
		Namespace:      "deferred",
		Name:           "late_counter",
		Help:           "registered after finalization",
		StabilityLevel: ALPHA,
	})
	registry.MustRegister(lateCounter)
	if !lateCounter.IsCreated() {
		t.Error("late counter should be created immediately after finalization")
	}
}

func TestDeferredGatherSafetyNet(t *testing.T) {
	resetMetricsGlobalState(t)

	registry := newKubeRegistry(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	registry.EnableDeferredCreation()

	counter := NewCounter(&CounterOpts{
		Namespace:      "deferred_gather",
		Name:           "test_counter",
		Help:           "counter help",
		StabilityLevel: ALPHA,
	})
	registry.MustRegister(counter)

	// Counter should not be created yet (deferred mode).
	if counter.IsCreated() {
		t.Error("counter should not be created before Gather")
	}

	// Gather should finalize deferred metrics automatically.
	output := `
		# HELP deferred_gather_test_counter [ALPHA] counter help
		# TYPE deferred_gather_test_counter counter
		deferred_gather_test_counter 0
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(output), "deferred_gather_test_counter"); err != nil {
		t.Fatal(err)
	}
	if !counter.IsCreated() {
		t.Error("counter should be created after Gather")
	}
}

func TestFinalizeDeferredMetricsIdempotent(t *testing.T) {
	resetMetricsGlobalState(t)

	registry := newKubeRegistry(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	registry.EnableDeferredCreation()

	counter := NewCounter(&CounterOpts{
		Namespace:      "idempotent",
		Name:           "test_counter",
		Help:           "counter help",
		StabilityLevel: ALPHA,
	})
	registry.MustRegister(counter)

	// Multiple calls should not panic or double-register.
	registry.FinalizeDeferredMetrics()
	registry.FinalizeDeferredMetrics()
	if !counter.IsCreated() {
		t.Error("counter should be created after finalization")
	}
}

func TestVecHookTriggersFinalize(t *testing.T) {
	resetMetricsGlobalState(t)

	registry := newKubeRegistry(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	registry.EnableDeferredCreation()

	counter := NewCounterVec(&CounterOpts{
		Namespace:      "vec_hook",
		Name:           "test_counter",
		Help:           "counter help",
		StabilityLevel: ALPHA,
	}, []string{"label"})
	registry.MustRegister(counter)

	if counter.IsCreated() {
		t.Error("vec should not be created while in deferred mode")
	}

	// Point the hook at this registry, mimicking what legacyregistry
	// does at init() time for its default registry.
	oldHook := FinalizeDeferredRegistries
	FinalizeDeferredRegistries = registry.FinalizeDeferredMetrics
	defer func() { FinalizeDeferredRegistries = oldHook }()

	counter.WithLabelValues("foo").Inc()
	if !counter.IsCreated() {
		t.Error("vec should be created automatically after Inc")
	}

	output := `
		# HELP vec_hook_test_counter [ALPHA] counter help
		# TYPE vec_hook_test_counter counter
		vec_hook_test_counter{label="foo"} 1
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(output), "vec_hook_test_counter"); err != nil {
		t.Fatal(err)
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
