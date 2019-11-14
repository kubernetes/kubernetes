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

	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

var (
	v115         = semver.MustParse("1.15.0")
	v114         = semver.MustParse("1.14.0")
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
		registryVersion         *semver.Version
		expectedErrors          []error
		expectedIsCreatedValues []bool
		expectedIsDeprecated    []bool
		expectedIsHidden        []bool
	}{
		{
			desc:                    "test alpha metric",
			metrics:                 []*Counter{alphaCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{false},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test registering same metric multiple times",
			metrics:                 []*Counter{alphaCounter, alphaCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{true, true},
			expectedIsDeprecated:    []bool{false, false},
			expectedIsHidden:        []bool{false, false},
		},
		{
			desc:                    "test alpha deprecated metric",
			metrics:                 []*Counter{alphaDeprecatedCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test alpha hidden metric",
			metrics:                 []*Counter{alphaHiddenCounter},
			registryVersion:         &v115,
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
				if err != test.expectedErrors[i] && err.Error() != test.expectedErrors[i].Error() {
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
		{
			desc:            "test must registering same hidden metric",
			metrics:         []*Counter{alphaHiddenCounter, alphaHiddenCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, false}, // hidden metrics no-opt
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
	assert.Equalf(t, expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), expectedMetricCount)

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
	assert.Equalf(t, expectedMetricCount, len(ms), "Got %v metrics, Want: %v metrics", len(ms), expectedMetricCount)
	assert.Nil(t, err, "Gather failed %v", err)

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
