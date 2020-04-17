/*
Copyright 2020 The Kubernetes Authors.

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

package testutil

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
)

func TestNewFakeKubeRegistry(t *testing.T) {
	registryVersion := "1.18.0"
	counter := metrics.NewCounter(
		&metrics.CounterOpts{
			Name: "test_counter_name",
			Help: "counter help",
		},
	)
	deprecatedCounter := metrics.NewCounter(
		&metrics.CounterOpts{
			Name:              "test_deprecated_counter",
			Help:              "counter help",
			DeprecatedVersion: "1.18.0",
		},
	)
	hiddenCounter := metrics.NewCounter(
		&metrics.CounterOpts{
			Name:              "test_hidden_counter",
			Help:              "counter help",
			DeprecatedVersion: "1.17.0",
		},
	)

	var tests = []struct {
		name     string
		metric   *metrics.Counter
		expected string
	}{
		{
			name:   "normal",
			metric: counter,
			expected: `
				# HELP test_counter_name [ALPHA] counter help
				# TYPE test_counter_name counter
				test_counter_name 0
				`,
		},
		{
			name:   "deprecated",
			metric: deprecatedCounter,
			expected: `
				# HELP test_deprecated_counter [ALPHA] (Deprecated since 1.18.0) counter help
				# TYPE test_deprecated_counter counter
				test_deprecated_counter 0
				`,
		},
		{
			name:     "hidden",
			metric:   hiddenCounter,
			expected: ``,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			registry := NewFakeKubeRegistry(registryVersion)
			registry.MustRegister(tc.metric)
			if err := GatherAndCompare(registry, strings.NewReader(tc.expected), tc.metric.FQName()); err != nil {
				t.Fatal(err)
			}
		})
	}
}
