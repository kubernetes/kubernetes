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

package framework

import (
	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"testing"
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
			DeprecatedVersion: &v115,
		},
	)
	alphaHiddenCounter = NewCounter(
		&CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_hidden_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    ALPHA,
			Help:              "counter help",
			DeprecatedVersion: &v114,
		},
	)
	stableCounter = NewCounter(
		&CounterOpts{
			Namespace:      "some_namespace",
			Name:           "test_some_other_counter",
			Subsystem:      "subsystem",
			StabilityLevel: STABLE,
			Help:           "counter help",
		},
	)
)

func TestRegister(t *testing.T) {
	var tests = []struct {
		desc                    string
		metrics                 []*kubeCounter
		registryVersion         *semver.Version
		expectedErrors          []error
		expectedIsCreatedValues []bool
		expectedIsDeprecated    []bool
		expectedIsHidden        []bool
	}{
		{
			desc:                    "test alpha metric",
			metrics:                 []*kubeCounter{alphaCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{false},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test registering same metric multiple times",
			metrics:                 []*kubeCounter{alphaCounter, alphaCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{true, true},
			expectedIsDeprecated:    []bool{false, false},
			expectedIsHidden:        []bool{false, false},
		},
		{
			desc:                    "test alpha deprecated metric",
			metrics:                 []*kubeCounter{alphaDeprecatedCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{true},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{false},
		},
		{
			desc:                    "test alpha hidden metric",
			metrics:                 []*kubeCounter{alphaHiddenCounter},
			registryVersion:         &v115,
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{false},
			expectedIsDeprecated:    []bool{true},
			expectedIsHidden:        []bool{true},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(*test.registryVersion)
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
		metrics         []*kubeCounter
		registryVersion *semver.Version
		expectedPanics  []bool
	}{
		{
			desc:            "test alpha metric",
			metrics:         []*kubeCounter{alphaCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
		{
			desc:            "test registering same metric multiple times",
			metrics:         []*kubeCounter{alphaCounter, alphaCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, true},
		},
		{
			desc:            "test alpha deprecated metric",
			metrics:         []*kubeCounter{alphaDeprecatedCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
		{
			desc:            "test must registering same deprecated metric",
			metrics:         []*kubeCounter{alphaDeprecatedCounter, alphaDeprecatedCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, true},
		},
		{
			desc:            "test alpha hidden metric",
			metrics:         []*kubeCounter{alphaHiddenCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false},
		},
		{
			desc:            "test must registering same hidden metric",
			metrics:         []*kubeCounter{alphaHiddenCounter, alphaHiddenCounter},
			registryVersion: &v115,
			expectedPanics:  []bool{false, false}, // hidden metrics no-opt
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			registry := newKubeRegistry(*test.registryVersion)
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
