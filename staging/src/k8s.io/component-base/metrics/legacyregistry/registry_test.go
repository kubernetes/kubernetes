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

package legacyregistry

import (
	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"k8s.io/component-base/metrics"
	"testing"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func init() {
	SetRegistryFactoryVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
}

var (
	alphaCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      "some_namespace",
			Name:           "test_counter_name",
			Subsystem:      "subsystem",
			StabilityLevel: metrics.ALPHA,
			Help:           "counter help",
		},
	)
	alphaDeprecatedCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_dep_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    metrics.ALPHA,
			Help:              "counter help",
			DeprecatedVersion: "1.15.0",
		},
	)
	alphaHiddenCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:         "some_namespace",
			Name:              "test_alpha_hidden_counter",
			Subsystem:         "subsystem",
			StabilityLevel:    metrics.ALPHA,
			Help:              "counter help",
			DeprecatedVersion: "1.14.0",
		},
	)
)

func TestRegister(t *testing.T) {
	var tests = []struct {
		desc                    string
		metrics                 []*metrics.Counter
		registryVersion         *semver.Version
		expectedErrors          []error
		expectedIsCreatedValues []bool
		expectedIsDeprecated    []bool
		expectedIsHidden        []bool
	}{
		{
			desc:                    "test registering same metric multiple times",
			metrics:                 []*metrics.Counter{alphaCounter, alphaCounter},
			expectedErrors:          []error{nil, prometheus.AlreadyRegisteredError{}},
			expectedIsCreatedValues: []bool{true, true},
			expectedIsDeprecated:    []bool{false, false},
			expectedIsHidden:        []bool{false, false},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			//t.Errorf("len %v - %v\n", len(test.metrics), len(test.expectedErrors))
			for i, m := range test.metrics {
				//t.Errorf("m %v\n", m)
				err := Register(m)
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
		metrics         []*metrics.Counter
		registryVersion *semver.Version
		expectedPanics  []bool
	}{
		{
			desc:           "test must registering same deprecated metric",
			metrics:        []*metrics.Counter{alphaDeprecatedCounter, alphaDeprecatedCounter},
			expectedPanics: []bool{false, true},
		},
		{
			desc:           "test alpha hidden metric",
			metrics:        []*metrics.Counter{alphaHiddenCounter},
			expectedPanics: []bool{false},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			for i, m := range test.metrics {
				if test.expectedPanics[i] {
					assert.Panics(t,
						func() { MustRegister(m) },
						"Did not panic even though we expected it.")
				} else {
					MustRegister(m)
				}
			}
		})
	}
}

func TestDeferredRegister(t *testing.T) {
	// reset the global registry for this test.
	globalRegistryFactory = metricsRegistryFactory{
		registerQueue:     make([]metrics.Registerable, 0),
		mustRegisterQueue: make([]metrics.Registerable, 0),
		globalRegistry:    noopRegistry{},
	}
	var err error
	err = Register(alphaDeprecatedCounter)
	if err != nil {
		t.Errorf("Got err == %v, expected no error", err)
	}
	err = Register(alphaDeprecatedCounter)
	if err != nil {
		t.Errorf("Got err == %v, expected no error", err)
	}
	// set the global registry version
	errs := SetRegistryFactoryVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	if len(errs) != 1 {
		t.Errorf("Got %d errs, expected 1", len(errs))
		for _, err := range errs {
			t.Logf("\t Got %v", err)
		}
	}
}

func TestDeferredMustRegister(t *testing.T) {
	// reset the global registry for this test.
	globalRegistryFactory = metricsRegistryFactory{
		registerQueue:     make([]metrics.Registerable, 0),
		mustRegisterQueue: make([]metrics.Registerable, 0),
		globalRegistry:    noopRegistry{},
	}
	MustRegister(alphaDeprecatedCounter)

	MustRegister(alphaDeprecatedCounter)
	assert.Panics(t,
		func() {
			SetRegistryFactoryVersion(apimachineryversion.Info{
				Major:      "1",
				Minor:      "15",
				GitVersion: "v1.15.0-alpha-1.12345",
			})
		},
		"Did not panic even though we expected it.")
}

func TestPreloadedMetrics(t *testing.T) {
	// reset the global registry for this test.
	globalRegistryFactory = metricsRegistryFactory{
		registerQueue:     make([]metrics.Registerable, 0),
		mustRegisterQueue: make([]metrics.Registerable, 0),
	}

	SetRegistryFactoryVersion(apimachineryversion.Info{
		Major:      "1",
		Minor:      "15",
		GitVersion: "v1.15.0-alpha-1.12345",
	})
	// partial list of some preregistered metrics we expect
	expectedMetricNames := []string{"go_gc_duration_seconds", "process_start_time_seconds"}

	mf, err := globalRegistryFactory.globalRegistry.Gather()
	if err != nil {
		t.Errorf("Got unexpected error %v ", err)
	}
	metricNames := map[string]struct{}{}
	for _, f := range mf {
		metricNames[f.GetName()] = struct{}{}
	}
	for _, expectedMetric := range expectedMetricNames {
		if _, ok := metricNames[expectedMetric]; !ok {
			t.Errorf("Expected %v to be preregistered", expectedMetric)
		}
	}
}
