/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	metricsNamespace = "apiserver"
	metricsSubsystem = "validating_admission_policy"
)

var (
	// Metrics provides access to validation admission metrics.
	Metrics = newValidationAdmissionMetrics()
)

// ValidatingAdmissionPolicyMetrics aggregates Prometheus metrics related to validation admission control.
type ValidatingAdmissionPolicyMetrics struct {
	policyCheck      *metrics.CounterVec
	policyDefinition *metrics.CounterVec
	policyLatency    *metrics.HistogramVec
}

func newValidationAdmissionMetrics() *ValidatingAdmissionPolicyMetrics {
	check := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      metricsNamespace,
			Subsystem:      metricsSubsystem,
			Name:           "check_total",
			Help:           "Validation admission policy check total, labeled by policy and param resource, and further identified by binding, validation expression, enforcement action taken, and state.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"policy", "policy_binding", "validation_expression", "enforcement_action", "params", "state"},
	)
	definition := metrics.NewCounterVec(&metrics.CounterOpts{
		Namespace:      metricsNamespace,
		Subsystem:      metricsSubsystem,
		Name:           "definition_total",
		Help:           "Validation admission policy count total, labeled by state and enforcement action.",
		StabilityLevel: metrics.ALPHA,
	},
		[]string{"state", "enforcement_action"},
	)
	latency := metrics.NewHistogramVec(&metrics.HistogramOpts{
		Namespace: metricsNamespace,
		Subsystem: metricsSubsystem,
		Name:      "check_duration_seconds",
		Help:      "Validation admission latency for individual validation expressions in seconds, labeled by policy and param resource, further including binding, state and enforcement action taken.",
		// the bucket distribution here is based oo the benchmark suite at
		// github.com/DangerOnTheRanger/cel-benchmark performed on 16-core Intel Xeon
		// the lowest bucket was based around the 180ns/op figure for BenchmarkAccess,
		// plus some additional leeway to account for the apiserver doing other things
		// the largest bucket was chosen based on the fact that benchmarks indicate the
		// same Xeon running a CEL expression close to the estimated cost limit takes
		// around 760ms, so that bucket should only ever have the slowest CEL expressions
		// in it
		Buckets:        []float64{0.0000005, 0.001, 0.01, 0.1, 1.0},
		StabilityLevel: metrics.ALPHA,
	},
		[]string{"policy", "policy_binding", "validation_expression", "enforcement_action", "params", "state"},
	)

	legacyregistry.MustRegister(check)
	legacyregistry.MustRegister(definition)
	legacyregistry.MustRegister(latency)
	return &ValidatingAdmissionPolicyMetrics{policyCheck: check, policyDefinition: definition, policyLatency: latency}
}

// Reset resets all validation admission-related Prometheus metrics.
func (m *ValidatingAdmissionPolicyMetrics) Reset() {
	m.policyCheck.Reset()
	m.policyDefinition.Reset()
	m.policyLatency.Reset()
}

// ObserveDefinition observes a policy definition.
func (m *ValidatingAdmissionPolicyMetrics) ObserveDefinition(ctx context.Context, state, enforcementAction string) {
	m.policyDefinition.WithContext(ctx).WithLabelValues(state, enforcementAction).Inc()
}

// ObserveCheck observes a policy validation check.
func (m *ValidatingAdmissionPolicyMetrics) ObserveCheck(ctx context.Context, elapsed time.Duration, policy, binding, expression, enforcementAction, params, state string) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, expression, enforcementAction, params, state).Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, expression, enforcementAction, params, state).Observe(elapsed.Seconds())
}
