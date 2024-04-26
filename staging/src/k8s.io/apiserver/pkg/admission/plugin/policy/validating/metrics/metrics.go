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

// ValidationErrorType defines different error types that happen to a validation expression
type ValidationErrorType string

const (
	// ValidationCompileError indicates that the expression fails to compile.
	ValidationCompileError ValidationErrorType = "compile_error"
	// ValidatingInternalError indicates that the expression fails due to internal
	// errors that are out of the control of the user.
	ValidatingInternalError ValidationErrorType = "internal_error"
	// ValidatingOutOfBudget indicates that the expression fails due to running
	// out of cost budget, or the budget cannot be obtained.
	ValidatingOutOfBudget ValidationErrorType = "out_of_budget"
	// ValidationNoError indicates that the expression returns without an error.
	ValidationNoError ValidationErrorType = "no_error"
)

var (
	// Metrics provides access to validation admission metrics.
	Metrics = newValidationAdmissionMetrics()
)

// ValidatingAdmissionPolicyMetrics aggregates Prometheus metrics related to validation admission control.
type ValidatingAdmissionPolicyMetrics struct {
	policyCheck   *metrics.CounterVec
	policyLatency *metrics.HistogramVec
}

func newValidationAdmissionMetrics() *ValidatingAdmissionPolicyMetrics {
	check := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      metricsNamespace,
			Subsystem:      metricsSubsystem,
			Name:           "check_total",
			Help:           "Validation admission policy check total, labeled by policy and further identified by binding and enforcement action taken.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"policy", "policy_binding", "enforcement_action"},
	)
	latency := metrics.NewHistogramVec(&metrics.HistogramOpts{
		Namespace: metricsNamespace,
		Subsystem: metricsSubsystem,
		Name:      "check_duration_seconds",
		Help:      "Validation admission latency for individual validation expressions in seconds, labeled by policy and further including binding and enforcement action taken.",
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
		[]string{"policy", "policy_binding", "enforcement_action"},
	)

	legacyregistry.MustRegister(check)
	legacyregistry.MustRegister(latency)
	return &ValidatingAdmissionPolicyMetrics{policyCheck: check, policyLatency: latency}
}

// Reset resets all validation admission-related Prometheus metrics.
func (m *ValidatingAdmissionPolicyMetrics) Reset() {
	m.policyCheck.Reset()
	m.policyLatency.Reset()
}

// ObserveAdmission observes a policy validation, with an optional error to indicate the error that may occur but ignored.
func (m *ValidatingAdmissionPolicyMetrics) ObserveAdmission(ctx context.Context, elapsed time.Duration, policy, binding string, errorType ValidationErrorType) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, "allow").Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, "allow").Observe(elapsed.Seconds())
}

// ObserveRejection observes a policy validation error that was at least one of the reasons for a deny.
func (m *ValidatingAdmissionPolicyMetrics) ObserveRejection(ctx context.Context, elapsed time.Duration, policy, binding string) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, "deny").Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, "deny").Observe(elapsed.Seconds())
}

// ObserveAudit observes a policy validation audit annotation was published for a validation failure.
func (m *ValidatingAdmissionPolicyMetrics) ObserveAudit(ctx context.Context, elapsed time.Duration, policy, binding string) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, "audit").Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, "audit").Observe(elapsed.Seconds())
}

// ObserveWarn observes a policy validation warning was published for a validation failure.
func (m *ValidatingAdmissionPolicyMetrics) ObserveWarn(ctx context.Context, elapsed time.Duration, policy, binding string) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, "warn").Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, "warn").Observe(elapsed.Seconds())
}
