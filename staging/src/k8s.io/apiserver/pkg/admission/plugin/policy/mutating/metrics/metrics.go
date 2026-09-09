/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	metricsNamespace = "apiserver"
	metricsSubsystem = "mutating_admission_policy"
)

// MutationErrorType defines different error types that happen to a mutation expression
type MutationErrorType string

const (
	// MutationCompileError indicates that the expression fails to compile.
	MutationCompileError MutationErrorType = "compile_error"
	// MutatingInvalidError indicates that the expression fails due to internal
	// errors that are out of the control of the user.
	MutatingInvalidError MutationErrorType = "invalid_error"
	// MutatingOutOfBudget indicates that the expression fails due to running
	// out of cost budget, or the budget cannot be obtained.
	MutatingOutOfBudget MutationErrorType = "out_of_budget"
	// MutationNoError indicates that the expression returns without an error.
	MutationNoError MutationErrorType = "no_error"
)

var (
	// Metrics provides access to mutation admission metrics.
	Metrics = newMutationAdmissionMetrics()
)

// MutatingAdmissionPolicyMetrics aggregates Prometheus metrics related to mutation admission control.
type MutatingAdmissionPolicyMetrics struct {
	policyCheck   *metrics.CounterVec
	policyLatency *metrics.HistogramVec
}

func newMutationAdmissionMetrics() *MutatingAdmissionPolicyMetrics {
	check := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      metricsNamespace,
			Subsystem:      metricsSubsystem,
			Name:           "check_total",
			Help:           "Mutation admission policy check total, labeled by policy and further identified by binding.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"policy", "policy_binding", "error_type"},
	)
	latency := metrics.NewHistogramVec(&metrics.HistogramOpts{
		Namespace: metricsNamespace,
		Subsystem: metricsSubsystem,
		Name:      "check_duration_seconds",
		Help:      "Mutation admission latency for individual mutation expressions in seconds, labeled by policy and binding.",
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
		[]string{"policy", "policy_binding", "error_type"},
	)

	legacyregistry.MustRegister(check)
	legacyregistry.MustRegister(latency)
	return &MutatingAdmissionPolicyMetrics{policyCheck: check, policyLatency: latency}
}

// Reset resets all mutation admission-related Prometheus metrics.
func (m *MutatingAdmissionPolicyMetrics) Reset() {
	m.policyCheck.Reset()
	m.policyLatency.Reset()
}

// ObserveAdmission observes a policy mutation, with an optional error to indicate the error that may occur but ignored.
func (m *MutatingAdmissionPolicyMetrics) ObserveAdmission(ctx context.Context, elapsed time.Duration, policy, binding string, errorType MutationErrorType) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, string(errorType)).Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, string(errorType)).Observe(elapsed.Seconds())
}

// ObserveRejection observes a policy mutation error that was at least one of the reasons for a deny.
func (m *MutatingAdmissionPolicyMetrics) ObserveRejection(ctx context.Context, elapsed time.Duration, policy, binding string, errorType MutationErrorType) {
	m.policyCheck.WithContext(ctx).WithLabelValues(policy, binding, string(errorType)).Inc()
	m.policyLatency.WithContext(ctx).WithLabelValues(policy, binding, string(errorType)).Observe(elapsed.Seconds())
}
