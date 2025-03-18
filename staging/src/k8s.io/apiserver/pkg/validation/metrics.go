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

package validation

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver" // Keep it consistent; apiserver is handling it
	subsystem = "validation"
)

// ValidationMetrics is the interface for validation metrics.
type ValidationMetrics interface {
	IncDeclarativeValidationMismatchMetric()
	IncDeclarativeValidationPanicMetric()
	Reset()
}

var validationMetricsInstance = &validationMetrics{
	DeclarativeValidationMismatchCounter: metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "declarative_validation_mismatch_total",
			Help:           "Number of times declarative validation results differed from handwritten validation results for core types.",
			StabilityLevel: metrics.BETA,
		},
	),
	DeclarativeValidationPanicCounter: metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "declarative_validation_panic_total",
			Help:           "Number of times declarative validation has panicked during validation.",
			StabilityLevel: metrics.BETA,
		},
	),
}

// Metrics provides access to validation metrics.
var Metrics ValidationMetrics = validationMetricsInstance

func init() {
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationMismatchCounter)
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationPanicCounter)
}

type validationMetrics struct {
	DeclarativeValidationMismatchCounter *metrics.Counter
	DeclarativeValidationPanicCounter    *metrics.Counter
}

// Reset resets the validation metrics.
func (m *validationMetrics) Reset() {
	m.DeclarativeValidationMismatchCounter.Reset()
	m.DeclarativeValidationPanicCounter.Reset()
}

// IncDeclarativeValidationMismatchMetric increments the counter for the declarative_validation_mismatch_total metric.
func (m *validationMetrics) IncDeclarativeValidationMismatchMetric() {
	m.DeclarativeValidationMismatchCounter.Inc()
}

// IncDeclarativeValidationPanicMetric increments the counter for the declarative_validation_panic_total metric.
func (m *validationMetrics) IncDeclarativeValidationPanicMetric() {
	m.DeclarativeValidationPanicCounter.Inc()
}

func ResetValidationMetricsInstance() {
	validationMetricsInstance.Reset()
}
