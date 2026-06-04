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
	IncDeclarativeValidationMismatchMetric(validationIdentifier string)
	IncDeclarativeValidationPanicMetric(validationIdentifier string)
	IncDuplicateValidationErrorMetric()
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
	DeclarativeValidationMismatchCounterVector: metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "declarative_validation_parity_discrepancies_total",
			Help:           "Number of discrepancies between declarative and handwritten validation, broken down by validation identifier.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"validation_identifier"},
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
	DeclarativeValidationPanicCounterVector: metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "declarative_validation_panics_total",
			Help:           "Number of panics in declarative validation, broken down by validation identifier.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"validation_identifier"},
	),
	DuplicateValidationErrorCounter: metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "duplicate_validation_error_total",
			Help:           "Number of duplicate validation errors during validation.",
			StabilityLevel: metrics.INTERNAL,
		},
	),
}

// Metrics provides access to validation metrics.
var Metrics ValidationMetrics = validationMetricsInstance

func init() {
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationMismatchCounter)
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationMismatchCounterVector)
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationPanicCounter)
	legacyregistry.MustRegister(validationMetricsInstance.DeclarativeValidationPanicCounterVector)
	legacyregistry.MustRegister(validationMetricsInstance.DuplicateValidationErrorCounter)
}

type validationMetrics struct {
	DeclarativeValidationMismatchCounter       *metrics.Counter
	DeclarativeValidationMismatchCounterVector *metrics.CounterVec
	DeclarativeValidationPanicCounter          *metrics.Counter
	DeclarativeValidationPanicCounterVector    *metrics.CounterVec
	DuplicateValidationErrorCounter            *metrics.Counter
}

// Reset resets the validation metrics.
func (m *validationMetrics) Reset() {
	m.DeclarativeValidationMismatchCounter.Reset()
	m.DeclarativeValidationMismatchCounterVector.Reset()
	m.DeclarativeValidationPanicCounter.Reset()
	m.DeclarativeValidationPanicCounterVector.Reset()
	m.DuplicateValidationErrorCounter.Reset()
}

// IncDeclarativeValidationMismatchMetric increments the counter for the declarative_validation_mismatch_total metric.
func (m *validationMetrics) IncDeclarativeValidationMismatchMetric(validationIdentifier string) {
	m.DeclarativeValidationMismatchCounter.Inc()
	m.DeclarativeValidationMismatchCounterVector.WithLabelValues(validationIdentifier).Inc()
}

// IncDeclarativeValidationPanicMetric increments the counter for the declarative_validation_panic_total metric.
func (m *validationMetrics) IncDeclarativeValidationPanicMetric(validationIdentifier string) {
	m.DeclarativeValidationPanicCounter.Inc()
	m.DeclarativeValidationPanicCounterVector.WithLabelValues(validationIdentifier).Inc()
}

// IncDuplicateValidationErrorMetric increments the counter for the duplicate_validation_error_total metric.
func (m *validationMetrics) IncDuplicateValidationErrorMetric() {
	m.DuplicateValidationErrorCounter.Inc()
}

func ResetValidationMetricsInstance() {
	validationMetricsInstance.Reset()
}
