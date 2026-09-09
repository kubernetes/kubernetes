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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const testIdentifier = "test_validation_identifier"
const anotherTestIdentifier = "another_test_validation_identifier"

// TestDeclarativeValidationMismatchMetric tests that the mismatch metric correctly increments once
func TestDeclarativeValidationMismatchMetric(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric once
	Metrics.IncDeclarativeValidationMismatchMetric(testIdentifier)

	expected := `
	# HELP apiserver_validation_declarative_validation_parity_discrepancies_total [ALPHA] Number of discrepancies between declarative and handwritten validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_parity_discrepancies_total counter
	apiserver_validation_declarative_validation_parity_discrepancies_total{validation_identifier="test_validation_identifier"} 1
	# HELP apiserver_validation_declarative_validation_mismatch_total [BETA] Number of times declarative validation results differed from handwritten validation results for core types.
	# TYPE apiserver_validation_declarative_validation_mismatch_total counter
	apiserver_validation_declarative_validation_mismatch_total 1
	
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_validation_declarative_validation_parity_discrepancies_total", "apiserver_validation_declarative_validation_mismatch_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDeclarativeValidationPanicMetric tests that the panic metric correctly increments once
func TestDeclarativeValidationPanicMetric(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric once
	Metrics.IncDeclarativeValidationPanicMetric(testIdentifier)

	expected := `
	# HELP apiserver_validation_declarative_validation_panics_total [ALPHA] Number of panics in declarative validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_panics_total counter
	apiserver_validation_declarative_validation_panics_total{validation_identifier="test_validation_identifier"} 1
	# HELP apiserver_validation_declarative_validation_panic_total [BETA] Number of times declarative validation has panicked during validation.
	# TYPE apiserver_validation_declarative_validation_panic_total counter
	apiserver_validation_declarative_validation_panic_total 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected),
		"apiserver_validation_declarative_validation_panic_total", "apiserver_validation_declarative_validation_panics_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDuplicateValidationErrorMetric tests that the duplicate error metric correctly increments once
func TestDuplicateValidationErrorMetric(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric once
	Metrics.IncDuplicateValidationErrorMetric()

	expected := `
	# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
	# TYPE apiserver_validation_duplicate_validation_error_total counter
	apiserver_validation_duplicate_validation_error_total 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_validation_duplicate_validation_error_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDeclarativeValidationMismatchMetricMultiple tests that the mismatch metric correctly increments multiple times
func TestDeclarativeValidationMismatchMetricMultiple(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric three times
	Metrics.IncDeclarativeValidationMismatchMetric(testIdentifier)
	Metrics.IncDeclarativeValidationMismatchMetric(testIdentifier)
	Metrics.IncDeclarativeValidationMismatchMetric(anotherTestIdentifier)

	expected := `
	# HELP apiserver_validation_declarative_validation_parity_discrepancies_total [ALPHA] Number of discrepancies between declarative and handwritten validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_parity_discrepancies_total counter
	apiserver_validation_declarative_validation_parity_discrepancies_total{validation_identifier="test_validation_identifier"} 2
	apiserver_validation_declarative_validation_parity_discrepancies_total{validation_identifier="another_test_validation_identifier"} 1
	# HELP apiserver_validation_declarative_validation_mismatch_total [BETA] Number of times declarative validation results differed from handwritten validation results for core types.
	# TYPE apiserver_validation_declarative_validation_mismatch_total counter
	apiserver_validation_declarative_validation_mismatch_total 3
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_validation_declarative_validation_parity_discrepancies_total", "apiserver_validation_declarative_validation_mismatch_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDeclarativeValidationPanicMetricMultiple tests that the panic metric correctly increments multiple times
func TestDeclarativeValidationPanicMetricMultiple(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric three times
	Metrics.IncDeclarativeValidationPanicMetric(testIdentifier)
	Metrics.IncDeclarativeValidationPanicMetric(testIdentifier)
	Metrics.IncDeclarativeValidationPanicMetric(anotherTestIdentifier)

	expected := `
	# HELP apiserver_validation_declarative_validation_panics_total [ALPHA] Number of panics in declarative validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_panics_total counter
	apiserver_validation_declarative_validation_panics_total{validation_identifier="test_validation_identifier"} 2
	apiserver_validation_declarative_validation_panics_total{validation_identifier="another_test_validation_identifier"} 1
	# HELP apiserver_validation_declarative_validation_panic_total [BETA] Number of times declarative validation has panicked during validation.
	# TYPE apiserver_validation_declarative_validation_panic_total counter
	apiserver_validation_declarative_validation_panic_total 3
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_validation_declarative_validation_panic_total", "apiserver_validation_declarative_validation_panics_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDuplicateValidationErrorMetricMultiple tests that the duplicate error metric correctly increments multiple times
func TestDuplicateValidationErrorMetricMultiple(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment the metric three times
	Metrics.IncDuplicateValidationErrorMetric()
	Metrics.IncDuplicateValidationErrorMetric()
	Metrics.IncDuplicateValidationErrorMetric()

	expected := `
	# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
	# TYPE apiserver_validation_duplicate_validation_error_total counter
	apiserver_validation_duplicate_validation_error_total 3
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_validation_duplicate_validation_error_total"); err != nil {
		t.Fatal(err)
	}
}

// TestDeclarativeValidationMetricsReset tests that the Reset function correctly resets the metrics to zero
func TestDeclarativeValidationMetricsReset(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetValidationMetricsInstance()

	// Increment both metrics
	Metrics.IncDeclarativeValidationMismatchMetric(testIdentifier)
	Metrics.IncDeclarativeValidationPanicMetric(testIdentifier)
	Metrics.IncDuplicateValidationErrorMetric()

	// Reset the metrics
	Metrics.Reset()

	// Verify they've been reset to zero
	expected := `
	# HELP apiserver_validation_declarative_validation_mismatch_total [BETA] Number of times declarative validation results differed from handwritten validation results for core types.
	# TYPE apiserver_validation_declarative_validation_mismatch_total counter
	apiserver_validation_declarative_validation_mismatch_total 0
	# HELP apiserver_validation_declarative_validation_panic_total [BETA] Number of times declarative validation has panicked during validation.
	# TYPE apiserver_validation_declarative_validation_panic_total counter
	apiserver_validation_declarative_validation_panic_total 0
	# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
	# TYPE apiserver_validation_duplicate_validation_error_total counter
	apiserver_validation_duplicate_validation_error_total 0
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected),
		"apiserver_validation_declarative_validation_mismatch_total",
		"apiserver_validation_declarative_validation_panic_total",
		"apiserver_validation_duplicate_validation_error_total"); err != nil {
		t.Fatal(err)
	}

	// Increment the metrics again to ensure they're still functional
	Metrics.IncDeclarativeValidationMismatchMetric(testIdentifier)
	Metrics.IncDeclarativeValidationPanicMetric(testIdentifier)
	Metrics.IncDuplicateValidationErrorMetric()

	// Verify they've been incremented correctly
	expected = `
	# HELP apiserver_validation_declarative_validation_mismatch_total [BETA] Number of times declarative validation results differed from handwritten validation results for core types.
	# TYPE apiserver_validation_declarative_validation_mismatch_total counter
	apiserver_validation_declarative_validation_mismatch_total 1
	# HELP apiserver_validation_declarative_validation_panic_total [BETA] Number of times declarative validation has panicked during validation.
	# TYPE apiserver_validation_declarative_validation_panic_total counter
	apiserver_validation_declarative_validation_panic_total 1
	# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
	# TYPE apiserver_validation_duplicate_validation_error_total counter
	apiserver_validation_duplicate_validation_error_total 1
	# HELP apiserver_validation_declarative_validation_parity_discrepancies_total [ALPHA] Number of discrepancies between declarative and handwritten validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_parity_discrepancies_total counter
	apiserver_validation_declarative_validation_parity_discrepancies_total{validation_identifier="test_validation_identifier"} 1
	# HELP apiserver_validation_declarative_validation_panics_total [ALPHA] Number of panics in declarative validation, broken down by validation identifier.
	# TYPE apiserver_validation_declarative_validation_panics_total counter
	apiserver_validation_declarative_validation_panics_total{validation_identifier="test_validation_identifier"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected),
		"apiserver_validation_declarative_validation_mismatch_total",
		"apiserver_validation_declarative_validation_parity_discrepancies_total",
		"apiserver_validation_declarative_validation_panic_total",
		"apiserver_validation_declarative_validation_panics_total",
		"apiserver_validation_duplicate_validation_error_total"); err != nil {
		t.Fatal(err)
	}
}
