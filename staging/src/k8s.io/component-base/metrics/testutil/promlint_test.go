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

	"github.com/prometheus/client_golang/prometheus/testutil/promlint"
)

func TestLinter(t *testing.T) {
	var tests = []struct {
		name   string
		metric string
		expect string
	}{
		{
			name: "problematic metric should be reported",
			metric: `
				# HELP test_problematic_total [ALPHA] non-counter metrics should not have total suffix
				# TYPE test_problematic_total gauge
				test_problematic_total{some_label="some_value"} 1
				`,
			expect: `non-counter metrics should not have "_total" suffix`,
		},
		// Don't need to test metrics in exception list, they will be covered by e2e test.
		// In addition, we don't need to update this test when we remove metrics from exception list in the future.
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			linter := NewPromLinter(strings.NewReader(tc.metric))
			problems, err := linter.Lint()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(problems) == 0 {
				t.Fatalf("expecte a problem but got none")
			}

			if problems[0].Text != tc.expect {
				t.Fatalf("expect: %s, but got: %s", tc.expect, problems[0])
			}
		})
	}
}

func TestMergeProblems(t *testing.T) {
	problemOne := Problem{
		Metric: "metric_one",
		Text:   "problem one",
	}
	problemTwo := Problem{
		Metric: "metric_two",
		Text:   "problem two",
	}

	var tests = []struct {
		name     string
		problems []Problem
		expected string
	}{
		{
			name:     "no problem",
			problems: nil,
			expected: "",
		},
		{
			name:     "one problem",
			problems: []Problem{problemOne},
			expected: "metric_one:problem one",
		},
		{
			name:     "more than one problem",
			problems: []Problem{problemOne, problemTwo},
			expected: "metric_one:problem one,metric_two:problem two",
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			got := mergeProblems(tc.problems)
			if tc.expected != got {
				t.Errorf("expected: %s, but got: %s", tc.expected, got)
			}
		})
	}
}

func TestCheckUnusedExceptions(t *testing.T) {
	// Save original exceptionMetrics
	origExceptions := exceptionMetrics
	defer func() { exceptionMetrics = origExceptions }()

	// Set a controlled list for testing
	exceptionMetrics = []string{"test_metric_1", "test_metric_2"}

	var tests = []struct {
		name        string
		problems    []promlint.Problem
		expectError bool
		errorStr    string
	}{
		{
			name: "all exceptions used",
			problems: []promlint.Problem{
				{Metric: "test_metric_1", Text: "some error"},
				{Metric: "test_metric_2", Text: "some error"},
			},
			expectError: false,
		},
		{
			name: "one exception unused",
			problems: []promlint.Problem{
				{Metric: "test_metric_1", Text: "some error"},
			},
			expectError: true,
			errorStr:    "metrics in exception list but have no violations: test_metric_2",
		},
		{
			name:        "all exceptions unused",
			problems:    []promlint.Problem{},
			expectError: true,
			errorStr:    "metrics in exception list but have no violations: test_metric_1, test_metric_2",
		},
		{
			name: "problems not in exception list are ignored in this check",
			problems: []promlint.Problem{
				{Metric: "test_metric_1", Text: "some error"},
				{Metric: "test_metric_2", Text: "some error"},
				{Metric: "some_other_metric", Text: "some error"},
			},
			expectError: false,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			err := CheckUnusedExceptions(tc.problems)
			if tc.expectError {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				if err.Error() != tc.errorStr {
					t.Fatalf("expected error: %s, but got: %s", tc.errorStr, err.Error())
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
