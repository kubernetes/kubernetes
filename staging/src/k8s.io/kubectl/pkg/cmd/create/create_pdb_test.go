/*
Copyright 2017 The Kubernetes Authors.

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

package create

import (
	"testing"

	policyv1 "k8s.io/api/policy/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestCreatePdbValidation(t *testing.T) {
	selectorOpts := "app=nginx"
	podAmountNumber := "3"
	podAmountPercent := "50%"

	tests := map[string]struct {
		options  *PodDisruptionBudgetOpts
		expected string
	}{
		"test-missing-name-param": {
			options: &PodDisruptionBudgetOpts{
				Selector:     selectorOpts,
				MinAvailable: podAmountNumber,
			},
			expected: "name must be specified",
		},
		"test-missing-selector-param": {
			options: &PodDisruptionBudgetOpts{
				Name:         "my-pdb",
				MinAvailable: podAmountNumber,
			},
			expected: "a selector must be specified",
		},
		"test-missing-max-unavailable-max-unavailable-param": {
			options: &PodDisruptionBudgetOpts{
				Name:     "my-pdb",
				Selector: selectorOpts,
			},
			expected: "one of min-available or max-unavailable must be specified",
		},
		"test-both-min-available-max-unavailable-param": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MinAvailable:   podAmountNumber,
				MaxUnavailable: podAmountPercent,
			},
			expected: "min-available and max-unavailable cannot be both specified",
		},
		"test-invalid-min-available-format": {
			options: &PodDisruptionBudgetOpts{
				Name:         "my-pdb",
				Selector:     selectorOpts,
				MinAvailable: "10GB",
			},
			expected: "invalid format specified for min-available",
		},
		"test-invalid-max-unavailable-format": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MaxUnavailable: "10GB",
			},
			expected: "invalid format specified for max-unavailable",
		},
		"test-valid-min-available-format": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MaxUnavailable: podAmountNumber,
			},
			expected: "",
		},
		"test-valid-max-unavailable-format": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MaxUnavailable: podAmountPercent,
			},
			expected: "",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {

			o := &PodDisruptionBudgetOpts{
				Name:           tc.options.Name,
				Selector:       tc.options.Selector,
				MinAvailable:   tc.options.MinAvailable,
				MaxUnavailable: tc.options.MaxUnavailable,
			}

			err := o.Validate()
			if err != nil && err.Error() != tc.expected {
				t.Errorf("unexpected error: %v", err)
			}
			if tc.expected != "" && err == nil {
				t.Errorf("expected error, got no error")
			}
		})
	}
}

func TestCreatePdb(t *testing.T) {
	selectorOpts := "app=nginx"
	podAmountNumber := "3"
	podAmountPercent := "50%"

	selector, err := metav1.ParseToLabelSelector(selectorOpts)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	minAvailableNumber := intstr.Parse(podAmountNumber)
	minAvailablePercent := intstr.Parse(podAmountPercent)

	maxUnavailableNumber := intstr.Parse(podAmountNumber)
	maxUnavailablePercent := intstr.Parse(podAmountPercent)

	tests := map[string]struct {
		options  *PodDisruptionBudgetOpts
		expected *policyv1.PodDisruptionBudget
	}{
		"test-valid-min-available-pods-number": {
			options: &PodDisruptionBudgetOpts{
				Name:         "my-pdb",
				Selector:     selectorOpts,
				MinAvailable: podAmountNumber,
			},
			expected: &policyv1.PodDisruptionBudget{
				TypeMeta: metav1.TypeMeta{
					Kind:       "PodDisruptionBudget",
					APIVersion: "policy/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-pdb",
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					Selector:     selector,
					MinAvailable: &minAvailableNumber,
				},
			},
		},
		"test-valid-min-available-pods-percentage": {
			options: &PodDisruptionBudgetOpts{
				Name:         "my-pdb",
				Selector:     selectorOpts,
				MinAvailable: podAmountPercent,
			},
			expected: &policyv1.PodDisruptionBudget{
				TypeMeta: metav1.TypeMeta{
					Kind:       "PodDisruptionBudget",
					APIVersion: "policy/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-pdb",
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					Selector:     selector,
					MinAvailable: &minAvailablePercent,
				},
			},
		},
		"test-valid-max-unavailable-pods-number": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MaxUnavailable: podAmountNumber,
			},
			expected: &policyv1.PodDisruptionBudget{
				TypeMeta: metav1.TypeMeta{
					Kind:       "PodDisruptionBudget",
					APIVersion: "policy/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-pdb",
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					Selector:       selector,
					MaxUnavailable: &maxUnavailableNumber,
				},
			},
		},
		"test-valid-max-unavailable-pods-percentage": {
			options: &PodDisruptionBudgetOpts{
				Name:           "my-pdb",
				Selector:       selectorOpts,
				MaxUnavailable: podAmountPercent,
			},
			expected: &policyv1.PodDisruptionBudget{
				TypeMeta: metav1.TypeMeta{
					Kind:       "PodDisruptionBudget",
					APIVersion: "policy/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-pdb",
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					Selector:       selector,
					MaxUnavailable: &maxUnavailablePercent,
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {

			podDisruptionBudget, err := tc.options.createPodDisruptionBudgets()
			if err != nil {
				t.Errorf("unexpected error:\n%#v\n", err)
				return
			}
			if !apiequality.Semantic.DeepEqual(podDisruptionBudget, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, podDisruptionBudget)
			}
		})
	}
}
