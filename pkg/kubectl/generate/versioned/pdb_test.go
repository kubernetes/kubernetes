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

package versioned

import (
	"reflect"
	"testing"

	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestPodDisruptionBudgetV1Generate(t *testing.T) {
	name := "foo"
	minAvailable := "5"
	minAvailableIS := intstr.Parse(minAvailable)
	defaultMinAvailableIS := intstr.Parse("1")
	selector := "app=foo"
	labelSelector, err := metav1.ParseToLabelSelector(selector)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := []struct {
		name         string
		params       map[string]interface{}
		expectErrMsg string
		expectPDB    *policy.PodDisruptionBudget
	}{
		{
			name: "test-valid-use",
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectPDB: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailableIS,
					Selector:     labelSelector,
				},
			},
		},
		{
			name: "test-missing-name-param",
			params: map[string]interface{}{
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		{
			name: "test-blank-name-param",
			params: map[string]interface{}{
				"name":          "",
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		{
			name: "test-invalid-name-param",
			params: map[string]interface{}{
				"name":          1,
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "expected string, found int for 'name'",
		},
		{
			name: "test-missing-min-available-param",
			params: map[string]interface{}{
				"name":     name,
				"selector": selector,
			},
			expectErrMsg: "expected string, found <nil> for 'min-available'",
		},
		{
			name: "test-blank-min-available-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": "",
				"selector":      selector,
			},
			expectPDB: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &defaultMinAvailableIS,
					Selector:     labelSelector,
				},
			},
		},
		{
			name: "test-invalid-min-available-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": 1,
				"selector":      selector,
			},
			expectErrMsg: "expected string, found int for 'min-available'",
		},
		{
			name: "test-missing-selector-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
			},
			expectErrMsg: "Parameter: selector is required",
		},
		{
			name: "test-blank-selector-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
				"selector":      "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		{
			name: "test-invalid-selector-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
				"selector":      1,
			},
			expectErrMsg: "expected string, found int for 'selector'",
		},
	}

	generator := PodDisruptionBudgetV1Generator{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			switch {
			case tt.expectErrMsg != "" && err != nil:
				if err.Error() != tt.expectErrMsg {
					t.Errorf("test '%s': expect error '%s', but saw '%s'", tt.name, tt.expectErrMsg, err.Error())
				}
				return
			case tt.expectErrMsg != "" && err == nil:
				t.Errorf("test '%s': expected error '%s' and didn't get one", tt.name, tt.expectErrMsg)
				return
			case tt.expectErrMsg == "" && err != nil:
				t.Errorf("test '%s': unexpected error %s", tt.name, err.Error())
				return
			}
			if !reflect.DeepEqual(obj.(*policy.PodDisruptionBudget), tt.expectPDB) {
				t.Errorf("test '%s': expected:\n%#v\nsaw:\n%#v", tt.name, tt.expectPDB, obj.(*policy.PodDisruptionBudget))
			}
		})
	}
}

func TestPodDisruptionBudgetV2Generate(t *testing.T) {
	name := "foo"
	minAvailable := "1"
	minAvailableIS := intstr.Parse(minAvailable)
	maxUnavailable := "5%"
	maxUnavailableIS := intstr.Parse(maxUnavailable)
	selector := "app=foo"
	labelSelector, err := metav1.ParseToLabelSelector(selector)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := []struct {
		name         string
		params       map[string]interface{}
		expectErrMsg string
		expectPDB    *policy.PodDisruptionBudget
	}{
		{
			name: "test-valid-min-available",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   minAvailable,
				"max-unavailable": "",
				"selector":        selector,
			},
			expectPDB: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailableIS,
					Selector:     labelSelector,
				},
			},
		},
		{
			name: "test-valid-max-available",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": maxUnavailable,
				"selector":        selector,
			},
			expectPDB: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MaxUnavailable: &maxUnavailableIS,
					Selector:       labelSelector,
				},
			},
		},
		{
			name: "test-missing-name-param",
			params: map[string]interface{}{
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		{
			name: "test-blank-name-param",
			params: map[string]interface{}{
				"name":            "",
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		{
			name: "test-invalid-name-param",
			params: map[string]interface{}{
				"name":            1,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'name'",
		},
		{
			name: "test-missing-min-available-param",
			params: map[string]interface{}{
				"name":            name,
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found <nil> for 'min-available'",
		},
		{
			name: "test-invalid-min-available-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   1,
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'min-available'",
		},
		{
			name: "test-missing-max-available-param",
			params: map[string]interface{}{
				"name":          name,
				"min-available": "",
				"selector":      selector,
			},
			expectErrMsg: "expected string, found <nil> for 'max-unavailable'",
		},
		{
			name: "test-invalid-max-available-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": 1,
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'max-unavailable'",
		},
		{
			name: "test-blank-min-available-max-unavailable-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "one of min-available or max-unavailable must be specified",
		},
		{
			name: "test-min-available-max-unavailable-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   minAvailable,
				"max-unavailable": maxUnavailable,
				"selector":        selector,
			},
			expectErrMsg: "min-available and max-unavailable cannot be both specified",
		},
		{
			name: "test-missing-selector-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		{
			name: "test-blank-selector-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		{
			name: "test-invalid-selector-param",
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        1,
			},
			expectErrMsg: "expected string, found int for 'selector'",
		},
	}

	generator := PodDisruptionBudgetV2Generator{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			switch {
			case tt.expectErrMsg != "" && err != nil:
				if err.Error() != tt.expectErrMsg {
					t.Errorf("test '%s': expect error '%s', but saw '%s'", tt.name, tt.expectErrMsg, err.Error())
				}
				return
			case tt.expectErrMsg != "" && err == nil:
				t.Errorf("test '%s': expected error '%s' and didn't get one", tt.name, tt.expectErrMsg)
				return
			case tt.expectErrMsg == "" && err != nil:
				t.Errorf("test '%s': unexpected error %s", tt.name, err.Error())
				return
			}
			if !reflect.DeepEqual(obj.(*policy.PodDisruptionBudget), tt.expectPDB) {
				t.Errorf("test '%s': expected:\n%#v\nsaw:\n%#v", tt.name, tt.expectPDB, obj.(*policy.PodDisruptionBudget))
			}
		})
	}
}
