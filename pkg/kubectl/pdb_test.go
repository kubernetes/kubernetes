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

package kubectl

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

	tests := map[string]struct {
		params       map[string]interface{}
		expectErrMsg string
		expectPDB    *policy.PodDisruptionBudget
	}{
		"test-valid-use": {
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
		"test-missing-name-param": {
			params: map[string]interface{}{
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		"test-blank-name-param": {
			params: map[string]interface{}{
				"name":          "",
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		"test-invalid-name-param": {
			params: map[string]interface{}{
				"name":          1,
				"min-available": minAvailable,
				"selector":      selector,
			},
			expectErrMsg: "expected string, found int for 'name'",
		},
		"test-missing-min-available-param": {
			params: map[string]interface{}{
				"name":     name,
				"selector": selector,
			},
			expectErrMsg: "expected string, found <nil> for 'min-available'",
		},
		"test-blank-min-available-param": {
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
		"test-invalid-min-available-param": {
			params: map[string]interface{}{
				"name":          name,
				"min-available": 1,
				"selector":      selector,
			},
			expectErrMsg: "expected string, found int for 'min-available'",
		},
		"test-missing-selector-param": {
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
			},
			expectErrMsg: "Parameter: selector is required",
		},
		"test-blank-selector-param": {
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
				"selector":      "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		"test-invalid-selector-param": {
			params: map[string]interface{}{
				"name":          name,
				"min-available": minAvailable,
				"selector":      1,
			},
			expectErrMsg: "expected string, found int for 'selector'",
		},
	}

	generator := PodDisruptionBudgetV1Generator{}
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		switch {
		case test.expectErrMsg != "" && err != nil:
			if err.Error() != test.expectErrMsg {
				t.Errorf("test '%s': expect error '%s', but saw '%s'", name, test.expectErrMsg, err.Error())
			}
			continue
		case test.expectErrMsg != "" && err == nil:
			t.Errorf("test '%s': expected error '%s' and didn't get one", name, test.expectErrMsg)
			continue
		case test.expectErrMsg == "" && err != nil:
			t.Errorf("test '%s': unexpected error %s", name, err.Error())
			continue
		}
		if !reflect.DeepEqual(obj.(*policy.PodDisruptionBudget), test.expectPDB) {
			t.Errorf("test '%s': expected:\n%#v\nsaw:\n%#v", name, test.expectPDB, obj.(*policy.PodDisruptionBudget))
		}
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

	tests := map[string]struct {
		params       map[string]interface{}
		expectErrMsg string
		expectPDB    *policy.PodDisruptionBudget
	}{
		"test-valid-min-available": {
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
		"test-valid-max-available": {
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
		"test-missing-name-param": {
			params: map[string]interface{}{
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		"test-blank-name-param": {
			params: map[string]interface{}{
				"name":            "",
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "Parameter: name is required",
		},
		"test-invalid-name-param": {
			params: map[string]interface{}{
				"name":            1,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'name'",
		},
		"test-missing-min-available-param": {
			params: map[string]interface{}{
				"name":            name,
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found <nil> for 'min-available'",
		},
		"test-invalid-min-available-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   1,
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'min-available'",
		},
		"test-missing-max-available-param": {
			params: map[string]interface{}{
				"name":          name,
				"min-available": "",
				"selector":      selector,
			},
			expectErrMsg: "expected string, found <nil> for 'max-unavailable'",
		},
		"test-invalid-max-available-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": 1,
				"selector":        selector,
			},
			expectErrMsg: "expected string, found int for 'max-unavailable'",
		},
		"test-blank-min-available-max-unavailable-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        selector,
			},
			expectErrMsg: "one of min-available or max-unavailable must be specified",
		},
		"test-min-available-max-unavailable-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   minAvailable,
				"max-unavailable": maxUnavailable,
				"selector":        selector,
			},
			expectErrMsg: "min-available and max-unavailable cannot be both specified",
		},
		"test-missing-selector-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		"test-blank-selector-param": {
			params: map[string]interface{}{
				"name":            name,
				"min-available":   "",
				"max-unavailable": "",
				"selector":        "",
			},
			expectErrMsg: "Parameter: selector is required",
		},
		"test-invalid-selector-param": {
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
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		switch {
		case test.expectErrMsg != "" && err != nil:
			if err.Error() != test.expectErrMsg {
				t.Errorf("test '%s': expect error '%s', but saw '%s'", name, test.expectErrMsg, err.Error())
			}
			continue
		case test.expectErrMsg != "" && err == nil:
			t.Errorf("test '%s': expected error '%s' and didn't get one", name, test.expectErrMsg)
			continue
		case test.expectErrMsg == "" && err != nil:
			t.Errorf("test '%s': unexpected error %s", name, err.Error())
			continue
		}
		if !reflect.DeepEqual(obj.(*policy.PodDisruptionBudget), test.expectPDB) {
			t.Errorf("test '%s': expected:\n%#v\nsaw:\n%#v", name, test.expectPDB, obj.(*policy.PodDisruptionBudget))
		}
	}
}
