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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

func TestValidatePriorityClass(t *testing.T) {
	successCases := map[string]scheduling.PriorityClass{
		"no description": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: ""},
			Value:      100,
		},
		"with description": {
			ObjectMeta:    metav1.ObjectMeta{Name: "tier1", Namespace: ""},
			Value:         200,
			GlobalDefault: false,
			Description:   "Used for the highest priority pods.",
		},
	}

	for k, v := range successCases {
		if errs := ValidatePriorityClass(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]scheduling.PriorityClass{
		"with namespace": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "foo"},
			Value:      100,
		},
		"invalid name": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier&1", Namespace: ""},
			Value:      100,
		},
	}

	for k, v := range errorCases {
		if errs := ValidatePriorityClass(&v); len(errs) == 0 {
			t.Errorf("Expected error for %s, but it succeeded", k)
		}
	}
}

func TestValidatePriorityClassUpdate(t *testing.T) {
	old := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "1"},
		Value:      100,
	}
	successCases := map[string]scheduling.PriorityClass{
		"no change": {
			ObjectMeta:  metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:       100,
			Description: "Used for the highest priority pods.",
		},
		"change description": {
			ObjectMeta:  metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:       100,
			Description: "A different description.",
		},
		"remove description": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:      100,
		},
		"change globalDefault": {
			ObjectMeta:    metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:         100,
			GlobalDefault: true,
		},
	}

	for k, v := range successCases {
		if errs := ValidatePriorityClassUpdate(&v, &old); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]struct {
		P scheduling.PriorityClass
		T field.ErrorType
	}{
		"add namespace": {
			P: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "foo", ResourceVersion: "2"},
				Value:      100,
			},
			T: field.ErrorTypeInvalid,
		},
		"change name": {
			P: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: "tier2", Namespace: "", ResourceVersion: "2"},
				Value:      100,
			},
			T: field.ErrorTypeInvalid,
		},
		"remove value": {
			P: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			},
			T: field.ErrorTypeForbidden,
		},
		"change value": {
			P: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
				Value:      101,
			},
			T: field.ErrorTypeForbidden,
		},
	}

	for k, v := range errorCases {
		errs := ValidatePriorityClassUpdate(&v.P, &old)
		if len(errs) == 0 {
			t.Errorf("Expected error for %s, but it succeeded", k)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
		}
	}
}
