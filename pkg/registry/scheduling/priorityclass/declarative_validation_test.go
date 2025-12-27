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

package priorityclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

var apiVersions = []string{"v1", "v1alpha1", "v1beta1"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "scheduling.k8s.io",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		input        scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid user priority class with positive value": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "high-priority",
				},
				Value: 1000,
			},
			expectedErrs: field.ErrorList{},
		},
		"valid user priority class at maximum boundary": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "max-user-priority",
				},
				Value: 1000000000,
			},
			expectedErrs: field.ErrorList{},
		},
		"valid priority class with negative value": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "negative-priority",
				},
				Value: -1000,
			},
			expectedErrs: field.ErrorList{},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "scheduling.k8s.io",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		old          scheduling.PriorityClass
		update       scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update of description": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       1000,
				Description: "old description",
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       1000,
				Description: "new description",
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid update of value field (immutable)": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       1000,
				Description: "test description",
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       2000,
				Description: "test description",
			},
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}
