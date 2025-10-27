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
		"valid system priority class with high value": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-cluster-critical",
				},
				Value: 2000000000,
			},
			expectedErrs: field.ErrorList{},
		},
		"valid priority class with description": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "documented-priority",
				},
				Value:       500,
				Description: "Priority class for important workloads",
			},
			expectedErrs: field.ErrorList{},
		},
		"valid priority class with globalDefault": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "default-priority",
				},
				Value:         100,
				GlobalDefault: true,
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid - user priority class exceeding maximum": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "too-high-priority",
				},
				Value: 1500000000, // Exceeds HighestUserDefinablePriority (1000000000)
			},
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("value"), "maximum allowed value of a user defined priority is 1000000000"),
			},
		},
		"invalid - system prefix without being system priority": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-custom",
				},
				Value: 1000,
			},
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "name"), ""),
			},
		},
		"invalid - missing name": {
			input: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "",
				},
				Value: 1000,
			},
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "name"), ""),
			},
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
		"valid update of globalDefault from false to true": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:         1000,
				GlobalDefault: false,
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:         1000,
				GlobalDefault: true,
			},
			expectedErrs: field.ErrorList{},
		},
		"valid update of globalDefault from true to false": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:         1000,
				GlobalDefault: true,
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:         1000,
				GlobalDefault: false,
			},
			expectedErrs: field.ErrorList{},
		},
		"valid update adding description": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value: 1000,
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       1000,
				Description: "newly added description",
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid update - value changed": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value: 1000,
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value: 2000,
			},
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("value"), "may not be changed in an update."),
			},
		},
		"valid update with no changes": {
			old: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       500,
				Description: "same description",
			},
			update: scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-priority",
					ResourceVersion: "1",
				},
				Value:       500,
				Description: "same description",
			},
			expectedErrs: field.ErrorList{},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}
