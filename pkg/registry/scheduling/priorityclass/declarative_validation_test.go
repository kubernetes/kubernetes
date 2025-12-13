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

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidate(t, apiVersion)
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})
	testCases := map[string]struct {
		input        scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid PriorityClass": {
			input: mkValidPriorityClass(),
		},
		"missing name": {
			input: mkValidPriorityClass(func(pc *scheduling.PriorityClass) {
				pc.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "name"), ""),
			},
		},
		"invalid name": {
			input: mkValidPriorityClass(func(pc *scheduling.PriorityClass) {
				pc.Name = "Invalid_Name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "Invalid_Name", ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateUpdate(t, apiVersion)
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		Name:              "valid-priority",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		old          scheduling.PriorityClass
		update       scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkValidPriorityClass(func(pc *scheduling.PriorityClass) { pc.ResourceVersion = "1" }),
			update: mkValidPriorityClass(func(pc *scheduling.PriorityClass) { pc.ResourceVersion = "1"; pc.Description = "updated" }),
		},
		"update value": {
			old: mkValidPriorityClass(func(pc *scheduling.PriorityClass) { pc.ResourceVersion = "1" }),
			update: mkValidPriorityClass(func(pc *scheduling.PriorityClass) {
				pc.ResourceVersion = "1"
				pc.Value = 2000
			}),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("value"), "may not be changed in an update."),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidPriorityClass(tweaks ...func(pc *scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-priority",
		},
		Value:         1000,
		GlobalDefault: false,
		Description:   "A valid priority class",
	}
	for _, tweak := range tweaks {
		tweak(&pc)
	}
	return pc
}
