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
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
)

var apiVersions = []string{"v1", "v1alpha1", "v1beta1"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testDeclarativeValidateForDeclarative(t, v)
		})
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "scheduling.k8s.io",
			APIVersion: apiVersion,
		},
	)

	testCases := map[string]struct {
		input        scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPriorityClass(),
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testValidateUpdateForDeclarative(t, v)
		})
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "scheduling.k8s.io",
			APIVersion: apiVersion,
		},
	)

	validPC := mkValidPriorityClass()
	testCases := map[string]struct {
		old, update  scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    validPC,
			update: validPC,
		},
		"invalid update changing value": {
			old:    validPC,
			update: mkValidPriorityClass(tweakValue(200)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), 200, "field is immutable").WithOrigin("immutable"),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidPriorityClass(tweaks ...func(*scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-priority-class",
		},
		Value:       100,
		Description: "This is a valid priority class",
	}

	for _, tweak := range tweaks {
		tweak(&pc)
	}
	return pc
}

func tweakValue(value int32) func(*scheduling.PriorityClass) {
	return func(pc *scheduling.PriorityClass) {
		pc.Value = value
	}
}
