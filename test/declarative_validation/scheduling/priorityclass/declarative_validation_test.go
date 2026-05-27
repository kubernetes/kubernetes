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
	registry "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass"

	// Ensure all API groups are registered with the scheme.
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
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
		"valid": {
			input: mkValidPriorityClass(),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       scheduling.PriorityClass
		updateObj    scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(),
		},
		"value changed": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(func(pc *scheduling.PriorityClass) { pc.Value = 20 }),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"value set to unset": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(tweakValue()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"value unset to set": {
			oldObj:    mkValidPriorityClass(tweakValue()),
			updateObj: mkValidPriorityClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "scheduling.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "priorityclasses",
				Name:              "valid-priority-class",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidPriorityClass(tweaks ...func(*scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-priority-class",
		},
		Value: 10,
	}
	for _, tweak := range tweaks {
		tweak(&pc)
	}
	return pc
}

func tweakValue() func(*scheduling.PriorityClass) {
	return func(pc *scheduling.PriorityClass) {
		pc.Value = 0
	}
}
