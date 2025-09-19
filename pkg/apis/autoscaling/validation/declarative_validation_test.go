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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

// TestScaleDeclarativeValidation verifies that the validation rules that are applied uniformly to the Scale API.
func TestScaleDeclarativeValidation(t *testing.T) {
	apiGroup := "autoscaling"
	apiVersion := "v1"
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    apiGroup,
		APIVersion:  apiVersion,
		Subresource: "scale",
	})

	testCases := map[string]struct {
		input        autoscaling.Scale
		expectedErrs field.ErrorList
	}{
		// spec.replicas
		"spec.replicas: 0 replicas": {
			input: mkScale(setScaleSpecReplicas(0)),
		},
		"spec.replicas: positive replicas": {
			input: mkScale(setScaleSpecReplicas(100)),
		},
		"spec.replicas: negative replicas": {
			input: mkScale(setScaleSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			// The Scale API has no handwritten validation to compare against. So we simply test that all the versions
			// of the Scale subresource are correct for our test cases.
			// All resources that have a scale subresource are expected to test Scale validation against any handwritten
			// validation code defined on that resource.
			tester := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			obj, _ := legacyscheme.Scheme.ConvertToVersion(&tc.input, schema.GroupVersion{Group: apiGroup, Version: apiVersion})
			tester.Test(t, tc.expectedErrs, legacyscheme.Scheme.Validate(ctx, nil, obj, "scale"))

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.input, nil, "scale")
		})
	}
}

// mkScale produces a Scale which passes validation with no tweaks.
func mkScale(tweaks ...func(rc *autoscaling.Scale)) autoscaling.Scale {
	rc := autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: autoscaling.ScaleSpec{
			Replicas: 1,
		},
	}
	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}
func setScaleSpecReplicas(val int32) func(rc *autoscaling.Scale) {
	return func(rc *autoscaling.Scale) {
		rc.Spec.Replicas = val
	}
}
