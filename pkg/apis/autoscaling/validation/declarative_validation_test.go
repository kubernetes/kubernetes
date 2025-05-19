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
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidateScaleForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    "autoscaling",
		APIVersion:  "v1",
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
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				t.Run(fmt.Sprintf("gates=%v", gateVal), func(t *testing.T) {
					// We only need to test both gate enabled and disabled together, because
					// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
					// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)

					errs := rest.ValidateDeclaratively(ctx, legacyscheme.Scheme, &tc.input)
					if gateVal {
						declarativeTakeoverErrs = errs
					} else {
						imperativeErrs = errs
					}
					// The errOutputMatcher is used to verify the output matches the expected errors in test cases.
					errOutputMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
					if len(tc.expectedErrs) > 0 {
						errOutputMatcher.Test(t, tc.expectedErrs, errs)
					} else if len(errs) != 0 {
						t.Errorf("expected no errors, but got: %v", errs)
					}
				})
			}
			// The equivalenceMatcher is used to verify the output errors from handwritten imperative validation
			// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)

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
