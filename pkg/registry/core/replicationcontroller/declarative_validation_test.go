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

package replicationcontroller

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"empty resource": {
			input: mkValidReplicationController(),
		},
		// spec.replicas
		"nil replicas": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"0 replicas": {
			input: mkValidReplicationController(setSpecReplicas(0)),
		},
		"1 replicas": {
			input: mkValidReplicationController(setSpecReplicas(1)),
		},
		"positive replicas": {
			input: mkValidReplicationController(setSpecReplicas(100)),
		},
		"negative replicas": {
			input: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"0 minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"1 minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(1)),
		},
		"positive minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(100)),
		},
		"negative minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				// We only need to test both gate enabled and disabled together, because
				// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
				// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)

				errs := Strategy.Validate(ctx, &tc.input)
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
			}
			// The equivalenceMatcher is used to verify the output errors from hand-written imperative validation
			// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.input, nil)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.ReplicationController
		update       api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"baseline": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(),
		},
		// spec.replicas
		"nil replicas": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"0 replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(0)),
		},
		"1 replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(1)),
		},
		"positive replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(100)),
		},
		"negative replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"0 minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"1 minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(1)),
		},
		"positive minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(3)),
		},
		"negative minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			var declarativeTakeoverErrs field.ErrorList
			var imperativeErrs field.ErrorList
			for _, gateVal := range []bool{true, false} {
				// We only need to test both gate enabled and disabled together, because
				// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
				// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, gateVal)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidationTakeover, gateVal)
				errs := Strategy.ValidateUpdate(ctx, &tc.update, &tc.old)
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
			}
			// The equivalenceMatcher is used to verify the output errors from hand-written imperative validation
			// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
			equivalenceMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			// TODO: remove this once RC's validation is fixed to not return duplicate errors.
			dedupedImperativeErrs := field.ErrorList{}
			for _, err := range imperativeErrs {
				found := false
				for _, existingErr := range dedupedImperativeErrs {
					if equivalenceMatcher.Matches(existingErr, err) {
						found = true
						break
					}
				}
				if !found {
					dedupedImperativeErrs = append(dedupedImperativeErrs, err)
				}
			}
			equivalenceMatcher.Test(t, dedupedImperativeErrs, declarativeTakeoverErrs)

			apitesting.VerifyVersionedValidationEquivalence(t, &tc.update, &tc.old)
		})
	}
}

// mkValidReplicationController produces a ReplicationController which passes
// validation with no tweaks.
func mkValidReplicationController(tweaks ...func(rc *api.ReplicationController)) api.ReplicationController {
	rc := api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: api.ReplicationControllerSpec{
			Replicas: ptr.To[int32](1),
			Selector: map[string]string{"a": "b"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func setSpecReplicas(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.Replicas = ptr.To(val)
	}
}

func setSpecMinReadySeconds(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.MinReadySeconds = val
	}
}
