/*
Copyright The Kubernetes Authors.

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

package mutatingadmissionpolicy

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	registry "k8s.io/kubernetes/pkg/registry/admissionregistration/mutatingadmissionpolicy"
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
		APIGroup:          "admissionregistration.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "mutatingadmissionpolicy",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		setFeatures  featuregatetesting.FeatureOverrides
		minVersion   *version.Version
		input        admissionregistration.MutatingAdmissionPolicy
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPolicy(),
		},
		"spec.variables too many items": {
			setFeatures: featuregatetesting.FeatureOverrides{},
			minVersion:  version.MustParseMajorMinor("1.37"),
			input:       mkValidPolicy(add11Variables),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "variables"), 11, 10).WithOrigin("maxItems"),
			},
		},
	}

	strategy := registry.NewStrategy(nil, nil)
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.setFeatures)
			var opts []apitesting.ValidationTestConfig
			if tc.minVersion != nil {
				opts = append(opts, apitesting.WithMinEmulationVersion(tc.minVersion))
			}
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy, tc.expectedErrs, opts...)
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
		setFeatures  featuregatetesting.FeatureOverrides
		minVersion   *version.Version
		oldObj       admissionregistration.MutatingAdmissionPolicy
		updateObj    admissionregistration.MutatingAdmissionPolicy
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPolicy(),
			updateObj: mkValidPolicy(),
		},
		"update with too many variables": {
			setFeatures: featuregatetesting.FeatureOverrides{},
			minVersion:  version.MustParseMajorMinor("1.37"),
			oldObj:      mkValidPolicy(),
			updateObj:   mkValidPolicy(add11Variables),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "variables"), 11, 10).WithOrigin("maxItems"),
			},
		},
	}

	strategy := registry.NewStrategy(nil, nil)

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.setFeatures)
			var opts []apitesting.ValidationTestConfig
			if tc.minVersion != nil {
				opts = append(opts, apitesting.WithMinEmulationVersion(tc.minVersion))
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "mutatingadmissionpolicy",
				Name:              tc.oldObj.Name,
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy, tc.expectedErrs, opts...)
		})
	}
}

func mkValidPolicy(tweaks ...func(obj *admissionregistration.MutatingAdmissionPolicy)) admissionregistration.MutatingAdmissionPolicy {
	obj := admissionregistration.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-policy",
		},
		Spec: admissionregistration.MutatingAdmissionPolicySpec{
			MatchConstraints: &admissionregistration.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				ResourceRules: []admissionregistration.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistration.RuleWithOperations{
						Operations: []admissionregistration.OperationType{admissionregistration.Create},
						Rule: admissionregistration.Rule{
							APIGroups:   []string{""},
							APIVersions: []string{"v1"},
							Resources:   []string{"configmaps"},
						},
					},
				}},
				MatchPolicy: new(admissionregistration.Equivalent),
			},
			FailurePolicy:      new(admissionregistration.Fail),
			ReinvocationPolicy: admissionregistration.IfNeededReinvocationPolicy,
			Mutations: []admissionregistration.Mutation{{
				PatchType:          admissionregistration.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistration.ApplyConfiguration{Expression: `Object{}`},
			}},
		},
	}
	obj.ResourceVersion = "1"
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func add11Variables(obj *admissionregistration.MutatingAdmissionPolicy) {
	obj.Spec.Variables = append(
		obj.Spec.Variables,
		admissionregistration.Variable{Name: "v1", Expression: "true"},
		admissionregistration.Variable{Name: "v2", Expression: "true"},
		admissionregistration.Variable{Name: "v3", Expression: "true"},
		admissionregistration.Variable{Name: "v4", Expression: "true"},
		admissionregistration.Variable{Name: "v5", Expression: "true"},
		admissionregistration.Variable{Name: "v6", Expression: "true"},
		admissionregistration.Variable{Name: "v7", Expression: "true"},
		admissionregistration.Variable{Name: "v8", Expression: "true"},
		admissionregistration.Variable{Name: "v9", Expression: "true"},
		admissionregistration.Variable{Name: "v10", Expression: "true"},
		admissionregistration.Variable{Name: "v11", Expression: "true"},
	)
}
