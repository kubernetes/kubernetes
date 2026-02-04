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

package validatingadmissionpolicybinding

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"testing"
)

var apiVersions = []string{"v1beta1", "v1", "v1alpha1"}

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
		Resource:          "validatingadmissionpolicybindings",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        admissionregistration.ValidatingAdmissionPolicyBinding
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidBinding(),
		},
		"spec.policyName is required": {
			input: mkValidBinding(tweakPolicyName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "policyName"), ""),
			},
		},
		"spec.validationActions is required": {
			input: mkValidBinding(tweakValidateActions([]admissionregistration.ValidationAction{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "validationActions"), ""),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
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
		oldObj       admissionregistration.ValidatingAdmissionPolicyBinding
		updateObj    admissionregistration.ValidatingAdmissionPolicyBinding
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidBinding(),
			updateObj: mkValidBinding(),
		},
		"update with empty spec.policyName": {
			oldObj:    mkValidBinding(func(obj *admissionregistration.ValidatingAdmissionPolicyBinding) { obj.ResourceVersion = "1" }),
			updateObj: mkValidBinding(tweakPolicyName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "policyName"), ""),
			},
		},
		"update with empty spec.validationActions": {
			oldObj:    mkValidBinding(),
			updateObj: mkValidBinding(tweakValidateActions([]admissionregistration.ValidationAction{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "validationActions"), ""),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "validatingadmissionpolicybindings",
				Name:              "valid-binding",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidBinding(tweaks ...func(obj *admissionregistration.ValidatingAdmissionPolicyBinding)) admissionregistration.ValidatingAdmissionPolicyBinding {
	obj := admissionregistration.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-binding",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        "test-policy",
			ValidationActions: []admissionregistration.ValidationAction{admissionregistration.Deny},
		},
	}
	obj.ResourceVersion = "1"
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func tweakPolicyName(policyName string) func(obj *admissionregistration.ValidatingAdmissionPolicyBinding) {
	return func(obj *admissionregistration.ValidatingAdmissionPolicyBinding) {
		obj.Spec.PolicyName = policyName
	}
}

func tweakValidateActions(actions []admissionregistration.ValidationAction) func(*admissionregistration.ValidatingAdmissionPolicyBinding) {
	return func(obj *admissionregistration.ValidatingAdmissionPolicyBinding) {
		obj.Spec.ValidationActions = actions
	}
}
