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

package ingressclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	networking "k8s.io/kubernetes/pkg/apis/networking"
)

// TestDeclarativeValidateParametersName verifies that the declarative
// validation generated from the `+required` / `+k8s:required` markers on
// IngressClassParametersReference.Name matches the existing handwritten
// validation logic in Strategy.Validate.
func TestDeclarativeValidateParametersName(t *testing.T) {
	// IngressClass exists in networking.k8s.io/v1 and (historically) v1beta1,
	// so exercise both served versions like we do in RuntimeClass tests.
	apiVersions := []string{"v1", "v1beta1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:          "networking.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "ingressclasses",
					IsResourceRequest: true,
					Verb:              "create",
				},
			)

			testCases := map[string]struct {
				input        networking.IngressClass
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkValidIngressClass(),
				},
				"missing parameters.name": {
					input: mkValidIngressClass(func(obj *networking.IngressClass) {
						// Simulate bad user input: required field left empty.
						obj.Spec.Parameters.Name = ""
					}),
					expectedErrs: field.ErrorList{
						// Handwritten validation (validateIngressClassParametersReference /
						// validateIngressTypedLocalObjectReference) treats an empty name
						// as "Required", with message "name is required".
						field.Required(
							field.NewPath("spec", "parameters", "name"),
							"name is required",
						),
					},
				},
			}

			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(
						t,
						ctx,
						&tc.input,
						Strategy.Validate,
						tc.expectedErrs,
					)
				})
			}
		})
	}
}

// TestDeclarativeValidateUpdateParametersName does the same check for update
// calls, making sure both the handwritten Update validation and the
// declarative validation surface the same required-field error when
// spec.parameters.name is cleared.
func TestDeclarativeValidateUpdateParametersName(t *testing.T) {
	apiVersions := []string{"v1", "v1beta1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testCases := map[string]struct {
				oldObj       networking.IngressClass
				updateObj    networking.IngressClass
				expectedErrs field.ErrorList
			}{
				"valid update": {
					oldObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
					}),
				},
				"invalid update clears parameters.name": {
					oldObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
						obj.Spec.Parameters.Name = ""
					}),
					expectedErrs: field.ErrorList{
						// Still required on update. Unlike RuntimeClass.Handler,
						// Parameters.Name is not immutable, so we only expect the
						// "Required" error and NOT an additional "Forbidden" error.
						field.Required(
							field.NewPath("spec", "parameters", "name"),
							"name is required",
						),
					},
				},
			}

			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					ctx := genericapirequest.WithRequestInfo(
						genericapirequest.NewDefaultContext(),
						&genericapirequest.RequestInfo{
							APIPrefix:         "apis",
							APIGroup:          "networking.k8s.io",
							APIVersion:        apiVersion,
							Resource:          "ingressclasses",
							Name:              "valid-ingress-class",
							IsResourceRequest: true,
							Verb:              "update",
						},
					)

					apitesting.VerifyUpdateValidationEquivalence(
						t,
						ctx,
						&tc.updateObj,
						&tc.oldObj,
						Strategy.ValidateUpdate,
						tc.expectedErrs,
					)
				})
			}
		})
	}
}

// mkValidIngressClass returns a semantically valid IngressClass, including a
// valid spec.parameters block, then applies any tweaks.
//
// The goal is to satisfy handwritten validation in
// validateIngressClassParametersReference:
//   - parameters.kind must be non-empty
//   - parameters.name must be non-empty
//   - parameters.scope must be set to a supported value ("Cluster"/"Namespace")
//   - parameters.namespace must be unset when scope == "Cluster"
//
// We choose scope == "Cluster" so namespace must remain unset.
func mkValidIngressClass(tweaks ...func(obj *networking.IngressClass)) networking.IngressClass {
	apiGroup := "example.com"
	scope := networking.IngressClassParametersReferenceScopeCluster

	obj := networking.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-ingress-class",
		},
		Spec: networking.IngressClassSpec{
			Controller: "example.com/ingress-controller",
			Parameters: &networking.IngressClassParametersReference{
				APIGroup:  &apiGroup,
				Kind:      "IngressParameters",
				Name:      "valid-params",
				Scope:     &scope,
				Namespace: nil, // must be nil when Scope == "Cluster"
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
