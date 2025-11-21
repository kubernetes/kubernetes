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

func TestDeclarativeValidateParametersName(t *testing.T) {
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
						obj.Spec.Parameters.Name = ""
					}),
					expectedErrs: field.ErrorList{
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

func TestDeclarativeValidateParametersKind(t *testing.T) {
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
				"missing parameters.kind": {
					input: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.Spec.Parameters.Kind = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(
							field.NewPath("spec", "parameters", "kind"),
							"kind is required",
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

func TestDeclarativeValidateUpdateParametersKind(t *testing.T) {
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
				"invalid update clears parameters.kind": {
					oldObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidIngressClass(func(obj *networking.IngressClass) {
						obj.ResourceVersion = "1"
						obj.Spec.Parameters.Kind = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(
							field.NewPath("spec", "parameters", "kind"),
							"kind is required",
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
				Namespace: nil,
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
