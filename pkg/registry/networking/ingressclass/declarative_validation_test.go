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

/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package ingressclass

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	netapi "k8s.io/kubernetes/pkg/apis/networking"
)

func TestDeclarativeValidate_IngressClass_Name(t *testing.T) {
	// IngressClass v1beta1 is no longer served (v1.22+), so we only test v1.
	apiVersions := []string{"v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIPrefix:         "apis",
					APIGroup:          "networking.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "ingressclasses",
					IsResourceRequest: true,
					Verb:              "create",
				},
			)

			long63 := strings.Repeat("a", 63)
			tooLong64 := strings.Repeat("a", 64)

			testCases := map[string]struct {
				input        netapi.IngressClass
				expectedErrs field.ErrorList
			}{
				"valid name": {
					input: mkValidIngressClass(),
				},
				"empty name (required)": {
					input: mkValidIngressClass(func(obj *netapi.IngressClass) {
						obj.Name = ""
					}),
					expectedErrs: field.ErrorList{
						// generic object meta: name or generateName required
						field.Required(field.NewPath("metadata", "name"), ""),
					},
				},
				"uppercase not allowed": {
					input: mkValidIngressClass(func(obj *netapi.IngressClass) {
						obj.Name = "Invalid-Upper"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("metadata", "name"), "Invalid-Upper", ""),
					},
				},
				"must start with a letter (dash first)": {
					input: mkValidIngressClass(func(obj *netapi.IngressClass) {
						obj.Name = "-bad"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("metadata", "name"), "-bad", ""),
					},
				},
				"max length 63 (exact 63 ok)": {
					input: mkValidIngressClass(func(obj *netapi.IngressClass) {
						obj.Name = long63
					}),
				},
				"too long (>63)": {
					input: mkValidIngressClass(func(obj *netapi.IngressClass) {
						obj.Name = tooLong64
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("metadata", "name"), tooLong64, ""),
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

func TestDeclarativeValidateUpdate_IngressClass_Name(t *testing.T) {
	// Name is immutable; we verify update parity while keeping name the same.
	apiVersions := []string{"v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			oldObj := mkValidIngressClass(func(obj *netapi.IngressClass) {
				obj.ResourceVersion = "1"
			})
			updateObj := mkValidIngressClass(func(obj *netapi.IngressClass) {
				obj.ResourceVersion = "1"
			})

			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIPrefix:         "apis",
					APIGroup:          "networking.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "ingressclasses",
					Name:              oldObj.Name,
					IsResourceRequest: true,
					Verb:              "update",
				},
			)

			apitesting.VerifyUpdateValidationEquivalence(
				t,
				ctx,
				&updateObj,
				&oldObj,
				Strategy.ValidateUpdate,
				nil, // no additional expected errors
			)
		})
	}
}

// mkValidIngressClass returns a semantically valid IngressClass and then applies any tweaks.
func mkValidIngressClass(tweaks ...func(obj *netapi.IngressClass)) netapi.IngressClass {
	obj := netapi.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-ingress-class",
		},
		Spec: netapi.IngressClassSpec{
			// Controller must be a qualified name; use a common one.
			Controller: "k8s.io/ingress-nginx",
			// Parameters intentionally omitted for base "valid" object.
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
