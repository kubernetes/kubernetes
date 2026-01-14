/*
Copyright 2026 The Kubernetes Authors.

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

package ipaddress

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	networking "k8s.io/kubernetes/pkg/apis/networking"
)

var apiVersions = []string{"v1alpha1", "v1beta1", "v1"}

func TestDeclarativeValidateIPAddress(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:          "networking.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "ipaddresses",
					IsResourceRequest: true,
					Verb:              "create",
				},
			)

			testCases := map[string]struct {
				input        networking.IPAddress
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkValidIPAddress(),
				},
				"missing parentRef": {
					input: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.Spec.ParentRef = nil
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "parentRef"), ""),
					},
				},
				"missing parentRef resource": {
					input: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.Spec.ParentRef.Resource = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "parentRef", "resource"), ""),
					},
				},
				"missing parentRef name": {
					input: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.Spec.ParentRef.Name = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "parentRef", "name"), ""),
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

func TestDeclarativeValidateIPAddressUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testCases := map[string]struct {
				oldObj       networking.IPAddress
				updateObj    networking.IPAddress
				expectedErrs field.ErrorList
			}{
				"valid update": {
					oldObj: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.ResourceVersion = "1"
					}),
				},
				"immutable parentRef": {
					oldObj: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidIPAddress(func(obj *networking.IPAddress) {
						obj.ResourceVersion = "1"
						obj.Spec.ParentRef.Name = "new-name"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "parentRef"), &networking.ParentReference{
							Group:    "apps",
							Resource: "deployments",
							Name:     "new-name",
						}, "field is immutable"),
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
							Resource:          "ipaddresses",
							Name:              "192.168.1.1",
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

func mkValidIPAddress(tweaks ...func(obj *networking.IPAddress)) networking.IPAddress {
	obj := networking.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: "192.168.1.1",
		},
		Spec: networking.IPAddressSpec{
			ParentRef: &networking.ParentReference{
				Group:    "apps",
				Resource: "deployments",
				Name:     "my-deployment",
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
