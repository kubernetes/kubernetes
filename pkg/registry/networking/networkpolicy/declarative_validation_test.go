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

package networkpolicy

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	networking "k8s.io/kubernetes/pkg/apis/networking"
)

var apiVersions = []string{"v1"}

func TestDeclarativeValidateIPBlockCIDR(t *testing.T) {

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:          "networking.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "networkpolicies",
					IsResourceRequest: true,
					Verb:              "create",
				},
			)

			testCases := map[string]struct {
				input        networking.NetworkPolicy
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkValidNetworkPolicy(),
				},
				"missing ipBlock.cidr": {
					input: mkValidNetworkPolicy(func(obj *networking.NetworkPolicy) {
						obj.Spec.Ingress[0].From[0].IPBlock.CIDR = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(
							field.NewPath("spec", "ingress").Index(0).Child("from").Index(0).Child("ipBlock", "cidr"),
							"",
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

func TestDeclarativeValidateIPBlockCIDRUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testCases := map[string]struct {
				oldObj       networking.NetworkPolicy
				updateObj    networking.NetworkPolicy
				expectedErrs field.ErrorList
			}{
				"invalid update clears ipBlock.cidr": {
					oldObj: mkValidNetworkPolicy(func(obj *networking.NetworkPolicy) {
						obj.ResourceVersion = "1"
					}),
					updateObj: mkValidNetworkPolicy(func(obj *networking.NetworkPolicy) {
						obj.ResourceVersion = "1"
						obj.Spec.Ingress[0].From[0].IPBlock.CIDR = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(
							field.NewPath("spec", "ingress").Index(0).Child("from").Index(0).Child("ipBlock", "cidr"),
							"",
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
							Resource:          "networkpolicies",
							Name:              "valid-network-policy",
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

func mkValidNetworkPolicy(tweaks ...func(obj *networking.NetworkPolicy)) networking.NetworkPolicy {
	obj := networking.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-network-policy",
			Namespace: "default",
		},
		Spec: networking.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			Ingress: []networking.NetworkPolicyIngressRule{
				{
					From: []networking.NetworkPolicyPeer{
						{
							IPBlock: &networking.IPBlock{
								CIDR: "192.168.1.0/24",
							},
						},
					},
				},
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}