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

package role

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
)

var apiVersions = []string{"v1", "v1alpha1", "v1beta1"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "rbac.authorization.k8s.io",
					APIVersion: v,
				},
			)

			testCases := map[string]struct {
				input        rbac.Role
				expectedErrs field.ErrorList
			}{
				"valid: minimal resource rule": {
					input: newRoleDeclarative("ns", "role-valid",
						[]rbac.PolicyRule{{
							Verbs:     []string{"get"},
							APIGroups: []string{""},
							Resources: []string{"pods"},
						}},
					),
				},
				"invalid: verbs empty (required)": {
					input: newRoleDeclarative("ns", "role-no-verbs",
						[]rbac.PolicyRule{{
							Verbs:     []string{}, 
							APIGroups: []string{""},
							Resources: []string{"pods"},
						}},
					),
					expectedErrs: field.ErrorList{
						field.Required(
							field.NewPath("rules").Index(0).Child("verbs"),
							"verbs must contain at least one value",
						),
					},
				},
				"invalid: namespaced rule has nonResourceURLs": {
					input: newRoleDeclarative("ns", "role-nonresource",
						[]rbac.PolicyRule{{
							Verbs:           []string{"get"},
							NonResourceURLs: []string{"/healthz"},
						}},
					),
					expectedErrs: field.ErrorList{
						field.Invalid(
							field.NewPath("rules").Index(0).Child("nonResourceURLs"),
							[]string{"/healthz"},
							"namespaced rules cannot apply to non-resource URLs",
						),
					},
				},
				// TODO: Add more test cases
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

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "rbac.authorization.k8s.io",
					APIVersion: v,
				},
			)

			testCases := map[string]struct {
				old, update  rbac.Role
				expectedErrs field.ErrorList
			}{
				// TODO: Add more test cases
			}

			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					apitesting.VerifyUpdateValidationEquivalence(
						t,
						ctx,
						&tc.update,
						&tc.old,
						Strategy.ValidateUpdate,
						tc.expectedErrs,
					)
				})
			}
		})
	}
}

// Helper to create a base Role for declarative tests.
func newRoleDeclarative(ns, name string, rules []rbac.PolicyRule) rbac.Role {
	return rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       ns,
			Name:            name,
			ResourceVersion: "1",
		},
		Rules: rules,
	}
}
