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

package clusterrole

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
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "rbac.authorization.k8s.io",
		APIVersion: apiVersion,
	})
	testCases := map[string]struct {
		input        rbac.ClusterRole
		expectedErrs field.ErrorList
	}{
		"missing verbs": {
			input: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "test-role"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
					},
				},
			},
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), "must have at least one verb"),
			},
		},
		"valid rule": {
			input: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-role"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
						Verbs:     []string{"get", "list"},
					},
				},
			},
			expectedErrs: field.ErrorList{},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "rbac.authorization.k8s.io",
		APIVersion: apiVersion,
	})
	testCases := map[string]struct {
		old          rbac.ClusterRole
		update       rbac.ClusterRole
		expectedErrs field.ErrorList
	}{
		"update missing verbs": {
			old: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "role"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
						Verbs:     []string{"get"},
					},
				},
			},
			update: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "role", ResourceVersion: "1"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
					},
				},
			},
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), "must have at least one verb"),
			},
		},
		"valid update": {
			old: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "role"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
						Verbs:     []string{"get"},
					},
				},
			},
			update: rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{Name: "role", ResourceVersion: "1"},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods"},
						Verbs:     []string{"get", "list"},
					},
				},
			},
			expectedErrs: field.ErrorList{},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}
