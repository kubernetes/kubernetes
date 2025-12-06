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

package clusterrolebinding

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
		input        rbac.ClusterRoleBinding
		expectedErrs field.ErrorList
	}{
		"valid: subject name is present": {
			input:        newClusterRoleBindingDeclarative("crb-valid-subject-name", "role-a", "ClusterRole"),
			expectedErrs: nil,
		},
		"invalid: subject name is empty string (required)": {
			input: func() rbac.ClusterRoleBinding {
				crb := newClusterRoleBindingDeclarative("crb-missing-name", "role-a", "ClusterRole")
				crb.Subjects[0].Name = ""
				return crb
			}(),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("subjects").Index(0).Child("name"), "name is required for Subject"),
			},
		},
		"invalid: multiple subjects, one name missing (required)": {
			input: func() rbac.ClusterRoleBinding {
				crb := newClusterRoleBindingDeclarative("crb-two-subjects-one-missing", "role-a", "ClusterRole")
				crb.Subjects = append(crb.Subjects, rbac.Subject{
					Kind:     "User",
					Name:     "",
					APIGroup: rbac.GroupName,
				})
				return crb
			}(),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("subjects").Index(1).Child("name"), "name is required for Subject"),
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
		old          rbac.ClusterRoleBinding
		update       rbac.ClusterRoleBinding
		expectedErrs field.ErrorList
	}{
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// Helper to create a base ClusterRoleBinding for declarative tests
func newClusterRoleBindingDeclarative(name, roleName, roleKind string) rbac.ClusterRoleBinding {
	return rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: "1",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     roleKind,
			Name:     roleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind:     "User",
				Name:     "test-user",
				APIGroup: rbac.GroupName,
			},
		},
	}
}
