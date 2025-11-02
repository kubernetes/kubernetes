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
		"valid": {
			input: mkValidClusterRoleBinding(),
		},
		"missing roleRef.name": {
			input: mkValidClusterRoleBinding(tweakRoleRef(rbac.RoleRef{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "ClusterRole",
			})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("roleRef", "name"), "").MarkAlpha(),
			},
		},
		"missing subjects[0].name": {
			input: mkValidClusterRoleBinding(tweakSubjectName(0, "")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("subjects").Index(0).Child("name"), "").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy, tc.expectedErrs)
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
	validCRB := mkValidClusterRoleBinding()
	testCases := map[string]struct {
		old          rbac.ClusterRoleBinding
		update       rbac.ClusterRoleBinding
		expectedErrs field.ErrorList
	}{
		"no change to roleRef - valid": {
			old:    validCRB,
			update: validCRB,
		},
		"roleRef changed - invalid": {
			old:    validCRB,
			update: mkValidClusterRoleBinding(tweakRoleRefName("new-role")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole", Name: "new-role"}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"roleRef set from unset - invalid": {
			old:    mkValidClusterRoleBinding(tweakRoleRef(rbac.RoleRef{})),
			update: validCRB,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"roleRef unset from set - invalid": {
			old:    validCRB,
			update: mkValidClusterRoleBinding(tweakRoleRef(rbac.RoleRef{})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
				field.Required(field.NewPath("roleRef", "name"), "").MarkShortCircuitedInDV(),
				field.NotSupported(field.NewPath("roleRef", "kind"), "", []string{}).MarkFromImperative(),
				field.NotSupported(field.NewPath("roleRef", "apiGroup"), "", []string{}).MarkFromImperative(),
			},
		},
		"invalid update clearing subjects[0].name": {
			old:    validCRB,
			update: mkValidClusterRoleBinding(tweakSubjectName(0, "")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("subjects").Index(0).Child("name"), "").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy, tc.expectedErrs)
		})
	}
}

func mkValidClusterRoleBinding(tweaks ...func(*rbac.ClusterRoleBinding)) rbac.ClusterRoleBinding {
	crb := rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-cluster-role-binding",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "valid-cluster-role",
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				APIGroup:  "",
				Name:      "valid-service-account",
				Namespace: "valid-namespace",
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&crb)
	}
	return crb
}

func tweakRoleRef(roleRef rbac.RoleRef) func(*rbac.ClusterRoleBinding) {
	return func(crb *rbac.ClusterRoleBinding) {
		crb.RoleRef = roleRef
	}
}

func tweakRoleRefName(name string) func(*rbac.ClusterRoleBinding) {
	return func(crb *rbac.ClusterRoleBinding) {
		crb.RoleRef.Name = name
	}
}

func tweakSubjectName(index int, name string) func(*rbac.ClusterRoleBinding) {
	return func(crb *rbac.ClusterRoleBinding) {
		if index < len(crb.Subjects) {
			crb.Subjects[index].Name = name
		}
	}
}
