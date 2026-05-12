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

package rolebinding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	registry "k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func mkValidRoleBinding(tweaks ...func(rb *rbac.RoleBinding)) rbac.RoleBinding {
	rb := rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "test-binding", Namespace: "test-ns"},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
			Name:     "admin",
		},
		Subjects: []rbac.Subject{
			{Kind: rbac.UserKind, APIGroup: rbac.GroupName, Name: "user1"},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rb)
	}
	return rb
}

func tweakRoleRefName(name string) func(rb *rbac.RoleBinding) {
	return func(rb *rbac.RoleBinding) {
		rb.RoleRef.Name = name
	}
}

func tweakSubjectName(index int, name string) func(rb *rbac.RoleBinding) {
	return func(rb *rbac.RoleBinding) {
		rb.Subjects[index].Name = name
	}
}

func tweakRoleRef(roleRef rbac.RoleRef) func(rb *rbac.RoleBinding) {
	return func(rb *rbac.RoleBinding) {
		rb.RoleRef = roleRef
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "rbac.authorization.k8s.io",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		input        rbac.RoleBinding
		expectedErrs field.ErrorList
	}{
		"valid binding": {
			input:        mkValidRoleBinding(),
			expectedErrs: field.ErrorList{},
		},
		"missing roleRef.name": {
			input: mkValidRoleBinding(tweakRoleRef(rbac.RoleRef{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "ClusterRole",
			})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("roleRef", "name"), "name is required").MarkAlpha(),
			},
		},

		"missing subject.name": {
			input: mkValidRoleBinding(tweakSubjectName(0, "")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("subjects").Index(0).Child("name"), "name is required").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
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
		oldObj       rbac.RoleBinding
		updateObj    rbac.RoleBinding
		expectedErrs field.ErrorList
	}{
		"no change to roleRef - valid": {
			oldObj:    mkValidRoleBinding(),
			updateObj: mkValidRoleBinding(),
		},
		"roleRef changed - invalid": {
			oldObj:    mkValidRoleBinding(),
			updateObj: mkValidRoleBinding(tweakRoleRefName("different-role")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{
					APIGroup: "rbac.authorization.k8s.io",
					Kind:     "Role",
					Name:     "different-role",
				}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"roleRef set from unset - invalid": {
			oldObj:    mkValidRoleBinding(tweakRoleRef(rbac.RoleRef{})),
			updateObj: mkValidRoleBinding(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"roleRef unset from set - invalid": {
			updateObj: mkValidRoleBinding(tweakRoleRef(rbac.RoleRef{})),
			oldObj:    mkValidRoleBinding(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("roleRef"), rbac.RoleRef{}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
				field.Required(field.NewPath("roleRef", "name"), "name is required").MarkShortCircuitedInDV(),
				field.NotSupported(field.NewPath("roleRef", "kind"), "", []string{}).MarkFromImperative(),
				field.NotSupported(field.NewPath("roleRef", "apiGroup"), "", []string{}).MarkFromImperative(),
			},
		},
		// TODO: Add more test cases
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "rbac.authorization.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "rolebindings",
				Name:              "test-binding",
				IsResourceRequest: true,
				Verb:              "update",
			})
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}
}
