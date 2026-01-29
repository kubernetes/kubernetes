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
			testDeclarativeValidateForDeclarative(t, v)
		})
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "rbac.authorization.k8s.io",
			APIVersion: apiVersion,
		},
	)

	testCases := map[string]struct {
		input        rbac.Role
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidRole(),
		},
		"invalid Role missing verbs": {
			input: mkValidRole(tweakVerbs(nil)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), ""),
			},
		},
		"invalid Role empty verbs": {
			input: mkValidRole(tweakVerbs([]string{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), ""),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testValidateUpdateForDeclarative(t, v)
		})
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "rbac.authorization.k8s.io",
			APIVersion: apiVersion,
		},
	)

	testCases := map[string]struct {
		old, update  rbac.Role
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkValidRole(),
			update: mkValidRole(),
		},
		"invalid update clearing verbs": {
			old:    mkValidRole(),
			update: mkValidRole(tweakVerbs([]string{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), ""),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidRole(tweaks ...func(*rbac.Role)) rbac.Role {
	r := rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-role",
			Namespace: "default",
		},
		Rules: []rbac.PolicyRule{{
			APIGroups: []string{"rbac.authorization.k8s.io"},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list"},
		}},
	}

	for _, tweak := range tweaks {
		tweak(&r)
	}
	return r
}

func tweakVerbs(verbs []string) func(*rbac.Role) {
	return func(r *rbac.Role) {
		if len(r.Rules) > 0 {
			r.Rules[0].Verbs = verbs
		}
	}
}
