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
		"valid": {
			input: mkValidClusterRole(),
		},
		"invalid ClusterRole missing verbs": {
			input: mkValidClusterRole(tweakVerbs(0, nil)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("verbs"), ""),
			},
		},
		"invalid ClusterRole missing resources": {
			input: mkValidClusterRole(tweakResources(0, nil)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("resources"), ""),
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
		old          rbac.ClusterRole
		update       rbac.ClusterRole
		expectedErrs field.ErrorList
	}{
		"valid update adding resources": {
			old:    mkValidClusterRole(tweakResourceVersion("1")),
			update: mkValidClusterRole(tweakResourceVersion("2"), tweakAddResource(0, "services")),
		},
		"invalid update clearing resources": {
			old:    mkValidClusterRole(tweakResourceVersion("1")),
			update: mkValidClusterRole(tweakResourceVersion("2"), tweakResources(0, []string{})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("rules").Index(0).Child("resources"), ""),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidClusterRole(tweaks ...func(*rbac.ClusterRole)) rbac.ClusterRole {
	cr := rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-cluster-role",
		},
		Rules: []rbac.PolicyRule{{
			APIGroups: []string{"rbac.authorization.k8s.io"},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list"},
		}},
	}

	for _, tweak := range tweaks {
		tweak(&cr)
	}
	return cr
}

func tweakResourceVersion(rv string) func(*rbac.ClusterRole) {
	return func(cr *rbac.ClusterRole) {
		cr.ResourceVersion = rv
	}
}

func tweakVerbs(ruleIndex int, verbs []string) func(*rbac.ClusterRole) {
	return func(cr *rbac.ClusterRole) {
		if len(cr.Rules) > ruleIndex {
			cr.Rules[ruleIndex].Verbs = verbs
		}
	}
}

func tweakResources(ruleIndex int, resources []string) func(*rbac.ClusterRole) {
	return func(cr *rbac.ClusterRole) {
		if len(cr.Rules) > ruleIndex {
			cr.Rules[ruleIndex].Resources = resources
		}
	}
}

func tweakAddResource(ruleIndex int, resource string) func(*rbac.ClusterRole) {
	return func(cr *rbac.ClusterRole) {
		if len(cr.Rules) > ruleIndex {
			cr.Rules[ruleIndex].Resources = append(cr.Rules[ruleIndex].Resources, resource)
		}
	}
}
