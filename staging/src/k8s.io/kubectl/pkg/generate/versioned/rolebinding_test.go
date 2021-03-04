/*
Copyright 2017 The Kubernetes Authors.

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

package versioned

import (
	"reflect"
	"testing"

	rbac "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRoleBindingGenerate(t *testing.T) {
	tests := []struct {
		name          string
		params        map[string]interface{}
		expectErrMsg  string
		expectBinding *rbac.RoleBinding
	}{
		{
			name: "test-missing-name",
			params: map[string]interface{}{
				"role":           "fake-role",
				"groups":         []string{"fake-group"},
				"serviceaccount": []string{"fake-namespace:fake-account"},
			},
			expectErrMsg: "Parameter: name is required",
		},
		{
			name: "test-missing-role-and-clusterrole",
			params: map[string]interface{}{
				"name":           "fake-binding",
				"group":          []string{"fake-group"},
				"serviceaccount": []string{"fake-namespace:fake-account"},
			},
			expectErrMsg: "exactly one of clusterrole or role must be specified",
		},
		{
			name: "test-both-role-and-clusterrole-provided",
			params: map[string]interface{}{
				"name":           "fake-binding",
				"role":           "fake-role",
				"clusterrole":    "fake-clusterrole",
				"group":          []string{"fake-group"},
				"serviceaccount": []string{"fake-namespace:fake-account"},
			},
			expectErrMsg: "exactly one of clusterrole or role must be specified",
		},
		{
			name: "test-invalid-parameter-type",
			params: map[string]interface{}{
				"name":           "fake-binding",
				"role":           []string{"fake-role"},
				"group":          []string{"fake-group"},
				"serviceaccount": []string{"fake-namespace:fake-account"},
			},
			expectErrMsg: "expected string, saw [fake-role] for 'role'",
		},
		{
			name: "test-invalid-serviceaccount",
			params: map[string]interface{}{
				"name":           "fake-binding",
				"role":           "fake-role",
				"group":          []string{"fake-group"},
				"serviceaccount": []string{"fake-account"},
			},
			expectErrMsg: "serviceaccount must be <namespace>:<name>",
		},
		{
			name: "test-valid-case",
			params: map[string]interface{}{
				"name":           "fake-binding",
				"role":           "fake-role",
				"user":           []string{"fake-user"},
				"group":          []string{"fake-group"},
				"serviceaccount": []string{"fake-namespace:fake-account"},
			},
			expectBinding: &rbac.RoleBinding{
				ObjectMeta: v1.ObjectMeta{
					Name: "fake-binding",
				},
				RoleRef: rbac.RoleRef{
					APIGroup: rbac.GroupName,
					Kind:     "Role",
					Name:     "fake-role",
				},
				Subjects: []rbac.Subject{
					{
						Kind:     rbac.UserKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-user",
					},
					{
						Kind:     rbac.GroupKind,
						APIGroup: "rbac.authorization.k8s.io",
						Name:     "fake-group",
					},
					{
						Kind:      rbac.ServiceAccountKind,
						Namespace: "fake-namespace",
						Name:      "fake-account",
					},
				},
			},
		},
	}

	generator := RoleBindingGeneratorV1{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			switch {
			case tt.expectErrMsg != "" && err != nil:
				if err.Error() != tt.expectErrMsg {
					t.Errorf("test '%s': expect error '%s', but saw '%s'", tt.name, tt.expectErrMsg, err.Error())
				}
				return
			case tt.expectErrMsg != "" && err == nil:
				t.Errorf("test '%s': expected error '%s' and didn't get one", tt.name, tt.expectErrMsg)
				return
			case tt.expectErrMsg == "" && err != nil:
				t.Errorf("test '%s': unexpected error %s", tt.name, err.Error())
				return
			}
			if !reflect.DeepEqual(obj.(*rbac.RoleBinding), tt.expectBinding) {
				t.Errorf("test '%s': expected:\n%#v\nsaw:\n%#v", tt.name, tt.expectBinding, obj.(*rbac.RoleBinding))
			}
		})
	}
}
