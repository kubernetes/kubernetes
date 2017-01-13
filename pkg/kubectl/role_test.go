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

package kubectl

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

func TestRoleGenerate(t *testing.T) {
	tests := map[string]struct {
		params    map[string]interface{}
		expected  *rbac.Role
		expectErr bool
	}{
		"test-missing-verb": {
			params: map[string]interface{}{
				"name": "foo",
			},
			expectErr: true,
		},
		"test-missing-resource": {
			params: map[string]interface{}{
				"name": "foo",
				"verb": []string{"get"},
			},
			expectErr: true,
		},
		"test-missing-api-group": {
			params: map[string]interface{}{
				"name":     "foo",
				"verb":     []string{"get"},
				"resource": []string{"pods"},
			},
			expected: &rbac.Role{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						Resources: []string{"pods"},
						APIGroups: []string{""},
					},
				},
			},
			expectErr: false,
		},
		"test-invalid-params": {
			params: map[string]interface{}{
				"name":     "foo",
				"verb":     "get",
				"resource": []string{"pods"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			params: map[string]interface{}{
				"name":             "foo",
				"verb":             []string{"get", "post"},
				"resource":         []string{"deployments"},
				"api-group":        []string{"extensions"},
				"resource-name":    []string{"pods"},
				"non-resource-url": []string{"foo-url"},
			},
			expected: &rbac.Role{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:           []string{"get", "post"},
						Resources:       []string{"deployments"},
						APIGroups:       []string{"extensions"},
						ResourceNames:   []string{"pods"},
						NonResourceURLs: []string{"foo-url"},
					},
				},
			},
			expectErr: false,
		},
	}

	generator := RoleGeneratorV1{}
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*rbac.Role), test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, obj.(*rbac.Role))
		}
	}
}
