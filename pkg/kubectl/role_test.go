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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

type fakeRESTMapper struct {
	InvalidResource string
}

func (f fakeRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	return schema.GroupVersionKind{}, nil
}

func (f fakeRESTMapper) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	return nil, nil
}

func (f fakeRESTMapper) ResourceFor(input schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	if input.Resource == f.InvalidResource {
		return schema.GroupVersionResource{}, fmt.Errorf("invalid resource")
	}
	return schema.GroupVersionResource{Resource: input.Resource}, nil
}

func (f fakeRESTMapper) ResourcesFor(input schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return nil, nil
}

func (f fakeRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return nil, nil
}

func (f fakeRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	return nil, nil
}

func (f fakeRESTMapper) AliasesForResource(resource string) ([]string, bool) {
	return nil, false
}

func (f fakeRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return "", nil
}

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
				ObjectMeta: v1.ObjectMeta{
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
		"test-invalid-verb": {
			params: map[string]interface{}{
				"name":     "foo",
				"verb":     []string{"invalid-verb"},
				"resource": []string{"pods"},
			},
			expectErr: true,
		},
		"test-invalid-resource": {
			params: map[string]interface{}{
				"name":     "foo",
				"verb":     []string{"get"},
				"resource": []string{"invalid-resource"},
			},
			expectErr: true,
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
				"name":          "foo",
				"verb":          []string{"get", "post"},
				"resource":      []string{"pods"},
				"api-group":     []string{""},
				"resource-name": []string{"abc"},
			},
			expected: &rbac.Role{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:         []string{"get", "post"},
						Resources:     []string{"pods"},
						APIGroups:     []string{""},
						ResourceNames: []string{"abc"},
					},
				},
			},
			expectErr: false,
		},
	}

	generator := &RoleGeneratorV1{Mapper: fakeRESTMapper{InvalidResource: "invalid-resource"}}
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
