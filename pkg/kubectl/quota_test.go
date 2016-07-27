/*
Copyright 2016 The Kubernetes Authors.

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
)

func TestQuotaGenerate(t *testing.T) {
	hard := "cpu=10,memory=5G,pods=10,services=7"
	resourceQuotaSpecList, err := populateResourceList(hard)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := map[string]struct {
		params    map[string]interface{}
		expected  *api.ResourceQuota
		expectErr bool
	}{
		"test-valid-use": {
			params: map[string]interface{}{
				"name": "foo",
				"hard": hard,
			},
			expected: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ResourceQuotaSpec{Hard: resourceQuotaSpecList},
			},
			expectErr: false,
		},
		"test-missing-required-param": {
			params: map[string]interface{}{
				"name": "foo",
			},
			expectErr: true,
		},
		"test-valid-scopes": {
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "BestEffort,NotTerminating",
			},
			expected: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ResourceQuotaSpec{
					Hard: resourceQuotaSpecList,
					Scopes: []api.ResourceQuotaScope{
						api.ResourceQuotaScopeBestEffort,
						api.ResourceQuotaScopeNotTerminating,
					},
				},
			},
			expectErr: false,
		},
		"test-empty-scopes": {
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "",
			},
			expected: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ResourceQuotaSpec{Hard: resourceQuotaSpecList},
			},
			expectErr: false,
		},
		"test-invalid-scopes": {
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "abc,",
			},
			expectErr: true,
		},
	}

	generator := ResourceQuotaGeneratorV1{}
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.ResourceQuota), test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, obj.(*api.ResourceQuota))
		}
	}
}
