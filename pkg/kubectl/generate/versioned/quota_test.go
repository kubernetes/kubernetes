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

package versioned

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestQuotaGenerate(t *testing.T) {
	hard := "cpu=10,memory=5G,pods=10,services=7"
	resourceQuotaSpecList, err := populateResourceListV1(hard)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *v1.ResourceQuota
		expectErr bool
	}{
		{
			name: "test-valid-use",
			params: map[string]interface{}{
				"name": "foo",
				"hard": hard,
			},
			expected: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1.ResourceQuotaSpec{Hard: resourceQuotaSpecList},
			},
			expectErr: false,
		},
		{
			name: "test-missing-required-param",
			params: map[string]interface{}{
				"name": "foo",
			},
			expectErr: true,
		},
		{
			name: "test-valid-scopes",
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "BestEffort,NotTerminating",
			},
			expected: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: resourceQuotaSpecList,
					Scopes: []v1.ResourceQuotaScope{
						v1.ResourceQuotaScopeBestEffort,
						v1.ResourceQuotaScopeNotTerminating,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "test-empty-scopes",
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "",
			},
			expected: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1.ResourceQuotaSpec{Hard: resourceQuotaSpecList},
			},
			expectErr: false,
		},
		{
			name: "test-invalid-scopes",
			params: map[string]interface{}{
				"name":   "foo",
				"hard":   hard,
				"scopes": "abc,",
			},
			expectErr: true,
		},
	}

	generator := ResourceQuotaGeneratorV1{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("%s: unexpected error: %v", tt.name, err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.ResourceQuota), tt.expected) {
				t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", tt.name, tt.expected, obj.(*v1.ResourceQuota))
			}
		})
	}
}
