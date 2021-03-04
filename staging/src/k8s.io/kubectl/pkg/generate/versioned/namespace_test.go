/*
Copyright 2015 The Kubernetes Authors.

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

func TestNamespaceGenerate(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *v1.Namespace
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			expectErr: false,
		},
		{
			name:      "test2",
			params:    map[string]interface{}{},
			expectErr: true,
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name": 1,
			},
			expectErr: true,
		},
		{
			name: "test4",
			params: map[string]interface{}{
				"name": "",
			},
			expectErr: true,
		},
		{
			name: "test5",
			params: map[string]interface{}{
				"name": nil,
			},
			expectErr: true,
		},
		{
			name: "test6",
			params: map[string]interface{}{
				"name_wrong_key": "some_value",
			},
			expectErr: true,
		},
		{
			name: "test7",
			params: map[string]interface{}{
				"NAME": "some_value",
			},
			expectErr: true,
		},
	}
	generator := NamespaceGeneratorV1{}
	for index, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			switch {
			case tt.expectErr && err != nil:
				return // loop, since there's no output to check
			case tt.expectErr && err == nil:
				t.Errorf("%v: expected error and didn't get one", index)
				return // loop, no expected output object
			case !tt.expectErr && err != nil:
				t.Errorf("%v: unexpected error %v", index, err)
				return // loop, no output object
			case !tt.expectErr && err == nil:
				// do nothing and drop through
			}
			if !reflect.DeepEqual(obj.(*v1.Namespace), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*v1.Namespace))
			}
		})
	}
}
