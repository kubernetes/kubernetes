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
	scheduling "k8s.io/api/scheduling/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"reflect"
	"testing"
)

func TestPriorityClassV1Generator(t *testing.T) {
	tests := map[string]struct {
		params    map[string]interface{}
		expected  *scheduling.PriorityClass
		expectErr bool
	}{
		"test valid case": {
			params: map[string]interface{}{
				"name":           "foo",
				"value":          int32(1000),
				"global-default": false,
				"description":    "high priority class",
			},
			expected: &scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Value:         int32(1000),
				GlobalDefault: false,
				Description:   "high priority class",
			},
			expectErr: false,
		},
		"test valid case that as default priority": {
			params: map[string]interface{}{
				"name":           "foo",
				"value":          int32(1000),
				"global-default": true,
				"description":    "high priority class",
			},
			expected: &scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Value:         int32(1000),
				GlobalDefault: true,
				Description:   "high priority class",
			},
			expectErr: false,
		},
		"test missing required param": {
			params: map[string]interface{}{
				"name":           "foo",
				"global-default": true,
				"description":    "high priority class",
			},
			expectErr: true,
		},
	}

	generator := PriorityClassV1Generator{}
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*scheduling.PriorityClass), test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, obj.(*scheduling.PriorityClass))
		}
	}
}
