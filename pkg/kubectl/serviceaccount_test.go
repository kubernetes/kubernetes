/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

func TestServiceAccountGenerate(t *testing.T) {
	tests := []struct {
		name      string
		expected  *api.ServiceAccount
		expectErr bool
	}{
		{
			name: "foo",
			expected: &api.ServiceAccount{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			expectErr: false,
		},
		{
			expectErr: true,
		},
	}
	for _, test := range tests {
		generator := ServiceAccountGeneratorV1{
			Name: test.name,
		}
		obj, err := generator.StructuredGenerate()
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.ServiceAccount), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.ServiceAccount))
		}
	}
}
