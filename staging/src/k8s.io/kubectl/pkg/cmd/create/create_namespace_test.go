/*
Copyright 2014 The Kubernetes Authors.

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

package create

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

func TestCreateNamespace(t *testing.T) {
	tests := map[string]struct {
		options   *NamespaceOptions
		expected  *corev1.Namespace
		expectErr bool
	}{
		"success_create": {
			options: &NamespaceOptions{
				Name: "my-namespace",
			},
			expected: &corev1.Namespace{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Namespace",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-namespace",
				},
			},
			expectErr: false,
		},
		"create_with_empty_name": {
			options: &NamespaceOptions{
				Name: "",
			},
			expectErr: true,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			namespace, err := tc.options.createNamespace()
			switch {
			case tc.expectErr && err != nil:
				return // loop, since there's no output to check
			case tc.expectErr && err == nil:
				t.Errorf("%v: expected error and didn't get one", name)
				return // loop, no expected output object
			case !tc.expectErr && err != nil:
				t.Errorf("%v: unexpected error %v", name, err)
				return // loop, no output object
			case !tc.expectErr && err == nil:
				// do nothing and drop through
			}
			if !apiequality.Semantic.DeepEqual(namespace, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, namespace)
			}
		})
	}
}
