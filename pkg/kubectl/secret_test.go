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

package kubectl

import (
	"os"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestSecretGenerate(t *testing.T) {
	tests := []struct {
		setup     func(t *testing.T, params map[string]interface{}) func()
		params    map[string]interface{}
		expected  *api.Secret
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name": "foo",
				"type": "my-type",
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
				Type: "my-type",
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1=value1", "key2=value2"},
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"key1": []byte("value1"),
					"key2": []byte("value2"),
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1value1"},
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"key1=/file=2"},
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"key1==value"},
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1==value1"},
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
		{
			setup: setupEnvFile("key1=value1", "#", "", "key2=value2"),
			params: map[string]interface{}{
				"name":          "valid_env",
				"from-env-file": "file.env",
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_env",
				},
				Data: map[string][]byte{
					"key1": []byte("value1"),
					"key2": []byte("value2"),
				},
			},
			expectErr: false,
		},
		{
			setup: func() func(t *testing.T, params map[string]interface{}) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			params: map[string]interface{}{
				"name":          "getenv",
				"from-env-file": "file.env",
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "getenv",
				},
				Data: map[string][]byte{
					"g_key1": []byte("1"),
					"g_key2": []byte(""),
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":          "too_many_args",
				"from-literal":  []string{"key1=value1"},
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{
			setup: setupEnvFile("key#1=value1"),
			params: map[string]interface{}{
				"name":          "invalid_key",
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{
			setup: setupEnvFile("  key1=  value1"),
			params: map[string]interface{}{
				"name":          "with_spaces",
				"from-env-file": "file.env",
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces",
				},
				Data: map[string][]byte{
					"key1": []byte("  value1"),
				},
			},
			expectErr: false,
		},
	}
	generator := SecretGeneratorV1{}
	for _, test := range tests {
		if test.setup != nil {
			if teardown := test.setup(t, test.params); teardown != nil {
				defer teardown()
			}
		}
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.Secret), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.Secret))
		}
	}
}
