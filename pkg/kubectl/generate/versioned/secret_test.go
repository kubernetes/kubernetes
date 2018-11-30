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
	"os"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestSecretGenerate(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(t *testing.T, params map[string]interface{}) func()
		params    map[string]interface{}
		expected  *v1.Secret
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test2",
			params: map[string]interface{}{
				"name":        "foo",
				"append-hash": true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-949tdgdkgg",
				},
				Data: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name": "foo",
				"type": "my-type",
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
				Type: "my-type",
			},
			expectErr: false,
		},
		{
			name: "test4",
			params: map[string]interface{}{
				"name":        "foo",
				"type":        "my-type",
				"append-hash": true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-dg474f9t76",
				},
				Data: map[string][]byte{},
				Type: "my-type",
			},
			expectErr: false,
		},
		{
			name: "test5",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1=value1", "key2=value2"},
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
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
			name: "test6",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1=value1", "key2=value2"},
				"append-hash":  true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-tf72c228m4",
				},
				Data: map[string][]byte{
					"key1": []byte("value1"),
					"key2": []byte("value2"),
				},
			},
			expectErr: false,
		},
		{
			name: "test7",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1value1"},
			},
			expectErr: true,
		},
		{
			name: "test8",
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"key1=/file=2"},
			},
			expectErr: true,
		},
		{
			name: "test9",
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"key1==value"},
			},
			expectErr: true,
		},
		{
			name: "test10",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1==value1"},
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
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
			name: "test11",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1==value1"},
				"append-hash":  true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-fdcc8tkhh5",
				},
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
		{
			name:  "test12",
			setup: setupEnvFile("key1=value1", "#", "", "key2=value2"),
			params: map[string]interface{}{
				"name":          "valid_env",
				"from-env-file": "file.env",
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
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
			name:  "test13",
			setup: setupEnvFile("key1=value1", "#", "", "key2=value2"),
			params: map[string]interface{}{
				"name":          "valid_env",
				"from-env-file": "file.env",
				"append-hash":   true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_env-bkb2m2965h",
				},
				Data: map[string][]byte{
					"key1": []byte("value1"),
					"key2": []byte("value2"),
				},
			},
			expectErr: false,
		},
		{
			name: "test14",
			setup: func() func(t *testing.T, params map[string]interface{}) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			params: map[string]interface{}{
				"name":          "getenv",
				"from-env-file": "file.env",
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
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
			name: "test15",
			setup: func() func(t *testing.T, params map[string]interface{}) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			params: map[string]interface{}{
				"name":          "getenv",
				"from-env-file": "file.env",
				"append-hash":   true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "getenv-m7kg2khdb4",
				},
				Data: map[string][]byte{
					"g_key1": []byte("1"),
					"g_key2": []byte(""),
				},
			},
			expectErr: false,
		},
		{
			name: "test16",
			params: map[string]interface{}{
				"name":          "too_many_args",
				"from-literal":  []string{"key1=value1"},
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{
			name:  "test17",
			setup: setupEnvFile("key#1=value1"),
			params: map[string]interface{}{
				"name":          "invalid_key",
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{
			name:  "test18",
			setup: setupEnvFile("  key1=  value1"),
			params: map[string]interface{}{
				"name":          "with_spaces",
				"from-env-file": "file.env",
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces",
				},
				Data: map[string][]byte{
					"key1": []byte("  value1"),
				},
			},
			expectErr: false,
		},
		{
			name:  "test19",
			setup: setupEnvFile("  key1=  value1"),
			params: map[string]interface{}{
				"name":          "with_spaces",
				"from-env-file": "file.env",
				"append-hash":   true,
			},
			expected: &v1.Secret{
				// this is ok because we know exactly how we want to be serialized
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Secret"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces-4488d5b57d",
				},
				Data: map[string][]byte{
					"key1": []byte("  value1"),
				},
			},
			expectErr: false,
		},
	}
	generator := SecretGeneratorV1{}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setup != nil {
				if teardown := tt.setup(t, tt.params); teardown != nil {
					defer teardown()
				}
			}
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("case %d, unexpected error: %v", i, err)
				return
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.Secret), tt.expected) {
				t.Errorf("\ncase %d, expected:\n%#v\nsaw:\n%#v", i, tt.expected, obj.(*v1.Secret))
			}
		})
	}
}
