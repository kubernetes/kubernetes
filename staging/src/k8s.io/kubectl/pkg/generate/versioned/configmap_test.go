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
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestConfigMapGenerate(t *testing.T) {
	tests := []struct {
		name      string
		setup     func(t *testing.T, params map[string]interface{}) func()
		params    map[string]interface{}
		expected  *v1.ConfigMap
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test2",
			params: map[string]interface{}{
				"name":        "foo",
				"append-hash": true,
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-867km9574f",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name": "foo",
				"type": "my-type",
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
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
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-867km9574f",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test5",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1=value1", "key2=value2"},
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
				},
				BinaryData: map[string][]byte{},
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
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-gcb75dd9gb",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
				},
				BinaryData: map[string][]byte{},
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
			name:  "test10",
			setup: setupBinaryFile([]byte{0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64}),
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"foo1"},
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{"foo1": "hello world"},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name:  "test11",
			setup: setupBinaryFile([]byte{0xff, 0xfd}),
			params: map[string]interface{}{
				"name":      "foo",
				"from-file": []string{"foo1"},
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{"foo1": {0xff, 0xfd}},
			},
			expectErr: false,
		},
		{
			name: "test12",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1==value1"},
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"key1": "=value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test13",
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1==value1"},
				"append-hash":  true,
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-bdgk9ttt7m",
				},
				Data: map[string]string{
					"key1": "=value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name:  "test14",
			setup: setupEnvFile("key1=value1", "#", "", "key2=value2"),
			params: map[string]interface{}{
				"name":          "valid_env",
				"from-env-file": "file.env",
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_env",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name:  "test15",
			setup: setupEnvFile("key1=value1", "#", "", "key2=value2"),
			params: map[string]interface{}{
				"name":          "valid_env",
				"from-env-file": "file.env",
				"append-hash":   true,
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_env-2cgh8552ch",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test16",
			setup: func() func(t *testing.T, params map[string]interface{}) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			params: map[string]interface{}{
				"name":          "getenv",
				"from-env-file": "file.env",
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "getenv",
				},
				Data: map[string]string{
					"g_key1": "1",
					"g_key2": "",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test17",
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
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "getenv-b4hh92hgdk",
				},
				Data: map[string]string{
					"g_key1": "1",
					"g_key2": "",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name: "test18",
			params: map[string]interface{}{
				"name":          "too_many_args",
				"from-literal":  []string{"key1=value1"},
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{name: "test19",
			setup: setupEnvFile("key#1=value1"),
			params: map[string]interface{}{
				"name":          "invalid_key",
				"from-env-file": "file.env",
			},
			expectErr: true,
		},
		{
			name:  "test20",
			setup: setupEnvFile("  key1=  value1"),
			params: map[string]interface{}{
				"name":          "with_spaces",
				"from-env-file": "file.env",
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces",
				},
				Data: map[string]string{
					"key1": "  value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		{
			name:  "test21",
			setup: setupEnvFile("  key1=  value1"),
			params: map[string]interface{}{
				"name":          "with_spaces",
				"from-env-file": "file.env",
				"append-hash":   true,
			},
			expected: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces-bfc558b4ct",
				},
				Data: map[string]string{
					"key1": "  value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
	}
	generator := ConfigMapGeneratorV1{}
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
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.ConfigMap), tt.expected) {
				t.Errorf("\ncase %d, expected:\n%#v\nsaw:\n%#v", i, tt.expected, obj.(*v1.ConfigMap))
			}
		})
	}
}

func setupEnvFile(lines ...string) func(*testing.T, map[string]interface{}) func() {
	return func(t *testing.T, params map[string]interface{}) func() {
		f, err := ioutil.TempFile("", "cme")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, l := range lines {
			f.WriteString(l)
			f.WriteString("\r\n")
		}
		f.Close()
		params["from-env-file"] = f.Name()
		return func() {
			os.Remove(f.Name())
		}
	}
}

func setupBinaryFile(data []byte) func(*testing.T, map[string]interface{}) func() {
	return func(t *testing.T, params map[string]interface{}) func() {
		tmp, _ := ioutil.TempDir("", "")
		f := tmp + "/foo1"
		ioutil.WriteFile(f, data, 0644)
		params["from-file"] = []string{f}
		return func() {
			os.Remove(f)
		}
	}
}
