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
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestConfigMapGenerate(t *testing.T) {
	tests := []struct {
		setup     func(t *testing.T, params map[string]interface{}) func()
		params    map[string]interface{}
		expected  *v1.ConfigMap
		expectErr bool
	}{
		{
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
	for i, test := range tests {
		if test.setup != nil {
			if teardown := test.setup(t, test.params); teardown != nil {
				defer teardown()
			}
		}
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("case %d, unexpected error: %v", i, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*v1.ConfigMap), test.expected) {
			t.Errorf("\ncase %d, expected:\n%#v\nsaw:\n%#v", i, test.expected, obj.(*v1.ConfigMap))
		}
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
