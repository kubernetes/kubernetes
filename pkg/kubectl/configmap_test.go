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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestConfigMapGenerate(t *testing.T) {
	tests := []struct {
		setup     func(t *testing.T, params map[string]interface{}) func()
		params    map[string]interface{}
		expected  *api.ConfigMap
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name": "foo",
				"type": "my-type",
			},
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":         "foo",
				"from-literal": []string{"key1=value1", "key2=value2"},
			},
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
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
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"key1": "=value1",
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
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_env",
				},
				Data: map[string]string{
					"key1": "value1",
					"key2": "value2",
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
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "getenv",
				},
				Data: map[string]string{
					"g_key1": "1",
					"g_key2": "",
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
			expected: &api.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "with_spaces",
				},
				Data: map[string]string{
					"key1": "  value1",
				},
			},
			expectErr: false,
		},
	}
	generator := ConfigMapGeneratorV1{}
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
		if !reflect.DeepEqual(obj.(*api.ConfigMap), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.ConfigMap))
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
