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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestSecretGenerate(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *api.Secret
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name": "foo",
			},
			expected: &api.Secret{
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
	}
	generator := SecretGeneratorV1{}
	for _, test := range tests {
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

func TestHandleSecretGenericReplace(t *testing.T) {
	generatorDefaultParams := map[string]interface{}{
		"name":         "foo",
		"type":         "my-type",
		"from-literal": []string{"key1=value1", "key2=value2"},
		"from-file":    []string{},
	}

	tests := []struct {
		replaceParams map[string][]string
		expected      *api.Secret
		expectErr     bool
	}{
		{
			replaceParams: map[string][]string{
				"from-literal": {"key11=value11", "key22=value22"},
				"from-file":    {},
			},
			expected: &api.Secret{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Type: "my-type",
				Data: map[string][]byte{
					"key11": []byte("value11"),
					"key22": []byte("value22"),
				},
			},
			expectErr: false,
		},
		{
			replaceParams: map[string][]string{
				"from-literal": {"key1==value1"},
				"from-file":    {},
			},
			expected: &api.Secret{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Type: "my-type",
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
		{
			replaceParams: map[string][]string{
				"from-literal": {"key1value1"},
				"from-file":    {},
			},
			expectErr: true,
		},
		{
			replaceParams: map[string][]string{
				"from-literal": {},
				"from-file":    {"key1=/file=2"},
			},
			expectErr: true,
		},
		{
			replaceParams: map[string][]string{
				"from-literal": {},
				"from-file":    {"key1==value"},
			},
			expectErr: true,
		},
	}
	generator := SecretGeneratorV1{}
	for _, test := range tests {
		obj, err := generator.Generate(generatorDefaultParams)
		err = HandleSecretGenericReplace(obj.(*api.Secret), test.replaceParams["from-file"], test.replaceParams["from-literal"])
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
