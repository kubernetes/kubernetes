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

package create

import (
	"io/ioutil"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateConfigMap(t *testing.T) {
	tests := map[string]struct {
		configMapName string
		configMapType string
		appendHash    bool
		fromLiteral   []string
		fromFile      []string
		fromEnvFile   string
		setup         func(t *testing.T, configMapOptions *ConfigMapOptions) func()

		expected  *corev1.ConfigMap
		expectErr bool
	}{
		"create_foo_configmap": {
			configMapName: "foo",
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_foo_hash_configmap": {
			configMapName: "foo",
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-867km9574f",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_foo_type_configmap": {
			configMapName: "foo",
			configMapType: "my-type",
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_foo_type_hash_configmap": {
			configMapName: "foo",
			configMapType: "my-type",
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-867km9574f",
				},
				Data:       map[string]string{},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_foo_two_literal_configmap": {
			configMapName: "foo",
			fromLiteral:   []string{"key1=value1", "key2=value2"},
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_foo_two_literal_hash_configmap": {
			configMapName: "foo",
			fromLiteral:   []string{"key1=value1", "key2=value2"},
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_foo_key1_=value1_configmap": {
			configMapName: "foo",
			fromLiteral:   []string{"key1==value1"},
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_foo_key1_=value1_hash_configmap": {
			configMapName: "foo",
			fromLiteral:   []string{"key1==value1"},
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_foo_from_file_foo1_foo2_configmap": {
			configMapName: "foo",
			setup:         setupBinaryFile([]byte{0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64}, "foo1", "foo2"),
			fromFile:      []string{"foo1", "foo2"},
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"foo1": "hello world",
					"foo2": "hello world",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_foo_from_file_foo1_foo2_and_configmap": {
			configMapName: "foo",
			setup:         setupBinaryFile([]byte{0xff, 0xfd}, "foo1", "foo2"),
			fromFile:      []string{"foo1", "foo2"},
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{},
				BinaryData: map[string][]byte{
					"foo1": {0xff, 0xfd},
					"foo2": {0xff, 0xfd},
				},
			},
			expectErr: false,
		},
		"create_valid_env_from_env_file_configmap": {
			configMapName: "valid_env",
			setup:         setupEnvFile("key1=value1", "#", "", "key2=value2"),
			fromEnvFile:   "file.env",
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_valid_env_from_env_file_hash_configmap": {
			configMapName: "valid_env",
			setup:         setupEnvFile("key1=value1", "#", "", "key2=value2"),
			fromEnvFile:   "file.env",
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
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
		"create_get_env_from_env_file_configmap": {
			configMapName: "get_env",
			setup: func() func(t *testing.T, configMapOptions *ConfigMapOptions) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			fromEnvFile: "file.env",
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "get_env",
				},
				Data: map[string]string{
					"g_key1": "1",
					"g_key2": "",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_get_env_from_env_file_hash_configmap": {
			configMapName: "get_env",
			setup: func() func(t *testing.T, configMapOptions *ConfigMapOptions) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupEnvFile("g_key1", "g_key2=")
			}(),
			fromEnvFile: "file.env",
			appendHash:  true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "get_env-54k882kkd2",
				},
				Data: map[string]string{
					"g_key1": "1",
					"g_key2": "",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_value_with_space_from_env_file_configmap": {
			configMapName: "value_with_space",
			setup:         setupEnvFile("key1=  value1"),
			fromEnvFile:   "file.env",
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "value_with_space",
				},
				Data: map[string]string{
					"key1": "  value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_value_with_space_from_env_file_hash_configmap": {
			configMapName: "valid_with_space",
			setup:         setupEnvFile("key1=  value1"),
			fromEnvFile:   "file.env",
			appendHash:    true,
			expected: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_with_space-b4448m7gdm",
				},
				Data: map[string]string{
					"key1": "  value1",
				},
				BinaryData: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_invalid_configmap_filepath_contains_=": {
			configMapName: "foo",
			fromFile:      []string{"key1=/file=2"},
			expectErr:     true,
		},
		"create_invalid_configmap_filepath_key_contains_=": {
			configMapName: "foo",
			fromFile:      []string{"=key=/file1"},
			expectErr:     true,
		},
		"create_invalid_configmap_literal_key_contains_=": {
			configMapName: "foo",
			fromFile:      []string{"=key=value1"},
			expectErr:     true,
		},
		"create_invalid_configmap_duplicate_key1": {
			configMapName: "foo",
			fromLiteral:   []string{"key1=value1", "key1=value2"},
			expectErr:     true,
		},
		"create_invalid_configmap_no_file": {
			configMapName: "foo",
			fromFile:      []string{"key1=/file1"},
			expectErr:     true,
		},
		"create_invalid_configmap_invalid_literal": {
			configMapName: "foo",
			fromLiteral:   []string{"key1value1"},
			expectErr:     true,
		},
		"create_invalid_configmap_invalid_filepath": {
			configMapName: "foo",
			fromFile:      []string{"key1==file1"},
			expectErr:     true,
		},
		"create_invalid_configmap_too_many_args": {
			configMapName: "too_many_args",
			fromFile:      []string{"key1=/file1"},
			fromEnvFile:   "foo",
			expectErr:     true,
		},
		"create_invalid_configmap_too_many_args_1": {
			configMapName: "too_many_args_1",
			fromLiteral:   []string{"key1=value1"},
			fromEnvFile:   "foo",
			expectErr:     true,
		},
	}

	// run all the tests
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			configMapOptions := ConfigMapOptions{
				Name:           test.configMapName,
				Type:           test.configMapType,
				AppendHash:     test.appendHash,
				FileSources:    test.fromFile,
				LiteralSources: test.fromLiteral,
				EnvFileSource:  test.fromEnvFile,
			}

			if test.setup != nil {
				if teardown := test.setup(t, &configMapOptions); teardown != nil {
					defer teardown()
				}
			}

			configMap, err := configMapOptions.createConfigMap()
			if !test.expectErr && err != nil {
				t.Errorf("test %s, unexpected error: %v", name, err)
			}
			if test.expectErr && err == nil {
				t.Errorf("test %s was expecting an error but no error occurred", name)
			}
			if !apiequality.Semantic.DeepEqual(configMap, test.expected) {
				t.Errorf("test %s expected:\n%#v\ngot:\n%#v", name, test.expected, configMap)
			}
		})
	}
}

func setupEnvFile(lines ...string) func(*testing.T, *ConfigMapOptions) func() {
	return func(t *testing.T, configMapOptions *ConfigMapOptions) func() {
		f, err := ioutil.TempFile("", "cme")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, l := range lines {
			f.WriteString(l)
			f.WriteString("\r\n")
		}
		f.Close()
		configMapOptions.EnvFileSource = f.Name()
		return func() {
			os.Remove(f.Name())
		}
	}
}

func setupBinaryFile(data []byte, files ...string) func(*testing.T, *ConfigMapOptions) func() {
	return func(t *testing.T, configMapOptions *ConfigMapOptions) func() {
		tmp, _ := ioutil.TempDir("", "")
		for i, file := range files {
			f := tmp + "/" + file
			ioutil.WriteFile(f, data, 0644)
			configMapOptions.FileSources[i] = f
		}
		return func() {
			for _, file := range files {
				f := tmp + "/" + file
				os.Remove(f)
			}
		}
	}
}
