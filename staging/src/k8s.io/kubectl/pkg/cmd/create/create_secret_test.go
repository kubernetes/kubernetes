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
	"io/ioutil"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateSecretObject(t *testing.T) {
	secretObject := newSecretObj("foo", "foo-namespace", corev1.SecretTypeDockerConfigJson)
	expectedSecretObject := &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "foo-namespace",
		},
		Type: corev1.SecretTypeDockerConfigJson,
		Data: map[string][]byte{},
	}
	t.Run("Creating a Secret Object", func(t *testing.T) {
		if !apiequality.Semantic.DeepEqual(secretObject, expectedSecretObject) {
			t.Errorf("expected:\n%#v\ngot:\n%#v", secretObject, expectedSecretObject)
		}
	})
}

func TestCreateSecretGeneric(t *testing.T) {
	tests := map[string]struct {
		secretName  string
		secretType  string
		fromLiteral []string
		fromFile    []string
		fromEnvFile string
		appendHash  bool
		setup       func(t *testing.T, secretGenericOptions *CreateSecretOptions) func()

		expected  *corev1.Secret
		expectErr bool
	}{
		"create_secret_foo": {
			secretName: "foo",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_secret_foo_hash": {
			secretName: "foo",
			appendHash: true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-949tdgdkgg",
				},
				Data: map[string][]byte{},
			},
			expectErr: false,
		},
		"create_secret_foo_type": {
			secretName: "foo",
			secretType: "my-type",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{},
				Type: "my-type",
			},
			expectErr: false,
		},
		"create_secret_foo_type_hash": {
			secretName: "foo",
			secretType: "my-type",
			appendHash: true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-dg474f9t76",
				},
				Data: map[string][]byte{},
				Type: "my-type",
			},
			expectErr: false,
		},
		"create_secret_foo_two_literal": {
			secretName:  "foo",
			fromLiteral: []string{"key1=value1", "key2=value2"},
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
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
		"create_secret_foo_two_literal_hash": {
			secretName:  "foo",
			fromLiteral: []string{"key1=value1", "key2=value2"},
			appendHash:  true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
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
		"create_secret_foo_key1_=value1": {
			secretName:  "foo",
			fromLiteral: []string{"key1==value1"},
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
		"create_secret_foo_key1_=value1_hash": {
			secretName:  "foo",
			fromLiteral: []string{"key1==value1"},
			appendHash:  true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-fdcc8tkhh5",
				},
				Data: map[string][]byte{
					"key1": []byte("=value1"),
				},
			},
			expectErr: false,
		},
		"create_secret_foo_from_file_foo1_foo2_secret": {
			secretName: "foo",
			setup:      setupSecretBinaryFile([]byte{0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64}, "foo1", "foo2"),
			fromFile:   []string{"foo1", "foo2"},
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"foo1": []byte("hello world"),
					"foo2": []byte("hello world"),
				},
			},
			expectErr: false,
		},
		"create_secret_foo_from_file_foo1_foo2_hash": {
			secretName: "foo",
			setup:      setupSecretBinaryFile([]byte{0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64}, "foo1", "foo2"),
			fromFile:   []string{"foo1", "foo2"},
			appendHash: true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-hbkh2cdb57",
				},
				Data: map[string][]byte{
					"foo1": []byte("hello world"),
					"foo2": []byte("hello world"),
				},
			},
			expectErr: false,
		},
		"create_secret_foo_from_file_foo1_foo2_and": {
			secretName: "foo",
			setup:      setupSecretBinaryFile([]byte{0xff, 0xfd}, "foo1", "foo2"),
			fromFile:   []string{"foo1", "foo2"},
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					"foo1": {0xff, 0xfd},
					"foo2": {0xff, 0xfd},
				},
			},
			expectErr: false,
		},
		"create_secret_foo_from_file_foo1_foo2_and_hash": {
			secretName: "foo",
			setup:      setupSecretBinaryFile([]byte{0xff, 0xfd}, "foo1", "foo2"),
			fromFile:   []string{"foo1", "foo2"},
			appendHash: true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-mkhg4ktk4d",
				},
				Data: map[string][]byte{
					"foo1": {0xff, 0xfd},
					"foo2": {0xff, 0xfd},
				},
			},
			expectErr: false,
		},
		"create_secret_valid_env_from_env_file": {
			secretName:  "valid_env",
			setup:       setupSecretEnvFile("key1=value1", "#", "", "key2=value2"),
			fromEnvFile: "file.env",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
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
		"create_secret_valid_env_from_env_file_hash": {
			secretName:  "valid_env",
			setup:       setupSecretEnvFile("key1=value1", "#", "", "key2=value2"),
			fromEnvFile: "file.env",
			appendHash:  true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
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
		"create_secret_get_env_from_env_file": {
			secretName: "get_env",
			setup: func() func(t *testing.T, secretGenericOptions *CreateSecretOptions) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupSecretEnvFile("g_key1", "g_key2=")
			}(),
			fromEnvFile: "file.env",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "get_env",
				},
				Data: map[string][]byte{
					"g_key1": []byte("1"),
					"g_key2": []byte(""),
				},
			},
			expectErr: false,
		},
		"create_secret_get_env_from_env_file_hash": {
			secretName: "get_env",
			setup: func() func(t *testing.T, secretGenericOptions *CreateSecretOptions) func() {
				os.Setenv("g_key1", "1")
				os.Setenv("g_key2", "2")
				return setupSecretEnvFile("g_key1", "g_key2=")
			}(),
			fromEnvFile: "file.env",
			appendHash:  true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "get_env-68mt8f2kkt",
				},
				Data: map[string][]byte{
					"g_key1": []byte("1"),
					"g_key2": []byte(""),
				},
			},
			expectErr: false,
		},
		"create_secret_value_with_space_from_env_file": {
			secretName:  "value_with_space",
			setup:       setupSecretEnvFile("   key1=  value1"),
			fromEnvFile: "file.env",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "value_with_space",
				},
				Data: map[string][]byte{
					"key1": []byte("  value1"),
				},
			},
			expectErr: false,
		},
		"create_secret_value_with_space_from_env_file_hash": {
			secretName:  "valid_with_space",
			setup:       setupSecretEnvFile("   key1=  value1"),
			fromEnvFile: "file.env",
			appendHash:  true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid_with_space-bhkb4gfck6",
				},
				Data: map[string][]byte{
					"key1": []byte("  value1"),
				},
			},
			expectErr: false,
		},
		"create_invalid_secret_filepath_contains_=": {
			secretName: "foo",
			fromFile:   []string{"key1=/file=2"},
			expectErr:  true,
		},
		"create_invalid_secret_filepath_key_contains_=": {
			secretName: "foo",
			fromFile:   []string{"=key=/file1"},
			expectErr:  true,
		},
		"create_invalid_secret_literal_key_contains_=": {
			secretName: "foo",
			fromFile:   []string{"=key=value1"},
			expectErr:  true,
		},
		"create_invalid_secret_env_key_contains_#": {
			secretName:  "invalid_key",
			setup:       setupSecretEnvFile("key#1=value1"),
			fromEnvFile: "file.env",
			expectErr:   true,
		},
		"create_invalid_secret_duplicate_key1": {
			secretName:  "foo",
			fromLiteral: []string{"key1=value1", "key1=value2"},
			expectErr:   true,
		},
		"create_invalid_secret_no_file": {
			secretName: "foo",
			fromFile:   []string{"key1=/file1"},
			expectErr:  true,
		},
		"create_invalid_secret_invalid_literal": {
			secretName:  "foo",
			fromLiteral: []string{"key1value1"},
			expectErr:   true,
		},
		"create_invalid_secret_invalid_filepath": {
			secretName: "foo",
			fromFile:   []string{"key1==file1"},
			expectErr:  true,
		},
		"create_invalid_secret_no_name": {
			expectErr: true,
		},
		"create_invalid_secret_too_many_args": {
			secretName:  "too_many_args",
			fromFile:    []string{"key1=/file1"},
			fromEnvFile: "foo",
			expectErr:   true,
		},
		"create_invalid_secret_too_many_args_1": {
			secretName:  "too_many_args_1",
			fromLiteral: []string{"key1=value1"},
			fromEnvFile: "foo",
			expectErr:   true,
		},
		"create_invalid_secret_too_many_args_2": {
			secretName:  "too_many_args_2",
			fromFile:    []string{"key1=/file1"},
			fromLiteral: []string{"key1=value1"},
			fromEnvFile: "foo",
			expectErr:   true,
		},
	}

	// run all the tests
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var secret *corev1.Secret = nil
			secretOptions := CreateSecretOptions{
				Name:           test.secretName,
				Type:           test.secretType,
				AppendHash:     test.appendHash,
				FileSources:    test.fromFile,
				LiteralSources: test.fromLiteral,
				EnvFileSource:  test.fromEnvFile,
			}
			if test.setup != nil {
				if teardown := test.setup(t, &secretOptions); teardown != nil {
					defer teardown()
				}
			}
			err := secretOptions.Validate()
			if err == nil {
				secret, err = secretOptions.createSecret()
			}

			if !test.expectErr && err != nil {
				t.Errorf("test %s, unexpected error: %v", name, err)
			}
			if test.expectErr && err == nil {
				t.Errorf("test %s was expecting an error but no error occurred", name)
			}
			if !apiequality.Semantic.DeepEqual(secret, test.expected) {
				t.Errorf("test %s\n expected:\n%#v\ngot:\n%#v", name, test.expected, secret)
			}
		})
	}
}

func setupSecretEnvFile(lines ...string) func(*testing.T, *CreateSecretOptions) func() {
	return func(t *testing.T, secretOptions *CreateSecretOptions) func() {
		f, err := ioutil.TempFile("", "cme")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, l := range lines {
			f.WriteString(l)
			f.WriteString("\r\n")
		}
		f.Close()
		secretOptions.EnvFileSource = f.Name()
		return func() {
			os.Remove(f.Name())
		}
	}
}

func setupSecretBinaryFile(data []byte, files ...string) func(*testing.T, *CreateSecretOptions) func() {
	return func(t *testing.T, secretOptions *CreateSecretOptions) func() {
		tmp, _ := ioutil.TempDir("", "")
		for i, file := range files {
			f := tmp + "/" + file
			ioutil.WriteFile(f, data, 0644)
			secretOptions.FileSources[i] = f
		}
		return func() {
			for _, file := range files {
				f := tmp + "/" + file
				os.RemoveAll(f)
			}
		}
	}
}
