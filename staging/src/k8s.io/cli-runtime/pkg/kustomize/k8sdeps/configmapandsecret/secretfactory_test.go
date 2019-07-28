/*
Copyright 2018 The Kubernetes Authors.

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

package configmapandsecret

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/types"
)

func makeEnvSecret(name string) *corev1.Secret {
	return &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string][]byte{
			"DB_PASSWORD": []byte("somepw"),
			"DB_USERNAME": []byte("admin"),
		},
		Type: "Opaque",
	}
}

func makeFileSecret(name string) *corev1.Secret {
	return &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string][]byte{
			"app-init.ini": []byte(`FOO=bar
BAR=baz
`),
		},
		Type: "Opaque",
	}
}

func makeLiteralSecret(name string) *corev1.Secret {
	s := &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string][]byte{
			"a": []byte("x"),
			"b": []byte("y"),
		},
		Type: "Opaque",
	}
	s.SetLabels(map[string]string{"foo": "bar"})
	return s
}

func TestConstructSecret(t *testing.T) {
	type testCase struct {
		description string
		input       types.SecretArgs
		options     *types.GeneratorOptions
		expected    *corev1.Secret
	}

	testCases := []testCase{
		{
			description: "construct secret from env",
			input: types.SecretArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "envSecret",
					DataSources: types.DataSources{
						EnvSource: "secret/app.env",
					},
				},
			},
			options:  nil,
			expected: makeEnvSecret("envSecret"),
		},
		{
			description: "construct secret from file",
			input: types.SecretArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "fileSecret",
					DataSources: types.DataSources{
						FileSources: []string{"secret/app-init.ini"},
					},
				},
			},
			options:  nil,
			expected: makeFileSecret("fileSecret"),
		},
		{
			description: "construct secret from literal",
			input: types.SecretArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "literalSecret",
					DataSources: types.DataSources{
						LiteralSources: []string{"a=x", "b=y"},
					},
				},
			},
			options: &types.GeneratorOptions{
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			expected: makeLiteralSecret("literalSecret"),
		},
	}

	fSys := fs.MakeFakeFS()
	fSys.WriteFile("/secret/app.env", []byte("DB_USERNAME=admin\nDB_PASSWORD=somepw\n"))
	fSys.WriteFile("/secret/app-init.ini", []byte("FOO=bar\nBAR=baz\n"))
	f := NewSecretFactory(loader.NewFileLoaderAtRoot(fSys))
	for _, tc := range testCases {
		cm, err := f.MakeSecret(&tc.input, tc.options)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(*cm, *tc.expected) {
			t.Fatalf("in testcase: %q updated:\n%#v\ndoesn't match expected:\n%#v\n", tc.description, *cm, tc.expected)
		}
	}
}
