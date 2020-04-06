// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package configmapandsecret

import (
	"path/filepath"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/kv"
	"sigs.k8s.io/kustomize/api/loader"
	valtest_test "sigs.k8s.io/kustomize/api/testutils/valtest"
	"sigs.k8s.io/kustomize/api/types"
)

func makeEnvConfigMap(name string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"DB_USERNAME": "admin",
			"DB_PASSWORD": "somepw",
		},
	}
}

func makeFileConfigMap(name string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"app-init.ini": `FOO=bar
BAR=baz
`,
		},
		BinaryData: map[string][]byte{
			"app.bin": {0xff, 0xfd},
		},
	}
}

func makeLiteralConfigMap(name string, labels, annotations map[string]string) *corev1.ConfigMap {
	cm := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"a": "x",
			"b": "y",
			"c": "Hello World",
			"d": "true",
		},
	}
	if labels != nil {
		cm.SetLabels(labels)
	}
	if annotations != nil {
		cm.SetAnnotations(annotations)
	}
	return cm
}

func TestConstructConfigMap(t *testing.T) {
	type testCase struct {
		description string
		input       types.ConfigMapArgs
		options     *types.GeneratorOptions
		expected    *corev1.ConfigMap
	}

	testCases := []testCase{
		{
			description: "construct config map from env",
			input: types.ConfigMapArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "envConfigMap",
					KvPairSources: types.KvPairSources{
						EnvSources: []string{
							filepath.Join("configmap", "app.env"),
						},
					},
				},
			},
			options:  nil,
			expected: makeEnvConfigMap("envConfigMap"),
		},
		{
			description: "construct config map from file",
			input: types.ConfigMapArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "fileConfigMap",
					KvPairSources: types.KvPairSources{
						FileSources: []string{
							filepath.Join("configmap", "app-init.ini"),
							filepath.Join("configmap", "app.bin"),
						},
					},
				},
			},
			options:  nil,
			expected: makeFileConfigMap("fileConfigMap"),
		},
		{
			description: "construct config map from literal",
			input: types.ConfigMapArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "literalConfigMap",
					KvPairSources: types.KvPairSources{
						LiteralSources: []string{"a=x", "b=y", "c=\"Hello World\"", "d='true'"},
					},
				},
			},
			options: &types.GeneratorOptions{
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			expected: makeLiteralConfigMap("literalConfigMap", map[string]string{
				"foo": "bar",
			}, nil),
		},
		{
			description: "construct config map from literal with GeneratorOptions in ConfigMapArgs",
			input: types.ConfigMapArgs{
				GeneratorArgs: types.GeneratorArgs{
					Name: "literalConfigMap",
					KvPairSources: types.KvPairSources{
						LiteralSources: []string{"a=x", "b=y", "c=\"Hello World\"", "d='true'"},
					},
					GeneratorOptions: &types.GeneratorOptions{
						Labels: map[string]string{
							"foo": "changed",
							"cat": "dog",
						},
						Annotations: map[string]string{
							"foo": "changed",
							"cat": "dog",
						},
					},
				},
			},
			options: &types.GeneratorOptions{
				Labels: map[string]string{
					"foo": "bar",
				},
				Annotations: map[string]string{
					"foo": "bar",
				},
			},
			// GeneratorOptions from the ConfigMapArgs take precedence over the
			// factory level GeneratorOptions and should overwrite
			// labels/annotations set in the factory level if there are common
			// labels/annotations
			expected: makeLiteralConfigMap("literalConfigMap", map[string]string{
				"foo": "changed",
				"cat": "dog",
			}, map[string]string{
				"foo": "changed",
				"cat": "dog",
			}),
		},
	}

	fSys := filesys.MakeFsInMemory()
	fSys.WriteFile(
		filesys.RootedPath("configmap", "app.env"),
		[]byte("DB_USERNAME=admin\nDB_PASSWORD=somepw\n"))
	fSys.WriteFile(
		filesys.RootedPath("configmap", "app-init.ini"),
		[]byte("FOO=bar\nBAR=baz\n"))
	fSys.WriteFile(
		filesys.RootedPath("configmap", "app.bin"),
		[]byte{0xff, 0xfd})
	kvLdr := kv.NewLoader(
		loader.NewFileLoaderAtRoot(fSys),
		valtest_test.MakeFakeValidator())
	for _, tc := range testCases {
		f := NewFactory(kvLdr, tc.options)
		cm, err := f.MakeConfigMap(&tc.input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(*cm, *tc.expected) {
			t.Fatalf("in testcase: %q updated:\n%#v\ndoesn't match expected:\n%#v\n", tc.description, *cm, tc.expected)
		}
	}
}
