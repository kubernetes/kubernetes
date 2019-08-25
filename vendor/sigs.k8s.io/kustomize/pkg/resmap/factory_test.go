// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmap_test

import (
	"encoding/base64"
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/loader"
	. "sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resmaptest"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/pkg/validators"
)

func TestFromFile(t *testing.T) {

	resourceStr := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: dply1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dply2
---
# some comment
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dply2
  namespace: test
---
`
	l := loadertest.NewFakeLoader("/whatever/project")
	if ferr := l.AddFile("/whatever/project/deployment.yaml", []byte(resourceStr)); ferr != nil {
		t.Fatalf("Error adding fake file: %v\n", ferr)
	}
	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "dply1",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "dply2",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "dply2",
				"namespace": "test",
			}}).ResMap()

	m, _ := rmF.FromFile(l, "deployment.yaml")
	if m.Size() != 3 {
		t.Fatalf("result should contain 3, but got %d", m.Size())
	}
	if err := expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestFromBytes(t *testing.T) {
	encoded := []byte(`apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm2
`)
	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
			}}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm2",
			}}).ResMap()
	m, err := rmF.NewResMapFromBytes(encoded)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(m, expected) {
		t.Fatalf("%#v doesn't match expected %#v", m, expected)
	}
}

var cmap = gvk.Gvk{Version: "v1", Kind: "ConfigMap"}

func TestNewFromConfigMaps(t *testing.T) {
	type testCase struct {
		description string
		input       []types.ConfigMapArgs
		filepath    string
		content     string
		expected    ResMap
	}

	l := loadertest.NewFakeLoader("/whatever/project")
	testCases := []testCase{
		{
			description: "construct config map from env",
			input: []types.ConfigMapArgs{
				{
					GeneratorArgs: types.GeneratorArgs{
						Name: "envConfigMap",
						DataSources: types.DataSources{
							EnvSources: []string{"app.env"},
						},
					},
				},
			},
			filepath: "/whatever/project/app.env",
			content:  "DB_USERNAME=admin\nDB_PASSWORD=somepw",
			expected: resmaptest_test.NewRmBuilder(t, rf).Add(
				map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "envConfigMap",
					},
					"data": map[string]interface{}{
						"DB_USERNAME": "admin",
						"DB_PASSWORD": "somepw",
					}}).ResMap(),
		},

		{
			description: "construct config map from file",
			input: []types.ConfigMapArgs{{
				GeneratorArgs: types.GeneratorArgs{
					Name: "fileConfigMap",
					DataSources: types.DataSources{
						FileSources: []string{"app-init.ini"},
					},
				},
			},
			},
			filepath: "/whatever/project/app-init.ini",
			content:  "FOO=bar\nBAR=baz\n",
			expected: resmaptest_test.NewRmBuilder(t, rf).Add(
				map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "fileConfigMap",
					},
					"data": map[string]interface{}{
						"app-init.ini": `FOO=bar
BAR=baz
`,
					},
				}).ResMap(),
		},
		{
			description: "construct config map from literal",
			input: []types.ConfigMapArgs{
				{
					GeneratorArgs: types.GeneratorArgs{
						Name: "literalConfigMap",
						DataSources: types.DataSources{
							LiteralSources: []string{"a=x", "b=y", "c=\"Good Morning\"", "d=\"false\""},
						},
					},
				},
			},
			expected: resmaptest_test.NewRmBuilder(t, rf).Add(
				map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "literalConfigMap",
					},
					"data": map[string]interface{}{
						"a": "x",
						"b": "y",
						"c": "Good Morning",
						"d": "false",
					},
				}).ResMap(),
		},

		// TODO: add testcase for data coming from multiple sources like
		// files/literal/env etc.
	}
	for _, tc := range testCases {
		if ferr := l.AddFile(tc.filepath, []byte(tc.content)); ferr != nil {
			t.Fatalf("Error adding fake file: %v\n", ferr)
		}
		r, err := rmF.NewResMapFromConfigMapArgs(l, nil, tc.input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err = tc.expected.ErrorIfNotEqualLists(r); err != nil {
			t.Fatalf("testcase: %q, err: %v", tc.description, err)
		}
	}
}

func TestNewResMapFromSecretArgs(t *testing.T) {
	secrets := []types.SecretArgs{
		{
			GeneratorArgs: types.GeneratorArgs{
				Name: "apple",
				DataSources: types.DataSources{
					LiteralSources: []string{
						"DB_USERNAME=admin",
						"DB_PASSWORD=somepw",
					},
				},
			},
			Type: ifc.SecretTypeOpaque,
		},
	}
	fakeFs := fs.MakeFakeFS()
	fakeFs.Mkdir(".")
	actual, err := rmF.NewResMapFromSecretArgs(
		loader.NewFileLoaderAtRoot(validators.MakeFakeValidator(), fakeFs), nil, secrets)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := resmaptest_test.NewRmBuilder(t, rf).Add(
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Secret",
			"metadata": map[string]interface{}{
				"name": "apple",
			},
			"type": ifc.SecretTypeOpaque,
			"data": map[string]interface{}{
				"DB_USERNAME": base64.StdEncoding.EncodeToString([]byte("admin")),
				"DB_PASSWORD": base64.StdEncoding.EncodeToString([]byte("somepw")),
			},
		}).ResMap()
	if err = expected.ErrorIfNotEqualLists(actual); err != nil {
		t.Fatalf("error: %s", err)
	}
}
