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

package kunstruct

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/pkg/ifc"
)

func TestSliceFromBytes(t *testing.T) {
	factory := NewKunstructuredFactoryImpl()
	testConfigMap := factory.FromMap(
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "winnie",
			},
		})
	testList := factory.FromMap(
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "List",
			"items": []interface{}{
				testConfigMap.Map(),
				testConfigMap.Map(),
			},
		})
	testConfigMapList := factory.FromMap(
		map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMapList",
			"items": []interface{}{
				testConfigMap.Map(),
				testConfigMap.Map(),
			},
		})

	tests := []struct {
		name        string
		input       []byte
		expectedOut []ifc.Kunstructured
		expectedErr bool
	}{
		{
			name:        "garbage",
			input:       []byte("garbageIn: garbageOut"),
			expectedOut: []ifc.Kunstructured{},
			expectedErr: true,
		},
		{
			name:        "noBytes",
			input:       []byte{},
			expectedOut: []ifc.Kunstructured{},
			expectedErr: false,
		},
		{
			name: "goodJson",
			input: []byte(`
{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"winnie"}}
`),
			expectedOut: []ifc.Kunstructured{testConfigMap},
			expectedErr: false,
		},
		{
			name: "goodYaml1",
			input: []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: winnie
`),
			expectedOut: []ifc.Kunstructured{testConfigMap},
			expectedErr: false,
		},
		{
			name: "goodYaml2",
			input: []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: winnie
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: winnie
`),
			expectedOut: []ifc.Kunstructured{testConfigMap, testConfigMap},
			expectedErr: false,
		},
		{
			name: "garbageInOneOfTwoObjects",
			input: []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: winnie
---
WOOOOOOOOOOOOOOOOOOOOOOOOT:  woot
`),
			expectedOut: []ifc.Kunstructured{},
			expectedErr: true,
		},
		{
			name: "emptyObjects",
			input: []byte(`
---
#a comment

---

`),
			expectedOut: []ifc.Kunstructured{},
			expectedErr: false,
		},
		{
			name: "Missing .metadata.name in object",
			input: []byte(`
apiVersion: v1
kind: Namespace
metadata:
  annotations:
    foo: bar
`),
			expectedOut: nil,
			expectedErr: true,
		},
		{
			name: "List",
			input: []byte(`
apiVersion: v1
kind: List
items:
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: winnie
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: winnie
`),
			expectedOut: []ifc.Kunstructured{testList},
			expectedErr: false,
		},
		{
			name: "ConfigMapList",
			input: []byte(`
apiVersion: v1
kind: ConfigMapList
items:
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: winnie
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: winnie
`),
			expectedOut: []ifc.Kunstructured{testConfigMapList},
			expectedErr: false,
		},
	}

	for _, test := range tests {
		rs, err := factory.SliceFromBytes(test.input)
		if test.expectedErr && err == nil {
			t.Fatalf("%v: should return error", test.name)
		}
		if !test.expectedErr && err != nil {
			t.Fatalf("%v: unexpected error: %s", test.name, err)
		}
		if len(rs) != len(test.expectedOut) {
			t.Fatalf("%s: length mismatch %d != %d",
				test.name, len(rs), len(test.expectedOut))
		}
		for i := range rs {
			if !reflect.DeepEqual(test.expectedOut[i], rs[i]) {
				t.Fatalf("%s: Got: %v\nexpected:%v",
					test.name, test.expectedOut[i], rs[i])
			}
		}
	}
}
