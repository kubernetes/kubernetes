/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"encoding/json"
	"reflect"
	"testing"

	jsonpatch "github.com/evanphx/json-patch"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestJSONPatchForConfigMap(t *testing.T) {
	cm := corev1.ConfigMap{
		Data: map[string]string{
			"mutation-start": "yes",
		},
	}
	cmJS, err := json.Marshal(cm)
	if err != nil {
		t.Fatal(err)
	}

	patchObj, err := jsonpatch.DecodePatch([]byte(patch1))
	if err != nil {
		t.Fatal(err)
	}
	patchedJS, err := patchObj.Apply(cmJS)
	patchedObj := corev1.ConfigMap{}
	err = json.Unmarshal(patchedJS, &patchedObj)
	if err != nil {
		t.Fatal(err)
	}
	expected := corev1.ConfigMap{
		Data: map[string]string{
			"mutation-start":   "yes",
			"mutation-stage-1": "yes",
		},
	}

	if !reflect.DeepEqual(patchedObj, expected) {
		t.Errorf("\nexpected %#v\n, got %#v", expected, patchedObj)
	}
}

func TestJSONPatchForUnstructured(t *testing.T) {
	cr := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "Something",
			"apiVersion": "somegroup/v1",
			"data": map[string]interface{}{
				"mutation-start": "yes",
			},
		},
	}
	crJS, err := json.Marshal(cr)
	if err != nil {
		t.Fatal(err)
	}

	patchObj, err := jsonpatch.DecodePatch([]byte(patch1))
	if err != nil {
		t.Fatal(err)
	}
	patchedJS, err := patchObj.Apply(crJS)
	patchedObj := unstructured.Unstructured{}
	err = json.Unmarshal(patchedJS, &patchedObj)
	if err != nil {
		t.Fatal(err)
	}
	expectedData := map[string]interface{}{
		"mutation-start":   "yes",
		"mutation-stage-1": "yes",
	}

	if !reflect.DeepEqual(patchedObj.Object["data"], expectedData) {
		t.Errorf("\nexpected %#v\n, got %#v", expectedData, patchedObj.Object["data"])
	}
}
