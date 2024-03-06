/*
Copyright 2024 The Kubernetes Authors.

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

package json_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

// Illustrates the diff between decoding into UnstructuredList using strict
// (i.e. SetUnstructuredContent()) and non-strict (i.e. unstructuredJSONScheme.decodeToList())
// modes.
func TestDecodeUnstructuredList(t *testing.T) {
	const in = `{"apiVersion": "json.example.com/v1alpha1", "kind": "FooList", "items": [{"spec":{"abc": 123}}]}`

	strict := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, unstructuredscheme.NewUnstructuredObjectTyper(), json.SerializerOptions{Strict: true})
	var listFromStrict unstructured.UnstructuredList
	if _, _, err := strict.Decode([]byte(in), nil, &listFromStrict); err != nil {
		t.Fatal(err)
	}

	lax := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, unstructuredscheme.NewUnstructuredObjectTyper(), json.SerializerOptions{Strict: false})
	var listFromLax unstructured.UnstructuredList
	if _, _, err := lax.Decode([]byte(in), nil, &listFromLax); err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(listFromStrict, listFromLax); diff != "" {
		t.Errorf("diff strict lax:\n%s", diff)
	}

}
