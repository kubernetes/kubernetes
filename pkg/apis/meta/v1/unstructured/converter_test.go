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

package unstructured_test

import (
	"math/rand"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/json"
	"k8s.io/kubernetes/pkg/util/sets"
)

func doRoundTrip(t *testing.T, group testapi.TestGroup, kind string) {
	// We do fuzzing on the internal version of the object, and only then
	// convert to the external version. This is because custom fuzzing
	// function are only supported for internal objects.
	internalObj, err := api.Scheme.New(group.InternalGroupVersion().WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create internal object %v: %v", kind, err)
	}
	seed := rand.Int63()
	apitesting.FuzzerFor(t, group.InternalGroupVersion(), rand.NewSource(seed)).Fuzz(internalObj)

	item, err := api.Scheme.New(group.GroupVersion().WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create external object %v: %v", kind, err)
	}
	if err := api.Scheme.Convert(internalObj, item, nil); err != nil {
		t.Fatalf("Conversion for %v failed: %v", kind, err)
	}

	data, err := json.Marshal(item)
	if err != nil {
		t.Errorf("Error when marshaling object: %v", err)
		return
	}
	unstr := make(map[string]interface{})
	err = json.Unmarshal(data, &unstr)
	if err != nil {
		t.Errorf("Error when unmarshaling to unstructured: %v", err)
		return
	}

	data, err = json.Marshal(unstr)
	if err != nil {
		t.Errorf("Error when marshaling unstructured: %v", err)
		return
	}
	unmarshalledObj := reflect.New(reflect.TypeOf(item).Elem()).Interface()
	err = json.Unmarshal(data, &unmarshalledObj)
	if err != nil {
		t.Errorf("Error when unmarshaling to object: %v", err)
		return
	}
	if !api.Semantic.DeepEqual(item, unmarshalledObj) {
		t.Errorf("Object changed during JSON operations, diff: %v", diff.ObjectReflectDiff(item, unmarshalledObj))
	}

	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = unstructured.NewConverter().FromUnstructured(unstr, newObj)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}

	if !api.Semantic.DeepEqual(item, newObj) {
		t.Errorf("Object changed, diff: %v", diff.ObjectReflectDiff(item, newObj))
	}
}

var nonRoundTrippableTypes = sets.NewString(
	"ExportOptions",
	"GetOptions",
	// WatchEvent does not include kind and version and can only be deserialized
	// implicitly (if the caller expects the specific object). The watch call defines
	// the schema by content type, rather than via kind/version included in each
	// object
	"WatchEvent",
)

func TestRoundTrip(t *testing.T) {
	for groupKey, group := range testapi.Groups {
		for kind := range group.ExternalTypes() {
			if nonRoundTrippableTypes.Has(kind) {
				continue
			}
			t.Logf("Testing: %v in %v", kind, groupKey)
			for i := 0; i < 20; i++ {
				doRoundTrip(t, group, kind)
				if t.Failed() {
					break
				}
			}
		}
	}
}
