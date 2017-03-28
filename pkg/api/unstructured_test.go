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

package api_test

import (
	"math/rand"
	"reflect"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/v1"

	"k8s.io/apimachinery/pkg/conversion/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"

	"github.com/google/gofuzz"
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
	apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(seed)).
		// We are explicitly overwriting custom fuzzing functions, to ensure
		// that InitContainers and their statuses are not generated. This is
		// because in thise test we are simply doing json operations, in which
		// those disappear.
		Funcs(
			func(s *api.PodSpec, c fuzz.Continue) {
				c.FuzzNoCustom(s)
				s.InitContainers = nil
			},
			func(s *api.PodStatus, c fuzz.Continue) {
				c.FuzzNoCustom(s)
				s.InitContainerStatuses = nil
			},
		).Fuzz(internalObj)

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
	if !apiequality.Semantic.DeepEqual(item, unmarshalledObj) {
		t.Errorf("Object changed during JSON operations, diff: %v", diff.ObjectReflectDiff(item, unmarshalledObj))
		return
	}

	newUnstr := make(map[string]interface{})
	err = unstructured.DefaultConverter.ToUnstructured(item, &newUnstr)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}

	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = unstructured.DefaultConverter.FromUnstructured(newUnstr, newObj)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}

	if !apiequality.Semantic.DeepEqual(item, newObj) {
		t.Errorf("Object changed, diff: %v", diff.ObjectReflectDiff(item, newObj))
	}
}

func TestRoundTrip(t *testing.T) {
	for groupKey, group := range testapi.Groups {
		for kind := range group.ExternalTypes() {
			if nonRoundTrippableTypes.Has(kind) {
				continue
			}
			t.Logf("Testing: %v in %v", kind, groupKey)
			for i := 0; i < 50; i++ {
				doRoundTrip(t, group, kind)
				if t.Failed() {
					break
				}
			}
		}
	}
}

func BenchmarkToFromUnstructured(b *testing.B) {
	items := benchmarkItems(b)
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unstr := map[string]interface{}{}
		if err := unstructured.DefaultConverter.ToUnstructured(&items[i%size], &unstr); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		obj := v1.Pod{}
		if err := unstructured.DefaultConverter.FromUnstructured(unstr, &obj); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkToFromUnstructuredViaJSON(b *testing.B) {
	items := benchmarkItems(b)
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(&items[i%size])
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		unstr := map[string]interface{}{}
		if err := json.Unmarshal(data, &unstr); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		data, err = json.Marshal(unstr)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		obj := v1.Pod{}
		if err := json.Unmarshal(data, &obj); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}
