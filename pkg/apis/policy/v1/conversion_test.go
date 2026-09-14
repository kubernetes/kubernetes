/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"reflect"
	"testing"

	"k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/policy"
	internalfuzzer "k8s.io/kubernetes/pkg/apis/policy/fuzzer"
	"math/rand"
)

func TestEvictionMemoryIdenticalConversion(t *testing.T) {

	scheme := runtime.NewScheme()
	if err := policy.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	codecs := serializer.NewCodecFactory(scheme)
	f := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, internalfuzzer.Funcs), rand.NewSource(1), codecs).NilChance(0).NumElements(1, 1)
	t.Run("v1 to internal", func(t *testing.T) {
		in := &v1.Eviction{}
		f.Fill(in)
		out := &policy.Eviction{}
		if err := scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "Eviction", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
	t.Run("internal to v1", func(t *testing.T) {
		in := &policy.Eviction{}
		f.Fill(in)
		out := &v1.Eviction{}
		if err := scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "Eviction", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
}

func TestPodDisruptionBudgetMemoryIdenticalConversion(t *testing.T) {

	scheme := runtime.NewScheme()
	if err := policy.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	codecs := serializer.NewCodecFactory(scheme)
	f := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, internalfuzzer.Funcs), rand.NewSource(1), codecs).NilChance(0).NumElements(1, 1)
	t.Run("v1 to internal", func(t *testing.T) {
		in := &v1.PodDisruptionBudget{}
		f.Fill(in)
		out := &policy.PodDisruptionBudget{}
		if err := scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "PodDisruptionBudget", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
	t.Run("internal to v1", func(t *testing.T) {
		in := &policy.PodDisruptionBudget{}
		f.Fill(in)
		out := &v1.PodDisruptionBudget{}
		if err := scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "PodDisruptionBudget", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
}

func assertMemoryIdentical(t *testing.T, path string, a, b reflect.Value) {
	t.Helper()
	if a.Kind() != b.Kind() {
		t.Errorf("%s: unexpected kind mismatch: %s, %s", path, a.Kind(), b.Kind())
		return
	}
	switch a.Kind() {
	case reflect.Struct: // allow for copied structs since conversion copies status/spec today
		if a.Type().Size() != b.Type().Size() {
			t.Errorf("%s: unexpected struct size mismatch: %d, %d", path, a.Type().Size(), b.Type().Size())
			return
		}
		if a.NumField() != b.NumField() {
			t.Errorf("%s: unexpected field count mismatch: %d, %d", path, a.NumField(), b.NumField())
			return
		}
		for i := 0; i < a.NumField(); i++ {
			aTypeField := a.Type().Field(i)
			bTypeField := b.Type().Field(i)
			if aTypeField.Name != bTypeField.Name {
				t.Errorf("%s: unexpected field name mismatch: %s, %s", path, aTypeField.Name, bTypeField.Name)
			}
			if aTypeField.Offset != bTypeField.Offset {
				t.Errorf("%s.%s: unexpected field offset mismatch: %d, %d", path, aTypeField.Name, aTypeField.Offset, bTypeField.Offset)
			}
			assertMemoryIdentical(t, path+"."+aTypeField.Name, a.Field(i), b.Field(i))
		}
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if a.IsNil() != b.IsNil() {
			t.Errorf("%s: nilable pointer mismatch: %v, %v", path, !a.IsNil(), !b.IsNil())
			return
		}
		if !a.IsNil() && a.UnsafePointer() != b.UnsafePointer() {
			t.Errorf("%s: nilable type was unexpectedly copied", path)
		}
	case reflect.Bool, reflect.Int, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.String:
		// Assume scalars are copied by value
	default:
		t.Errorf("%s: unexpected kind: %v", path, a.Kind())
	}
}
