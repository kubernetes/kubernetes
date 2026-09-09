/*
Copyright The Kubernetes Authors.

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

package v1_test

import (
	"math/rand"
	"reflect"
	"testing"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/discovery"
	discoveryfuzzer "k8s.io/kubernetes/pkg/apis/discovery/fuzzer"
)

// TestEndpointSliceMemoryIdenticalConversion ensures the internal and v1 EndpointSlice types
// remain memory-identical. These types must be kept memory-identical
// for performance reasons.
func TestEndpointSliceMemoryIdenticalConversion(t *testing.T) {
	f := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, discoveryfuzzer.Funcs), rand.NewSource(1), legacyscheme.Codecs).NilChance(0).NumElements(1, 1)
	t.Run("v1 to internal", func(t *testing.T) {
		in := &discoveryv1.EndpointSlice{}
		f.Fill(in)
		out := &discovery.EndpointSlice{}
		if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "EndpointSlice", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
	t.Run("internal to v1", func(t *testing.T) {
		in := &discovery.EndpointSlice{}
		f.Fill(in)
		out := &discoveryv1.EndpointSlice{}
		if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "EndpointSlice", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
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
