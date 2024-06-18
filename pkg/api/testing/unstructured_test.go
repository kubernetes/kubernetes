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

package testing

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metaunstruct "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	cborserializer "k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	jsonserializer "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func doRoundTrip(t *testing.T, internalVersion schema.GroupVersion, externalVersion schema.GroupVersion, kind string) {
	// We do fuzzing on the internal version of the object, and only then
	// convert to the external version. This is because custom fuzzing
	// function are only supported for internal objects.
	internalObj, err := legacyscheme.Scheme.New(internalVersion.WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create internal object %v: %v", kind, err)
	}
	seed := rand.Int63()
	fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs).
		// We are explicitly overwriting custom fuzzing functions, to ensure
		// that InitContainers and their statuses are not generated. This is
		// because in this test we are simply doing json operations, in which
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

	item, err := legacyscheme.Scheme.New(externalVersion.WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create external object %v: %v", kind, err)
	}
	if err := legacyscheme.Scheme.Convert(internalObj, item, nil); err != nil {
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
		t.Errorf("Object changed during JSON operations, diff: %v", cmp.Diff(item, unmarshalledObj))
		return
	}

	newUnstr, err := runtime.DefaultUnstructuredConverter.ToUnstructured(item)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}

	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(newUnstr, newObj)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}

	if !apiequality.Semantic.DeepEqual(item, newObj) {
		t.Errorf("Object changed, diff: %v", cmp.Diff(item, newObj))
	}
}

func TestRoundTrip(t *testing.T) {
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if nonRoundTrippableTypes.Has(gvk.Kind) {
			continue
		}
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		t.Logf("Testing: %v in %v", gvk.Kind, gvk.GroupVersion().String())
		for i := 0; i < 50; i++ {
			doRoundTrip(t, schema.GroupVersion{Group: gvk.Group, Version: runtime.APIVersionInternal}, gvk.GroupVersion(), gvk.Kind)
			if t.Failed() {
				break
			}
		}
	}
}

// TestRoundtripToUnstructured verifies the roundtrip faithfulness of all external types from native
// to unstructured and back using both the JSON and CBOR serializers. The intermediate unstructured
// objects produced by both encodings must be identical and be themselves roundtrippable to JSON and
// CBOR.
func TestRoundtripToUnstructured(t *testing.T) {
	// These are GVKs that whose CBOR roundtrippability is blocked by a known issue that must be
	// resolved as a prerequisite for alpha.
	knownFailureReasons := map[string][]schema.GroupVersionKind{
		// If a RawExtension's bytes are invalid JSON, its containing object can't be encoded to JSON.
		"rawextension needs to work in programs that assume json": {
			{Version: "v1", Kind: "List"},
			{Group: "apps", Version: "v1beta1", Kind: "ControllerRevision"},
			{Group: "apps", Version: "v1beta1", Kind: "ControllerRevisionList"},
			{Group: "apps", Version: "v1beta2", Kind: "ControllerRevision"},
			{Group: "apps", Version: "v1beta2", Kind: "ControllerRevisionList"},
			{Group: "apps", Version: "v1", Kind: "ControllerRevision"},
			{Group: "apps", Version: "v1", Kind: "ControllerRevisionList"},
			{Group: "admission.k8s.io", Version: "v1beta1", Kind: "AdmissionReview"},
			{Group: "admission.k8s.io", Version: "v1", Kind: "AdmissionReview"},
			{Group: "resource.k8s.io", Version: "v1alpha3", Kind: "ResourceClaim"},
			{Group: "resource.k8s.io", Version: "v1alpha3", Kind: "ResourceClaimList"},
		},
	}

	seed := int64(time.Now().Nanosecond())
	if override := os.Getenv("TEST_RAND_SEED"); len(override) > 0 {
		overrideSeed, err := strconv.ParseInt(override, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		seed = overrideSeed
		t.Logf("using overridden seed: %d", seed)
	} else {
		t.Logf("seed (override with TEST_RAND_SEED if desired): %d", seed)
	}

	var buf bytes.Buffer
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if nonRoundTrippableTypes.Has(gvk.Kind) {
			continue
		}
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}

		subtestName := fmt.Sprintf("%s.%s/%s", gvk.Version, gvk.Group, gvk.Kind)
		if gvk.Group == "" {
			subtestName = fmt.Sprintf("%s/%s", gvk.Version, gvk.Kind)
		}

		t.Run(subtestName, func(t *testing.T) {
			for reason, gvks := range knownFailureReasons {
				for _, each := range gvks {
					if gvk == each {
						t.Skip(reason)
					}
				}
			}

			fuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs)

			for i := 0; i < 50; i++ {
				// We do fuzzing on the internal version of the object, and only then
				// convert to the external version. This is because custom fuzzing
				// function are only supported for internal objects.
				internalObj, err := legacyscheme.Scheme.New(schema.GroupVersion{Group: gvk.Group, Version: runtime.APIVersionInternal}.WithKind(gvk.Kind))
				if err != nil {
					t.Fatalf("couldn't create internal object %v: %v", gvk.Kind, err)
				}
				fuzzer.Fuzz(internalObj)

				item, err := legacyscheme.Scheme.New(gvk)
				if err != nil {
					t.Fatalf("couldn't create external object %v: %v", gvk.Kind, err)
				}
				if err := legacyscheme.Scheme.Convert(internalObj, item, nil); err != nil {
					t.Fatalf("conversion for %v failed: %v", gvk.Kind, err)
				}

				// Decoding into Unstructured requires that apiVersion and kind be
				// serialized, so populate TypeMeta.
				item.GetObjectKind().SetGroupVersionKind(gvk)

				jsonSerializer := jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, legacyscheme.Scheme, legacyscheme.Scheme, jsonserializer.SerializerOptions{})
				cborSerializer := cborserializer.NewSerializer(legacyscheme.Scheme, legacyscheme.Scheme)

				// original->JSON->Unstructured
				buf.Reset()
				if err := jsonSerializer.Encode(item, &buf); err != nil {
					t.Fatalf("error encoding native to json: %v", err)
				}
				var uJSON runtime.Object = &metaunstruct.Unstructured{}
				uJSON, _, err = jsonSerializer.Decode(buf.Bytes(), &gvk, uJSON)
				if err != nil {
					t.Fatalf("error decoding json to unstructured: %v", err)
				}

				// original->CBOR->Unstructured
				buf.Reset()
				if err := cborSerializer.Encode(item, &buf); err != nil {
					t.Fatalf("error encoding native to cbor: %v", err)
				}
				var uCBOR runtime.Object = &metaunstruct.Unstructured{}
				uCBOR, _, err = cborSerializer.Decode(buf.Bytes(), &gvk, uCBOR)
				if err != nil {
					diag, _ := cbor.Diagnose(buf.Bytes())
					t.Fatalf("error decoding cbor to unstructured: %v, diag: %s", err, diag)
				}

				// original->JSON->Unstructured == original->CBOR->Unstructured
				if !apiequality.Semantic.DeepEqual(uJSON, uCBOR) {
					t.Fatalf("unstructured via json differed from unstructured via cbor: %v", cmp.Diff(uJSON, uCBOR))
				}

				// original->JSON/CBOR->Unstructured == original->JSON/CBOR->Unstructured->JSON->Unstructured
				buf.Reset()
				if err := jsonSerializer.Encode(uJSON, &buf); err != nil {
					t.Fatalf("error encoding unstructured to json: %v", err)
				}
				var uJSON2 runtime.Object = &metaunstruct.Unstructured{}
				uJSON2, _, err = jsonSerializer.Decode(buf.Bytes(), &gvk, uJSON2)
				if err != nil {
					t.Fatalf("error decoding json to unstructured: %v", err)
				}
				if !apiequality.Semantic.DeepEqual(uJSON, uJSON2) {
					t.Errorf("object changed during native-json-unstructured-json-unstructured roundtrip, diff: %s", cmp.Diff(uJSON, uJSON2))
				}

				// original->JSON/CBOR->Unstructured == original->JSON/CBOR->Unstructured->CBOR->Unstructured
				buf.Reset()
				if err := cborSerializer.Encode(uCBOR, &buf); err != nil {
					t.Fatalf("error encoding unstructured to cbor: %v", err)
				}
				var uCBOR2 runtime.Object = &metaunstruct.Unstructured{}
				uCBOR2, _, err = cborSerializer.Decode(buf.Bytes(), &gvk, uCBOR2)
				if err != nil {
					diag, _ := cbor.Diagnose(buf.Bytes())
					t.Fatalf("error decoding cbor to unstructured: %v, diag: %s", err, diag)
				}
				if !apiequality.Semantic.DeepEqual(uCBOR, uCBOR2) {
					t.Errorf("object changed during native-cbor-unstructured-cbor-unstructured roundtrip, diff: %s", cmp.Diff(uCBOR, uCBOR2))
				}

				// original->JSON/CBOR->Unstructured->JSON->final == original
				buf.Reset()
				if err := jsonSerializer.Encode(uJSON, &buf); err != nil {
					t.Fatalf("error encoding unstructured to json: %v", err)
				}
				finalJSON, _, err := jsonSerializer.Decode(buf.Bytes(), &gvk, nil)
				if err != nil {
					t.Fatalf("error decoding json to native: %v", err)
				}
				if !apiequality.Semantic.DeepEqual(item, finalJSON) {
					t.Errorf("object changed during native-json-unstructured-json-native roundtrip, diff: %s", cmp.Diff(item, finalJSON))
				}

				// original->JSON/CBOR->Unstructured->CBOR->final == original
				buf.Reset()
				if err := cborSerializer.Encode(uCBOR, &buf); err != nil {
					t.Fatalf("error encoding unstructured to cbor: %v", err)
				}
				finalCBOR, _, err := cborSerializer.Decode(buf.Bytes(), &gvk, nil)
				if err != nil {
					diag, _ := cbor.Diagnose(buf.Bytes())
					t.Fatalf("error decoding cbor to native: %v, diag: %s", err, diag)
				}
				if !apiequality.Semantic.DeepEqual(item, finalCBOR) {
					t.Errorf("object changed during native-cbor-unstructured-cbor-native roundtrip, diff: %s", cmp.Diff(item, finalCBOR))
				}
			}
		})
	}
}

func TestRoundTripWithEmptyCreationTimestamp(t *testing.T) {
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if nonRoundTrippableTypes.Has(gvk.Kind) {
			continue
		}
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}

		item, err := legacyscheme.Scheme.New(gvk)
		if err != nil {
			t.Fatalf("Couldn't create external object %v: %v", gvk, err)
		}
		t.Logf("Testing: %v in %v", gvk.Kind, gvk.GroupVersion().String())

		unstrBody, err := runtime.DefaultUnstructuredConverter.ToUnstructured(item)
		if err != nil {
			t.Fatalf("ToUnstructured failed: %v", err)
		}

		unstructObj := &metaunstruct.Unstructured{}
		unstructObj.Object = unstrBody

		if meta, err := meta.Accessor(unstructObj); err == nil {
			meta.SetCreationTimestamp(metav1.Time{})
		} else {
			t.Fatalf("Unable to set creation timestamp: %v", err)
		}

		// attempt to re-convert unstructured object - conversion should not fail
		// based on empty metadata fields, such as creationTimestamp
		newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(unstructObj.Object, newObj)
		if err != nil {
			t.Fatalf("FromUnstructured failed: %v", err)
		}
	}
}

func BenchmarkToUnstructured(b *testing.B) {
	items := benchmarkItems(b)
	size := len(items)
	convertor := runtime.DefaultUnstructuredConverter
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unstr, err := convertor.ToUnstructured(&items[i%size])
		if err != nil || unstr == nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkFromUnstructured(b *testing.B) {
	items := benchmarkItems(b)
	convertor := runtime.DefaultUnstructuredConverter
	var unstr []map[string]interface{}
	for i := range items {
		item, err := convertor.ToUnstructured(&items[i])
		if err != nil || item == nil {
			b.Fatalf("unexpected error: %v", err)
		}
		unstr = append(unstr, item)
	}
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := convertor.FromUnstructured(unstr[i%size], &obj); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkToUnstructuredViaJSON(b *testing.B) {
	items := benchmarkItems(b)
	var data [][]byte
	for i := range items {
		item, err := json.Marshal(&items[i])
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		data = append(data, item)
	}
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unstr := map[string]interface{}{}
		if err := json.Unmarshal(data[i%size], &unstr); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkFromUnstructuredViaJSON(b *testing.B) {
	items := benchmarkItems(b)
	var unstr []map[string]interface{}
	for i := range items {
		data, err := json.Marshal(&items[i])
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		item := map[string]interface{}{}
		if err := json.Unmarshal(data, &item); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		unstr = append(unstr, item)
	}
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		item, err := json.Marshal(unstr[i%size])
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		obj := v1.Pod{}
		if err := json.Unmarshal(item, &obj); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}
