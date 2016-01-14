/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"math/rand"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
	flag "github.com/spf13/pflag"
	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/extensions"
	_ "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

var fuzzIters = flag.Int("fuzz-iters", 20, "How many fuzzing iterations to do.")

var codecsToTest = []func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, error){
	func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, error) {
		return testapi.GetCodecForObject(item)
	},
}

func fuzzInternalObject(t *testing.T, forVersion unversioned.GroupVersion, item runtime.Object, seed int64) runtime.Object {
	apitesting.FuzzerFor(t, forVersion, rand.NewSource(seed)).Fuzz(item)

	j, err := meta.TypeAccessor(item)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, item)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return item
}

func roundTrip(t *testing.T, codec runtime.Codec, item runtime.Object) {
	printer := spew.ConfigState{DisableMethods: true}

	name := reflect.TypeOf(item).Elem().Name()
	data, err := runtime.Encode(codec, item)
	if err != nil {
		t.Errorf("%v: %v (%s)", name, err, printer.Sprintf("%#v", item))
		return
	}

	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("0: %v: %v\nCodec: %v\nData: %s\nSource: %#v", name, err, codec, string(data), printer.Sprintf("%#v", item))
		return
	}
	if !api.Semantic.DeepEqual(item, obj2) {
		t.Errorf("\n1: %v: diff: %v\nCodec: %v\nSource:\n\n%#v\n\nEncoded:\n\n%s\n\nFinal:\n\n%#v", name, util.ObjectGoPrintDiff(item, obj2), codec, printer.Sprintf("%#v", item), string(data), printer.Sprintf("%#v", obj2))
		return
	}

	obj3 := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = runtime.DecodeInto(codec, data, obj3)
	if err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !api.Semantic.DeepEqual(item, obj3) {
		t.Errorf("3: %v: diff: %v\nCodec: %v", name, util.ObjectDiff(item, obj3), codec)
		return
	}
}

// roundTripSame verifies the same source object is tested in all API versions.
func roundTripSame(t *testing.T, item runtime.Object, except ...string) {
	set := sets.NewString(except...)
	seed := rand.Int63()
	fuzzInternalObject(t, testapi.Default.InternalGroupVersion(), item, seed)

	version := *testapi.Default.GroupVersion()
	codecs := []runtime.Codec{}
	for _, fn := range codecsToTest {
		codec, err := fn(version, item)
		if err != nil {
			t.Errorf("unable to get codec: %v", err)
			return
		}
		codecs = append(codecs, codec)
	}

	if !set.Has(version.String()) {
		fuzzInternalObject(t, version, item, seed)
		for _, codec := range codecs {
			roundTrip(t, codec, item)
		}
	}
}

// For debugging problems
func TestSpecificKind(t *testing.T) {
	// api.Scheme.Log(t)
	// defer api.Scheme.Log(nil)

	kind := "Pod"
	for i := 0; i < *fuzzIters; i++ {
		doRoundTripTest(kind, t)
		if t.Failed() {
			break
		}
	}
}

func TestList(t *testing.T) {
	// api.Scheme.Log(t)
	// defer api.Scheme.Log(nil)

	kind := "List"
	item, err := api.Scheme.New(api.SchemeGroupVersion.WithKind(kind))
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	roundTripSame(t, item)
}

var nonRoundTrippableTypes = sets.NewString("ExportOptions")

var nonInternalRoundTrippableTypes = sets.NewString("List", "ListOptions", "ExportOptions")
var nonRoundTrippableTypesByVersion = map[string][]string{}

func TestRoundTripTypes(t *testing.T) {
	// api.Scheme.Log(t)
	// defer api.Scheme.Log(nil)

	for kind := range api.Scheme.KnownTypes(testapi.Default.InternalGroupVersion()) {
		t.Logf("working on %v in %v", kind, testapi.Default.InternalGroupVersion())
		if nonRoundTrippableTypes.Has(kind) {
			continue
		}
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			doRoundTripTest(kind, t)
			if t.Failed() {
				break
			}
		}
	}
}

func doRoundTripTest(kind string, t *testing.T) {
	item, err := api.Scheme.New(testapi.Default.InternalGroupVersion().WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't make a %v? %v", kind, err)
	}
	if _, err := meta.TypeAccessor(item); err != nil {
		t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableTypes: %v", kind, err)
	}
	if api.Scheme.Recognizes(testapi.Default.GroupVersion().WithKind(kind)) {
		roundTripSame(t, item, nonRoundTrippableTypesByVersion[kind]...)
	}
	if !nonInternalRoundTrippableTypes.Has(kind) {
		roundTrip(t, api.Codec, fuzzInternalObject(t, testapi.Default.InternalGroupVersion(), item, rand.Int63()))
	}
}

func TestEncode_Ptr(t *testing.T) {
	grace := int64(30)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,

			SecurityContext: &api.PodSecurityContext{},
		},
	}
	obj := runtime.Object(pod)
	data, err := testapi.Default.Codec().Encode(obj)
	obj2, err2 := testapi.Default.Codec().Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !api.Semantic.DeepEqual(obj2, pod) {
		t.Errorf("\nExpected:\n\n %#v,\n\nGot:\n\n %#vDiff: %v\n\n", pod, obj2, util.ObjectDiff(obj2, pod))

	}
}

func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := testapi.Default.Codec().Decode(badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := testapi.Default.Codec().Decode(badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}

func TestUnversionedTypes(t *testing.T) {
	testcases := []runtime.Object{
		&unversioned.Status{Status: "Failure", Message: "something went wrong"},
		&unversioned.APIVersions{Versions: []string{"A", "B", "C"}},
		&unversioned.APIGroupList{Groups: []unversioned.APIGroup{{Name: "mygroup"}}},
		&unversioned.APIGroup{Name: "mygroup"},
		&unversioned.APIResourceList{GroupVersion: "mygroup/myversion"},
	}

	for _, obj := range testcases {
		// Make sure the unversioned codec can encode
		unversionedJSON, err := api.Codec.Encode(obj)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}

		// Make sure the versioned codec under test can decode
		versionDecodedObject, err := testapi.Default.Codec().Decode(unversionedJSON)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}
		// Make sure it decodes correctly
		if !reflect.DeepEqual(obj, versionDecodedObject) {
			t.Errorf("%v: expected %#v, got %#v", obj, obj, versionDecodedObject)
			continue
		}
	}
}

const benchmarkSeed = 100

func benchmarkItems() []v1.Pod {
	apiObjectFuzzer := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
	items := make([]v1.Pod, 2)
	for i := range items {
		apiObjectFuzzer.Fuzz(&items[i])
	}
	return items
}

// BenchmarkEncodeCodec measures the cost of performing a codec encode, which includes
// reflection (to clear APIVersion and Kind)
func BenchmarkEncodeCodec(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := testapi.Default.Codec().Encode(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeJSONMarshal provides a baseline for regular JSON encode performance
func BenchmarkEncodeJSONMarshal(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := json.Marshal(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Decode(codec, encoded[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeIntoCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := runtime.DecodeInto(codec, encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeJSON provides a baseline for regular JSON decode performance
func BenchmarkDecodeIntoJSON(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := json.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeJSON provides a baseline for codecgen JSON decode performance
func BenchmarkDecodeIntoJSONCodecGen(b *testing.B) {
	kcodec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(kcodec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}
	handler := &codec.JsonHandle{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := codec.NewDecoderBytes(encoded[i%width], handler).Decode(&obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}
