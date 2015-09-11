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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"

	_ "k8s.io/kubernetes/pkg/apis/experimental"
	_ "k8s.io/kubernetes/pkg/apis/experimental/v1"

	flag "github.com/spf13/pflag"
)

var fuzzIters = flag.Int("fuzz-iters", 20, "How many fuzzing iterations to do.")

func fuzzInternalObject(t *testing.T, forVersion string, item runtime.Object, seed int64) runtime.Object {
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
	data, err := codec.Encode(item)
	if err != nil {
		t.Errorf("%v: %v (%s)", name, err, printer.Sprintf("%#v", item))
		return
	}

	obj2, err := codec.Decode(data)
	if err != nil {
		t.Errorf("0: %v: %v\nCodec: %v\nData: %s\nSource: %#v", name, err, codec, string(data), printer.Sprintf("%#v", item))
		return
	}
	if !api.Semantic.DeepEqual(item, obj2) {
		t.Errorf("1: %v: diff: %v\nCodec: %v\nSource:\n\n%#v\n\nEncoded:\n\n%s\n\nFinal:\n\n%#v", name, util.ObjectGoPrintDiff(item, obj2), codec, printer.Sprintf("%#v", item), string(data), printer.Sprintf("%#v", obj2))
		return
	}

	obj3 := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = codec.DecodeInto(data, obj3)
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
	fuzzInternalObject(t, "", item, seed)
	version := testapi.Default.Version()
	if !set.Has(version) {
		fuzzInternalObject(t, version, item, seed)
		roundTrip(t, testapi.Default.Codec(), item)
	}
}

// For debugging problems
func TestSpecificKind(t *testing.T) {
	api.Scheme.Log(t)
	defer api.Scheme.Log(nil)

	kind := "PodList"
	doRoundTripTest(kind, t)
}

func TestList(t *testing.T) {
	api.Scheme.Log(t)
	defer api.Scheme.Log(nil)

	kind := "List"
	item, err := api.Scheme.New("", kind)
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	roundTripSame(t, item)
}

var nonRoundTrippableTypes = sets.NewString()
var nonInternalRoundTrippableTypes = sets.NewString("List", "ListOptions", "PodExecOptions", "PodAttachOptions")
var nonRoundTrippableTypesByVersion = map[string][]string{}

func TestRoundTripTypes(t *testing.T) {
	// api.Scheme.Log(t)
	// defer api.Scheme.Log(nil)

	for kind := range api.Scheme.KnownTypes("") {
		if nonRoundTrippableTypes.Has(kind) {
			continue
		}
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			doRoundTripTest(kind, t)
		}
	}
}

func doRoundTripTest(kind string, t *testing.T) {
	item, err := api.Scheme.New("", kind)
	if err != nil {
		t.Fatalf("Couldn't make a %v? %v", kind, err)
	}
	if _, err := meta.TypeAccessor(item); err != nil {
		t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableTypes: %v", kind, err)
	}
	roundTripSame(t, item, nonRoundTrippableTypesByVersion[kind]...)
	if !nonInternalRoundTrippableTypes.Has(kind) {
		roundTrip(t, api.Codec, fuzzInternalObject(t, "", item, rand.Int63()))
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
		},
	}
	obj := runtime.Object(pod)
	data, err := latest.Codec.Encode(obj)
	obj2, err2 := latest.Codec.Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !api.Semantic.DeepEqual(obj2, pod) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", pod, obj2)
	}
}

func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := latest.Codec.Decode(badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := latest.Codec.Decode(badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}

const benchmarkSeed = 100

func BenchmarkEncode(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	for i := 0; i < b.N; i++ {
		latest.Codec.Encode(&pod)
	}
}

// BenchmarkEncodeJSON provides a baseline for regular JSON encode performance
func BenchmarkEncodeJSON(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	for i := 0; i < b.N; i++ {
		json.Marshal(&pod)
	}
}

func BenchmarkDecode(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		latest.Codec.Decode(data)
	}
}

func BenchmarkDecodeInto(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		latest.Codec.DecodeInto(data, &obj)
	}
}

// BenchmarkDecodeJSON provides a baseline for regular JSON decode performance
func BenchmarkDecodeJSON(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		json.Unmarshal(data, &obj)
	}
}
