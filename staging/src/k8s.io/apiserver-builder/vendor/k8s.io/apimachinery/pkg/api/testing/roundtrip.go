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
	"encoding/hex"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/protobuf/proto"
	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
)

type InstallFunc func(groupFactoryRegistry announced.APIGroupFactoryRegistry, registry *registered.APIRegistrationManager, scheme *runtime.Scheme)

// RoundTripTestForAPIGroup is convenient to call from your install package to make sure that a "bare" install of your group provides
// enough information to round trip
func RoundTripTestForAPIGroup(t *testing.T, installFn InstallFunc, fuzzingFuncs []interface{}) {
	groupFactoryRegistry := make(announced.APIGroupFactoryRegistry)
	registry := registered.NewOrDie("")
	scheme := runtime.NewScheme()
	installFn(groupFactoryRegistry, registry, scheme)

	RoundTripTestForScheme(t, scheme, fuzzingFuncs)
}

// RoundTripTestForScheme is convenient to call if you already have a scheme and want to make sure that its well-formed
func RoundTripTestForScheme(t *testing.T, scheme *runtime.Scheme, fuzzingFuncs []interface{}) {
	codecFactory := runtimeserializer.NewCodecFactory(scheme)
	fuzzer := DefaultFuzzers(t, codecFactory, fuzzingFuncs)
	RoundTripTypesWithoutProtobuf(t, scheme, codecFactory, fuzzer, nil)
}

// RoundTripProtobufTestForAPIGroup is convenient to call from your install package to make sure that a "bare" install of your group provides
// enough information to round trip
func RoundTripProtobufTestForAPIGroup(t *testing.T, installFn InstallFunc, fuzzingFuncs []interface{}) {
	groupFactoryRegistry := make(announced.APIGroupFactoryRegistry)
	registry := registered.NewOrDie("")
	scheme := runtime.NewScheme()
	installFn(groupFactoryRegistry, registry, scheme)

	RoundTripProtobufTestForScheme(t, scheme, fuzzingFuncs)
}

// RoundTripProtobufTestForScheme is convenient to call if you already have a scheme and want to make sure that its well-formed
func RoundTripProtobufTestForScheme(t *testing.T, scheme *runtime.Scheme, fuzzingFuncs []interface{}) {
	codecFactory := runtimeserializer.NewCodecFactory(scheme)
	fuzzer := DefaultFuzzers(t, codecFactory, fuzzingFuncs)
	RoundTripTypes(t, scheme, codecFactory, fuzzer, nil)
}

var FuzzIters = flag.Int("fuzz-iters", 20, "How many fuzzing iterations to do.")

// globalNonRoundTrippableTypes are kinds that are effectively reserved across all GroupVersions
// They don't roundtrip
var globalNonRoundTrippableTypes = sets.NewString(
	"ExportOptions",
	"GetOptions",
	// WatchEvent does not include kind and version and can only be deserialized
	// implicitly (if the caller expects the specific object). The watch call defines
	// the schema by content type, rather than via kind/version included in each
	// object.
	"WatchEvent",
	// ListOptions is now part of the meta group
	"ListOptions",
	// Delete options is only read in metav1
	"DeleteOptions",
)

// RoundTripTypesWithoutProtobuf applies the round-trip test to all round-trippable Kinds
// in the scheme.  It will skip all the GroupVersionKinds in the skip list.
func RoundTripTypesWithoutProtobuf(t *testing.T, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool) {
	roundTripTypes(t, scheme, codecFactory, fuzzer, nonRoundTrippableTypes, true)
}

func RoundTripTypes(t *testing.T, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool) {
	roundTripTypes(t, scheme, codecFactory, fuzzer, nonRoundTrippableTypes, false)
}

func roundTripTypes(t *testing.T, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool, skipProtobuf bool) {
	for _, group := range groupsFromScheme(scheme) {
		t.Logf("starting group %q", group)
		internalVersion := schema.GroupVersion{Group: group, Version: runtime.APIVersionInternal}
		internalKindToGoType := scheme.KnownTypes(internalVersion)

		for kind := range internalKindToGoType {
			if globalNonRoundTrippableTypes.Has(kind) {
				continue
			}

			internalGVK := internalVersion.WithKind(kind)
			roundTripSpecificKind(t, internalGVK, scheme, codecFactory, fuzzer, nonRoundTrippableTypes, skipProtobuf)
		}

		t.Logf("finished group %q", group)
	}
}

func RoundTripSpecificKindWithoutProtobuf(t *testing.T, internalGVK schema.GroupVersionKind, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool) {
	roundTripSpecificKind(t, internalGVK, scheme, codecFactory, fuzzer, nonRoundTrippableTypes, true)
}

func RoundTripSpecificKind(t *testing.T, internalGVK schema.GroupVersionKind, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool) {
	roundTripSpecificKind(t, internalGVK, scheme, codecFactory, fuzzer, nonRoundTrippableTypes, false)
}

func roundTripSpecificKind(t *testing.T, internalGVK schema.GroupVersionKind, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, nonRoundTrippableTypes map[schema.GroupVersionKind]bool, skipProtobuf bool) {
	if nonRoundTrippableTypes[internalGVK] {
		t.Logf("skipping %v", internalGVK)
		return
	}
	t.Logf("round tripping %v", internalGVK)

	// Try a few times, since runTest uses random values.
	for i := 0; i < *FuzzIters; i++ {
		roundTripToAllExternalVersions(t, scheme, codecFactory, fuzzer, internalGVK, nonRoundTrippableTypes, skipProtobuf)
		if t.Failed() {
			break
		}
	}
}

// fuzzInternalObject fuzzes an arbitrary runtime object using the appropriate
// fuzzer registered with the apitesting package.
func fuzzInternalObject(t *testing.T, fuzzer *fuzz.Fuzzer, object runtime.Object) runtime.Object {
	fuzzer.Fuzz(object)

	j, err := meta.TypeAccessor(object)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, object)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return object
}

func groupsFromScheme(scheme *runtime.Scheme) []string {
	ret := sets.String{}
	for gvk := range scheme.AllKnownTypes() {
		ret.Insert(gvk.Group)
	}
	return ret.List()
}

func roundTripToAllExternalVersions(t *testing.T, scheme *runtime.Scheme, codecFactory runtimeserializer.CodecFactory, fuzzer *fuzz.Fuzzer, internalGVK schema.GroupVersionKind, nonRoundTrippableTypes map[schema.GroupVersionKind]bool, skipProtobuf bool) {
	object, err := scheme.New(internalGVK)
	if err != nil {
		t.Fatalf("Couldn't make a %v? %v", internalGVK, err)
	}
	if _, err := meta.TypeAccessor(object); err != nil {
		t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableInternalTypes: %v", internalGVK, err)
	}

	fuzzInternalObject(t, fuzzer, object)

	// find all potential serializations in the scheme.
	// TODO fix this up to handle kinds that cross registered with different names.
	for externalGVK, externalGoType := range scheme.AllKnownTypes() {
		if externalGVK.Version == runtime.APIVersionInternal {
			continue
		}
		if externalGVK.GroupKind() != internalGVK.GroupKind() {
			continue
		}
		if nonRoundTrippableTypes[externalGVK] {
			t.Logf("\tskipping  %v %v", externalGVK, externalGoType)
			continue
		}
		t.Logf("\tround tripping to %v %v", externalGVK, externalGoType)

		roundTrip(t, scheme, TestCodec(codecFactory, externalGVK.GroupVersion()), object)

		// TODO remove this hack after we're past the intermediate steps
		if !skipProtobuf && externalGVK.Group != "kubeadm.k8s.io" {
			s := protobuf.NewSerializer(scheme, scheme, "application/arbitrary.content.type")
			protobufCodec := codecFactory.CodecForVersions(s, s, externalGVK.GroupVersion(), nil)
			roundTrip(t, scheme, protobufCodec, object)
		}
	}
}

// roundTrip applies a single round-trip test to the given runtime object
// using the given codec.  The round-trip test ensures that an object can be
// deep-copied and converted from internal -> versioned -> internal without
// loss of data.
func roundTrip(t *testing.T, scheme *runtime.Scheme, codec runtime.Codec, object runtime.Object) {
	printer := spew.ConfigState{DisableMethods: true}
	original := object

	// deep copy the original object
	copied, err := scheme.DeepCopy(object)
	if err != nil {
		panic(fmt.Sprintf("unable to copy: %v", err))
	}
	object = copied.(runtime.Object)
	name := reflect.TypeOf(object).Elem().Name()

	// encode (serialize) the deep copy using the provided codec
	data, err := runtime.Encode(codec, object)
	if err != nil {
		if runtime.IsNotRegisteredError(err) {
			t.Logf("%v: not registered: %v (%s)", name, err, printer.Sprintf("%#v", object))
		} else {
			t.Errorf("%v: %v (%s)", name, err, printer.Sprintf("%#v", object))
		}
		return
	}

	// ensure that the deep copy is equal to the original; neither the deep
	// copy or conversion should alter the object
	// TODO eliminate this global
	if !apiequality.Semantic.DeepEqual(original, object) {
		t.Errorf("0: %v: encode altered the object, diff: %v", name, diff.ObjectReflectDiff(original, object))
		return
	}

	// decode (deserialize) the encoded data back into an object
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("0: %v: %v\nCodec: %#v\nData: %s\nSource: %#v", name, err, codec, dataAsString(data), printer.Sprintf("%#v", object))
		panic("failed")
	}

	// ensure that the object produced from decoding the encoded data is equal
	// to the original object
	if !apiequality.Semantic.DeepEqual(original, obj2) {
		t.Errorf("1: %v: diff: %v\nCodec: %#v\nSource:\n\n%#v\n\nEncoded:\n\n%s\n\nFinal:\n\n%#v", name, diff.ObjectReflectDiff(object, obj2), codec, printer.Sprintf("%#v", object), dataAsString(data), printer.Sprintf("%#v", obj2))
		return
	}

	// decode the encoded data into a new object (instead of letting the codec
	// create a new object)
	obj3 := reflect.New(reflect.TypeOf(object).Elem()).Interface().(runtime.Object)
	if err := runtime.DecodeInto(codec, data, obj3); err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}

	// ensure that the new runtime object is equal to the original after being
	// decoded into
	if !apiequality.Semantic.DeepEqual(object, obj3) {
		t.Errorf("3: %v: diff: %v\nCodec: %#v", name, diff.ObjectReflectDiff(object, obj3), codec)
		return
	}

	// do structure-preserving fuzzing of the deep-copied object. If it shares anything with the original,
	// the deep-copy was actually only a shallow copy. Then original and obj3 will be different after fuzzing.
	// NOTE: we use the encoding+decoding here as an alternative, guaranteed deep-copy to compare against.
	ValueFuzz(object)
	if !apiequality.Semantic.DeepEqual(original, obj3) {
		t.Errorf("0: %v: fuzzing a copy altered the original, diff: %v", name, diff.ObjectReflectDiff(original, object))
		return
	}
}

// dataAsString returns the given byte array as a string; handles detecting
// protocol buffers.
func dataAsString(data []byte) string {
	dataString := string(data)
	if !strings.HasPrefix(dataString, "{") {
		dataString = "\n" + hex.Dump(data)
		proto.NewBuffer(make([]byte, 0, 1024)).DebugPrint("decoded object", data)
	}
	return dataString
}
