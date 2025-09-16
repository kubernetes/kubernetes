/*
Copyright 2014 The Kubernetes Authors.

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

package serializer

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"github.com/google/go-cmp/cmp"
	flag "github.com/spf13/pflag"
	"sigs.k8s.io/randfill"
	"sigs.k8s.io/yaml"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

type testMetaFactory struct{}

func (testMetaFactory) Interpret(data []byte) (*schema.GroupVersionKind, error) {
	findKind := struct {
		APIVersion string `json:"myVersionKey,omitempty"`
		ObjectKind string `json:"myKindKey,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	if err := yaml.Unmarshal(data, &findKind); err != nil {
		return nil, fmt.Errorf("couldn't get version/kind: %v", err)
	}
	gv, err := schema.ParseGroupVersion(findKind.APIVersion)
	if err != nil {
		return nil, err
	}
	return &schema.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: findKind.ObjectKind}, nil
}

// TestObjectFuzzer can randomly populate all the above objects.
var TestObjectFuzzer = randfill.New().NilChance(.5).NumElements(1, 100).Funcs(
	func(j *runtimetesting.MyWeirdCustomEmbeddedVersionKindField, c randfill.Continue) {
		c.FillNoCustom(j)
		j.APIVersion = ""
		j.ObjectKind = ""
	},
)

// Returns a new Scheme set up with the test objects.
func GetTestScheme() (*runtime.Scheme, runtime.Codec) {
	internalGV := schema.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Version: "v1"}
	externalGV2 := schema.GroupVersion{Version: "v2"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &runtimetesting.TestType1{}, &runtimetesting.TestType2{}, &runtimetesting.ExternalInternalSame{})
	s.AddKnownTypes(externalGV, &runtimetesting.ExternalInternalSame{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType2"), &runtimetesting.ExternalTestType2{})
	s.AddKnownTypeWithName(internalGV.WithKind("TestType3"), &runtimetesting.TestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType3"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV2.WithKind("TestType1"), &runtimetesting.ExternalTestType1{})

	s.AddUnversionedTypes(externalGV, &metav1.Status{})

	utilruntime.Must(runtimetesting.RegisterConversions(s))

	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: true}))
	codec := cf.LegacyCodec(schema.GroupVersion{Version: "v1"})
	return s, codec
}

var semantic = conversion.EqualitiesOrDie(
	func(a, b runtimetesting.MyWeirdCustomEmbeddedVersionKindField) bool {
		a.APIVersion, a.ObjectKind = "", ""
		b.APIVersion, b.ObjectKind = "", ""
		return a == b
	},
)

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	TestObjectFuzzer.Fill(source)

	_, codec := GetTestScheme()
	data, err := runtime.Encode(codec, source.(runtime.Object))
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v: %v (%v)", name, err, string(data))
		return
	}
	if !semantic.DeepEqual(source, obj2) {
		t.Errorf("1: %v: diff: %v", name, diff.ObjectGoPrintSideBySide(source, obj2))
		return
	}
	obj3 := reflect.New(reflect.TypeOf(source).Elem()).Interface()
	if err := runtime.DecodeInto(codec, data, obj3.(runtime.Object)); err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !semantic.DeepEqual(source, obj3) {
		t.Errorf("3: %v: diff: %v", name, cmp.Diff(source, obj3))
		return
	}
}

func TestTypes(t *testing.T) {
	table := []interface{}{
		&runtimetesting.TestType1{},
		&runtimetesting.ExternalInternalSame{},
	}
	for _, item := range table {
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			runTest(t, item)
		}
	}
}

func TestVersionedEncoding(t *testing.T) {
	s, _ := GetTestScheme()
	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: true}))
	info, _ := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := info.Serializer

	codec := cf.EncoderForVersion(encoder, schema.GroupVersion{Version: "v2"})
	out, err := runtime.Encode(codec, &runtimetesting.TestType1{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"myVersionKey":"v2","myKindKey":"TestType1"}`+"\n" {
		t.Fatal(string(out))
	}

	codec = cf.EncoderForVersion(encoder, schema.GroupVersion{Version: "v3"})
	_, err = runtime.Encode(codec, &runtimetesting.TestType1{})
	if err == nil {
		t.Fatal(err)
	}

	// unversioned encode with no versions is written directly to wire
	codec = cf.EncoderForVersion(encoder, runtime.InternalGroupVersioner)
	out, err = runtime.Encode(codec, &runtimetesting.TestType1{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{}`+"\n" {
		t.Fatal(string(out))
	}
}

func TestMultipleNames(t *testing.T) {
	_, codec := GetTestScheme()

	obj, _, err := codec.Decode([]byte(`{"myKindKey":"TestType3","myVersionKey":"v1","A":"value"}`), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	internal := obj.(*runtimetesting.TestType1)
	if internal.A != "value" {
		t.Fatalf("unexpected decoded object: %#v", internal)
	}

	out, err := runtime.Encode(codec, internal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(string(out), `"myKindKey":"TestType1"`) {
		t.Errorf("unexpected encoded output: %s", string(out))
	}
}

func TestStrictOption(t *testing.T) {
	s, _ := GetTestScheme()
	duplicateKeys := `{"myKindKey":"TestType3","myVersionKey":"v1","myVersionKey":"v1","A":"value"}`

	strictCodec := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: true})).LegacyCodec()
	_, _, err := strictCodec.Decode([]byte(duplicateKeys), nil, nil)
	if !runtime.IsStrictDecodingError(err) {
		t.Fatalf("StrictDecodingError not returned on object with duplicate keys: %v, type: %v", err, reflect.TypeOf(err))
	}

	nonStrictCodec := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: false})).LegacyCodec()
	_, _, err = nonStrictCodec.Decode([]byte(duplicateKeys), nil, nil)
	if runtime.IsStrictDecodingError(err) {
		t.Fatalf("Non-Strict decoder returned a StrictDecodingError: %v", err)
	}
}

func TestConvertTypesWhenDefaultNamesMatch(t *testing.T) {
	internalGV := schema.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Version: "v1"}

	s := runtime.NewScheme()
	// create two names internally, with TestType1 being preferred
	s.AddKnownTypeWithName(internalGV.WithKind("TestType1"), &runtimetesting.TestType1{})
	s.AddKnownTypeWithName(internalGV.WithKind("OtherType1"), &runtimetesting.TestType1{})
	// create two names externally, with TestType1 being preferred
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &runtimetesting.ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("OtherType1"), &runtimetesting.ExternalTestType1{})
	if err := runtimetesting.RegisterConversions(s); err != nil {
		t.Fatalf("unexpected error; %v", err)
	}

	ext := &runtimetesting.ExternalTestType1{}
	ext.APIVersion = "v1"
	ext.ObjectKind = "OtherType1"
	ext.A = "test"
	data, err := json.Marshal(ext)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expect := &runtimetesting.TestType1{A: "test"}

	codec := newCodecFactory(
		s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: true}),
	).LegacyCodec(schema.GroupVersion{Version: "v1"})

	obj, err := runtime.Decode(codec, data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !semantic.DeepEqual(expect, obj) {
		t.Errorf("unexpected object: %#v", obj)
	}

	into := &runtimetesting.TestType1{}
	if err := runtime.DecodeInto(codec, data, into); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !semantic.DeepEqual(expect, into) {
		t.Errorf("unexpected object: %#v", obj)
	}
}

func TestEncode_Ptr(t *testing.T) {
	_, codec := GetTestScheme()
	tt := &runtimetesting.TestType1{A: "I am a pointer object"}
	data, err := runtime.Encode(codec, tt)
	obj2, err2 := runtime.Decode(codec, data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'\n%s", err, err2, data)
	}
	if _, ok := obj2.(*runtimetesting.TestType1); !ok {
		t.Fatalf("Got wrong type")
	}
	if !semantic.DeepEqual(obj2, tt) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", tt, obj2)
	}
}

func TestBadJSONRejection(t *testing.T) {
	log.SetOutput(os.Stderr)
	_, codec := GetTestScheme()
	badJSONs := [][]byte{
		[]byte(`{"myVersionKey":"v1"}`),                          // Missing kind
		[]byte(`{"myVersionKey":"v1","myKindKey":"bar"}`),        // Unknown kind
		[]byte(`{"myVersionKey":"bar","myKindKey":"TestType1"}`), // Unknown version
		[]byte(`{"myKindKey":"TestType1"}`),                      // Missing version
	}
	for _, b := range badJSONs {
		if _, err := runtime.Decode(codec, b); err == nil {
			t.Errorf("Did not reject bad json: %s", string(b))
		}
	}
	badJSONKindMismatch := []byte(`{"myVersionKey":"v1","myKindKey":"ExternalInternalSame"}`)
	if err := runtime.DecodeInto(codec, badJSONKindMismatch, &runtimetesting.TestType1{}); err == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}
	if err := runtime.DecodeInto(codec, []byte(``), &runtimetesting.TestType1{}); err != nil {
		t.Errorf("Should allow empty decode: %v", err)
	}
	if _, _, err := codec.Decode([]byte(``), &schema.GroupVersionKind{Kind: "ExternalInternalSame"}, nil); err == nil {
		t.Errorf("Did not give error for empty data with only kind default")
	}
	if _, _, err := codec.Decode([]byte(`{"myVersionKey":"v1"}`), &schema.GroupVersionKind{Kind: "ExternalInternalSame"}, nil); err != nil {
		t.Errorf("Gave error for version and kind default")
	}
	if _, _, err := codec.Decode([]byte(`{"myKindKey":"ExternalInternalSame"}`), &schema.GroupVersionKind{Version: "v1"}, nil); err != nil {
		t.Errorf("Gave error for version and kind default")
	}
	if _, _, err := codec.Decode([]byte(``), &schema.GroupVersionKind{Kind: "ExternalInternalSame", Version: "v1"}, nil); err != nil {
		t.Errorf("Gave error for version and kind defaulted: %v", err)
	}
	if _, err := runtime.Decode(codec, []byte(``)); err == nil {
		t.Errorf("Did not give error for empty data")
	}
}

// Returns a new Scheme set up with the test objects needed by TestDirectCodec.
func GetDirectCodecTestScheme() *runtime.Scheme {
	internalGV := schema.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Version: "v1"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &runtimetesting.TestType1{})
	s.AddKnownTypes(externalGV, &runtimetesting.ExternalTestType1{})

	s.AddUnversionedTypes(externalGV, &metav1.Status{})

	utilruntime.Must(runtimetesting.RegisterConversions(s))
	return s
}

func TestDirectCodec(t *testing.T) {
	s := GetDirectCodecTestScheme()
	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}, CodecFactoryOptions{Pretty: true, Strict: true}))
	info, _ := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeJSON)
	serializer := info.Serializer
	df := cf.WithoutConversion()
	ignoredGV, err := schema.ParseGroupVersion("ignored group/ignored version")
	if err != nil {
		t.Fatal(err)
	}
	directEncoder := df.EncoderForVersion(serializer, ignoredGV)
	directDecoder := df.DecoderToVersion(serializer, ignoredGV)
	out, err := runtime.Encode(directEncoder, &runtimetesting.ExternalTestType1{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"myVersionKey":"v1","myKindKey":"ExternalTestType1"}`+"\n" {
		t.Fatal(string(out))
	}
	a, _, err := directDecoder.Decode(out, nil, nil)
	if err != nil {
		t.Fatalf("error on Decode: %v", err)
	}
	e := &runtimetesting.ExternalTestType1{
		MyWeirdCustomEmbeddedVersionKindField: runtimetesting.MyWeirdCustomEmbeddedVersionKindField{
			APIVersion: "v1",
			ObjectKind: "ExternalTestType1",
		},
	}
	if !semantic.DeepEqual(e, a) {
		t.Fatalf("expect %v, got %v", e, a)
	}
}
