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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"

	"github.com/ghodss/yaml"
	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

type testMetaFactory struct{}

func (testMetaFactory) Interpret(data []byte) (*unversioned.GroupVersionKind, error) {
	findKind := struct {
		APIVersion string `json:"myVersionKey,omitempty"`
		ObjectKind string `json:"myKindKey,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	if err := yaml.Unmarshal(data, &findKind); err != nil {
		return nil, fmt.Errorf("couldn't get version/kind: %v", err)
	}
	gv, err := unversioned.ParseGroupVersion(findKind.APIVersion)
	if err != nil {
		return nil, err
	}
	return &unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: findKind.ObjectKind}, nil
}

// Test a weird version/kind embedding format.
type MyWeirdCustomEmbeddedVersionKindField struct {
	ID         string `json:"ID,omitempty"`
	APIVersion string `json:"myVersionKey,omitempty"`
	ObjectKind string `json:"myKindKey,omitempty"`
	Z          string `json:"Z,omitempty"`
	Y          uint64 `json:"Y,omitempty"`
}

type TestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string               `json:"A,omitempty"`
	B                                     int                  `json:"B,omitempty"`
	C                                     int8                 `json:"C,omitempty"`
	D                                     int16                `json:"D,omitempty"`
	E                                     int32                `json:"E,omitempty"`
	F                                     int64                `json:"F,omitempty"`
	G                                     uint                 `json:"G,omitempty"`
	H                                     uint8                `json:"H,omitempty"`
	I                                     uint16               `json:"I,omitempty"`
	J                                     uint32               `json:"J,omitempty"`
	K                                     uint64               `json:"K,omitempty"`
	L                                     bool                 `json:"L,omitempty"`
	M                                     map[string]int       `json:"M,omitempty"`
	N                                     map[string]TestType2 `json:"N,omitempty"`
	O                                     *TestType2           `json:"O,omitempty"`
	P                                     []TestType2          `json:"Q,omitempty"`
}

type TestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

type ExternalTestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}
type ExternalTestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string                       `json:"A,omitempty"`
	B                                     int                          `json:"B,omitempty"`
	C                                     int8                         `json:"C,omitempty"`
	D                                     int16                        `json:"D,omitempty"`
	E                                     int32                        `json:"E,omitempty"`
	F                                     int64                        `json:"F,omitempty"`
	G                                     uint                         `json:"G,omitempty"`
	H                                     uint8                        `json:"H,omitempty"`
	I                                     uint16                       `json:"I,omitempty"`
	J                                     uint32                       `json:"J,omitempty"`
	K                                     uint64                       `json:"K,omitempty"`
	L                                     bool                         `json:"L,omitempty"`
	M                                     map[string]int               `json:"M,omitempty"`
	N                                     map[string]ExternalTestType2 `json:"N,omitempty"`
	O                                     *ExternalTestType2           `json:"O,omitempty"`
	P                                     []ExternalTestType2          `json:"Q,omitempty"`
}

type ExternalInternalSame struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     TestType2 `json:"A,omitempty"`
}

// TestObjectFuzzer can randomly populate all the above objects.
var TestObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 100).Funcs(
	func(j *MyWeirdCustomEmbeddedVersionKindField, c fuzz.Continue) {
		c.FuzzNoCustom(j)
		j.APIVersion = ""
		j.ObjectKind = ""
	},
)

func (obj *MyWeirdCustomEmbeddedVersionKindField) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *MyWeirdCustomEmbeddedVersionKindField) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.ObjectKind = gvk.ToAPIVersionAndKind()
}
func (obj *MyWeirdCustomEmbeddedVersionKindField) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.ObjectKind)
}

func (obj *ExternalInternalSame) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType1) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *ExternalTestType1) GetObjectKind() unversioned.ObjectKind {
	return &obj.MyWeirdCustomEmbeddedVersionKindField
}

func (obj *TestType2) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }
func (obj *ExternalTestType2) GetObjectKind() unversioned.ObjectKind {
	return unversioned.EmptyObjectKind
}

// Returns a new Scheme set up with the test objects.
func GetTestScheme() (*runtime.Scheme, runtime.Codec) {
	internalGV := unversioned.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Version: "v1"}
	externalGV2 := unversioned.GroupVersion{Version: "v2"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &TestType1{}, &TestType2{}, &ExternalInternalSame{})
	s.AddKnownTypes(externalGV, &ExternalInternalSame{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType2"), &ExternalTestType2{})
	s.AddKnownTypeWithName(internalGV.WithKind("TestType3"), &TestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType3"), &ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV2.WithKind("TestType1"), &ExternalTestType1{})

	s.AddUnversionedTypes(externalGV, &unversioned.Status{})

	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}))
	codec := cf.LegacyCodec(unversioned.GroupVersion{Version: "v1"})
	return s, codec
}

func objDiff(a, b interface{}) string {
	ab, err := json.Marshal(a)
	if err != nil {
		panic("a")
	}
	bb, err := json.Marshal(b)
	if err != nil {
		panic("b")
	}
	return diff.StringDiff(string(ab), string(bb))

	// An alternate diff attempt, in case json isn't showing you
	// the difference. (reflect.DeepEqual makes a distinction between
	// nil and empty slices, for example.)
	//return diff.StringDiff(
	//  fmt.Sprintf("%#v", a),
	//  fmt.Sprintf("%#v", b),
	//)
}

var semantic = conversion.EqualitiesOrDie(
	func(a, b MyWeirdCustomEmbeddedVersionKindField) bool {
		a.APIVersion, a.ObjectKind = "", ""
		b.APIVersion, b.ObjectKind = "", ""
		return a == b
	},
)

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	TestObjectFuzzer.Fuzz(source)

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
		t.Errorf("3: %v: diff: %v", name, objDiff(source, obj3))
		return
	}
}

func TestTypes(t *testing.T) {
	table := []interface{}{
		&TestType1{},
		&ExternalInternalSame{},
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
	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}))
	encoder, _ := cf.SerializerForFileExtension("json")

	codec := cf.CodecForVersions(encoder, nil, unversioned.GroupVersion{Version: "v2"}, nil)
	out, err := runtime.Encode(codec, &TestType1{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"myVersionKey":"v2","myKindKey":"TestType1"}`+"\n" {
		t.Fatal(string(out))
	}

	codec = cf.CodecForVersions(encoder, nil, unversioned.GroupVersion{Version: "v3"}, nil)
	_, err = runtime.Encode(codec, &TestType1{})
	if err == nil {
		t.Fatal(err)
	}

	// unversioned encode with no versions is written directly to wire
	codec = cf.CodecForVersions(encoder, nil, runtime.InternalGroupVersioner, nil)
	out, err = runtime.Encode(codec, &TestType1{})
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
	internal := obj.(*TestType1)
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

func TestConvertTypesWhenDefaultNamesMatch(t *testing.T) {
	internalGV := unversioned.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Version: "v1"}

	s := runtime.NewScheme()
	// create two names internally, with TestType1 being preferred
	s.AddKnownTypeWithName(internalGV.WithKind("TestType1"), &TestType1{})
	s.AddKnownTypeWithName(internalGV.WithKind("OtherType1"), &TestType1{})
	// create two names externally, with TestType1 being preferred
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("OtherType1"), &ExternalTestType1{})

	ext := &ExternalTestType1{}
	ext.APIVersion = "v1"
	ext.ObjectKind = "OtherType1"
	ext.A = "test"
	data, err := json.Marshal(ext)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expect := &TestType1{A: "test"}

	codec := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{})).LegacyCodec(unversioned.GroupVersion{Version: "v1"})

	obj, err := runtime.Decode(codec, data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !semantic.DeepEqual(expect, obj) {
		t.Errorf("unexpected object: %#v", obj)
	}

	into := &TestType1{}
	if err := runtime.DecodeInto(codec, data, into); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !semantic.DeepEqual(expect, into) {
		t.Errorf("unexpected object: %#v", obj)
	}
}

func TestEncode_Ptr(t *testing.T) {
	_, codec := GetTestScheme()
	tt := &TestType1{A: "I am a pointer object"}
	data, err := runtime.Encode(codec, tt)
	obj2, err2 := runtime.Decode(codec, data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'\n%s", err, err2, data)
	}
	if _, ok := obj2.(*TestType1); !ok {
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
	if err := runtime.DecodeInto(codec, badJSONKindMismatch, &TestType1{}); err == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}
	if err := runtime.DecodeInto(codec, []byte(``), &TestType1{}); err != nil {
		t.Errorf("Should allow empty decode: %v", err)
	}
	if _, _, err := codec.Decode([]byte(``), &unversioned.GroupVersionKind{Kind: "ExternalInternalSame"}, nil); err == nil {
		t.Errorf("Did not give error for empty data with only kind default")
	}
	if _, _, err := codec.Decode([]byte(`{"myVersionKey":"v1"}`), &unversioned.GroupVersionKind{Kind: "ExternalInternalSame"}, nil); err != nil {
		t.Errorf("Gave error for version and kind default")
	}
	if _, _, err := codec.Decode([]byte(`{"myKindKey":"ExternalInternalSame"}`), &unversioned.GroupVersionKind{Version: "v1"}, nil); err != nil {
		t.Errorf("Gave error for version and kind default")
	}
	if _, _, err := codec.Decode([]byte(``), &unversioned.GroupVersionKind{Kind: "ExternalInternalSame", Version: "v1"}, nil); err != nil {
		t.Errorf("Gave error for version and kind defaulted: %v", err)
	}
	if _, err := runtime.Decode(codec, []byte(``)); err == nil {
		t.Errorf("Did not give error for empty data")
	}
}

// Returns a new Scheme set up with the test objects needed by TestDirectCodec.
func GetDirectCodecTestScheme() *runtime.Scheme {
	internalGV := unversioned.GroupVersion{Version: runtime.APIVersionInternal}
	externalGV := unversioned.GroupVersion{Version: "v1"}

	s := runtime.NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &TestType1{})
	s.AddKnownTypes(externalGV, &ExternalTestType1{})

	s.AddUnversionedTypes(externalGV, &unversioned.Status{})
	return s
}

func TestDirectCodec(t *testing.T) {
	s := GetDirectCodecTestScheme()
	cf := newCodecFactory(s, newSerializersForScheme(s, testMetaFactory{}))
	serializer, _ := cf.SerializerForFileExtension("json")
	df := DirectCodecFactory{cf}
	ignoredGV, err := unversioned.ParseGroupVersion("ignored group/ignored version")
	if err != nil {
		t.Fatal(err)
	}
	directEncoder := df.EncoderForVersion(serializer, ignoredGV)
	directDecoder := df.DecoderToVersion(serializer, ignoredGV)
	out, err := runtime.Encode(directEncoder, &ExternalTestType1{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"myVersionKey":"v1","myKindKey":"ExternalTestType1"}`+"\n" {
		t.Fatal(string(out))
	}
	a, _, err := directDecoder.Decode(out, nil, nil)
	e := &ExternalTestType1{
		MyWeirdCustomEmbeddedVersionKindField: MyWeirdCustomEmbeddedVersionKindField{
			APIVersion: "v1",
			ObjectKind: "ExternalTestType1",
		},
	}
	if !semantic.DeepEqual(e, a) {
		t.Fatalf("expect %v, got %v", e, a)
	}
}
