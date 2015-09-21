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

package conversion

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util"

	"github.com/ghodss/yaml"
	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

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
		// We have to customize the randomization of MyWeirdCustomEmbeddedVersionKindFields because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.ObjectKind = ""
		j.ID = c.RandString()
	},
)

// Returns a new Scheme set up with the test objects.
func GetTestScheme() *Scheme {
	s := NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes("", &TestType1{}, &TestType2{}, &ExternalInternalSame{})
	s.AddKnownTypes("v1", &ExternalInternalSame{})
	s.AddKnownTypeWithName("v1", "TestType1", &ExternalTestType1{})
	s.AddKnownTypeWithName("v1", "TestType2", &ExternalTestType2{})
	s.AddKnownTypeWithName("", "TestType3", &TestType1{})
	s.AddKnownTypeWithName("v1", "TestType3", &ExternalTestType1{})
	s.InternalVersion = ""
	s.MetaFactory = testMetaFactory{}
	return s
}

type testMetaFactory struct{}

func (testMetaFactory) Interpret(data []byte) (version, kind string, err error) {
	findKind := struct {
		APIVersion string `json:"myVersionKey,omitempty"`
		ObjectKind string `json:"myKindKey,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, &findKind)
	if err != nil {
		return "", "", fmt.Errorf("couldn't get version/kind: %v", err)
	}
	return findKind.APIVersion, findKind.ObjectKind, nil
}

func (testMetaFactory) Update(version, kind string, obj interface{}) error {
	return UpdateVersionAndKind(nil, "APIVersion", version, "ObjectKind", kind, obj)
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
	return util.StringDiff(string(ab), string(bb))

	// An alternate diff attempt, in case json isn't showing you
	// the difference. (reflect.DeepEqual makes a distinction between
	// nil and empty slices, for example.)
	//return util.StringDiff(
	//	fmt.Sprintf("%#v", a),
	//	fmt.Sprintf("%#v", b),
	//)
}

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	TestObjectFuzzer.Fuzz(source)

	s := GetTestScheme()
	data, err := s.EncodeToVersion(source, "v1")
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}
	obj2, err := s.Decode(data)
	if err != nil {
		t.Errorf("%v: %v (%v)", name, err, string(data))
		return
	}
	if !reflect.DeepEqual(source, obj2) {
		t.Errorf("1: %v: diff: %v", name, objDiff(source, obj2))
		return
	}
	obj3 := reflect.New(reflect.TypeOf(source).Elem()).Interface()
	err = s.DecodeInto(data, obj3)
	if err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !reflect.DeepEqual(source, obj3) {
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

func TestMultipleNames(t *testing.T) {
	s := GetTestScheme()

	obj, err := s.Decode([]byte(`{"myKindKey":"TestType3","myVersionKey":"v1","A":"value"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	internal := obj.(*TestType1)
	if internal.A != "value" {
		t.Fatalf("unexpected decoded object: %#v", internal)
	}

	out, err := s.EncodeToVersion(internal, "v1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(string(out), `"myKindKey":"TestType1"`) {
		t.Errorf("unexpected encoded output: %s", string(out))
	}
}

func TestConvertTypesWhenDefaultNamesMatch(t *testing.T) {
	s := NewScheme()
	// create two names internally, with TestType1 being preferred
	s.AddKnownTypeWithName("", "TestType1", &TestType1{})
	s.AddKnownTypeWithName("", "OtherType1", &TestType1{})
	// create two names externally, with TestType1 being preferred
	s.AddKnownTypeWithName("v1", "TestType1", &ExternalTestType1{})
	s.AddKnownTypeWithName("v1", "OtherType1", &ExternalTestType1{})
	s.MetaFactory = testMetaFactory{}

	ext := &ExternalTestType1{}
	ext.APIVersion = "v1"
	ext.ObjectKind = "OtherType1"
	ext.A = "test"
	data, err := json.Marshal(ext)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expect := &TestType1{A: "test"}

	obj, err := s.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(expect, obj) {
		t.Errorf("unexpected object: %#v", obj)
	}

	into := &TestType1{}
	if err := s.DecodeInto(data, into); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(expect, obj) {
		t.Errorf("unexpected object: %#v", obj)
	}
}

func TestKnownTypes(t *testing.T) {
	s := GetTestScheme()
	if len(s.KnownTypes("v2")) != 0 {
		t.Errorf("should have no known types for v2")
	}

	types := s.KnownTypes("v1")
	for _, s := range []string{"TestType1", "TestType2", "TestType3", "ExternalInternalSame"} {
		if _, ok := types[s]; !ok {
			t.Errorf("missing type %q", s)
		}
	}
}

func TestConvertToVersion(t *testing.T) {
	s := GetTestScheme()
	tt := &TestType1{A: "I'm not a pointer object"}
	other, err := s.ConvertToVersion(tt, "v1")
	if err != nil {
		t.Fatalf("Failure: %v", err)
	}
	converted, ok := other.(*ExternalTestType1)
	if !ok {
		t.Fatalf("Got wrong type")
	}
	if tt.A != converted.A {
		t.Fatalf("Failed to convert object correctly: %#v", converted)
	}
}

func TestConvertToVersionErr(t *testing.T) {
	s := GetTestScheme()
	tt := TestType1{A: "I'm not a pointer object"}
	_, err := s.ConvertToVersion(tt, "v1")
	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestEncode_NonPtr(t *testing.T) {
	s := GetTestScheme()
	tt := TestType1{A: "I'm not a pointer object"}
	obj := interface{}(tt)
	data, err := s.EncodeToVersion(obj, "v1")
	obj2, err2 := s.Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*TestType1); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, &tt) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &tt, obj2)
	}
}

func TestEncode_Ptr(t *testing.T) {
	s := GetTestScheme()
	tt := &TestType1{A: "I am a pointer object"}
	obj := interface{}(tt)
	data, err := s.EncodeToVersion(obj, "v1")
	obj2, err2 := s.Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*TestType1); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, tt) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &tt, obj2)
	}
}

func TestBadJSONRejection(t *testing.T) {
	s := GetTestScheme()
	badJSONs := [][]byte{
		[]byte(`{"myVersionKey":"v1"}`),                          // Missing kind
		[]byte(`{"myVersionKey":"v1","myKindKey":"bar"}`),        // Unknown kind
		[]byte(`{"myVersionKey":"bar","myKindKey":"TestType1"}`), // Unknown version
	}
	for _, b := range badJSONs {
		if _, err := s.Decode(b); err == nil {
			t.Errorf("Did not reject bad json: %s", string(b))
		}
	}
	badJSONKindMismatch := []byte(`{"myVersionKey":"v1","myKindKey":"ExternalInternalSame"}`)
	if err := s.DecodeInto(badJSONKindMismatch, &TestType1{}); err == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}
	if err := s.DecodeInto([]byte(``), &TestType1{}); err == nil {
		t.Errorf("Did not give error for empty data")
	}
}

func TestBadJSONRejectionForSetInternalVersion(t *testing.T) {
	s := GetTestScheme()
	s.InternalVersion = "v1"
	badJSONs := [][]byte{
		[]byte(`{"myKindKey":"TestType1"}`), // Missing version
	}
	for _, b := range badJSONs {
		if _, err := s.Decode(b); err == nil {
			t.Errorf("Did not reject bad json: %s", string(b))
		}
	}
	badJSONKindMismatch := []byte(`{"myVersionKey":"v1","myKindKey":"ExternalInternalSame"}`)
	if err := s.DecodeInto(badJSONKindMismatch, &TestType1{}); err == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}
}
