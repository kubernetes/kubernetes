/*
Copyright 2014 Google Inc. All rights reserved.

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
	"flag"
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var fuzzIters = flag.Int("fuzz_iters", 10, "How many fuzzing iterations to do.")

// Intended to be compatible with the default implementation of MetaInsertionFactory
type JSONBase struct {
	ID         string `yaml:"ID,omitempty" json:"ID,omitempty"`
	APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
}

type TestType1 struct {
	JSONBase `json:",inline" yaml:",inline"`
	A        string               `yaml:"A,omitempty" json:"A,omitempty"`
	B        int                  `yaml:"B,omitempty" json:"B,omitempty"`
	C        int8                 `yaml:"C,omitempty" json:"C,omitempty"`
	D        int16                `yaml:"D,omitempty" json:"D,omitempty"`
	E        int32                `yaml:"E,omitempty" json:"E,omitempty"`
	F        int64                `yaml:"F,omitempty" json:"F,omitempty"`
	G        uint                 `yaml:"G,omitempty" json:"G,omitempty"`
	H        uint8                `yaml:"H,omitempty" json:"H,omitempty"`
	I        uint16               `yaml:"I,omitempty" json:"I,omitempty"`
	J        uint32               `yaml:"J,omitempty" json:"J,omitempty"`
	K        uint64               `yaml:"K,omitempty" json:"K,omitempty"`
	L        bool                 `yaml:"L,omitempty" json:"L,omitempty"`
	M        map[string]int       `yaml:"M,omitempty" json:"M,omitempty"`
	N        map[string]TestType2 `yaml:"N,omitempty" json:"N,omitempty"`
	O        *TestType2           `yaml:"O,omitempty" json:"O,omitempty"`
	P        []TestType2          `yaml:"Q,omitempty" json:"Q,omitempty"`
}

type TestType2 struct {
	A string `yaml:"A,omitempty" json:"A,omitempty"`
	B int    `yaml:"B,omitempty" json:"B,omitempty"`
}

// We depend on the name of the external and internal types matching. Ordinarily,
// we'd accomplish this with an additional package, but since this is a test, we
// can just enclose stuff in a function to simulate that.
func externalTypeReturn() interface{} {
	type TestType2 struct {
		A string `yaml:"A,omitempty" json:"A,omitempty"`
		B int    `yaml:"B,omitempty" json:"B,omitempty"`
	}
	type TestType1 struct {
		JSONBase `json:",inline" yaml:",inline"`
		A        string               `yaml:"A,omitempty" json:"A,omitempty"`
		B        int                  `yaml:"B,omitempty" json:"B,omitempty"`
		C        int8                 `yaml:"C,omitempty" json:"C,omitempty"`
		D        int16                `yaml:"D,omitempty" json:"D,omitempty"`
		E        int32                `yaml:"E,omitempty" json:"E,omitempty"`
		F        int64                `yaml:"F,omitempty" json:"F,omitempty"`
		G        uint                 `yaml:"G,omitempty" json:"G,omitempty"`
		H        uint8                `yaml:"H,omitempty" json:"H,omitempty"`
		I        uint16               `yaml:"I,omitempty" json:"I,omitempty"`
		J        uint32               `yaml:"J,omitempty" json:"J,omitempty"`
		K        uint64               `yaml:"K,omitempty" json:"K,omitempty"`
		L        bool                 `yaml:"L,omitempty" json:"L,omitempty"`
		M        map[string]int       `yaml:"M,omitempty" json:"M,omitempty"`
		N        map[string]TestType2 `yaml:"N,omitempty" json:"N,omitempty"`
		O        *TestType2           `yaml:"O,omitempty" json:"O,omitempty"`
		P        []TestType2          `yaml:"Q,omitempty" json:"Q,omitempty"`
	}
	return TestType1{}
}

type ExternalInternalSame struct {
	JSONBase `json:",inline" yaml:",inline"`
	A        TestType2 `yaml:"A,omitempty" json:"A,omitempty"`
}

// TestObjectFuzzer can randomly populate all the above objects.
var TestObjectFuzzer = util.NewFuzzer(
	func(j *JSONBase) {
		// We have to customize the randomization of JSONBases because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
		j.ID = util.RandString()
	},
	func(u *uint64) {
		// TODO: Fix JSON/YAML packages and/or write custom encoding
		// for uint64's. Somehow the LS *byte* of this is lost, but
		// only when all 8 bytes are set.
		*u = util.RandUint64() >> 8
	},
	func(u *uint) {
		// TODO: Fix JSON/YAML packages and/or write custom encoding
		// for uint64's. Somehow the LS *byte* of this is lost, but
		// only when all 8 bytes are set.
		*u = uint(util.RandUint64() >> 8)
	},
)

// Returns a new Scheme set up with the test objects.
func GetTestScheme() *Scheme {
	s := NewScheme()
	s.AddKnownTypes("", TestType1{}, ExternalInternalSame{})
	s.AddKnownTypes("v1", externalTypeReturn(), ExternalInternalSame{})
	s.ExternalVersion = "v1"
	s.InternalVersion = ""
	return s
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
	return util.StringDiff(
		fmt.Sprintf("%#v", a),
		fmt.Sprintf("%#v", b),
	)
}

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	TestObjectFuzzer.Fuzz(source)

	s := GetTestScheme()
	data, err := s.Encode(source)
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}
	obj2, err := s.Decode(data)
	if err != nil {
		t.Errorf("%v: %v (%v)", name, err, string(data))
		return
	} else {
		if !reflect.DeepEqual(source, obj2) {
			t.Errorf("1: %v: diff: %v", name, objDiff(source, obj2))
			return
		}
	}
	obj3 := reflect.New(reflect.TypeOf(source).Elem()).Interface()
	err = s.DecodeInto(data, obj3)
	if err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	} else {
		if !reflect.DeepEqual(source, obj3) {
			t.Errorf("3: %v: diff: %v", name, objDiff(source, obj3))
			return
		}
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

func TestEncode_NonPtr(t *testing.T) {
	s := GetTestScheme()
	tt := TestType1{A: "I'm not a pointer object"}
	obj := interface{}(tt)
	data, err := s.Encode(obj)
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
	data, err := s.Encode(obj)
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
		[]byte(`{"apiVersion":"v1"}`),                     // Missing kind
		[]byte(`{"kind":"TestType1"}`),                    // Missing version
		[]byte(`{"apiVersion":"v1","kind":"bar"}`),        // Unknown kind
		[]byte(`{"apiVersion":"bar","kind":"TestType1"}`), // Unknown version
	}
	for _, b := range badJSONs {
		if _, err := s.Decode(b); err == nil {
			t.Errorf("Did not reject bad json: %s", string(b))
		}
	}
	badJSONKindMismatch := []byte(`{"apiVersion":"v1","kind": "ExternalInternalSame"}`)
	if err := s.DecodeInto(badJSONKindMismatch, &TestType1{}); err == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}
}
