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
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util"

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
	internalGV := unversioned.GroupVersion{Version: "__internal"}
	externalGV := unversioned.GroupVersion{Version: "v1"}

	s := NewScheme()
	// Ordinarily, we wouldn't add TestType2, but because this is a test and
	// both types are from the same package, we need to get it into the system
	// so that converter will match it with ExternalType2.
	s.AddKnownTypes(internalGV, &TestType1{}, &TestType2{}, &ExternalInternalSame{})
	s.AddKnownTypes(externalGV, &ExternalInternalSame{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType1"), &ExternalTestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType2"), &ExternalTestType2{})
	s.AddKnownTypeWithName(internalGV.WithKind("TestType3"), &TestType1{})
	s.AddKnownTypeWithName(externalGV.WithKind("TestType3"), &ExternalTestType1{})
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
	//return util.StringDiff(
	//	fmt.Sprintf("%#v", a),
	//	fmt.Sprintf("%#v", b),
	//)
}

func TestKnownTypes(t *testing.T) {
	s := GetTestScheme()
	if len(s.KnownTypes(unversioned.GroupVersion{Group: "group", Version: "v2"})) != 0 {
		t.Errorf("should have no known types for v2")
	}

	types := s.KnownTypes(unversioned.GroupVersion{Version: "v1"})
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
