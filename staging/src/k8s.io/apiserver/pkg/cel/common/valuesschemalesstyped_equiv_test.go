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

package common

import (
	"testing"

	"github.com/google/cel-go/common/types"
	"k8s.io/apimachinery/pkg/runtime"
)

// Define Equiv structures mirroring the k8s converter_test.go types.
type EquivA struct {
	A int    `json:"aa,omitempty"`
	B string `json:"ab,omitempty"`
	C bool   `json:"ac,omitempty"`
}

type EquivB struct {
	A EquivA            `json:"ba"`
	B string            `json:"bb"`
	C map[string]string `json:"bc"`
	D []string          `json:"bd"`
}

type EquivC struct {
	A      []EquivA `json:"ca"`
	EquivB `json:""`
	C      string         `json:"cc"`
	D      *int64         `json:"cd"`
	E      map[string]int `json:"ce"`
	F      []bool         `json:"cf"`
	G      []int          `json:"cg"`
	H      float32        `json:"ch"`
	I      []interface{}  `json:"ci"`
}

type EquivD struct {
	A []interface{} `json:"da"`
}

type EquivE struct {
	A interface{} `json:"ea"`
}

type EquivF struct {
	A string            `json:"fa"`
	B map[string]string `json:"fb"`
	C []EquivA          `json:"fc"`
	D int               `json:"fd"`
	E float32           `json:"fe"`
	F []string          `json:"ff"`
	G []int             `json:"fg"`
	H []bool            `json:"fh"`
	I []float32         `json:"fi"`
	J []byte            `json:"fj"`
}

type EquivG struct {
	CustomValue1   EquivCustomValue      `json:"customValue1"`
	CustomValue2   *EquivCustomValue     `json:"customValue2"`
	CustomPointer1 EquivCustomPointer    `json:"customPointer1"`
	CustomPointer2 *EquivCustomPointer   `json:"customPointer2"`
	RawExtension1  runtime.RawExtension  `json:"rawExtension1"`
	RawExtension2  *runtime.RawExtension `json:"rawExtension2"`
}

type EquivCustomValue struct {
	Data []byte
}

func (c EquivCustomValue) MarshalJSON() ([]byte, error) {
	if len(c.Data) == 0 {
		return []byte("null"), nil
	}
	return c.Data, nil
}

type EquivCustomPointer struct {
	Data []byte
}

func (c *EquivCustomPointer) MarshalJSON() ([]byte, error) {
	if len(c.Data) == 0 {
		return []byte("null"), nil
	}
	return c.Data, nil
}

type EquivInlineTestPrimitive struct {
	NoNameTagPrimitive          int64 `json:""`
	NoNameTagInlinePrimitive    int64 `json:""`
	NoNameTagOmitemptyPrimitive int64 `json:",omitempty"`
}

type EquivInlineTestAnonymous struct {
	EquivNoTag
	EquivNoNameTag          `json:""`
	EquivNameTag            `json:"nameTagEmbedded"`
	EquivNoNameTagInline    `json:""`
	EquivNoNameTagOmitempty `json:",omitempty"`
}

type EquivInlineTestNamed struct {
	NoTag              EquivNoTag
	NoNameTag          EquivNoNameTag          `json:""`
	NameTag            EquivNameTag            `json:"nameTagEmbedded"`
	NoNameTagInline    EquivNoNameTagInline    `json:""`
	NoNameTagOmitempty EquivNoNameTagOmitempty `json:",omitempty"`
}

type EquivNoTag struct {
	Data0 int `json:"data0"`
}

type EquivNameTag struct {
	Data1 int `json:"data1"`
}

type EquivNoNameTag struct {
	Data2 int `json:"data2"`
}

type EquivNoNameTagInline struct {
	Data3 int `json:"data3"`
}

type EquivNoNameTagOmitempty struct {
	Data4 int `json:"data4"`
}

func assertEquivalence(t *testing.T, val interface{}) {
	unstrMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(val)
	if err != nil {
		t.Fatalf("ToUnstructured failed: %v", err)
	}

	expectedCEL := types.DefaultTypeAdapter.NativeToValue(unstrMap)
	gotCEL := SchemalessTypedToVal(val)

	if gotCEL.Equal(expectedCEL) != types.True {
		t.Errorf("Equivalence mismatch!\nExpected: %v (Type: %T, CEL: %v)\nGot:      %v (Type: %T, CEL: %v)",
			unstrMap, expectedCEL, expectedCEL.Type(), gotCEL.Value(), gotCEL, gotCEL.Type())
	}
}

func TestEquivalence_StructA(t *testing.T) {
	assertEquivalence(t, &EquivA{
		A: 12,
		B: "hello",
		C: true,
	})
	// Zero/empty
	assertEquivalence(t, &EquivA{})
}

func TestEquivalence_StructB(t *testing.T) {
	assertEquivalence(t, &EquivB{
		A: EquivA{A: 1, B: "a", C: true},
		B: "str",
		C: map[string]string{"k": "v"},
		D: []string{"a", "b"},
	})
}

func TestEquivalence_StructC(t *testing.T) {
	cd := int64(12345)
	assertEquivalence(t, &EquivC{
		A: []EquivA{
			{A: 1, B: "a"},
			{A: 2, B: "b"},
		},
		EquivB: EquivB{
			A: EquivA{A: 3, B: "c"},
			B: "embedB",
		},
		C: "helloC",
		D: &cd,
		E: map[string]int{"a": 10, "b": 20},
		F: []bool{true, false},
		G: []int{1, 2, 3},
		H: 3.14,
		I: []interface{}{"any", int64(42), float64(1.23)},
	})
}

func TestEquivalence_StructD(t *testing.T) {
	assertEquivalence(t, &EquivD{
		A: []interface{}{
			map[string]interface{}{"a": "b"},
			[]interface{}{"c", int64(1)},
		},
	})
}

func TestEquivalence_StructE(t *testing.T) {
	assertEquivalence(t, &EquivE{
		A: map[string]interface{}{"nested": "val"},
	})
}

func TestEquivalence_StructF(t *testing.T) {
	assertEquivalence(t, &EquivF{
		A: "fa",
		B: map[string]string{"fb": "val"},
		C: []EquivA{{A: 1}},
		D: 42,
		E: 3.14,
		F: []string{"a"},
		G: []int{1},
		H: []bool{true},
		I: []float32{1.1},
		J: []byte("hello"),
	})
}

func TestEquivalence_StructG_CustomMarshaling(t *testing.T) {
	assertEquivalence(t, &EquivG{
		CustomValue1:   EquivCustomValue{Data: []byte(`{"a":1}`)},
		CustomValue2:   &EquivCustomValue{Data: []byte(`[1,2]`)},
		CustomPointer1: EquivCustomPointer{Data: []byte(`"string"`)},
		CustomPointer2: &EquivCustomPointer{Data: []byte(`42`)},
		RawExtension1:  runtime.RawExtension{Raw: []byte(`{"nestedKey":"nestedVal"}`)},
		RawExtension2:  &runtime.RawExtension{Raw: []byte(`[1,2,3]`)},
	})
}

func TestEquivalence_InlinePrimitive(t *testing.T) {
	assertEquivalence(t, &EquivInlineTestPrimitive{})
	assertEquivalence(t, &EquivInlineTestPrimitive{
		NoNameTagPrimitive:          1,
		NoNameTagInlinePrimitive:    2,
		NoNameTagOmitemptyPrimitive: 3,
	})
}

func TestEquivalence_InlineAnonymous(t *testing.T) {
	assertEquivalence(t, &EquivInlineTestAnonymous{})
	assertEquivalence(t, &EquivInlineTestAnonymous{
		EquivNoTag:              EquivNoTag{Data0: 100},
		EquivNoNameTag:          EquivNoNameTag{Data2: 200},
		EquivNameTag:            EquivNameTag{Data1: 300},
		EquivNoNameTagInline:    EquivNoNameTagInline{Data3: 400},
		EquivNoNameTagOmitempty: EquivNoNameTagOmitempty{Data4: 500},
	})
}

func TestEquivalence_InlineNamed(t *testing.T) {
	assertEquivalence(t, &EquivInlineTestNamed{})
	assertEquivalence(t, &EquivInlineTestNamed{
		NoTag:              EquivNoTag{Data0: 10},
		NoNameTag:          EquivNoNameTag{Data2: 20},
		NameTag:            EquivNameTag{Data1: 30},
		NoNameTagInline:    EquivNoNameTagInline{Data3: 40},
		NoNameTagOmitempty: EquivNoNameTagOmitempty{Data4: 50},
	})
}
