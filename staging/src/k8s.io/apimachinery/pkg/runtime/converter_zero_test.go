/*
Copyright 2025 The Kubernetes Authors.

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

// These tests are in a separate package to break cyclic dependency in tests.
// Unstructured type depends on unstructured converter package but we want to test how the converter handles
// the Unstructured type so we need to import both.

package runtime

import (
	encodingjson "encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/json"

	"github.com/google/go-cmp/cmp"
)

type ZeroParent struct {
	Int               int                    `json:"int,omitzero"`
	IntP              *int                   `json:"intP,omitzero"`
	String            string                 `json:"string,omitzero"`
	StringP           *string                `json:"stringP,omitzero"`
	Bool              bool                   `json:"bool,omitzero"`
	BoolP             bool                   `json:"boolP,omitzero"`
	Slice             []int                  `json:"slice,omitzero"`
	SliceP            *[]int                 `json:"sliceP,omitzero"`
	Map               map[string]int         `json:"map,omitzero"`
	MapP              *map[string]int        `json:"mapP,omitzero"`
	Struct            ZeroChild              `json:"struct,omitzero"`
	StructP           *ZeroChild             `json:"structP,omitzero"`
	CustomPrimitive   ZeroCustomPrimitive    `json:"customPrimitive,omitzero"`
	CustomPrimitiveP  *ZeroCustomPrimitiveP  `json:"customPrimitiveP,omitzero"`
	CustomStruct      ZeroCustomStruct       `json:"customStruct,omitzero"`
	CustomStructP     *ZeroCustomStructP     `json:"customStructP,omitzero"`
	CustomPPrimitive  ZeroCustomPPrimitive   `json:"customPPrimitive,omitzero"`
	CustomPPrimitiveP *ZeroCustomPPrimitiveP `json:"customPPrimitiveP,omitzero"`
	CustomPStruct     ZeroCustomPStruct      `json:"customPStruct,omitzero"`
	CustomPStructP    *ZeroCustomPStructP    `json:"customPStructP,omitzero"`
}
type ZeroChild struct {
	Data int `json:"data"`
}

type ZeroCustomPrimitive int

func (z ZeroCustomPrimitive) IsZero() bool {
	return z == 42
}

type ZeroCustomPrimitiveP int

func (z ZeroCustomPrimitiveP) IsZero() bool {
	return z == 42
}

type ZeroCustomStruct struct {
	Data int `json:"data"`
}

func (z ZeroCustomStruct) IsZero() bool {
	return z.Data == 42
}

type ZeroCustomStructP struct {
	Data int `json:"data"`
}

func (z ZeroCustomStructP) IsZero() bool {
	return z.Data == 42
}

type ZeroCustomPPrimitive int

func (z *ZeroCustomPPrimitive) IsZero() bool {
	return *z == 42
}

type ZeroCustomPPrimitiveP int

func (z *ZeroCustomPPrimitiveP) IsZero() bool {
	return *z == 42
}

type ZeroCustomPStruct struct {
	Data int `json:"data"`
}

func (z *ZeroCustomPStruct) IsZero() bool {
	return z.Data == 42
}

type ZeroCustomPStructP struct {
	Data int `json:"data"`
}

func (z *ZeroCustomPStructP) IsZero() bool {
	return z.Data == 42
}

func TestOmitZero2(t *testing.T) {
	testcases := []struct {
		name   string
		obj    any
		expect map[string]any
	}{
		{
			name: "emptyzero",
			obj:  &ZeroParent{},
			expect: map[string]any{
				"customPPrimitive": int64(0),
				"customPStruct":    map[string]any{"data": int64(0)},
				"customPrimitive":  int64(0),
				"customStruct":     map[string]any{"data": int64(0)},
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			jsonData, err := json.Marshal(tc.obj)
			if err != nil {
				t.Fatal(err)
			}
			jsonUnstructured := map[string]any{}
			if err := json.Unmarshal(jsonData, &jsonUnstructured); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(jsonUnstructured, tc.expect) {
				t.Fatal(cmp.Diff(tc.expect, jsonUnstructured))
			}

			unstr, err := DefaultUnstructuredConverter.ToUnstructured(tc.obj)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(unstr, tc.expect) {
				t.Fatal(cmp.Diff(tc.expect, unstr))
			}
		})
	}
}

type NonZeroStruct struct{}

func (nzs NonZeroStruct) IsZero() bool {
	return false
}

type NoPanicStruct struct {
	Int int `json:"int,omitzero"`
}

func (nps *NoPanicStruct) IsZero() bool {
	return nps.Int != 0
}

type isZeroer interface {
	IsZero() bool
}

type OptionalsZero struct {
	Sr string `json:"sr"`
	So string `json:"so,omitzero"`
	Sw string `json:"-"`

	Ir int `json:"omitzero"` // actually named omitzero, not an option
	Io int `json:"io,omitzero"`

	Slr       []string `json:"slr,random"` //nolint:staticcheck // SA5008
	Slo       []string `json:"slo,omitzero"`
	SloNonNil []string `json:"slononnil,omitzero"`

	Mr  map[string]any `json:"mr"`
	Mo  map[string]any `json:",omitzero"`
	Moo map[string]any `json:"moo,omitzero"`

	Fr   float64    `json:"fr"`
	Fo   float64    `json:"fo,omitzero"`
	Foo  float64    `json:"foo,omitzero"`
	Foo2 [2]float64 `json:"foo2,omitzero"`

	Br bool `json:"br"`
	Bo bool `json:"bo,omitzero"`

	Ur uint `json:"ur"`
	Uo uint `json:"uo,omitzero"`

	Str struct{} `json:"str"`
	Sto struct{} `json:"sto,omitzero"`

	Time      time.Time     `json:"time,omitzero"`
	TimeLocal time.Time     `json:"timelocal,omitzero"`
	Nzs       NonZeroStruct `json:"nzs,omitzero"`

	NilIsZeroer    isZeroer       `json:"niliszeroer,omitzero"`    // nil interface
	NonNilIsZeroer isZeroer       `json:"nonniliszeroer,omitzero"` // non-nil interface
	NoPanicStruct0 isZeroer       `json:"nps0,omitzero"`           // non-nil interface with nil pointer
	NoPanicStruct1 isZeroer       `json:"nps1,omitzero"`           // non-nil interface with non-nil pointer
	NoPanicStruct2 *NoPanicStruct `json:"nps2,omitzero"`           // nil pointer
	NoPanicStruct3 *NoPanicStruct `json:"nps3,omitzero"`           // non-nil pointer
	NoPanicStruct4 NoPanicStruct  `json:"nps4,omitzero"`           // concrete type
}

func TestOmitZero(t *testing.T) {
	const want = `{
 "Mo": {},
 "br": false,
 "fr": 0,
 "mr": {},
 "nps1": {},
 "nps3": {},
 "nps4": {},
 "nzs": {},
 "omitzero": 0,
 "slononnil": [],
 "slr": null,
 "sr": "",
 "str": {},
 "ur": 0
}`
	var o OptionalsZero
	o.Sw = "something"
	o.SloNonNil = make([]string, 0)
	o.Mr = map[string]any{}
	o.Mo = map[string]any{}

	o.Foo = -0
	o.Foo2 = [2]float64{+0, -0}

	o.TimeLocal = time.Time{}.Local()

	o.NonNilIsZeroer = time.Time{}
	o.NoPanicStruct0 = (*NoPanicStruct)(nil)
	o.NoPanicStruct1 = &NoPanicStruct{}
	o.NoPanicStruct3 = &NoPanicStruct{}

	unstr, err := DefaultUnstructuredConverter.ToUnstructured(&o)
	if err != nil {
		t.Fatal(err)
	}

	got, err := encodingjson.MarshalIndent(unstr, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", got, want)
	}
}

func TestOmitZeroMap(t *testing.T) {
	const want = `{
 "foo": {
  "br": false,
  "fr": 0,
  "mr": null,
  "nps4": {},
  "nzs": {},
  "omitzero": 0,
  "slr": null,
  "sr": "",
  "str": {},
  "ur": 0
 }
}`

	m := map[string]OptionalsZero{"foo": {}}

	unstr, err := DefaultUnstructuredConverter.ToUnstructured(&m)
	if err != nil {
		t.Fatal(err)
	}

	got, err := encodingjson.MarshalIndent(unstr, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		fmt.Println(got)
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", got, want)
	}
}

type OptionalsEmptyZero struct {
	Sr string `json:"sr"`
	So string `json:"so,omitempty,omitzero"`
	Sw string `json:"-"`

	Io int `json:"io,omitempty,omitzero"`

	Slr       []string `json:"slr,random"` //nolint:staticcheck // SA5008
	Slo       []string `json:"slo,omitempty,omitzero"`
	SloNonNil []string `json:"slononnil,omitempty,omitzero"`

	Mr map[string]any `json:"mr"`
	Mo map[string]any `json:",omitempty,omitzero"`

	Fr float64 `json:"fr"`
	Fo float64 `json:"fo,omitempty,omitzero"`

	Br bool `json:"br"`
	Bo bool `json:"bo,omitempty,omitzero"`

	Ur uint `json:"ur"`
	Uo uint `json:"uo,omitempty,omitzero"`

	Str struct{} `json:"str"`
	Sto struct{} `json:"sto,omitempty,omitzero"`

	Time time.Time     `json:"time,omitempty,omitzero"`
	Nzs  NonZeroStruct `json:"nzs,omitempty,omitzero"`
}

func TestOmitEmptyZero(t *testing.T) {
	const want = `{
 "br": false,
 "fr": 0,
 "mr": {},
 "nzs": {},
 "slr": null,
 "sr": "",
 "str": {},
 "ur": 0
}`
	var o OptionalsEmptyZero
	o.Sw = "something"
	o.SloNonNil = make([]string, 0)
	o.Mr = map[string]any{}
	o.Mo = map[string]any{}

	unstr, err := DefaultUnstructuredConverter.ToUnstructured(&o)
	if err != nil {
		t.Fatal(err)
	}

	got, err := encodingjson.MarshalIndent(unstr, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", got, want)
	}
}
