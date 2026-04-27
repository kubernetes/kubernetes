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

package v1beta1

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/google/go-cmp/cmp"
)

type marshalTestable interface {
	json.Marshaler
	cbor.Marshaler
}

type marshalTestCase struct {
	name          string
	input         marshalTestable
	wantJSONError bool
	wantCBORError bool
	wantJSON      []byte
	wantCBOR      []byte
}

type unmarshalTestable interface {
	json.Unmarshaler
	cbor.Unmarshaler
}

type unmarshalTestCase struct {
	name          string
	inputJSON     []byte
	inputCBOR     []byte
	wantJSONError bool
	wantCBORError bool
	wantDecoded   unmarshalTestable
}

type roundTripTestable interface {
	marshalTestable
	unmarshalTestable
}

type roundTripTestCase struct {
	name        string
	input       roundTripTestable
	wantJSON    []byte
	wantCBOR    []byte
	wantDecoded roundTripTestable
}

func TestJSONSchemaPropsOrBool(t *testing.T) {
	nan := math.NaN()

	t.Run("Marshal", func(t *testing.T) {
		testCases := []marshalTestCase{
			{
				name: "unsupported value",
				input: &JSONSchemaPropsOrBool{
					Schema: &JSONSchemaProps{Maximum: &nan},
				},
				wantJSONError: true,
				wantCBORError: true,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", marshalJSONTest(tc.input, tc.wantJSONError, tc.wantJSON))
				t.Run("cbor", marshalCBORTest(tc.input, tc.wantCBORError, tc.wantCBOR))
			})
		}
	})
	t.Run("RoundTrip", func(t *testing.T) {
		testCases := []roundTripTestCase{
			{
				name:        "zero value",
				input:       &JSONSchemaPropsOrBool{},
				wantDecoded: &JSONSchemaPropsOrBool{},
				wantJSON:    []byte(`false`),
				wantCBOR:    []byte{cborFalseValue},
			},
			{
				name:        "bool false",
				input:       &JSONSchemaPropsOrBool{Allows: false},
				wantDecoded: &JSONSchemaPropsOrBool{Allows: false},
				wantJSON:    []byte(`false`),
				wantCBOR:    []byte{cborFalseValue},
			},
			{
				name:        "bool true",
				input:       &JSONSchemaPropsOrBool{Allows: true},
				wantDecoded: &JSONSchemaPropsOrBool{Allows: true},
				wantJSON:    []byte(`true`),
				wantCBOR:    []byte{cborTrueValue},
			},
			{
				name:        "with props",
				input:       &JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{Type: "string"}},
				wantDecoded: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{Type: "string"}},
				wantJSON:    []byte(`{"type":"string"}`),
				wantCBOR:    []byte{0xA1, 0x44, 't', 'y', 'p', 'e', 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", roundTripJSONTest(tc.input, tc.wantJSON, tc.wantDecoded, &JSONSchemaPropsOrBool{}))
				t.Run("cbor", roundTripCBORTest(tc.input, tc.wantCBOR, tc.wantDecoded, &JSONSchemaPropsOrBool{}))
			})
		}
	})
	t.Run("Unmarshal", func(t *testing.T) {
		testCases := []unmarshalTestCase{
			{
				name:        "legacy behavior",
				inputJSON:   []byte(`{}`),
				inputCBOR:   []byte{0xA0},
				wantDecoded: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{}},
			},
			{
				name:          "null",
				inputJSON:     []byte(`null`),
				inputCBOR:     []byte{cborNullValue},
				wantJSONError: true,
				wantCBORError: true,
			},
			{
				name:        "zero len input",
				inputJSON:   []byte{},
				inputCBOR:   []byte{},
				wantDecoded: &JSONSchemaPropsOrBool{},
			},
			{
				name:          "unsupported type",
				inputJSON:     []byte(`42`),
				inputCBOR:     []byte{0x18, 42},
				wantJSONError: true,
				wantCBORError: true,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", unmarshalJSONTest(tc.inputJSON, tc.wantJSONError, tc.wantDecoded, &JSONSchemaPropsOrBool{}))
				t.Run("cbor", unmarshalCBORTest(tc.inputCBOR, tc.wantCBORError, tc.wantDecoded, &JSONSchemaPropsOrBool{}))
			})
		}
	})
}

func TestJSONSchemaPropsOrArray(t *testing.T) {
	nan := math.NaN()

	t.Run("Marshal", func(t *testing.T) {
		testCases := []marshalTestCase{
			{
				name:          "unsupported value",
				input:         &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Maximum: &nan}},
				wantJSONError: true,
				wantCBORError: true,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", marshalJSONTest(tc.input, tc.wantJSONError, tc.wantJSON))
				t.Run("cbor", marshalCBORTest(tc.input, tc.wantCBORError, tc.wantCBOR))
			})
		}
	})
	t.Run("RoundTrip", func(t *testing.T) {
		testCases := []roundTripTestCase{
			{
				name:        "empty props",
				input:       &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}},
				wantDecoded: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}},
				wantJSON:    []byte(`{}`),
				wantCBOR:    []byte{0xA0},
			},
			{
				name:        "props",
				input:       &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}},
				wantDecoded: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}},
				wantJSON:    []byte(`{"type":"string"}`),
				wantCBOR:    []byte{0xA1, 0x44, 't', 'y', 'p', 'e', 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},
			{
				name:        "array with empty props",
				input:       &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}},
				wantDecoded: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}},
				wantJSON:    []byte(`[{}]`),
				wantCBOR:    []byte{0x81, 0xA0},
			},
			{
				name:        "array with empty props and props",
				input:       &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}},
				wantDecoded: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}},
				wantJSON:    []byte(`[{},{"type":"string"}]`),
				wantCBOR:    []byte{0x82, 0xA0, 0xA1, 0x44, 't', 'y', 'p', 'e', 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", roundTripJSONTest(tc.input, tc.wantJSON, tc.wantDecoded, &JSONSchemaPropsOrArray{}))
				t.Run("cbor", roundTripCBORTest(tc.input, tc.wantCBOR, tc.wantDecoded, &JSONSchemaPropsOrArray{}))
			})
		}
	})
	t.Run("Unmarshal", func(t *testing.T) {
		testCases := []unmarshalTestCase{
			{
				name:        "null",
				inputJSON:   []byte(`null`),
				inputCBOR:   []byte{cborNullValue},
				wantDecoded: &JSONSchemaPropsOrArray{},
			},
			{
				name:        "zero len input",
				inputJSON:   []byte{},
				inputCBOR:   []byte{},
				wantDecoded: &JSONSchemaPropsOrArray{},
			},
			{
				name:        "unsupported type",
				inputJSON:   []byte(`42`),
				inputCBOR:   []byte{0x18, 42},
				wantDecoded: &JSONSchemaPropsOrArray{},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", unmarshalJSONTest(tc.inputJSON, tc.wantJSONError, tc.wantDecoded, &JSONSchemaPropsOrArray{}))
				t.Run("cbor", unmarshalCBORTest(tc.inputCBOR, tc.wantCBORError, tc.wantDecoded, &JSONSchemaPropsOrArray{}))
			})
		}
	})
}

func TestJSONSchemaPropsOrStringArray(t *testing.T) {
	nan := math.NaN()
	t.Run("Marshal", func(t *testing.T) {
		testCases := []marshalTestCase{
			{
				name:          "unsupported value",
				input:         JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{Maximum: &nan}},
				wantJSONError: true,
				wantCBORError: true,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", marshalJSONTest(tc.input, tc.wantJSONError, tc.wantJSON))
				t.Run("cbor", marshalCBORTest(tc.input, tc.wantCBORError, tc.wantCBOR))
			})
		}
	})

	t.Run("RoundTrip", func(t *testing.T) {
		testCases := []roundTripTestCase{
			{
				name:        "empty props",
				input:       &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{}},
				wantDecoded: &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{}},
				wantJSON:    []byte(`{}`),
				wantCBOR:    []byte{0xA0},
			},
			{
				name:        "props",
				input:       &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{Type: "string"}},
				wantDecoded: &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{Type: "string"}},
				wantJSON:    []byte(`{"type":"string"}`),
				wantCBOR:    []byte{0xA1, 0x44, 't', 'y', 'p', 'e', 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},

			{
				name:        "empty array",
				input:       &JSONSchemaPropsOrStringArray{Property: []string{}},
				wantDecoded: &JSONSchemaPropsOrStringArray{Property: nil},
				wantJSON:    []byte(`null`),
				wantCBOR:    []byte{cborNullValue},
			},
			{
				name:        "array value",
				input:       &JSONSchemaPropsOrStringArray{Property: []string{"string"}},
				wantDecoded: &JSONSchemaPropsOrStringArray{Property: []string{"string"}},
				wantJSON:    []byte(`["string"]`),
				wantCBOR:    []byte{0x81, 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},
			{
				name:        "both props and array",
				input:       &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{Type: "props"}, Property: []string{"string"}},
				wantDecoded: &JSONSchemaPropsOrStringArray{Schema: nil, Property: []string{"string"}},
				wantJSON:    []byte(`["string"]`),
				wantCBOR:    []byte{0x81, 0x46, 's', 't', 'r', 'i', 'n', 'g'},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", roundTripJSONTest(tc.input, tc.wantJSON, tc.wantDecoded, &JSONSchemaPropsOrStringArray{}))
				t.Run("cbor", roundTripCBORTest(tc.input, tc.wantCBOR, tc.wantDecoded, &JSONSchemaPropsOrStringArray{}))
			})
		}
	})

	t.Run("Unmarshal", func(t *testing.T) {
		testCases := []unmarshalTestCase{
			{
				name:        "empty array",
				inputJSON:   []byte(`[]`),
				inputCBOR:   []byte{0x80},
				wantDecoded: &JSONSchemaPropsOrStringArray{Property: []string{}},
			},
			{
				name:        "null",
				inputJSON:   []byte(`null`),
				inputCBOR:   []byte{cborNullValue},
				wantDecoded: &JSONSchemaPropsOrStringArray{},
			},
			{
				name:        "zero len input",
				inputJSON:   []byte{},
				inputCBOR:   []byte{},
				wantDecoded: &JSONSchemaPropsOrStringArray{},
			},
			{
				name:        "unsupported type",
				inputJSON:   []byte(`42`),
				inputCBOR:   []byte{0x18, 42},
				wantDecoded: &JSONSchemaPropsOrStringArray{},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", unmarshalJSONTest(tc.inputJSON, tc.wantJSONError, tc.wantDecoded, &JSONSchemaPropsOrStringArray{}))
				t.Run("cbor", unmarshalCBORTest(tc.inputCBOR, tc.wantCBORError, tc.wantDecoded, &JSONSchemaPropsOrStringArray{}))
			})
		}
	})

}

func TestJSON(t *testing.T) {
	t.Run("RoundTrip", func(t *testing.T) {
		testCases := []roundTripTestCase{
			{
				name:        "nil raw",
				input:       &JSON{Raw: nil},
				wantDecoded: &JSON{Raw: nil},
				wantJSON:    []byte(`null`),
				wantCBOR:    []byte{cborNullValue},
			},
			{
				name:        "zero len raw",
				input:       &JSON{Raw: []byte{}},
				wantDecoded: &JSON{Raw: nil},
				wantJSON:    []byte(`null`),
				wantCBOR:    []byte{cborNullValue},
			},
			{
				name:        "empty",
				input:       &JSON{},
				wantDecoded: &JSON{},
				wantJSON:    []byte(`null`),
				wantCBOR:    []byte{cborNullValue},
			},
			{
				name:        "string",
				input:       &JSON{Raw: []byte(`"string"`)},
				wantDecoded: &JSON{Raw: []byte(`"string"`)},
				wantJSON:    []byte(`"string"`),
				wantCBOR:    []byte{0x46, 0x73, 0x74, 0x72, 0x69, 0x6E, 0x67},
			},
			{
				name:        "number",
				input:       &JSON{Raw: []byte(`42.01`)},
				wantDecoded: &JSON{Raw: []byte(`42.01`)},
				wantJSON:    []byte(`42.01`),
				wantCBOR:    []byte{0xFB, 0x40, 0x45, 0x01, 0x47, 0xAE, 0x14, 0x7A, 0xE1},
			},
			{
				name:        "bool",
				input:       &JSON{Raw: []byte(`true`)},
				wantDecoded: &JSON{Raw: []byte(`true`)},
				wantJSON:    []byte(`true`),
				wantCBOR:    []byte{0xF5},
			},
			{
				name:        "array",
				input:       &JSON{Raw: []byte(`[1,2,3]`)},
				wantDecoded: &JSON{Raw: []byte(`[1,2,3]`)},
				wantJSON:    []byte(`[1,2,3]`),
				wantCBOR:    []byte{0x83, 1, 2, 3},
			},
			{
				name:        "map",
				input:       &JSON{Raw: []byte(`{"foo":"bar"}`)},
				wantDecoded: &JSON{Raw: []byte(`{"foo":"bar"}`)},
				wantJSON:    []byte(`{"foo":"bar"}`),
				wantCBOR:    []byte{0xA1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'},
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("json", roundTripJSONTest(tc.input, tc.wantJSON, tc.wantDecoded, &JSON{}))
				t.Run("cbor", roundTripCBORTest(tc.input, tc.wantCBOR, tc.wantDecoded, &JSON{}))
			})
		}
	})
}

func marshalJSONTest(input marshalTestable, wantErr bool, expected []byte) func(t *testing.T) {
	return func(t *testing.T) {
		actual, err := input.MarshalJSON()
		if (err != nil) != wantErr {
			if wantErr {
				t.Fatal("expected error")
			}
			t.Fatalf("unexpected error: %v", err)
		}
		if err != nil {
			return
		}
		if diff := cmp.Diff(string(expected), string(actual)); len(diff) > 0 {
			t.Fatal(diff)
		}
	}
}

func marshalCBORTest(input marshalTestable, wantErr bool, expected []byte) func(t *testing.T) {
	return func(t *testing.T) {
		actual, err := input.MarshalCBOR()
		if (err != nil) != wantErr {
			if wantErr {
				t.Fatal("expected error")
			}
			t.Fatalf("unexpected error: %v", err)
		}
		if err != nil {
			return
		}
		if diff := cmp.Diff(expected, actual); len(diff) > 0 {
			t.Fatal(diff)
		}
	}
}

func unmarshalJSONTest(input []byte, wantErr bool, expectedDecoded unmarshalTestable, actualDecoded unmarshalTestable) func(t *testing.T) {
	return func(t *testing.T) {
		err := actualDecoded.UnmarshalJSON(input)
		if (err != nil) != wantErr {
			if wantErr {
				t.Fatal("expected error")
			}
			t.Fatalf("unexpected error: %v", err)
		}
		if err != nil {
			return
		}
		if diff := cmp.Diff(expectedDecoded, actualDecoded); len(diff) > 0 {
			t.Error("unexpected decoded value")
			t.Fatal(diff)
		}
	}
}

func unmarshalCBORTest(input []byte, wantErr bool, expectedDecoded unmarshalTestable, actualDecoded unmarshalTestable) func(t *testing.T) {
	return func(t *testing.T) {
		err := actualDecoded.UnmarshalCBOR(input)
		if (err != nil) != wantErr {
			if wantErr {
				t.Fatal("expected error")
			}
			t.Fatalf("unexpected error: %v", err)
		}
		if err != nil {
			return
		}
		if diff := cmp.Diff(expectedDecoded, actualDecoded); len(diff) != 0 {
			t.Error("unexpected decoded value")
			t.Fatal(diff)
		}
	}
}

func roundTripJSONTest(input roundTripTestable, expectedEncoded []byte, expectedDecoded roundTripTestable, actualDecoded roundTripTestable) func(t *testing.T) {
	return func(t *testing.T) {
		actualEncoded, err := input.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(string(expectedEncoded), string(actualEncoded)); len(diff) > 0 {
			t.Error("unexpected encoded value")
			t.Fatal(diff)
		}
		err = actualDecoded.UnmarshalJSON(actualEncoded)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(expectedDecoded, actualDecoded); len(diff) > 0 {
			t.Error("unexpected decoded value")
			t.Fatal(diff)
		}
	}
}

func roundTripCBORTest(input roundTripTestable, expectedEncoded []byte, expectedDecoded roundTripTestable, actualDecoded roundTripTestable) func(t *testing.T) {
	return func(t *testing.T) {
		actualEncoded, err := input.MarshalCBOR()
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(expectedEncoded, actualEncoded); len(diff) > 0 {
			t.Error("unexpected encoded value")
			t.Fatal(diff)
		}
		err = actualDecoded.UnmarshalCBOR(actualEncoded)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(expectedDecoded, actualDecoded); len(diff) > 0 {
			t.Error("unexpected decoded value")
			t.Fatal(diff)
		}
	}
}

func TestJSONUnderlyingArrayReuse(t *testing.T) {
	const want = `{"foo":"bar"}`

	b := []byte(want)

	var s JSON
	if err := s.UnmarshalJSON(b); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Underlying array is modified.
	copy(b[2:5], "bar")
	copy(b[8:11], "foo")

	// If UnmarshalJSON copied the bytes of its argument, then it should not have been affected
	// by the mutation.
	if got := string(s.Raw); got != want {
		t.Errorf("unexpected mutation, got %s want %s", got, want)
	}
}
