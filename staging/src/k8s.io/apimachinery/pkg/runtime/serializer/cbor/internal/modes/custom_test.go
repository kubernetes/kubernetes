/*
Copyright 2024 The Kubernetes Authors.

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

package modes_test

import (
	"encoding"
	"encoding/json"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"

	"github.com/fxamacker/cbor/v2"
)

type TextUnmarshalerPointer struct{}

func (*TextUnmarshalerPointer) UnmarshalText([]byte) error { return nil }

type Struct[F any] struct {
	Field F
}

type StructCBORMarshalerValue[F any] struct {
	Field F
}

func (StructCBORMarshalerValue[F]) MarshalCBOR() ([]byte, error) { return nil, nil }

type StructCBORMarshalerPointer[F any] struct {
	Field F
}

func (*StructCBORMarshalerPointer[F]) MarshalCBOR() ([]byte, error) { return nil, nil }

type StructCBORUnmarshalerPointer[F any] struct {
	Field F
}

func (*StructCBORUnmarshalerPointer[F]) UnmarshalCBOR([]byte) error { return nil }

type StructCBORUnmarshalerValueWithEmbeddedJSONMarshaler struct {
	json.Marshaler
}

func (StructCBORUnmarshalerValueWithEmbeddedJSONMarshaler) MarshalCBOR() ([]byte, error) {
	return nil, nil
}

type SafeCyclicTypeA struct {
	Bs []SafeCyclicTypeB
}

type SafeCyclicTypeB struct {
	As []SafeCyclicTypeA
}

type UnsafeCyclicTypeA struct {
	Bs []UnsafeCyclicTypeB
}

type UnsafeCyclicTypeB struct {
	As []UnsafeCyclicTypeA

	json.Marshaler
}

func TestCheckUnsupportedMarshalers(t *testing.T) {
	t.Run("accepted", func(t *testing.T) {
		for _, v := range []interface{}{
			// Unstructured types.
			nil,
			false,
			0,
			0.0,
			"",
			[]interface{}{nil, false, 0, 0.0, "", []interface{}{}, map[string]interface{}{}},
			map[string]interface{}{
				"nil":                      nil,
				"false":                    false,
				"0":                        0,
				"0.0":                      0.0,
				`""`:                       "",
				"[]interface{}{}":          []interface{}{},
				"map[string]interface{}{}": map[string]interface{}{},
			},
			// Zero-length array is statically OK, even though its element type is
			// json.Marshaler.
			[0]json.Marshaler{},
			[3]bool{},
			// Has to be dynamically checked.
			[1]interface{}{},
			map[int]interface{}{0: nil},
			Struct[string]{},
			StructCBORMarshalerValue[json.Marshaler]{},
			&StructCBORMarshalerValue[json.Marshaler]{},
			&StructCBORMarshalerPointer[json.Marshaler]{},
			new(string),
			map[cbor.Marshaler]cbor.Marshaler{},
			make([]string, 10),
			[]Struct[interface{}]{{Field: true}},
			&Struct[StructCBORUnmarshalerPointer[json.Unmarshaler]]{},
			// Checked dynamically because the dynamic type of an interface, even
			// encoding.TextMarshaler or json.Marshaler, might also implement
			// cbor.Marshaler. Accepted because there are no map entries.
			map[encoding.TextMarshaler]struct{}{},
			map[string]json.Marshaler{},
			// Methods of json.Marshaler and cbor.Marshaler are both promoted to the
			// struct's method set.
			struct {
				json.Marshaler
				StructCBORMarshalerValue[int]
			}{},

			// Embedded field's json.Marshaler implementation is promoted, but the
			// containing struct implements cbor.Marshaler.
			StructCBORUnmarshalerValueWithEmbeddedJSONMarshaler{},

			SafeCyclicTypeA{},
			SafeCyclicTypeB{},
		} {
			if err := modes.RejectCustomMarshalers(v); err != nil {
				t.Errorf("%#v: unexpected non-nil error: %v", v, err)
			}
		}
	})

	t.Run("rejected", func(t *testing.T) {
		for _, v := range []interface{}{
			Struct[json.Marshaler]{},
			StructCBORMarshalerValue[json.Unmarshaler]{},
			Struct[interface{}]{Field: [1]json.Unmarshaler{}},
			// cbor.Marshaler implemented with pointer receiver on non-pointer type.
			StructCBORMarshalerPointer[json.Marshaler]{},
			[1]json.Marshaler{},
			[1]interface{}{[1]json.Marshaler{}},
			map[int]interface{}{0: [1]json.Marshaler{}},
			[]interface{}{[1]json.Marshaler{}},
			map[string]interface{}{"": [1]json.Marshaler{}},
			[]Struct[interface{}]{{Field: [1]json.Marshaler{}}},
			map[encoding.TextMarshaler]struct{}{[1]encoding.TextMarshaler{}[0]: {}},
			map[string]json.Marshaler{"": [1]json.Marshaler{}[0]},
			&Struct[StructCBORMarshalerPointer[json.Marshaler]]{},
			Struct[TextUnmarshalerPointer]{},
			struct {
				json.Marshaler
			}{},
			UnsafeCyclicTypeA{Bs: []UnsafeCyclicTypeB{{}}},
			UnsafeCyclicTypeB{},
		} {
			if err := modes.RejectCustomMarshalers(v); err == nil {
				t.Errorf("%#v: unexpected nil error", v)
			}
		}
	})
}
