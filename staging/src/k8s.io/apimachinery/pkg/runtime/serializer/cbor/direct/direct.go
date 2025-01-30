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

// Package direct provides functions for marshaling and unmarshaling between arbitrary Go values and
// CBOR data, with behavior that is compatible with that of the CBOR serializer. In particular,
// types that implement cbor.Marshaler and cbor.Unmarshaler should use these functions.
package direct

import (
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"
)

// Marshal serializes a value to CBOR. If there is more than one way to encode the value, it will
// make the same choice as the CBOR implementation of runtime.Serializer.
//
// Note: Support for CBOR is at an alpha stage. If the value (or, for composite types, any of its
// nested values) implement any of the interfaces encoding.TextMarshaler, encoding.TextUnmarshaler,
// encoding/json.Marshaler, or encoding/json.Unmarshaler, a non-nil error will be returned unless
// the value also implements the corresponding CBOR interfaces. This limitation will ultimately be
// removed in favor of automatic transcoding to CBOR.
func Marshal(src interface{}) ([]byte, error) {
	if err := modes.RejectCustomMarshalers(src); err != nil {
		return nil, err
	}
	return modes.Encode.Marshal(src)
}

// Unmarshal deserializes from CBOR into an addressable value. If there is more than one way to
// unmarshal a value, it will make the same choice as the CBOR implementation of runtime.Serializer.
//
// Note: Support for CBOR is at an alpha stage. If the value (or, for composite types, any of its
// nested values) implement any of the interfaces encoding.TextMarshaler, encoding.TextUnmarshaler,
// encoding/json.Marshaler, or encoding/json.Unmarshaler, a non-nil error will be returned unless
// the value also implements the corresponding CBOR interfaces. This limitation will ultimately be
// removed in favor of automatic transcoding to CBOR.
func Unmarshal(src []byte, dst interface{}) error {
	if err := modes.RejectCustomMarshalers(dst); err != nil {
		return err
	}
	return modes.Decode.Unmarshal(src, dst)
}

// Diagnose accepts well-formed CBOR bytes and returns a string representing the same data item in
// human-readable diagnostic notation (RFC 8949 Section 8). The diagnostic notation is not meant to
// be parsed.
func Diagnose(src []byte) (string, error) {
	return modes.Diagnostic.Diagnose(src)
}
