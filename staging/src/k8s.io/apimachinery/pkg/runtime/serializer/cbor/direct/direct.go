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
	"bytes"
	"errors"
	"io"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"
)

func Marshal(src interface{}) ([]byte, error) {
	return modes.Encode.Marshal(src)
}

func Unmarshal(src []byte, dst interface{}) error {
	return modes.Decode.Unmarshal(src, dst)
}

func Diagnose(src []byte) (string, error) {
	return modes.Diagnostic.Diagnose(src)
}

// selfDescribedCBOR is the CBOR encoding of the head of the "self-described CBOR" tag.
var selfDescribedCBOR = []byte{0xd9, 0xd9, 0xf7}

// Sniff does a constant-time inspection of the provided bytes and returns whether or not they may
// contain a CBOR data item. If unknown is true, the determination could be wrong.
func Sniff(src []byte) (cbor bool, unknown bool) {
	// Objects Outputs from the CBOR encoder will always have this prefix
	if bytes.HasPrefix(src, selfDescribedCBOR) {
		return true, true
	}

	if len(src) > 32 {
		src = src[:32]
	}
	err := modes.DecodeLax.Wellformed(src)
	switch {
	case err == nil:
		return true, false
	case errors.Is(err, io.ErrUnexpectedEOF):
		// Ran out of input without seeing a wellformedness error.
		return true, true
	default:
		return false, false
	}
}
