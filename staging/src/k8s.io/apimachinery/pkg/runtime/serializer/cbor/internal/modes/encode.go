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

package modes

import (
	"github.com/fxamacker/cbor/v2"
)

var Encode cbor.EncMode = func() cbor.EncMode {
	encode, err := cbor.EncOptions{
		// Map keys need to be sorted to have deterministic output, and this is the order
		// defined in RFC 8949 4.2.1 "Core Deterministic Encoding Requirements".
		Sort: cbor.SortBytewiseLexical,

		// CBOR supports distinct types for IEEE-754 float16, float32, and float64. Store
		// floats in the smallest width that preserves value so that equivalent float32 and
		// float64 values encode to identical bytes, as they do in a JSON
		// encoding. Satisfies one of the "Core Deterministic Encoding Requirements".
		ShortestFloat: cbor.ShortestFloat16,

		// Error on attempt to encode NaN and infinite values. This is what the JSON
		// serializer does.
		NaNConvert: cbor.NaNConvertReject,
		InfConvert: cbor.InfConvertReject,

		// Prefer encoding math/big.Int to one of the 64-bit integer types if it fits. When
		// later decoded into Unstructured, the set of allowable concrete numeric types is
		// limited to int64 and float64, so the distinction between big integer and integer
		// can't be preserved.
		BigIntConvert: cbor.BigIntConvertShortest,

		// MarshalJSON for time.Time writes RFC3339 with nanos.
		Time: cbor.TimeRFC3339Nano,

		// The decoder must be able to accept RFC3339 strings with or without tag 0 (e.g. by
		// the end of time.Time -> JSON -> Unstructured -> CBOR, the CBOR encoder has no
		// reliable way of knowing that a particular string originated from serializing a
		// time.Time), so producing tag 0 has little use.
		TimeTag: cbor.EncTagNone,

		// Indefinite-length items have multiple encodings and aren't being used anyway, so
		// disable to avoid an opportunity for nondeterminism.
		IndefLength: cbor.IndefLengthForbidden,

		// Preserve distinction between nil and empty for slices and maps.
		NilContainers: cbor.NilContainerAsNull,

		// OK to produce tags.
		TagsMd: cbor.TagsAllowed,

		// Use the same definition of "empty" as encoding/json.
		OmitEmpty: cbor.OmitEmptyGoValue,

		// The CBOR types text string and byte string are structurally equivalent, with the
		// semantic difference that a text string whose content is an invalid UTF-8 sequence
		// is itself invalid. We reject all invalid text strings at decode time and do not
		// validate or sanitize all Go strings at encode time. Encoding Go strings to the
		// byte string type is comparable to the existing Protobuf behavior and cheaply
		// ensures that the output is valid CBOR.
		String: cbor.StringToByteString,

		// Encode struct field names to the byte string type rather than the text string
		// type.
		FieldName: cbor.FieldNameToByteString,
	}.EncMode()
	if err != nil {
		panic(err)
	}
	return encode
}()

var EncodeNondeterministic cbor.EncMode = func() cbor.EncMode {
	opts := Encode.EncOptions()
	opts.Sort = cbor.SortNone
	em, err := opts.EncMode()
	if err != nil {
		panic(err)
	}
	return em
}()
