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
	"io"

	"github.com/fxamacker/cbor/v2"
)

var Encode = EncMode{
	delegate: func() cbor.UserBufferEncMode {
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

			// Error on attempt to encode math/big.Int values, which can't be faithfully
			// roundtripped through Unstructured in general (the dynamic numeric types allowed
			// in Unstructured are limited to float64 and int64).
			BigIntConvert: cbor.BigIntConvertReject,

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

			// Marshal Go byte arrays to CBOR arrays of integers (as in JSON) instead of byte
			// strings.
			ByteArray: cbor.ByteArrayToArray,

			// Marshal []byte to CBOR byte string enclosed in tag 22 (expected later base64
			// encoding, https://www.rfc-editor.org/rfc/rfc8949.html#section-3.4.5.2), to
			// interoperate with the existing JSON behavior. This indicates to the decoder that,
			// when decoding into a string (or unstructured), the resulting value should be the
			// base64 encoding of the original bytes. No base64 encoding or decoding needs to be
			// performed for []byte-to-CBOR-to-[]byte roundtrips.
			ByteSliceLaterFormat: cbor.ByteSliceLaterFormatBase64,

			// Disable default recognition of types implementing encoding.BinaryMarshaler, which
			// is not recognized for JSON encoding.
			BinaryMarshaler: cbor.BinaryMarshalerNone,
		}.UserBufferEncMode()
		if err != nil {
			panic(err)
		}
		return encode
	}(),
}

var EncodeNondeterministic = EncMode{
	delegate: func() cbor.UserBufferEncMode {
		opts := Encode.options()
		opts.Sort = cbor.SortNone // TODO: Use cbor.SortFastShuffle after bump to v2.7.0.
		em, err := opts.UserBufferEncMode()
		if err != nil {
			panic(err)
		}
		return em
	}(),
}

type EncMode struct {
	delegate cbor.UserBufferEncMode
}

func (em EncMode) options() cbor.EncOptions {
	return em.delegate.EncOptions()
}

func (em EncMode) MarshalTo(v interface{}, w io.Writer) error {
	if buf, ok := w.(*buffer); ok {
		return em.delegate.MarshalToBuffer(v, &buf.Buffer)
	}

	buf := buffers.Get()
	defer buffers.Put(buf)
	if err := em.delegate.MarshalToBuffer(v, &buf.Buffer); err != nil {
		return err
	}

	if _, err := io.Copy(w, buf); err != nil {
		return err
	}

	return nil
}

func (em EncMode) Marshal(v interface{}) ([]byte, error) {
	buf := buffers.Get()
	defer buffers.Put(buf)

	if err := em.MarshalTo(v, &buf.Buffer); err != nil {
		return nil, err
	}

	clone := make([]byte, buf.Len())
	copy(clone, buf.Bytes())

	return clone, nil
}
