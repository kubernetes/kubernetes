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
	"reflect"

	"github.com/fxamacker/cbor/v2"
)

var simpleValues *cbor.SimpleValueRegistry = func() *cbor.SimpleValueRegistry {
	var opts []func(*cbor.SimpleValueRegistry) error
	for sv := 0; sv <= 255; sv++ {
		// Reject simple values 0-19, 23, and 32-255. The simple values 24-31 are reserved
		// and considered ill-formed by the CBOR specification. We only accept false (20),
		// true (21), and null (22).
		switch sv {
		case 20: // false
		case 21: // true
		case 22: // null
		case 24, 25, 26, 27, 28, 29, 30, 31: // reserved
		default:
			opts = append(opts, cbor.WithRejectedSimpleValue(cbor.SimpleValue(sv)))
		}
	}
	simpleValues, err := cbor.NewSimpleValueRegistryFromDefaults(opts...)
	if err != nil {
		panic(err)
	}
	return simpleValues
}()

var Decode cbor.DecMode = func() cbor.DecMode {
	decode, err := cbor.DecOptions{
		// Maps with duplicate keys are well-formed but invalid according to the CBOR spec
		// and never acceptable. Unlike the JSON serializer, inputs containing duplicate map
		// keys are rejected outright and not surfaced as a strict decoding error.
		DupMapKey: cbor.DupMapKeyEnforcedAPF,

		// For JSON parity, decoding an RFC3339 string into time.Time needs to be accepted
		// with or without tagging. If a tag number is present, it must be valid.
		TimeTag: cbor.DecTagOptional,

		// Observed depth up to 16 in fuzzed batch/v1 CronJobList. JSON implementation limit
		// is 10000.
		MaxNestedLevels: 64,

		MaxArrayElements: 1024,
		MaxMapPairs:      1024,

		// Indefinite-length sequences aren't produced by this serializer, but other
		// implementations can.
		IndefLength: cbor.IndefLengthAllowed,

		// Accept inputs that contain CBOR tags.
		TagsMd: cbor.TagsAllowed,

		// Decode type 0 (unsigned integer) as int64.
		// TODO: IntDecConvertSignedOrFail errors on overflow, JSON will try to fall back to float64.
		IntDec: cbor.IntDecConvertSignedOrFail,

		// Disable producing map[cbor.ByteString]interface{}, which is not acceptable for
		// decodes into interface{}.
		MapKeyByteString: cbor.MapKeyByteStringForbidden,

		// Error on map keys that don't map to a field in the destination struct.
		ExtraReturnErrors: cbor.ExtraDecErrorUnknownField,

		// Decode maps into concrete type map[string]interface{} when the destination is an
		// interface{}.
		DefaultMapType: reflect.TypeOf(map[string]interface{}(nil)),

		// A CBOR text string whose content is not a valid UTF-8 sequence is well-formed but
		// invalid according to the CBOR spec. Reject invalid inputs. Encoders are
		// responsible for ensuring that all text strings they produce contain valid UTF-8
		// sequences and may use the byte string major type to encode strings that have not
		// been validated.
		UTF8: cbor.UTF8RejectInvalid,

		// Never make a case-insensitive match between a map key and a struct field.
		FieldNameMatching: cbor.FieldNameMatchingCaseSensitive,

		// Produce string concrete values when decoding a CBOR byte string into interface{}.
		DefaultByteStringType: reflect.TypeOf(""),

		// Allow CBOR byte strings to be decoded into string destination values. If a byte
		// string is enclosed in an "expected later encoding" tag
		// (https://www.rfc-editor.org/rfc/rfc8949.html#section-3.4.5.2), then the text
		// encoding indicated by that tag (e.g. base64) will be applied to the contents of
		// the byte string.
		ByteStringToString: cbor.ByteStringToStringAllowedWithExpectedLaterEncoding,

		// Allow CBOR byte strings to match struct fields when appearing as a map key.
		FieldNameByteString: cbor.FieldNameByteStringAllowed,

		// When decoding an unrecognized tag to interface{}, return the decoded tag content
		// instead of the default, a cbor.Tag representing a (number, content) pair.
		UnrecognizedTagToAny: cbor.UnrecognizedTagContentToAny,

		// Decode time tags to interface{} as strings containing RFC 3339 timestamps.
		TimeTagToAny: cbor.TimeTagToRFC3339Nano,

		// For parity with JSON, strings can be decoded into time.Time if they are RFC 3339
		// timestamps.
		ByteStringToTime: cbor.ByteStringToTimeAllowed,

		// Reject NaN and infinite floating-point values since they don't have a JSON
		// representation (RFC 8259 Section 6).
		NaN: cbor.NaNDecodeForbidden,
		Inf: cbor.InfDecodeForbidden,

		// When unmarshaling a byte string into a []byte, assume that the byte string
		// contains base64-encoded bytes, unless explicitly counterindicated by an "expected
		// later encoding" tag. This is consistent with the because of unmarshaling a JSON
		// text into a []byte.
		ByteStringExpectedFormat: cbor.ByteStringExpectedBase64,

		// Reject the arbitrary-precision integer tags because they can't be faithfully
		// roundtripped through the allowable Unstructured types.
		BignumTag: cbor.BignumTagForbidden,

		// Reject anything other than the simple values true, false, and null.
		SimpleValues: simpleValues,
	}.DecMode()
	if err != nil {
		panic(err)
	}
	return decode
}()

// DecodeLax is derived from Decode, but does not complain about unknown fields in the input.
var DecodeLax cbor.DecMode = func() cbor.DecMode {
	opts := Decode.DecOptions()
	opts.ExtraReturnErrors &^= cbor.ExtraDecErrorUnknownField // clear bit
	dm, err := opts.DecMode()
	if err != nil {
		panic(err)
	}
	return dm
}()
