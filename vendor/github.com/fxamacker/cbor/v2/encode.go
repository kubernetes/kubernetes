// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"math/rand"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/x448/float16"
)

// Marshal returns the CBOR encoding of v using default encoding options.
// See EncOptions for encoding options.
//
// Marshal uses the following encoding rules:
//
// If value implements the Marshaler interface, Marshal calls its
// MarshalCBOR method.
//
// If value implements encoding.BinaryMarshaler, Marhsal calls its
// MarshalBinary method and encode it as CBOR byte string.
//
// Boolean values encode as CBOR booleans (type 7).
//
// Positive integer values encode as CBOR positive integers (type 0).
//
// Negative integer values encode as CBOR negative integers (type 1).
//
// Floating point values encode as CBOR floating points (type 7).
//
// String values encode as CBOR text strings (type 3).
//
// []byte values encode as CBOR byte strings (type 2).
//
// Array and slice values encode as CBOR arrays (type 4).
//
// Map values encode as CBOR maps (type 5).
//
// Struct values encode as CBOR maps (type 5).  Each exported struct field
// becomes a pair with field name encoded as CBOR text string (type 3) and
// field value encoded based on its type.  See struct tag option "keyasint"
// to encode field name as CBOR integer (type 0 and 1).  Also see struct
// tag option "toarray" for special field "_" to encode struct values as
// CBOR array (type 4).
//
// Marshal supports format string stored under the "cbor" key in the struct
// field's tag.  CBOR format string can specify the name of the field,
// "omitempty" and "keyasint" options, and special case "-" for field omission.
// If "cbor" key is absent, Marshal uses "json" key.
//
// Struct field name is treated as integer if it has "keyasint" option in
// its format string.  The format string must specify an integer as its
// field name.
//
// Special struct field "_" is used to specify struct level options, such as
// "toarray". "toarray" option enables Go struct to be encoded as CBOR array.
// "omitempty" is disabled by "toarray" to ensure that the same number
// of elements are encoded every time.
//
// Anonymous struct fields are marshaled as if their exported fields
// were fields in the outer struct.  Marshal follows the same struct fields
// visibility rules used by JSON encoding package.
//
// time.Time values encode as text strings specified in RFC3339 or numerical
// representation of seconds since January 1, 1970 UTC depending on
// EncOptions.Time setting.  Also See EncOptions.TimeTag to encode
// time.Time as CBOR tag with tag number 0 or 1.
//
// big.Int values encode as CBOR integers (type 0 and 1) if values fit.
// Otherwise, big.Int values encode as CBOR bignums (tag 2 and 3).  See
// EncOptions.BigIntConvert to always encode big.Int values as CBOR
// bignums.
//
// Pointer values encode as the value pointed to.
//
// Interface values encode as the value stored in the interface.
//
// Nil slice/map/pointer/interface values encode as CBOR nulls (type 7).
//
// Values of other types cannot be encoded in CBOR.  Attempting
// to encode such a value causes Marshal to return an UnsupportedTypeError.
func Marshal(v interface{}) ([]byte, error) {
	return defaultEncMode.Marshal(v)
}

// MarshalToBuffer encodes v into provided buffer (instead of using built-in buffer pool)
// and uses default encoding options.
//
// NOTE: Unlike Marshal, the buffer provided to MarshalToBuffer can contain
// partially encoded data if error is returned.
//
// See Marshal for more details.
func MarshalToBuffer(v interface{}, buf *bytes.Buffer) error {
	return defaultEncMode.MarshalToBuffer(v, buf)
}

// Marshaler is the interface implemented by types that can marshal themselves
// into valid CBOR.
type Marshaler interface {
	MarshalCBOR() ([]byte, error)
}

// MarshalerError represents error from checking encoded CBOR data item
// returned from MarshalCBOR for well-formedness and some very limited tag validation.
type MarshalerError struct {
	typ reflect.Type
	err error
}

func (e *MarshalerError) Error() string {
	return "cbor: error calling MarshalCBOR for type " +
		e.typ.String() +
		": " + e.err.Error()
}

func (e *MarshalerError) Unwrap() error {
	return e.err
}

// UnsupportedTypeError is returned by Marshal when attempting to encode value
// of an unsupported type.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) Error() string {
	return "cbor: unsupported type: " + e.Type.String()
}

// UnsupportedValueError is returned by Marshal when attempting to encode an
// unsupported value.
type UnsupportedValueError struct {
	msg string
}

func (e *UnsupportedValueError) Error() string {
	return "cbor: unsupported value: " + e.msg
}

// SortMode identifies supported sorting order.
type SortMode int

const (
	// SortNone encodes map pairs and struct fields in an arbitrary order.
	SortNone SortMode = 0

	// SortLengthFirst causes map keys or struct fields to be sorted such that:
	//     - If two keys have different lengths, the shorter one sorts earlier;
	//     - If two keys have the same length, the one with the lower value in
	//       (byte-wise) lexical order sorts earlier.
	// It is used in "Canonical CBOR" encoding in RFC 7049 3.9.
	SortLengthFirst SortMode = 1

	// SortBytewiseLexical causes map keys or struct fields to be sorted in the
	// bytewise lexicographic order of their deterministic CBOR encodings.
	// It is used in "CTAP2 Canonical CBOR" and "Core Deterministic Encoding"
	// in RFC 7049bis.
	SortBytewiseLexical SortMode = 2

	// SortShuffle encodes map pairs and struct fields in a shuffled
	// order. This mode does not guarantee an unbiased permutation, but it
	// does guarantee that the runtime of the shuffle algorithm used will be
	// constant.
	SortFastShuffle SortMode = 3

	// SortCanonical is used in "Canonical CBOR" encoding in RFC 7049 3.9.
	SortCanonical SortMode = SortLengthFirst

	// SortCTAP2 is used in "CTAP2 Canonical CBOR".
	SortCTAP2 SortMode = SortBytewiseLexical

	// SortCoreDeterministic is used in "Core Deterministic Encoding" in RFC 7049bis.
	SortCoreDeterministic SortMode = SortBytewiseLexical

	maxSortMode SortMode = 4
)

func (sm SortMode) valid() bool {
	return sm >= 0 && sm < maxSortMode
}

// StringMode specifies how to encode Go string values.
type StringMode int

const (
	// StringToTextString encodes Go string to CBOR text string (major type 3).
	StringToTextString StringMode = iota

	// StringToByteString encodes Go string to CBOR byte string (major type 2).
	StringToByteString
)

func (st StringMode) cborType() (cborType, error) {
	switch st {
	case StringToTextString:
		return cborTypeTextString, nil

	case StringToByteString:
		return cborTypeByteString, nil
	}
	return 0, errors.New("cbor: invalid StringType " + strconv.Itoa(int(st)))
}

// ShortestFloatMode specifies which floating-point format should
// be used as the shortest possible format for CBOR encoding.
// It is not used for encoding Infinity and NaN values.
type ShortestFloatMode int

const (
	// ShortestFloatNone makes float values encode without any conversion.
	// This is the default for ShortestFloatMode in v1.
	// E.g. a float32 in Go will encode to CBOR float32.  And
	// a float64 in Go will encode to CBOR float64.
	ShortestFloatNone ShortestFloatMode = iota

	// ShortestFloat16 specifies float16 as the shortest form that preserves value.
	// E.g. if float64 can convert to float32 while preserving value, then
	// encoding will also try to convert float32 to float16.  So a float64 might
	// encode as CBOR float64, float32 or float16 depending on the value.
	ShortestFloat16

	maxShortestFloat
)

func (sfm ShortestFloatMode) valid() bool {
	return sfm >= 0 && sfm < maxShortestFloat
}

// NaNConvertMode specifies how to encode NaN and overrides ShortestFloatMode.
// ShortestFloatMode is not used for encoding Infinity and NaN values.
type NaNConvertMode int

const (
	// NaNConvert7e00 always encodes NaN to 0xf97e00 (CBOR float16 = 0x7e00).
	NaNConvert7e00 NaNConvertMode = iota

	// NaNConvertNone never modifies or converts NaN to other representations
	// (float64 NaN stays float64, etc. even if it can use float16 without losing
	// any bits).
	NaNConvertNone

	// NaNConvertPreserveSignal converts NaN to the smallest form that preserves
	// value (quiet bit + payload) as described in RFC 7049bis Draft 12.
	NaNConvertPreserveSignal

	// NaNConvertQuiet always forces quiet bit = 1 and shortest form that preserves
	// NaN payload.
	NaNConvertQuiet

	// NaNConvertReject returns UnsupportedValueError on attempts to encode a NaN value.
	NaNConvertReject

	maxNaNConvert
)

func (ncm NaNConvertMode) valid() bool {
	return ncm >= 0 && ncm < maxNaNConvert
}

// InfConvertMode specifies how to encode Infinity and overrides ShortestFloatMode.
// ShortestFloatMode is not used for encoding Infinity and NaN values.
type InfConvertMode int

const (
	// InfConvertFloat16 always converts Inf to lossless IEEE binary16 (float16).
	InfConvertFloat16 InfConvertMode = iota

	// InfConvertNone never converts (used by CTAP2 Canonical CBOR).
	InfConvertNone

	// InfConvertReject returns UnsupportedValueError on attempts to encode an infinite value.
	InfConvertReject

	maxInfConvert
)

func (icm InfConvertMode) valid() bool {
	return icm >= 0 && icm < maxInfConvert
}

// TimeMode specifies how to encode time.Time values.
type TimeMode int

const (
	// TimeUnix causes time.Time to be encoded as epoch time in integer with second precision.
	TimeUnix TimeMode = iota

	// TimeUnixMicro causes time.Time to be encoded as epoch time in float-point rounded to microsecond precision.
	TimeUnixMicro

	// TimeUnixDynamic causes time.Time to be encoded as integer if time.Time doesn't have fractional seconds,
	// otherwise float-point rounded to microsecond precision.
	TimeUnixDynamic

	// TimeRFC3339 causes time.Time to be encoded as RFC3339 formatted string with second precision.
	TimeRFC3339

	// TimeRFC3339Nano causes time.Time to be encoded as RFC3339 formatted string with nanosecond precision.
	TimeRFC3339Nano

	maxTimeMode
)

func (tm TimeMode) valid() bool {
	return tm >= 0 && tm < maxTimeMode
}

// BigIntConvertMode specifies how to encode big.Int values.
type BigIntConvertMode int

const (
	// BigIntConvertShortest makes big.Int encode to CBOR integer if value fits.
	// E.g. if big.Int value can be converted to CBOR integer while preserving
	// value, encoder will encode it to CBOR integer (major type 0 or 1).
	BigIntConvertShortest BigIntConvertMode = iota

	// BigIntConvertNone makes big.Int encode to CBOR bignum (tag 2 or 3) without
	// converting it to another CBOR type.
	BigIntConvertNone

	// BigIntConvertReject returns an UnsupportedTypeError instead of marshaling a big.Int.
	BigIntConvertReject

	maxBigIntConvert
)

func (bim BigIntConvertMode) valid() bool {
	return bim >= 0 && bim < maxBigIntConvert
}

// NilContainersMode specifies how to encode nil slices and maps.
type NilContainersMode int

const (
	// NilContainerAsNull encodes nil slices and maps as CBOR null.
	// This is the default.
	NilContainerAsNull NilContainersMode = iota

	// NilContainerAsEmpty encodes nil slices and maps as
	// empty container (CBOR bytestring, array, or map).
	NilContainerAsEmpty

	maxNilContainersMode
)

func (m NilContainersMode) valid() bool {
	return m >= 0 && m < maxNilContainersMode
}

// OmitEmptyMode specifies how to encode struct fields with omitempty tag.
// The default behavior omits if field value would encode as empty CBOR value.
type OmitEmptyMode int

const (
	// OmitEmptyCBORValue specifies that struct fields tagged with "omitempty"
	// should be omitted from encoding if the field would be encoded as an empty
	// CBOR value, such as CBOR false, 0, 0.0, nil, empty byte, empty string,
	// empty array, or empty map.
	OmitEmptyCBORValue OmitEmptyMode = iota

	// OmitEmptyGoValue specifies that struct fields tagged with "omitempty"
	// should be omitted from encoding if the field has an empty Go value,
	// defined as false, 0, 0.0, a nil pointer, a nil interface value, and
	// any empty array, slice, map, or string.
	// This behavior is the same as the current (aka v1) encoding/json package
	// included in Go.
	OmitEmptyGoValue

	maxOmitEmptyMode
)

func (om OmitEmptyMode) valid() bool {
	return om >= 0 && om < maxOmitEmptyMode
}

// FieldNameMode specifies the CBOR type to use when encoding struct field names.
type FieldNameMode int

const (
	// FieldNameToTextString encodes struct fields to CBOR text string (major type 3).
	FieldNameToTextString FieldNameMode = iota

	// FieldNameToTextString encodes struct fields to CBOR byte string (major type 2).
	FieldNameToByteString

	maxFieldNameMode
)

func (fnm FieldNameMode) valid() bool {
	return fnm >= 0 && fnm < maxFieldNameMode
}

// ByteSliceLaterFormatMode specifies which later format conversion hint (CBOR tag 21-23)
// to include (if any) when encoding Go byte slice to CBOR byte string. The encoder will
// always encode unmodified bytes from the byte slice and just wrap it within
// CBOR tag 21, 22, or 23 if specified.
// See "Expected Later Encoding for CBOR-to-JSON Converters" in RFC 8949 Section 3.4.5.2.
type ByteSliceLaterFormatMode int

const (
	// ByteSliceLaterFormatNone encodes unmodified bytes from Go byte slice to CBOR byte string (major type 2)
	// without adding CBOR tag 21, 22, or 23.
	ByteSliceLaterFormatNone ByteSliceLaterFormatMode = iota

	// ByteSliceLaterFormatBase64URL encodes unmodified bytes from Go byte slice to CBOR byte string (major type 2)
	// inside CBOR tag 21 (expected later conversion to base64url encoding, see RFC 8949 Section 3.4.5.2).
	ByteSliceLaterFormatBase64URL

	// ByteSliceLaterFormatBase64 encodes unmodified bytes from Go byte slice to CBOR byte string (major type 2)
	// inside CBOR tag 22 (expected later conversion to base64 encoding, see RFC 8949 Section 3.4.5.2).
	ByteSliceLaterFormatBase64

	// ByteSliceLaterFormatBase16 encodes unmodified bytes from Go byte slice to CBOR byte string (major type 2)
	// inside CBOR tag 23 (expected later conversion to base16 encoding, see RFC 8949 Section 3.4.5.2).
	ByteSliceLaterFormatBase16
)

func (bsefm ByteSliceLaterFormatMode) encodingTag() (uint64, error) {
	switch bsefm {
	case ByteSliceLaterFormatNone:
		return 0, nil

	case ByteSliceLaterFormatBase64URL:
		return tagNumExpectedLaterEncodingBase64URL, nil

	case ByteSliceLaterFormatBase64:
		return tagNumExpectedLaterEncodingBase64, nil

	case ByteSliceLaterFormatBase16:
		return tagNumExpectedLaterEncodingBase16, nil
	}
	return 0, errors.New("cbor: invalid ByteSliceLaterFormat " + strconv.Itoa(int(bsefm)))
}

// ByteArrayMode specifies how to encode byte arrays.
type ByteArrayMode int

const (
	// ByteArrayToByteSlice encodes byte arrays the same way that a byte slice with identical
	// length and contents is encoded.
	ByteArrayToByteSlice ByteArrayMode = iota

	// ByteArrayToArray encodes byte arrays to the CBOR array type with one unsigned integer
	// item for each byte in the array.
	ByteArrayToArray

	maxByteArrayMode
)

func (bam ByteArrayMode) valid() bool {
	return bam >= 0 && bam < maxByteArrayMode
}

// BinaryMarshalerMode specifies how to encode types that implement encoding.BinaryMarshaler.
type BinaryMarshalerMode int

const (
	// BinaryMarshalerByteString encodes the output of MarshalBinary to a CBOR byte string.
	BinaryMarshalerByteString BinaryMarshalerMode = iota

	// BinaryMarshalerNone does not recognize BinaryMarshaler implementations during encode.
	BinaryMarshalerNone

	maxBinaryMarshalerMode
)

func (bmm BinaryMarshalerMode) valid() bool {
	return bmm >= 0 && bmm < maxBinaryMarshalerMode
}

// EncOptions specifies encoding options.
type EncOptions struct {
	// Sort specifies sorting order.
	Sort SortMode

	// ShortestFloat specifies the shortest floating-point encoding that preserves
	// the value being encoded.
	ShortestFloat ShortestFloatMode

	// NaNConvert specifies how to encode NaN and it overrides ShortestFloatMode.
	NaNConvert NaNConvertMode

	// InfConvert specifies how to encode Inf and it overrides ShortestFloatMode.
	InfConvert InfConvertMode

	// BigIntConvert specifies how to encode big.Int values.
	BigIntConvert BigIntConvertMode

	// Time specifies how to encode time.Time.
	Time TimeMode

	// TimeTag allows time.Time to be encoded with a tag number.
	// RFC3339 format gets tag number 0, and numeric epoch time tag number 1.
	TimeTag EncTagMode

	// IndefLength specifies whether to allow indefinite length CBOR items.
	IndefLength IndefLengthMode

	// NilContainers specifies how to encode nil slices and maps.
	NilContainers NilContainersMode

	// TagsMd specifies whether to allow CBOR tags (major type 6).
	TagsMd TagsMode

	// OmitEmptyMode specifies how to encode struct fields with omitempty tag.
	OmitEmpty OmitEmptyMode

	// String specifies which CBOR type to use when encoding Go strings.
	// - CBOR text string (major type 3) is default
	// - CBOR byte string (major type 2)
	String StringMode

	// FieldName specifies the CBOR type to use when encoding struct field names.
	FieldName FieldNameMode

	// ByteSliceLaterFormat specifies which later format conversion hint (CBOR tag 21-23)
	// to include (if any) when encoding Go byte slice to CBOR byte string. The encoder will
	// always encode unmodified bytes from the byte slice and just wrap it within
	// CBOR tag 21, 22, or 23 if specified.
	// See "Expected Later Encoding for CBOR-to-JSON Converters" in RFC 8949 Section 3.4.5.2.
	ByteSliceLaterFormat ByteSliceLaterFormatMode

	// ByteArray specifies how to encode byte arrays.
	ByteArray ByteArrayMode

	// BinaryMarshaler specifies how to encode types that implement encoding.BinaryMarshaler.
	BinaryMarshaler BinaryMarshalerMode
}

// CanonicalEncOptions returns EncOptions for "Canonical CBOR" encoding,
// defined in RFC 7049 Section 3.9 with the following rules:
//
//  1. "Integers must be as small as possible."
//  2. "The expression of lengths in major types 2 through 5 must be as short as possible."
//  3. The keys in every map must be sorted in length-first sorting order.
//     See SortLengthFirst for details.
//  4. "Indefinite-length items must be made into definite-length items."
//  5. "If a protocol allows for IEEE floats, then additional canonicalization rules might
//     need to be added.  One example rule might be to have all floats start as a 64-bit
//     float, then do a test conversion to a 32-bit float; if the result is the same numeric
//     value, use the shorter value and repeat the process with a test conversion to a
//     16-bit float.  (This rule selects 16-bit float for positive and negative Infinity
//     as well.)  Also, there are many representations for NaN.  If NaN is an allowed value,
//     it must always be represented as 0xf97e00."
func CanonicalEncOptions() EncOptions {
	return EncOptions{
		Sort:          SortCanonical,
		ShortestFloat: ShortestFloat16,
		NaNConvert:    NaNConvert7e00,
		InfConvert:    InfConvertFloat16,
		IndefLength:   IndefLengthForbidden,
	}
}

// CTAP2EncOptions returns EncOptions for "CTAP2 Canonical CBOR" encoding,
// defined in CTAP specification, with the following rules:
//
//  1. "Integers must be encoded as small as possible."
//  2. "The representations of any floating-point values are not changed."
//  3. "The expression of lengths in major types 2 through 5 must be as short as possible."
//  4. "Indefinite-length items must be made into definite-length items.""
//  5. The keys in every map must be sorted in bytewise lexicographic order.
//     See SortBytewiseLexical for details.
//  6. "Tags as defined in Section 2.4 in [RFC7049] MUST NOT be present."
func CTAP2EncOptions() EncOptions {
	return EncOptions{
		Sort:          SortCTAP2,
		ShortestFloat: ShortestFloatNone,
		NaNConvert:    NaNConvertNone,
		InfConvert:    InfConvertNone,
		IndefLength:   IndefLengthForbidden,
		TagsMd:        TagsForbidden,
	}
}

// CoreDetEncOptions returns EncOptions for "Core Deterministic" encoding,
// defined in RFC 7049bis with the following rules:
//
//  1. "Preferred serialization MUST be used. In particular, this means that arguments
//     (see Section 3) for integers, lengths in major types 2 through 5, and tags MUST
//     be as short as possible"
//     "Floating point values also MUST use the shortest form that preserves the value"
//  2. "Indefinite-length items MUST NOT appear."
//  3. "The keys in every map MUST be sorted in the bytewise lexicographic order of
//     their deterministic encodings."
func CoreDetEncOptions() EncOptions {
	return EncOptions{
		Sort:          SortCoreDeterministic,
		ShortestFloat: ShortestFloat16,
		NaNConvert:    NaNConvert7e00,
		InfConvert:    InfConvertFloat16,
		IndefLength:   IndefLengthForbidden,
	}
}

// PreferredUnsortedEncOptions returns EncOptions for "Preferred Serialization" encoding,
// defined in RFC 7049bis with the following rules:
//
//  1. "The preferred serialization always uses the shortest form of representing the argument
//     (Section 3);"
//  2. "it also uses the shortest floating-point encoding that preserves the value being
//     encoded (see Section 5.5)."
//     "The preferred encoding for a floating-point value is the shortest floating-point encoding
//     that preserves its value, e.g., 0xf94580 for the number 5.5, and 0xfa45ad9c00 for the
//     number 5555.5, unless the CBOR-based protocol specifically excludes the use of the shorter
//     floating-point encodings. For NaN values, a shorter encoding is preferred if zero-padding
//     the shorter significand towards the right reconstitutes the original NaN value (for many
//     applications, the single NaN encoding 0xf97e00 will suffice)."
//  3. "Definite length encoding is preferred whenever the length is known at the time the
//     serialization of the item starts."
func PreferredUnsortedEncOptions() EncOptions {
	return EncOptions{
		Sort:          SortNone,
		ShortestFloat: ShortestFloat16,
		NaNConvert:    NaNConvert7e00,
		InfConvert:    InfConvertFloat16,
	}
}

// EncMode returns EncMode with immutable options and no tags (safe for concurrency).
func (opts EncOptions) EncMode() (EncMode, error) { //nolint:gocritic // ignore hugeParam
	return opts.encMode()
}

// UserBufferEncMode returns UserBufferEncMode with immutable options and no tags (safe for concurrency).
func (opts EncOptions) UserBufferEncMode() (UserBufferEncMode, error) { //nolint:gocritic // ignore hugeParam
	return opts.encMode()
}

// EncModeWithTags returns EncMode with options and tags that are both immutable (safe for concurrency).
func (opts EncOptions) EncModeWithTags(tags TagSet) (EncMode, error) { //nolint:gocritic // ignore hugeParam
	return opts.UserBufferEncModeWithTags(tags)
}

// UserBufferEncModeWithTags returns UserBufferEncMode with options and tags that are both immutable (safe for concurrency).
func (opts EncOptions) UserBufferEncModeWithTags(tags TagSet) (UserBufferEncMode, error) { //nolint:gocritic // ignore hugeParam
	if opts.TagsMd == TagsForbidden {
		return nil, errors.New("cbor: cannot create EncMode with TagSet when TagsMd is TagsForbidden")
	}
	if tags == nil {
		return nil, errors.New("cbor: cannot create EncMode with nil value as TagSet")
	}
	em, err := opts.encMode()
	if err != nil {
		return nil, err
	}
	// Copy tags
	ts := tagSet(make(map[reflect.Type]*tagItem))
	syncTags := tags.(*syncTagSet)
	syncTags.RLock()
	for contentType, tag := range syncTags.t {
		if tag.opts.EncTag != EncTagNone {
			ts[contentType] = tag
		}
	}
	syncTags.RUnlock()
	if len(ts) > 0 {
		em.tags = ts
	}
	return em, nil
}

// EncModeWithSharedTags returns EncMode with immutable options and mutable shared tags (safe for concurrency).
func (opts EncOptions) EncModeWithSharedTags(tags TagSet) (EncMode, error) { //nolint:gocritic // ignore hugeParam
	return opts.UserBufferEncModeWithSharedTags(tags)
}

// UserBufferEncModeWithSharedTags returns UserBufferEncMode with immutable options and mutable shared tags (safe for concurrency).
func (opts EncOptions) UserBufferEncModeWithSharedTags(tags TagSet) (UserBufferEncMode, error) { //nolint:gocritic // ignore hugeParam
	if opts.TagsMd == TagsForbidden {
		return nil, errors.New("cbor: cannot create EncMode with TagSet when TagsMd is TagsForbidden")
	}
	if tags == nil {
		return nil, errors.New("cbor: cannot create EncMode with nil value as TagSet")
	}
	em, err := opts.encMode()
	if err != nil {
		return nil, err
	}
	em.tags = tags
	return em, nil
}

func (opts EncOptions) encMode() (*encMode, error) { //nolint:gocritic // ignore hugeParam
	if !opts.Sort.valid() {
		return nil, errors.New("cbor: invalid SortMode " + strconv.Itoa(int(opts.Sort)))
	}
	if !opts.ShortestFloat.valid() {
		return nil, errors.New("cbor: invalid ShortestFloatMode " + strconv.Itoa(int(opts.ShortestFloat)))
	}
	if !opts.NaNConvert.valid() {
		return nil, errors.New("cbor: invalid NaNConvertMode " + strconv.Itoa(int(opts.NaNConvert)))
	}
	if !opts.InfConvert.valid() {
		return nil, errors.New("cbor: invalid InfConvertMode " + strconv.Itoa(int(opts.InfConvert)))
	}
	if !opts.BigIntConvert.valid() {
		return nil, errors.New("cbor: invalid BigIntConvertMode " + strconv.Itoa(int(opts.BigIntConvert)))
	}
	if !opts.Time.valid() {
		return nil, errors.New("cbor: invalid TimeMode " + strconv.Itoa(int(opts.Time)))
	}
	if !opts.TimeTag.valid() {
		return nil, errors.New("cbor: invalid TimeTag " + strconv.Itoa(int(opts.TimeTag)))
	}
	if !opts.IndefLength.valid() {
		return nil, errors.New("cbor: invalid IndefLength " + strconv.Itoa(int(opts.IndefLength)))
	}
	if !opts.NilContainers.valid() {
		return nil, errors.New("cbor: invalid NilContainers " + strconv.Itoa(int(opts.NilContainers)))
	}
	if !opts.TagsMd.valid() {
		return nil, errors.New("cbor: invalid TagsMd " + strconv.Itoa(int(opts.TagsMd)))
	}
	if opts.TagsMd == TagsForbidden && opts.TimeTag == EncTagRequired {
		return nil, errors.New("cbor: cannot set TagsMd to TagsForbidden when TimeTag is EncTagRequired")
	}
	if !opts.OmitEmpty.valid() {
		return nil, errors.New("cbor: invalid OmitEmpty " + strconv.Itoa(int(opts.OmitEmpty)))
	}
	stringMajorType, err := opts.String.cborType()
	if err != nil {
		return nil, err
	}
	if !opts.FieldName.valid() {
		return nil, errors.New("cbor: invalid FieldName " + strconv.Itoa(int(opts.FieldName)))
	}
	byteSliceLaterEncodingTag, err := opts.ByteSliceLaterFormat.encodingTag()
	if err != nil {
		return nil, err
	}
	if !opts.ByteArray.valid() {
		return nil, errors.New("cbor: invalid ByteArray " + strconv.Itoa(int(opts.ByteArray)))
	}
	if !opts.BinaryMarshaler.valid() {
		return nil, errors.New("cbor: invalid BinaryMarshaler " + strconv.Itoa(int(opts.BinaryMarshaler)))
	}
	em := encMode{
		sort:                      opts.Sort,
		shortestFloat:             opts.ShortestFloat,
		nanConvert:                opts.NaNConvert,
		infConvert:                opts.InfConvert,
		bigIntConvert:             opts.BigIntConvert,
		time:                      opts.Time,
		timeTag:                   opts.TimeTag,
		indefLength:               opts.IndefLength,
		nilContainers:             opts.NilContainers,
		tagsMd:                    opts.TagsMd,
		omitEmpty:                 opts.OmitEmpty,
		stringType:                opts.String,
		stringMajorType:           stringMajorType,
		fieldName:                 opts.FieldName,
		byteSliceLaterFormat:      opts.ByteSliceLaterFormat,
		byteSliceLaterEncodingTag: byteSliceLaterEncodingTag,
		byteArray:                 opts.ByteArray,
		binaryMarshaler:           opts.BinaryMarshaler,
	}
	return &em, nil
}

// EncMode is the main interface for CBOR encoding.
type EncMode interface {
	Marshal(v interface{}) ([]byte, error)
	NewEncoder(w io.Writer) *Encoder
	EncOptions() EncOptions
}

// UserBufferEncMode is an interface for CBOR encoding, which extends EncMode by
// adding MarshalToBuffer to support user specified buffer rather than encoding
// into the built-in buffer pool.
type UserBufferEncMode interface {
	EncMode
	MarshalToBuffer(v interface{}, buf *bytes.Buffer) error

	// This private method is to prevent users implementing
	// this interface and so future additions to it will
	// not be breaking changes.
	// See https://go.dev/blog/module-compatibility
	unexport()
}

type encMode struct {
	tags                      tagProvider
	sort                      SortMode
	shortestFloat             ShortestFloatMode
	nanConvert                NaNConvertMode
	infConvert                InfConvertMode
	bigIntConvert             BigIntConvertMode
	time                      TimeMode
	timeTag                   EncTagMode
	indefLength               IndefLengthMode
	nilContainers             NilContainersMode
	tagsMd                    TagsMode
	omitEmpty                 OmitEmptyMode
	stringType                StringMode
	stringMajorType           cborType
	fieldName                 FieldNameMode
	byteSliceLaterFormat      ByteSliceLaterFormatMode
	byteSliceLaterEncodingTag uint64
	byteArray                 ByteArrayMode
	binaryMarshaler           BinaryMarshalerMode
}

var defaultEncMode, _ = EncOptions{}.encMode()

// These four decoding modes are used by getMarshalerDecMode.
// maxNestedLevels, maxArrayElements, and maxMapPairs are
// set to max allowed limits to avoid rejecting Marshaler
// output that would have been the allowable output of a
// non-Marshaler object that exceeds default limits.
var (
	marshalerForbidIndefLengthForbidTagsDecMode = decMode{
		maxNestedLevels:  maxMaxNestedLevels,
		maxArrayElements: maxMaxArrayElements,
		maxMapPairs:      maxMaxMapPairs,
		indefLength:      IndefLengthForbidden,
		tagsMd:           TagsForbidden,
	}

	marshalerAllowIndefLengthForbidTagsDecMode = decMode{
		maxNestedLevels:  maxMaxNestedLevels,
		maxArrayElements: maxMaxArrayElements,
		maxMapPairs:      maxMaxMapPairs,
		indefLength:      IndefLengthAllowed,
		tagsMd:           TagsForbidden,
	}

	marshalerForbidIndefLengthAllowTagsDecMode = decMode{
		maxNestedLevels:  maxMaxNestedLevels,
		maxArrayElements: maxMaxArrayElements,
		maxMapPairs:      maxMaxMapPairs,
		indefLength:      IndefLengthForbidden,
		tagsMd:           TagsAllowed,
	}

	marshalerAllowIndefLengthAllowTagsDecMode = decMode{
		maxNestedLevels:  maxMaxNestedLevels,
		maxArrayElements: maxMaxArrayElements,
		maxMapPairs:      maxMaxMapPairs,
		indefLength:      IndefLengthAllowed,
		tagsMd:           TagsAllowed,
	}
)

// getMarshalerDecMode returns one of four existing decoding modes
// which can be reused (safe for parallel use) for the purpose of
// checking if data returned by Marshaler is well-formed.
func getMarshalerDecMode(indefLength IndefLengthMode, tagsMd TagsMode) *decMode {
	switch {
	case indefLength == IndefLengthAllowed && tagsMd == TagsAllowed:
		return &marshalerAllowIndefLengthAllowTagsDecMode

	case indefLength == IndefLengthAllowed && tagsMd == TagsForbidden:
		return &marshalerAllowIndefLengthForbidTagsDecMode

	case indefLength == IndefLengthForbidden && tagsMd == TagsAllowed:
		return &marshalerForbidIndefLengthAllowTagsDecMode

	case indefLength == IndefLengthForbidden && tagsMd == TagsForbidden:
		return &marshalerForbidIndefLengthForbidTagsDecMode

	default:
		// This should never happen, unless we add new options to
		// IndefLengthMode or TagsMode without updating this function.
		return &decMode{
			maxNestedLevels:  maxMaxNestedLevels,
			maxArrayElements: maxMaxArrayElements,
			maxMapPairs:      maxMaxMapPairs,
			indefLength:      indefLength,
			tagsMd:           tagsMd,
		}
	}
}

// EncOptions returns user specified options used to create this EncMode.
func (em *encMode) EncOptions() EncOptions {
	return EncOptions{
		Sort:                 em.sort,
		ShortestFloat:        em.shortestFloat,
		NaNConvert:           em.nanConvert,
		InfConvert:           em.infConvert,
		BigIntConvert:        em.bigIntConvert,
		Time:                 em.time,
		TimeTag:              em.timeTag,
		IndefLength:          em.indefLength,
		NilContainers:        em.nilContainers,
		TagsMd:               em.tagsMd,
		OmitEmpty:            em.omitEmpty,
		String:               em.stringType,
		FieldName:            em.fieldName,
		ByteSliceLaterFormat: em.byteSliceLaterFormat,
		ByteArray:            em.byteArray,
		BinaryMarshaler:      em.binaryMarshaler,
	}
}

func (em *encMode) unexport() {}

func (em *encMode) encTagBytes(t reflect.Type) []byte {
	if em.tags != nil {
		if tagItem := em.tags.getTagItemFromType(t); tagItem != nil {
			return tagItem.cborTagNum
		}
	}
	return nil
}

// Marshal returns the CBOR encoding of v using em encoding mode.
//
// See the documentation for Marshal for details.
func (em *encMode) Marshal(v interface{}) ([]byte, error) {
	e := getEncodeBuffer()

	if err := encode(e, em, reflect.ValueOf(v)); err != nil {
		putEncodeBuffer(e)
		return nil, err
	}

	buf := make([]byte, e.Len())
	copy(buf, e.Bytes())

	putEncodeBuffer(e)
	return buf, nil
}

// MarshalToBuffer encodes v into provided buffer (instead of using built-in buffer pool)
// and uses em encoding mode.
//
// NOTE: Unlike Marshal, the buffer provided to MarshalToBuffer can contain
// partially encoded data if error is returned.
//
// See Marshal for more details.
func (em *encMode) MarshalToBuffer(v interface{}, buf *bytes.Buffer) error {
	if buf == nil {
		return fmt.Errorf("cbor: encoding buffer provided by user is nil")
	}
	return encode(buf, em, reflect.ValueOf(v))
}

// NewEncoder returns a new encoder that writes to w using em EncMode.
func (em *encMode) NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w, em: em}
}

// encodeBufferPool caches unused bytes.Buffer objects for later reuse.
var encodeBufferPool = sync.Pool{
	New: func() interface{} {
		e := new(bytes.Buffer)
		e.Grow(32) // TODO: make this configurable
		return e
	},
}

func getEncodeBuffer() *bytes.Buffer {
	return encodeBufferPool.Get().(*bytes.Buffer)
}

func putEncodeBuffer(e *bytes.Buffer) {
	e.Reset()
	encodeBufferPool.Put(e)
}

type encodeFunc func(e *bytes.Buffer, em *encMode, v reflect.Value) error
type isEmptyFunc func(em *encMode, v reflect.Value) (empty bool, err error)

func encode(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if !v.IsValid() {
		// v is zero value
		e.Write(cborNil)
		return nil
	}
	vt := v.Type()
	f, _ := getEncodeFunc(vt)
	if f == nil {
		return &UnsupportedTypeError{vt}
	}

	return f(e, em, v)
}

func encodeBool(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	b := cborFalse
	if v.Bool() {
		b = cborTrue
	}
	e.Write(b)
	return nil
}

func encodeInt(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	i := v.Int()
	if i >= 0 {
		encodeHead(e, byte(cborTypePositiveInt), uint64(i))
		return nil
	}
	i = i*(-1) - 1
	encodeHead(e, byte(cborTypeNegativeInt), uint64(i))
	return nil
}

func encodeUint(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	encodeHead(e, byte(cborTypePositiveInt), v.Uint())
	return nil
}

func encodeFloat(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	f64 := v.Float()
	if math.IsNaN(f64) {
		return encodeNaN(e, em, v)
	}
	if math.IsInf(f64, 0) {
		return encodeInf(e, em, v)
	}
	fopt := em.shortestFloat
	if v.Kind() == reflect.Float64 && (fopt == ShortestFloatNone || cannotFitFloat32(f64)) {
		// Encode float64
		// Don't use encodeFloat64() because it cannot be inlined.
		const argumentSize = 8
		const headSize = 1 + argumentSize
		var scratch [headSize]byte
		scratch[0] = byte(cborTypePrimitives) | byte(additionalInformationAsFloat64)
		binary.BigEndian.PutUint64(scratch[1:], math.Float64bits(f64))
		e.Write(scratch[:])
		return nil
	}

	f32 := float32(f64)
	if fopt == ShortestFloat16 {
		var f16 float16.Float16
		p := float16.PrecisionFromfloat32(f32)
		if p == float16.PrecisionExact {
			// Roundtrip float32->float16->float32 test isn't needed.
			f16 = float16.Fromfloat32(f32)
		} else if p == float16.PrecisionUnknown {
			// Try roundtrip float32->float16->float32 to determine if float32 can fit into float16.
			f16 = float16.Fromfloat32(f32)
			if f16.Float32() == f32 {
				p = float16.PrecisionExact
			}
		}
		if p == float16.PrecisionExact {
			// Encode float16
			// Don't use encodeFloat16() because it cannot be inlined.
			const argumentSize = 2
			const headSize = 1 + argumentSize
			var scratch [headSize]byte
			scratch[0] = byte(cborTypePrimitives) | additionalInformationAsFloat16
			binary.BigEndian.PutUint16(scratch[1:], uint16(f16))
			e.Write(scratch[:])
			return nil
		}
	}

	// Encode float32
	// Don't use encodeFloat32() because it cannot be inlined.
	const argumentSize = 4
	const headSize = 1 + argumentSize
	var scratch [headSize]byte
	scratch[0] = byte(cborTypePrimitives) | additionalInformationAsFloat32
	binary.BigEndian.PutUint32(scratch[1:], math.Float32bits(f32))
	e.Write(scratch[:])
	return nil
}

func encodeInf(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	f64 := v.Float()
	switch em.infConvert {
	case InfConvertReject:
		return &UnsupportedValueError{msg: "floating-point infinity"}

	case InfConvertFloat16:
		if f64 > 0 {
			e.Write(cborPositiveInfinity)
		} else {
			e.Write(cborNegativeInfinity)
		}
		return nil
	}
	if v.Kind() == reflect.Float64 {
		return encodeFloat64(e, f64)
	}
	return encodeFloat32(e, float32(f64))
}

func encodeNaN(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	switch em.nanConvert {
	case NaNConvert7e00:
		e.Write(cborNaN)
		return nil

	case NaNConvertNone:
		if v.Kind() == reflect.Float64 {
			return encodeFloat64(e, v.Float())
		}
		f32 := float32NaNFromReflectValue(v)
		return encodeFloat32(e, f32)

	case NaNConvertReject:
		return &UnsupportedValueError{msg: "floating-point NaN"}

	default: // NaNConvertPreserveSignal, NaNConvertQuiet
		if v.Kind() == reflect.Float64 {
			f64 := v.Float()
			f64bits := math.Float64bits(f64)
			if em.nanConvert == NaNConvertQuiet && f64bits&(1<<51) == 0 {
				f64bits |= 1 << 51 // Set quiet bit = 1
				f64 = math.Float64frombits(f64bits)
			}
			// The lower 29 bits are dropped when converting from float64 to float32.
			if f64bits&0x1fffffff != 0 {
				// Encode NaN as float64 because dropped coef bits from float64 to float32 are not all 0s.
				return encodeFloat64(e, f64)
			}
			// Create float32 from float64 manually because float32(f64) always turns on NaN's quiet bits.
			sign := uint32(f64bits>>32) & (1 << 31)
			exp := uint32(0x7f800000)
			coef := uint32((f64bits & 0xfffffffffffff) >> 29)
			f32bits := sign | exp | coef
			f32 := math.Float32frombits(f32bits)
			// The lower 13 bits are dropped when converting from float32 to float16.
			if f32bits&0x1fff != 0 {
				// Encode NaN as float32 because dropped coef bits from float32 to float16 are not all 0s.
				return encodeFloat32(e, f32)
			}
			// Encode NaN as float16
			f16, _ := float16.FromNaN32ps(f32) // Ignore err because it only returns error when f32 is not a NaN.
			return encodeFloat16(e, f16)
		}

		f32 := float32NaNFromReflectValue(v)
		f32bits := math.Float32bits(f32)
		if em.nanConvert == NaNConvertQuiet && f32bits&(1<<22) == 0 {
			f32bits |= 1 << 22 // Set quiet bit = 1
			f32 = math.Float32frombits(f32bits)
		}
		// The lower 13 bits are dropped coef bits when converting from float32 to float16.
		if f32bits&0x1fff != 0 {
			// Encode NaN as float32 because dropped coef bits from float32 to float16 are not all 0s.
			return encodeFloat32(e, f32)
		}
		f16, _ := float16.FromNaN32ps(f32) // Ignore err because it only returns error when f32 is not a NaN.
		return encodeFloat16(e, f16)
	}
}

func encodeFloat16(e *bytes.Buffer, f16 float16.Float16) error {
	const argumentSize = 2
	const headSize = 1 + argumentSize
	var scratch [headSize]byte
	scratch[0] = byte(cborTypePrimitives) | additionalInformationAsFloat16
	binary.BigEndian.PutUint16(scratch[1:], uint16(f16))
	e.Write(scratch[:])
	return nil
}

func encodeFloat32(e *bytes.Buffer, f32 float32) error {
	const argumentSize = 4
	const headSize = 1 + argumentSize
	var scratch [headSize]byte
	scratch[0] = byte(cborTypePrimitives) | additionalInformationAsFloat32
	binary.BigEndian.PutUint32(scratch[1:], math.Float32bits(f32))
	e.Write(scratch[:])
	return nil
}

func encodeFloat64(e *bytes.Buffer, f64 float64) error {
	const argumentSize = 8
	const headSize = 1 + argumentSize
	var scratch [headSize]byte
	scratch[0] = byte(cborTypePrimitives) | additionalInformationAsFloat64
	binary.BigEndian.PutUint64(scratch[1:], math.Float64bits(f64))
	e.Write(scratch[:])
	return nil
}

func encodeByteString(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	vk := v.Kind()
	if vk == reflect.Slice && v.IsNil() && em.nilContainers == NilContainerAsNull {
		e.Write(cborNil)
		return nil
	}
	if vk == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 && em.byteSliceLaterEncodingTag != 0 {
		encodeHead(e, byte(cborTypeTag), em.byteSliceLaterEncodingTag)
	}
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	slen := v.Len()
	if slen == 0 {
		return e.WriteByte(byte(cborTypeByteString))
	}
	encodeHead(e, byte(cborTypeByteString), uint64(slen))
	if vk == reflect.Array {
		for i := 0; i < slen; i++ {
			e.WriteByte(byte(v.Index(i).Uint()))
		}
		return nil
	}
	e.Write(v.Bytes())
	return nil
}

func encodeString(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	s := v.String()
	encodeHead(e, byte(em.stringMajorType), uint64(len(s)))
	e.WriteString(s)
	return nil
}

type arrayEncodeFunc struct {
	f encodeFunc
}

func (ae arrayEncodeFunc) encode(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if em.byteArray == ByteArrayToByteSlice && v.Type().Elem().Kind() == reflect.Uint8 {
		return encodeByteString(e, em, v)
	}
	if v.Kind() == reflect.Slice && v.IsNil() && em.nilContainers == NilContainerAsNull {
		e.Write(cborNil)
		return nil
	}
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	alen := v.Len()
	if alen == 0 {
		return e.WriteByte(byte(cborTypeArray))
	}
	encodeHead(e, byte(cborTypeArray), uint64(alen))
	for i := 0; i < alen; i++ {
		if err := ae.f(e, em, v.Index(i)); err != nil {
			return err
		}
	}
	return nil
}

// encodeKeyValueFunc encodes key/value pairs in map (v).
// If kvs is provided (having the same length as v), length of encoded key and value are stored in kvs.
// kvs is used for canonical encoding of map.
type encodeKeyValueFunc func(e *bytes.Buffer, em *encMode, v reflect.Value, kvs []keyValue) error

type mapEncodeFunc struct {
	e encodeKeyValueFunc
}

func (me mapEncodeFunc) encode(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if v.IsNil() && em.nilContainers == NilContainerAsNull {
		e.Write(cborNil)
		return nil
	}
	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}
	mlen := v.Len()
	if mlen == 0 {
		return e.WriteByte(byte(cborTypeMap))
	}

	encodeHead(e, byte(cborTypeMap), uint64(mlen))
	if em.sort == SortNone || em.sort == SortFastShuffle || mlen <= 1 {
		return me.e(e, em, v, nil)
	}

	kvsp := getKeyValues(v.Len()) // for sorting keys
	defer putKeyValues(kvsp)
	kvs := *kvsp

	kvBeginOffset := e.Len()
	if err := me.e(e, em, v, kvs); err != nil {
		return err
	}
	kvTotalLen := e.Len() - kvBeginOffset

	// Use the capacity at the tail of the encode buffer as a staging area to rearrange the
	// encoded pairs into sorted order.
	e.Grow(kvTotalLen)
	tmp := e.Bytes()[e.Len() : e.Len()+kvTotalLen] // Can use e.AvailableBuffer() in Go 1.21+.
	dst := e.Bytes()[kvBeginOffset:]

	if em.sort == SortBytewiseLexical {
		sort.Sort(&bytewiseKeyValueSorter{kvs: kvs, data: dst})
	} else {
		sort.Sort(&lengthFirstKeyValueSorter{kvs: kvs, data: dst})
	}

	// This is where the encoded bytes are actually rearranged in the output buffer to reflect
	// the desired order.
	sortedOffset := 0
	for _, kv := range kvs {
		copy(tmp[sortedOffset:], dst[kv.offset:kv.nextOffset])
		sortedOffset += kv.nextOffset - kv.offset
	}
	copy(dst, tmp[:kvTotalLen])

	return nil

}

// keyValue is the position of an encoded pair in a buffer. All offsets are zero-based and relative
// to the first byte of the first encoded pair.
type keyValue struct {
	offset      int
	valueOffset int
	nextOffset  int
}

type bytewiseKeyValueSorter struct {
	kvs  []keyValue
	data []byte
}

func (x *bytewiseKeyValueSorter) Len() int {
	return len(x.kvs)
}

func (x *bytewiseKeyValueSorter) Swap(i, j int) {
	x.kvs[i], x.kvs[j] = x.kvs[j], x.kvs[i]
}

func (x *bytewiseKeyValueSorter) Less(i, j int) bool {
	kvi, kvj := x.kvs[i], x.kvs[j]
	return bytes.Compare(x.data[kvi.offset:kvi.valueOffset], x.data[kvj.offset:kvj.valueOffset]) <= 0
}

type lengthFirstKeyValueSorter struct {
	kvs  []keyValue
	data []byte
}

func (x *lengthFirstKeyValueSorter) Len() int {
	return len(x.kvs)
}

func (x *lengthFirstKeyValueSorter) Swap(i, j int) {
	x.kvs[i], x.kvs[j] = x.kvs[j], x.kvs[i]
}

func (x *lengthFirstKeyValueSorter) Less(i, j int) bool {
	kvi, kvj := x.kvs[i], x.kvs[j]
	if keyLengthDifference := (kvi.valueOffset - kvi.offset) - (kvj.valueOffset - kvj.offset); keyLengthDifference != 0 {
		return keyLengthDifference < 0
	}
	return bytes.Compare(x.data[kvi.offset:kvi.valueOffset], x.data[kvj.offset:kvj.valueOffset]) <= 0
}

var keyValuePool = sync.Pool{}

func getKeyValues(length int) *[]keyValue {
	v := keyValuePool.Get()
	if v == nil {
		y := make([]keyValue, length)
		return &y
	}
	x := v.(*[]keyValue)
	if cap(*x) >= length {
		*x = (*x)[:length]
		return x
	}
	// []keyValue from the pool does not have enough capacity.
	// Return it back to the pool and create a new one.
	keyValuePool.Put(x)
	y := make([]keyValue, length)
	return &y
}

func putKeyValues(x *[]keyValue) {
	*x = (*x)[:0]
	keyValuePool.Put(x)
}

func encodeStructToArray(e *bytes.Buffer, em *encMode, v reflect.Value) (err error) {
	structType, err := getEncodingStructType(v.Type())
	if err != nil {
		return err
	}

	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}

	flds := structType.fields

	encodeHead(e, byte(cborTypeArray), uint64(len(flds)))
	for i := 0; i < len(flds); i++ {
		f := flds[i]

		var fv reflect.Value
		if len(f.idx) == 1 {
			fv = v.Field(f.idx[0])
		} else {
			// Get embedded field value.  No error is expected.
			fv, _ = getFieldValue(v, f.idx, func(reflect.Value) (reflect.Value, error) {
				// Write CBOR nil for null pointer to embedded struct
				e.Write(cborNil)
				return reflect.Value{}, nil
			})
			if !fv.IsValid() {
				continue
			}
		}

		if err := f.ef(e, em, fv); err != nil {
			return err
		}
	}
	return nil
}

func encodeStruct(e *bytes.Buffer, em *encMode, v reflect.Value) (err error) {
	structType, err := getEncodingStructType(v.Type())
	if err != nil {
		return err
	}

	flds := structType.getFields(em)

	start := 0
	if em.sort == SortFastShuffle {
		start = rand.Intn(len(flds)) //nolint:gosec // Don't need a CSPRNG for deck cutting.
	}

	if b := em.encTagBytes(v.Type()); b != nil {
		e.Write(b)
	}

	// Encode head with struct field count.
	// Head is rewritten later if actual encoded field count is different from struct field count.
	encodedHeadLen := encodeHead(e, byte(cborTypeMap), uint64(len(flds)))

	kvbegin := e.Len()
	kvcount := 0
	for offset := 0; offset < len(flds); offset++ {
		f := flds[(start+offset)%len(flds)]

		var fv reflect.Value
		if len(f.idx) == 1 {
			fv = v.Field(f.idx[0])
		} else {
			// Get embedded field value.  No error is expected.
			fv, _ = getFieldValue(v, f.idx, func(reflect.Value) (reflect.Value, error) {
				// Skip null pointer to embedded struct
				return reflect.Value{}, nil
			})
			if !fv.IsValid() {
				continue
			}
		}
		if f.omitEmpty {
			empty, err := f.ief(em, fv)
			if err != nil {
				return err
			}
			if empty {
				continue
			}
		}

		if !f.keyAsInt && em.fieldName == FieldNameToByteString {
			e.Write(f.cborNameByteString)
		} else { // int or text string
			e.Write(f.cborName)
		}

		if err := f.ef(e, em, fv); err != nil {
			return err
		}

		kvcount++
	}

	if len(flds) == kvcount {
		// Encoded element count in head is the same as actual element count.
		return nil
	}

	// Overwrite the bytes that were reserved for the head before encoding the map entries.
	var actualHeadLen int
	{
		headbuf := *bytes.NewBuffer(e.Bytes()[kvbegin-encodedHeadLen : kvbegin-encodedHeadLen : kvbegin])
		actualHeadLen = encodeHead(&headbuf, byte(cborTypeMap), uint64(kvcount))
	}

	if actualHeadLen == encodedHeadLen {
		// The bytes reserved for the encoded head were exactly the right size, so the
		// encoded entries are already in their final positions.
		return nil
	}

	// We reserved more bytes than needed for the encoded head, based on the number of fields
	// encoded. The encoded entries are offset to the right by the number of excess reserved
	// bytes. Shift the entries left to remove the gap.
	excessReservedBytes := encodedHeadLen - actualHeadLen
	dst := e.Bytes()[kvbegin-excessReservedBytes : e.Len()-excessReservedBytes]
	src := e.Bytes()[kvbegin:e.Len()]
	copy(dst, src)

	// After shifting, the excess bytes are at the end of the output buffer and they are
	// garbage.
	e.Truncate(e.Len() - excessReservedBytes)
	return nil
}

func encodeIntf(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if v.IsNil() {
		e.Write(cborNil)
		return nil
	}
	return encode(e, em, v.Elem())
}

func encodeTime(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	t := v.Interface().(time.Time)
	if t.IsZero() {
		e.Write(cborNil) // Even if tag is required, encode as CBOR null.
		return nil
	}
	if em.timeTag == EncTagRequired {
		tagNumber := 1
		if em.time == TimeRFC3339 || em.time == TimeRFC3339Nano {
			tagNumber = 0
		}
		encodeHead(e, byte(cborTypeTag), uint64(tagNumber))
	}
	switch em.time {
	case TimeUnix:
		secs := t.Unix()
		return encodeInt(e, em, reflect.ValueOf(secs))

	case TimeUnixMicro:
		t = t.UTC().Round(time.Microsecond)
		f := float64(t.UnixNano()) / 1e9
		return encodeFloat(e, em, reflect.ValueOf(f))

	case TimeUnixDynamic:
		t = t.UTC().Round(time.Microsecond)
		secs, nsecs := t.Unix(), uint64(t.Nanosecond())
		if nsecs == 0 {
			return encodeInt(e, em, reflect.ValueOf(secs))
		}
		f := float64(secs) + float64(nsecs)/1e9
		return encodeFloat(e, em, reflect.ValueOf(f))

	case TimeRFC3339:
		s := t.Format(time.RFC3339)
		return encodeString(e, em, reflect.ValueOf(s))

	default: // TimeRFC3339Nano
		s := t.Format(time.RFC3339Nano)
		return encodeString(e, em, reflect.ValueOf(s))
	}
}

func encodeBigInt(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if em.bigIntConvert == BigIntConvertReject {
		return &UnsupportedTypeError{Type: typeBigInt}
	}

	vbi := v.Interface().(big.Int)
	sign := vbi.Sign()
	bi := new(big.Int).SetBytes(vbi.Bytes()) // bi is absolute value of v
	if sign < 0 {
		// For negative number, convert to CBOR encoded number (-v-1).
		bi.Sub(bi, big.NewInt(1))
	}

	if em.bigIntConvert == BigIntConvertShortest {
		if bi.IsUint64() {
			if sign >= 0 {
				// Encode as CBOR pos int (major type 0)
				encodeHead(e, byte(cborTypePositiveInt), bi.Uint64())
				return nil
			}
			// Encode as CBOR neg int (major type 1)
			encodeHead(e, byte(cborTypeNegativeInt), bi.Uint64())
			return nil
		}
	}

	tagNum := 2
	if sign < 0 {
		tagNum = 3
	}
	// Write tag number
	encodeHead(e, byte(cborTypeTag), uint64(tagNum))
	// Write bignum byte string
	b := bi.Bytes()
	encodeHead(e, byte(cborTypeByteString), uint64(len(b)))
	e.Write(b)
	return nil
}

type binaryMarshalerEncoder struct {
	alternateEncode  encodeFunc
	alternateIsEmpty isEmptyFunc
}

func (bme binaryMarshalerEncoder) encode(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if em.binaryMarshaler != BinaryMarshalerByteString {
		return bme.alternateEncode(e, em, v)
	}

	vt := v.Type()
	m, ok := v.Interface().(encoding.BinaryMarshaler)
	if !ok {
		pv := reflect.New(vt)
		pv.Elem().Set(v)
		m = pv.Interface().(encoding.BinaryMarshaler)
	}
	data, err := m.MarshalBinary()
	if err != nil {
		return err
	}
	if b := em.encTagBytes(vt); b != nil {
		e.Write(b)
	}
	encodeHead(e, byte(cborTypeByteString), uint64(len(data)))
	e.Write(data)
	return nil
}

func (bme binaryMarshalerEncoder) isEmpty(em *encMode, v reflect.Value) (bool, error) {
	if em.binaryMarshaler != BinaryMarshalerByteString {
		return bme.alternateIsEmpty(em, v)
	}

	m, ok := v.Interface().(encoding.BinaryMarshaler)
	if !ok {
		pv := reflect.New(v.Type())
		pv.Elem().Set(v)
		m = pv.Interface().(encoding.BinaryMarshaler)
	}
	data, err := m.MarshalBinary()
	if err != nil {
		return false, err
	}
	return len(data) == 0, nil
}

func encodeMarshalerType(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if em.tagsMd == TagsForbidden && v.Type() == typeRawTag {
		return errors.New("cbor: cannot encode cbor.RawTag when TagsMd is TagsForbidden")
	}
	m, ok := v.Interface().(Marshaler)
	if !ok {
		pv := reflect.New(v.Type())
		pv.Elem().Set(v)
		m = pv.Interface().(Marshaler)
	}
	data, err := m.MarshalCBOR()
	if err != nil {
		return err
	}

	// Verify returned CBOR data item from MarshalCBOR() is well-formed and passes tag validity for builtin tags 0-3.
	d := decoder{data: data, dm: getMarshalerDecMode(em.indefLength, em.tagsMd)}
	err = d.wellformed(false, true)
	if err != nil {
		return &MarshalerError{typ: v.Type(), err: err}
	}

	e.Write(data)
	return nil
}

func encodeTag(e *bytes.Buffer, em *encMode, v reflect.Value) error {
	if em.tagsMd == TagsForbidden {
		return errors.New("cbor: cannot encode cbor.Tag when TagsMd is TagsForbidden")
	}

	t := v.Interface().(Tag)

	if t.Number == 0 && t.Content == nil {
		// Marshal uninitialized cbor.Tag
		e.Write(cborNil)
		return nil
	}

	// Marshal tag number
	encodeHead(e, byte(cborTypeTag), t.Number)

	vem := *em // shallow copy

	// For built-in tags, disable settings that may introduce tag validity errors when
	// marshaling certain Content values.
	switch t.Number {
	case tagNumRFC3339Time:
		vem.stringType = StringToTextString
		vem.stringMajorType = cborTypeTextString
	case tagNumUnsignedBignum, tagNumNegativeBignum:
		vem.byteSliceLaterFormat = ByteSliceLaterFormatNone
		vem.byteSliceLaterEncodingTag = 0
	}

	// Marshal tag content
	return encode(e, &vem, reflect.ValueOf(t.Content))
}

// encodeHead writes CBOR head of specified type t and returns number of bytes written.
func encodeHead(e *bytes.Buffer, t byte, n uint64) int {
	if n <= maxAdditionalInformationWithoutArgument {
		const headSize = 1
		e.WriteByte(t | byte(n))
		return headSize
	}

	if n <= math.MaxUint8 {
		const headSize = 2
		scratch := [headSize]byte{
			t | byte(additionalInformationWith1ByteArgument),
			byte(n),
		}
		e.Write(scratch[:])
		return headSize
	}

	if n <= math.MaxUint16 {
		const headSize = 3
		var scratch [headSize]byte
		scratch[0] = t | byte(additionalInformationWith2ByteArgument)
		binary.BigEndian.PutUint16(scratch[1:], uint16(n))
		e.Write(scratch[:])
		return headSize
	}

	if n <= math.MaxUint32 {
		const headSize = 5
		var scratch [headSize]byte
		scratch[0] = t | byte(additionalInformationWith4ByteArgument)
		binary.BigEndian.PutUint32(scratch[1:], uint32(n))
		e.Write(scratch[:])
		return headSize
	}

	const headSize = 9
	var scratch [headSize]byte
	scratch[0] = t | byte(additionalInformationWith8ByteArgument)
	binary.BigEndian.PutUint64(scratch[1:], n)
	e.Write(scratch[:])
	return headSize
}

var (
	typeMarshaler       = reflect.TypeOf((*Marshaler)(nil)).Elem()
	typeBinaryMarshaler = reflect.TypeOf((*encoding.BinaryMarshaler)(nil)).Elem()
	typeRawMessage      = reflect.TypeOf(RawMessage(nil))
	typeByteString      = reflect.TypeOf(ByteString(""))
)

func getEncodeFuncInternal(t reflect.Type) (ef encodeFunc, ief isEmptyFunc) {
	k := t.Kind()
	if k == reflect.Ptr {
		return getEncodeIndirectValueFunc(t), isEmptyPtr
	}
	switch t {
	case typeSimpleValue:
		return encodeMarshalerType, isEmptyUint

	case typeTag:
		return encodeTag, alwaysNotEmpty

	case typeTime:
		return encodeTime, alwaysNotEmpty

	case typeBigInt:
		return encodeBigInt, alwaysNotEmpty

	case typeRawMessage:
		return encodeMarshalerType, isEmptySlice

	case typeByteString:
		return encodeMarshalerType, isEmptyString
	}
	if reflect.PtrTo(t).Implements(typeMarshaler) {
		return encodeMarshalerType, alwaysNotEmpty
	}
	if reflect.PtrTo(t).Implements(typeBinaryMarshaler) {
		defer func() {
			// capture encoding method used for modes that disable BinaryMarshaler
			bme := binaryMarshalerEncoder{
				alternateEncode:  ef,
				alternateIsEmpty: ief,
			}
			ef = bme.encode
			ief = bme.isEmpty
		}()
	}
	switch k {
	case reflect.Bool:
		return encodeBool, isEmptyBool

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return encodeInt, isEmptyInt

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return encodeUint, isEmptyUint

	case reflect.Float32, reflect.Float64:
		return encodeFloat, isEmptyFloat

	case reflect.String:
		return encodeString, isEmptyString

	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return encodeByteString, isEmptySlice
		}
		fallthrough

	case reflect.Array:
		f, _ := getEncodeFunc(t.Elem())
		if f == nil {
			return nil, nil
		}
		return arrayEncodeFunc{f: f}.encode, isEmptySlice

	case reflect.Map:
		f := getEncodeMapFunc(t)
		if f == nil {
			return nil, nil
		}
		return f, isEmptyMap

	case reflect.Struct:
		// Get struct's special field "_" tag options
		if f, ok := t.FieldByName("_"); ok {
			tag := f.Tag.Get("cbor")
			if tag != "-" {
				if hasToArrayOption(tag) {
					return encodeStructToArray, isEmptyStruct
				}
			}
		}
		return encodeStruct, isEmptyStruct

	case reflect.Interface:
		return encodeIntf, isEmptyIntf
	}
	return nil, nil
}

func getEncodeIndirectValueFunc(t reflect.Type) encodeFunc {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	f, _ := getEncodeFunc(t)
	if f == nil {
		return nil
	}
	return func(e *bytes.Buffer, em *encMode, v reflect.Value) error {
		for v.Kind() == reflect.Ptr && !v.IsNil() {
			v = v.Elem()
		}
		if v.Kind() == reflect.Ptr && v.IsNil() {
			e.Write(cborNil)
			return nil
		}
		return f(e, em, v)
	}
}

func alwaysNotEmpty(_ *encMode, _ reflect.Value) (empty bool, err error) {
	return false, nil
}

func isEmptyBool(_ *encMode, v reflect.Value) (bool, error) {
	return !v.Bool(), nil
}

func isEmptyInt(_ *encMode, v reflect.Value) (bool, error) {
	return v.Int() == 0, nil
}

func isEmptyUint(_ *encMode, v reflect.Value) (bool, error) {
	return v.Uint() == 0, nil
}

func isEmptyFloat(_ *encMode, v reflect.Value) (bool, error) {
	return v.Float() == 0.0, nil
}

func isEmptyString(_ *encMode, v reflect.Value) (bool, error) {
	return v.Len() == 0, nil
}

func isEmptySlice(_ *encMode, v reflect.Value) (bool, error) {
	return v.Len() == 0, nil
}

func isEmptyMap(_ *encMode, v reflect.Value) (bool, error) {
	return v.Len() == 0, nil
}

func isEmptyPtr(_ *encMode, v reflect.Value) (bool, error) {
	return v.IsNil(), nil
}

func isEmptyIntf(_ *encMode, v reflect.Value) (bool, error) {
	return v.IsNil(), nil
}

func isEmptyStruct(em *encMode, v reflect.Value) (bool, error) {
	structType, err := getEncodingStructType(v.Type())
	if err != nil {
		return false, err
	}

	if em.omitEmpty == OmitEmptyGoValue {
		return false, nil
	}

	if structType.toArray {
		return len(structType.fields) == 0, nil
	}

	if len(structType.fields) > len(structType.omitEmptyFieldsIdx) {
		return false, nil
	}

	for _, i := range structType.omitEmptyFieldsIdx {
		f := structType.fields[i]

		// Get field value
		var fv reflect.Value
		if len(f.idx) == 1 {
			fv = v.Field(f.idx[0])
		} else {
			// Get embedded field value.  No error is expected.
			fv, _ = getFieldValue(v, f.idx, func(reflect.Value) (reflect.Value, error) {
				// Skip null pointer to embedded struct
				return reflect.Value{}, nil
			})
			if !fv.IsValid() {
				continue
			}
		}

		empty, err := f.ief(em, fv)
		if err != nil {
			return false, err
		}
		if !empty {
			return false, nil
		}
	}
	return true, nil
}

func cannotFitFloat32(f64 float64) bool {
	f32 := float32(f64)
	return float64(f32) != f64
}

// float32NaNFromReflectValue extracts float32 NaN from reflect.Value while preserving NaN's quiet bit.
func float32NaNFromReflectValue(v reflect.Value) float32 {
	// Keith Randall's workaround for issue https://github.com/golang/go/issues/36400
	p := reflect.New(v.Type())
	p.Elem().Set(v)
	f32 := p.Convert(reflect.TypeOf((*float32)(nil))).Elem().Interface().(float32)
	return f32
}
