// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"errors"
	"io"
	"reflect"
	"sync"
)

// MarshalOptions configures how Go data is serialized as JSON data.
// The zero value is equivalent to the default marshal settings.
type MarshalOptions struct {
	requireKeyedLiterals
	nonComparable

	// Marshalers is a list of type-specific marshalers to use.
	Marshalers *Marshalers

	// StringifyNumbers specifies that numeric Go types should be serialized
	// as a JSON string containing the equivalent JSON number value.
	//
	// According to RFC 8259, section 6, a JSON implementation may choose to
	// limit the representation of a JSON number to an IEEE 754 binary64 value.
	// This may cause decoders to lose precision for int64 and uint64 types.
	// Escaping JSON numbers as a JSON string preserves the exact precision.
	StringifyNumbers bool

	// DiscardUnknownMembers specifies that marshaling should ignore any
	// JSON object members stored in Go struct fields dedicated to storing
	// unknown JSON object members.
	DiscardUnknownMembers bool

	// formatDepth is the depth at which we respect the format flag.
	formatDepth int
	// format is custom formatting for the value at the specified depth.
	format string
}

// Marshal serializes a Go value as a []byte with default options.
// It is a thin wrapper over MarshalOptions.Marshal.
func Marshal(in any) (out []byte, err error) {
	return MarshalOptions{}.Marshal(EncodeOptions{}, in)
}

// MarshalFull serializes a Go value into an io.Writer with default options.
// It is a thin wrapper over MarshalOptions.MarshalFull.
func MarshalFull(out io.Writer, in any) error {
	return MarshalOptions{}.MarshalFull(EncodeOptions{}, out, in)
}

// Marshal serializes a Go value as a []byte according to the provided
// marshal and encode options. It does not terminate the output with a newline.
// See MarshalNext for details about the conversion of a Go value into JSON.
func (mo MarshalOptions) Marshal(eo EncodeOptions, in any) (out []byte, err error) {
	enc := getBufferedEncoder(eo)
	defer putBufferedEncoder(enc)
	enc.options.omitTopLevelNewline = true
	err = mo.MarshalNext(enc, in)
	// TODO(https://go.dev/issue/45038): Use bytes.Clone.
	return append([]byte(nil), enc.buf...), err
}

// MarshalFull serializes a Go value into an io.Writer according to the provided
// marshal and encode options. It does not terminate the output with a newline.
// See MarshalNext for details about the conversion of a Go value into JSON.
func (mo MarshalOptions) MarshalFull(eo EncodeOptions, out io.Writer, in any) error {
	enc := getStreamingEncoder(out, eo)
	defer putStreamingEncoder(enc)
	enc.options.omitTopLevelNewline = true
	err := mo.MarshalNext(enc, in)
	return err
}

// MarshalNext encodes a Go value as the next JSON value according to
// the provided marshal options.
//
// Type-specific marshal functions and methods take precedence
// over the default representation of a value.
// Functions or methods that operate on *T are only called when encoding
// a value of type T (by taking its address) or a non-nil value of *T.
// MarshalNext ensures that a value is always addressable
// (by boxing it on the heap if necessary) so that
// these functions and methods can be consistently called. For performance,
// it is recommended that MarshalNext be passed a non-nil pointer to the value.
//
// The input value is encoded as JSON according the following rules:
//
//   - If any type-specific functions in MarshalOptions.Marshalers match
//     the value type, then those functions are called to encode the value.
//     If all applicable functions return SkipFunc,
//     then the value is encoded according to subsequent rules.
//
//   - If the value type implements MarshalerV2,
//     then the MarshalNextJSON method is called to encode the value.
//
//   - If the value type implements MarshalerV1,
//     then the MarshalJSON method is called to encode the value.
//
//   - If the value type implements encoding.TextMarshaler,
//     then the MarshalText method is called to encode the value and
//     subsequently encode its result as a JSON string.
//
//   - Otherwise, the value is encoded according to the value's type
//     as described in detail below.
//
// Most Go types have a default JSON representation.
// Certain types support specialized formatting according to
// a format flag optionally specified in the Go struct tag
// for the struct field that contains the current value
// (see the “JSON Representation of Go structs” section for more details).
//
// The representation of each type is as follows:
//
//   - A Go boolean is encoded as a JSON boolean (e.g., true or false).
//     It does not support any custom format flags.
//
//   - A Go string is encoded as a JSON string.
//     It does not support any custom format flags.
//
//   - A Go []byte or [N]byte is encoded as a JSON string containing
//     the binary value encoded using RFC 4648.
//     If the format is "base64" or unspecified, then this uses RFC 4648, section 4.
//     If the format is "base64url", then this uses RFC 4648, section 5.
//     If the format is "base32", then this uses RFC 4648, section 6.
//     If the format is "base32hex", then this uses RFC 4648, section 7.
//     If the format is "base16" or "hex", then this uses RFC 4648, section 8.
//     If the format is "array", then the bytes value is encoded as a JSON array
//     where each byte is recursively JSON-encoded as each JSON array element.
//
//   - A Go integer is encoded as a JSON number without fractions or exponents.
//     If MarshalOptions.StringifyNumbers is specified, then the JSON number is
//     encoded within a JSON string. It does not support any custom format
//     flags.
//
//   - A Go float is encoded as a JSON number.
//     If MarshalOptions.StringifyNumbers is specified,
//     then the JSON number is encoded within a JSON string.
//     If the format is "nonfinite", then NaN, +Inf, and -Inf are encoded as
//     the JSON strings "NaN", "Infinity", and "-Infinity", respectively.
//     Otherwise, the presence of non-finite numbers results in a SemanticError.
//
//   - A Go map is encoded as a JSON object, where each Go map key and value
//     is recursively encoded as a name and value pair in the JSON object.
//     The Go map key must encode as a JSON string, otherwise this results
//     in a SemanticError. When encoding keys, MarshalOptions.StringifyNumbers
//     is automatically applied so that numeric keys encode as JSON strings.
//     The Go map is traversed in a non-deterministic order.
//     For deterministic encoding, consider using RawValue.Canonicalize.
//     If the format is "emitnull", then a nil map is encoded as a JSON null.
//     Otherwise by default, a nil map is encoded as an empty JSON object.
//
//   - A Go struct is encoded as a JSON object.
//     See the “JSON Representation of Go structs” section
//     in the package-level documentation for more details.
//
//   - A Go slice is encoded as a JSON array, where each Go slice element
//     is recursively JSON-encoded as the elements of the JSON array.
//     If the format is "emitnull", then a nil slice is encoded as a JSON null.
//     Otherwise by default, a nil slice is encoded as an empty JSON array.
//
//   - A Go array is encoded as a JSON array, where each Go array element
//     is recursively JSON-encoded as the elements of the JSON array.
//     The JSON array length is always identical to the Go array length.
//     It does not support any custom format flags.
//
//   - A Go pointer is encoded as a JSON null if nil, otherwise it is
//     the recursively JSON-encoded representation of the underlying value.
//     Format flags are forwarded to the encoding of the underlying value.
//
//   - A Go interface is encoded as a JSON null if nil, otherwise it is
//     the recursively JSON-encoded representation of the underlying value.
//     It does not support any custom format flags.
//
//   - A Go time.Time is encoded as a JSON string containing the timestamp
//     formatted in RFC 3339 with nanosecond resolution.
//     If the format matches one of the format constants declared
//     in the time package (e.g., RFC1123), then that format is used.
//     Otherwise, the format is used as-is with time.Time.Format if non-empty.
//
//   - A Go time.Duration is encoded as a JSON string containing the duration
//     formatted according to time.Duration.String.
//     If the format is "nanos", it is encoded as a JSON number
//     containing the number of nanoseconds in the duration.
//
//   - All other Go types (e.g., complex numbers, channels, and functions)
//     have no default representation and result in a SemanticError.
//
// JSON cannot represent cyclic data structures and
// MarshalNext does not handle them.
// Passing cyclic structures will result in an error.
func (mo MarshalOptions) MarshalNext(out *Encoder, in any) error {
	v := reflect.ValueOf(in)
	if !v.IsValid() || (v.Kind() == reflect.Pointer && v.IsNil()) {
		return out.WriteToken(Null)
	}
	// Shallow copy non-pointer values to obtain an addressable value.
	// It is beneficial to performance to always pass pointers to avoid this.
	if v.Kind() != reflect.Pointer {
		v2 := reflect.New(v.Type())
		v2.Elem().Set(v)
		v = v2
	}
	va := addressableValue{v.Elem()} // dereferenced pointer is always addressable
	t := va.Type()

	// Lookup and call the marshal function for this type.
	marshal := lookupArshaler(t).marshal
	if mo.Marshalers != nil {
		marshal, _ = mo.Marshalers.lookup(marshal, t)
	}
	if err := marshal(mo, out, va); err != nil {
		if !out.options.AllowDuplicateNames {
			out.tokens.invalidateDisabledNamespaces()
		}
		return err
	}
	return nil
}

// UnmarshalOptions configures how JSON data is deserialized as Go data.
// The zero value is equivalent to the default unmarshal settings.
type UnmarshalOptions struct {
	requireKeyedLiterals
	nonComparable

	// Unmarshalers is a list of type-specific unmarshalers to use.
	Unmarshalers *Unmarshalers

	// StringifyNumbers specifies that numeric Go types can be deserialized
	// from either a JSON number or a JSON string containing a JSON number
	// without any surrounding whitespace.
	StringifyNumbers bool

	// RejectUnknownMembers specifies that unknown members should be rejected
	// when unmarshaling a JSON object, regardless of whether there is a field
	// to store unknown members.
	RejectUnknownMembers bool

	// formatDepth is the depth at which we respect the format flag.
	formatDepth int
	// format is custom formatting for the value at the specified depth.
	format string
}

// Unmarshal deserializes a Go value from a []byte with default options.
// It is a thin wrapper over UnmarshalOptions.Unmarshal.
func Unmarshal(in []byte, out any) error {
	return UnmarshalOptions{}.Unmarshal(DecodeOptions{}, in, out)
}

// UnmarshalFull deserializes a Go value from an io.Reader with default options.
// It is a thin wrapper over UnmarshalOptions.UnmarshalFull.
func UnmarshalFull(in io.Reader, out any) error {
	return UnmarshalOptions{}.UnmarshalFull(DecodeOptions{}, in, out)
}

// Unmarshal deserializes a Go value from a []byte according to the
// provided unmarshal and decode options. The output must be a non-nil pointer.
// The input must be a single JSON value with optional whitespace interspersed.
// See UnmarshalNext for details about the conversion of JSON into a Go value.
func (uo UnmarshalOptions) Unmarshal(do DecodeOptions, in []byte, out any) error {
	dec := getBufferedDecoder(in, do)
	defer putBufferedDecoder(dec)
	return uo.unmarshalFull(dec, out)
}

// UnmarshalFull deserializes a Go value from an io.Reader according to the
// provided unmarshal and decode options. The output must be a non-nil pointer.
// The input must be a single JSON value with optional whitespace interspersed.
// It consumes the entirety of io.Reader until io.EOF is encountered.
// See UnmarshalNext for details about the conversion of JSON into a Go value.
func (uo UnmarshalOptions) UnmarshalFull(do DecodeOptions, in io.Reader, out any) error {
	dec := getStreamingDecoder(in, do)
	defer putStreamingDecoder(dec)
	return uo.unmarshalFull(dec, out)
}
func (uo UnmarshalOptions) unmarshalFull(in *Decoder, out any) error {
	switch err := uo.UnmarshalNext(in, out); err {
	case nil:
		return in.checkEOF()
	case io.EOF:
		return io.ErrUnexpectedEOF
	default:
		return err
	}
}

// UnmarshalNext decodes the next JSON value into a Go value according to
// the provided unmarshal options. The output must be a non-nil pointer.
//
// Type-specific unmarshal functions and methods take precedence
// over the default representation of a value.
// Functions or methods that operate on *T are only called when decoding
// a value of type T (by taking its address) or a non-nil value of *T.
// UnmarshalNext ensures that a value is always addressable
// (by boxing it on the heap if necessary) so that
// these functions and methods can be consistently called.
//
// The input is decoded into the output according the following rules:
//
//   - If any type-specific functions in UnmarshalOptions.Unmarshalers match
//     the value type, then those functions are called to decode the JSON
//     value. If all applicable functions return SkipFunc,
//     then the input is decoded according to subsequent rules.
//
//   - If the value type implements UnmarshalerV2,
//     then the UnmarshalNextJSON method is called to decode the JSON value.
//
//   - If the value type implements UnmarshalerV1,
//     then the UnmarshalJSON method is called to decode the JSON value.
//
//   - If the value type implements encoding.TextUnmarshaler,
//     then the input is decoded as a JSON string and
//     the UnmarshalText method is called with the decoded string value.
//     This fails with a SemanticError if the input is not a JSON string.
//
//   - Otherwise, the JSON value is decoded according to the value's type
//     as described in detail below.
//
// Most Go types have a default JSON representation.
// Certain types support specialized formatting according to
// a format flag optionally specified in the Go struct tag
// for the struct field that contains the current value
// (see the “JSON Representation of Go structs” section for more details).
// A JSON null may be decoded into every supported Go value where
// it is equivalent to storing the zero value of the Go value.
// If the input JSON kind is not handled by the current Go value type,
// then this fails with a SemanticError. Unless otherwise specified,
// the decoded value replaces any pre-existing value.
//
// The representation of each type is as follows:
//
//   - A Go boolean is decoded from a JSON boolean (e.g., true or false).
//     It does not support any custom format flags.
//
//   - A Go string is decoded from a JSON string.
//     It does not support any custom format flags.
//
//   - A Go []byte or [N]byte is decoded from a JSON string
//     containing the binary value encoded using RFC 4648.
//     If the format is "base64" or unspecified, then this uses RFC 4648, section 4.
//     If the format is "base64url", then this uses RFC 4648, section 5.
//     If the format is "base32", then this uses RFC 4648, section 6.
//     If the format is "base32hex", then this uses RFC 4648, section 7.
//     If the format is "base16" or "hex", then this uses RFC 4648, section 8.
//     If the format is "array", then the Go slice or array is decoded from a
//     JSON array where each JSON element is recursively decoded for each byte.
//     When decoding into a non-nil []byte, the slice length is reset to zero
//     and the decoded input is appended to it.
//     When decoding into a [N]byte, the input must decode to exactly N bytes,
//     otherwise it fails with a SemanticError.
//
//   - A Go integer is decoded from a JSON number.
//     It may also be decoded from a JSON string containing a JSON number
//     if UnmarshalOptions.StringifyNumbers is specified.
//     It fails with a SemanticError if the JSON number
//     has a fractional or exponent component.
//     It also fails if it overflows the representation of the Go integer type.
//     It does not support any custom format flags.
//
//   - A Go float is decoded from a JSON number.
//     It may also be decoded from a JSON string containing a JSON number
//     if UnmarshalOptions.StringifyNumbers is specified.
//     The JSON number is parsed as the closest representable Go float value.
//     If the format is "nonfinite", then the JSON strings
//     "NaN", "Infinity", and "-Infinity" are decoded as NaN, +Inf, and -Inf.
//     Otherwise, the presence of such strings results in a SemanticError.
//
//   - A Go map is decoded from a JSON object,
//     where each JSON object name and value pair is recursively decoded
//     as the Go map key and value. When decoding keys,
//     UnmarshalOptions.StringifyNumbers is automatically applied so that
//     numeric keys can decode from JSON strings. Maps are not cleared.
//     If the Go map is nil, then a new map is allocated to decode into.
//     If the decoded key matches an existing Go map entry, the entry value
//     is reused by decoding the JSON object value into it.
//     The only supported format is "emitnull" and has no effect when decoding.
//
//   - A Go struct is decoded from a JSON object.
//     See the “JSON Representation of Go structs” section
//     in the package-level documentation for more details.
//
//   - A Go slice is decoded from a JSON array, where each JSON element
//     is recursively decoded and appended to the Go slice.
//     Before appending into a Go slice, a new slice is allocated if it is nil,
//     otherwise the slice length is reset to zero.
//     The only supported format is "emitnull" and has no effect when decoding.
//
//   - A Go array is decoded from a JSON array, where each JSON array element
//     is recursively decoded as each corresponding Go array element.
//     Each Go array element is zeroed before decoding into it.
//     It fails with a SemanticError if the JSON array does not contain
//     the exact same number of elements as the Go array.
//     It does not support any custom format flags.
//
//   - A Go pointer is decoded based on the JSON kind and underlying Go type.
//     If the input is a JSON null, then this stores a nil pointer.
//     Otherwise, it allocates a new underlying value if the pointer is nil,
//     and recursively JSON decodes into the underlying value.
//     Format flags are forwarded to the decoding of the underlying type.
//
//   - A Go interface is decoded based on the JSON kind and underlying Go type.
//     If the input is a JSON null, then this stores a nil interface value.
//     Otherwise, a nil interface value of an empty interface type is initialized
//     with a zero Go bool, string, float64, map[string]any, or []any if the
//     input is a JSON boolean, string, number, object, or array, respectively.
//     If the interface value is still nil, then this fails with a SemanticError
//     since decoding could not determine an appropriate Go type to decode into.
//     For example, unmarshaling into a nil io.Reader fails since
//     there is no concrete type to populate the interface value with.
//     Otherwise an underlying value exists and it recursively decodes
//     the JSON input into it. It does not support any custom format flags.
//
//   - A Go time.Time is decoded from a JSON string containing the time
//     formatted in RFC 3339 with nanosecond resolution.
//     If the format matches one of the format constants declared in
//     the time package (e.g., RFC1123), then that format is used for parsing.
//     Otherwise, the format is used as-is with time.Time.Parse if non-empty.
//
//   - A Go time.Duration is decoded from a JSON string by
//     passing the decoded string to time.ParseDuration.
//     If the format is "nanos", it is instead decoded from a JSON number
//     containing the number of nanoseconds in the duration.
//
//   - All other Go types (e.g., complex numbers, channels, and functions)
//     have no default representation and result in a SemanticError.
//
// In general, unmarshaling follows merge semantics (similar to RFC 7396)
// where the decoded Go value replaces the destination value
// for any JSON kind other than an object.
// For JSON objects, the input object is merged into the destination value
// where matching object members recursively apply merge semantics.
func (uo UnmarshalOptions) UnmarshalNext(in *Decoder, out any) error {
	v := reflect.ValueOf(out)
	if !v.IsValid() || v.Kind() != reflect.Pointer || v.IsNil() {
		var t reflect.Type
		if v.IsValid() {
			t = v.Type()
			if t.Kind() == reflect.Pointer {
				t = t.Elem()
			}
		}
		err := errors.New("value must be passed as a non-nil pointer reference")
		return &SemanticError{action: "unmarshal", GoType: t, Err: err}
	}
	va := addressableValue{v.Elem()} // dereferenced pointer is always addressable
	t := va.Type()

	// Lookup and call the unmarshal function for this type.
	unmarshal := lookupArshaler(t).unmarshal
	if uo.Unmarshalers != nil {
		unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, t)
	}
	if err := unmarshal(uo, in, va); err != nil {
		if !in.options.AllowDuplicateNames {
			in.tokens.invalidateDisabledNamespaces()
		}
		return err
	}
	return nil
}

// addressableValue is a reflect.Value that is guaranteed to be addressable
// such that calling the Addr and Set methods do not panic.
//
// There is no compile magic that enforces this property,
// but rather the need to construct this type makes it easier to examine each
// construction site to ensure that this property is upheld.
type addressableValue struct{ reflect.Value }

// newAddressableValue constructs a new addressable value of type t.
func newAddressableValue(t reflect.Type) addressableValue {
	return addressableValue{reflect.New(t).Elem()}
}

// All marshal and unmarshal behavior is implemented using these signatures.
type (
	marshaler   = func(MarshalOptions, *Encoder, addressableValue) error
	unmarshaler = func(UnmarshalOptions, *Decoder, addressableValue) error
)

type arshaler struct {
	marshal    marshaler
	unmarshal  unmarshaler
	nonDefault bool
}

var lookupArshalerCache sync.Map // map[reflect.Type]*arshaler

func lookupArshaler(t reflect.Type) *arshaler {
	if v, ok := lookupArshalerCache.Load(t); ok {
		return v.(*arshaler)
	}

	fncs := makeDefaultArshaler(t)
	fncs = makeMethodArshaler(fncs, t)
	fncs = makeTimeArshaler(fncs, t)

	// Use the last stored so that duplicate arshalers can be garbage collected.
	v, _ := lookupArshalerCache.LoadOrStore(t, fncs)
	return v.(*arshaler)
}
