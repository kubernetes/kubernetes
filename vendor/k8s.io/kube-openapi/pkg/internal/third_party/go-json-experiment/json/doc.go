// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package json implements serialization of JSON
// as specified in RFC 4627, RFC 7159, RFC 7493, RFC 8259, and RFC 8785.
// JSON is a simple data interchange format that can represent
// primitive data types such as booleans, strings, and numbers,
// in addition to structured data types such as objects and arrays.
//
// # Terminology
//
// This package uses the terms "encode" and "decode" for syntactic functionality
// that is concerned with processing JSON based on its grammar, and
// uses the terms "marshal" and "unmarshal" for semantic functionality
// that determines the meaning of JSON values as Go values and vice-versa.
// It aims to provide a clear distinction between functionality that
// is purely concerned with encoding versus that of marshaling.
// For example, one can directly encode a stream of JSON tokens without
// needing to marshal a concrete Go value representing them.
// Similarly, one can decode a stream of JSON tokens without
// needing to unmarshal them into a concrete Go value.
//
// This package uses JSON terminology when discussing JSON, which may differ
// from related concepts in Go or elsewhere in computing literature.
//
//   - A JSON "object" refers to an unordered collection of name/value members.
//   - A JSON "array" refers to an ordered sequence of elements.
//   - A JSON "value" refers to either a literal (i.e., null, false, or true),
//     string, number, object, or array.
//
// See RFC 8259 for more information.
//
// # Specifications
//
// Relevant specifications include RFC 4627, RFC 7159, RFC 7493, RFC 8259,
// and RFC 8785. Each RFC is generally a stricter subset of another RFC.
// In increasing order of strictness:
//
//   - RFC 4627 and RFC 7159 do not require (but recommend) the use of UTF-8
//     and also do not require (but recommend) that object names be unique.
//   - RFC 8259 requires the use of UTF-8,
//     but does not require (but recommends) that object names be unique.
//   - RFC 7493 requires the use of UTF-8
//     and also requires that object names be unique.
//   - RFC 8785 defines a canonical representation. It requires the use of UTF-8
//     and also requires that object names be unique and in a specific ordering.
//     It specifies exactly how strings and numbers must be formatted.
//
// The primary difference between RFC 4627 and RFC 7159 is that the former
// restricted top-level values to only JSON objects and arrays, while
// RFC 7159 and subsequent RFCs permit top-level values to additionally be
// JSON nulls, booleans, strings, or numbers.
//
// By default, this package operates on RFC 7493, but can be configured
// to operate according to the other RFC specifications.
// RFC 7493 is a stricter subset of RFC 8259 and fully compliant with it.
// In particular, it makes specific choices about behavior that RFC 8259
// leaves as undefined in order to ensure greater interoperability.
//
// # JSON Representation of Go structs
//
// A Go struct is naturally represented as a JSON object,
// where each Go struct field corresponds with a JSON object member.
// When marshaling, all Go struct fields are recursively encoded in depth-first
// order as JSON object members except those that are ignored or omitted.
// When unmarshaling, JSON object members are recursively decoded
// into the corresponding Go struct fields.
// Object members that do not match any struct fields,
// also known as “unknown members”, are ignored by default or rejected
// if UnmarshalOptions.RejectUnknownMembers is specified.
//
// The representation of each struct field can be customized in the
// "json" struct field tag, where the tag is a comma separated list of options.
// As a special case, if the entire tag is `json:"-"`,
// then the field is ignored with regard to its JSON representation.
//
// The first option is the JSON object name override for the Go struct field.
// If the name is not specified, then the Go struct field name
// is used as the JSON object name. JSON names containing commas or quotes,
// or names identical to "" or "-", can be specified using
// a single-quoted string literal, where the syntax is identical to
// the Go grammar for a double-quoted string literal,
// but instead uses single quotes as the delimiters.
// By default, unmarshaling uses case-sensitive matching to identify
// the Go struct field associated with a JSON object name.
//
// After the name, the following tag options are supported:
//
//   - omitzero: When marshaling, the "omitzero" option specifies that
//     the struct field should be omitted if the field value is zero
//     as determined by the "IsZero() bool" method if present,
//     otherwise based on whether the field is the zero Go value.
//     This option has no effect when unmarshaling.
//
//   - omitempty: When marshaling, the "omitempty" option specifies that
//     the struct field should be omitted if the field value would have been
//     encoded as a JSON null, empty string, empty object, or empty array.
//     This option has no effect when unmarshaling.
//
//   - string: The "string" option specifies that
//     MarshalOptions.StringifyNumbers and UnmarshalOptions.StringifyNumbers
//     be set when marshaling or unmarshaling a struct field value.
//     This causes numeric types to be encoded as a JSON number
//     within a JSON string, and to be decoded from either a JSON number or
//     a JSON string containing a JSON number.
//     This extra level of encoding is often necessary since
//     many JSON parsers cannot precisely represent 64-bit integers.
//
//   - nocase: When unmarshaling, the "nocase" option specifies that
//     if the JSON object name does not exactly match the JSON name
//     for any of the struct fields, then it attempts to match the struct field
//     using a case-insensitive match that also ignores dashes and underscores.
//     If multiple fields match, the first declared field in breadth-first order
//     takes precedence. This option has no effect when marshaling.
//
//   - inline: The "inline" option specifies that
//     the JSON representable content of this field type is to be promoted
//     as if they were specified in the parent struct.
//     It is the JSON equivalent of Go struct embedding.
//     A Go embedded field is implicitly inlined unless an explicit JSON name
//     is specified. The inlined field must be a Go struct
//     (that does not implement any JSON methods), RawValue, map[string]T,
//     or an unnamed pointer to such types. When marshaling,
//     inlined fields from a pointer type are omitted if it is nil.
//     Inlined fields of type RawValue and map[string]T are called
//     “inlined fallbacks” as they can represent all possible
//     JSON object members not directly handled by the parent struct.
//     Only one inlined fallback field may be specified in a struct,
//     while many non-fallback fields may be specified. This option
//     must not be specified with any other option (including the JSON name).
//
//   - unknown: The "unknown" option is a specialized variant
//     of the inlined fallback to indicate that this Go struct field
//     contains any number of unknown JSON object members. The field type
//     must be a RawValue, map[string]T, or an unnamed pointer to such types.
//     If MarshalOptions.DiscardUnknownMembers is specified when marshaling,
//     the contents of this field are ignored.
//     If UnmarshalOptions.RejectUnknownMembers is specified when unmarshaling,
//     any unknown object members are rejected regardless of whether
//     an inlined fallback with the "unknown" option exists. This option
//     must not be specified with any other option (including the JSON name).
//
//   - format: The "format" option specifies a format flag
//     used to specialize the formatting of the field value.
//     The option is a key-value pair specified as "format:value" where
//     the value must be either a literal consisting of letters and numbers
//     (e.g., "format:RFC3339") or a single-quoted string literal
//     (e.g., "format:'2006-01-02'"). The interpretation of the format flag
//     is determined by the struct field type.
//
// The "omitzero" and "omitempty" options are mostly semantically identical.
// The former is defined in terms of the Go type system,
// while the latter in terms of the JSON type system.
// Consequently they behave differently in some circumstances.
// For example, only a nil slice or map is omitted under "omitzero", while
// an empty slice or map is omitted under "omitempty" regardless of nilness.
// The "omitzero" option is useful for types with a well-defined zero value
// (e.g., netip.Addr) or have an IsZero method (e.g., time.Time).
//
// Every Go struct corresponds to a list of JSON representable fields
// which is constructed by performing a breadth-first search over
// all struct fields (excluding unexported or ignored fields),
// where the search recursively descends into inlined structs.
// The set of non-inlined fields in a struct must have unique JSON names.
// If multiple fields all have the same JSON name, then the one
// at shallowest depth takes precedence and the other fields at deeper depths
// are excluded from the list of JSON representable fields.
// If multiple fields at the shallowest depth have the same JSON name,
// then all of those fields are excluded from the list. This is analogous to
// Go visibility rules for struct field selection with embedded struct types.
//
// Marshaling or unmarshaling a non-empty struct
// without any JSON representable fields results in a SemanticError.
// Unexported fields must not have any `json` tags except for `json:"-"`.
package json

// requireKeyedLiterals can be embedded in a struct to require keyed literals.
type requireKeyedLiterals struct{}

// nonComparable can be embedded in a struct to prevent comparability.
type nonComparable [0]func()
