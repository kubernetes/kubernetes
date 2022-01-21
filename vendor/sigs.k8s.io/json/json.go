/*
Copyright 2021 The Kubernetes Authors.

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

package json

import (
	gojson "encoding/json"
	"fmt"
	"io"

	internaljson "sigs.k8s.io/json/internal/golang/encoding/json"
)

// Decoder describes the decoding API exposed by `encoding/json#Decoder`
type Decoder interface {
	Decode(v interface{}) error
	Buffered() io.Reader
	Token() (gojson.Token, error)
	More() bool
	InputOffset() int64
}

// NewDecoderCaseSensitivePreserveInts returns a decoder that matches the behavior of encoding/json#NewDecoder, with the following changes:
// - When unmarshaling into a struct, JSON keys must case-sensitively match `json` tag names (for tagged struct fields)
//   or struct field names (for untagged struct fields), or they are treated as unknown fields and discarded.
// - When unmarshaling a number into an interface value, it is unmarshaled as an int64 if
//   the JSON data does not contain a "." character and parses as an integer successfully and
//   does not overflow int64. Otherwise, the number is unmarshaled as a float64.
// - If a syntax error is returned, it will not be of type encoding/json#SyntaxError,
//   but will be recognizeable by this package's IsSyntaxError() function.
func NewDecoderCaseSensitivePreserveInts(r io.Reader) Decoder {
	d := internaljson.NewDecoder(r)
	d.CaseSensitive()
	d.PreserveInts()
	return d
}

// UnmarshalCaseSensitivePreserveInts parses the JSON-encoded data and stores the result in the value pointed to by v.
//
// UnmarshalCaseSensitivePreserveInts matches the behavior of encoding/json#Unmarshal, with the following changes:
// - When unmarshaling into a struct, JSON keys must case-sensitively match `json` tag names (for tagged struct fields)
//   or struct field names (for untagged struct fields), or they are treated as unknown fields and discarded.
// - When unmarshaling a number into an interface value, it is unmarshaled as an int64 if
//   the JSON data does not contain a "." character and parses as an integer successfully and
//   does not overflow int64. Otherwise, the number is unmarshaled as a float64.
// - If a syntax error is returned, it will not be of type encoding/json#SyntaxError,
//   but will be recognizeable by this package's IsSyntaxError() function.
func UnmarshalCaseSensitivePreserveInts(data []byte, v interface{}) error {
	return internaljson.Unmarshal(
		data,
		v,
		internaljson.CaseSensitive,
		internaljson.PreserveInts,
	)
}

type StrictOption int

const (
	// DisallowDuplicateFields returns strict errors if data contains duplicate fields
	DisallowDuplicateFields StrictOption = 1

	// DisallowUnknownFields returns strict errors if data contains unknown fields when decoding into typed structs
	DisallowUnknownFields StrictOption = 2
)

// UnmarshalStrict parses the JSON-encoded data and stores the result in the value pointed to by v.
// Unmarshaling is performed identically to UnmarshalCaseSensitivePreserveInts(), returning an error on failure.
//
// If parsing succeeds, additional strict checks as selected by `strictOptions` are performed
// and a list of the strict failures (if any) are returned. If no `strictOptions` are selected,
// all supported strict checks are performed.
//
// Currently supported strict checks are:
// - DisallowDuplicateFields: ensure the data contains no duplicate fields
// - DisallowUnknownFields: ensure the data contains no unknown fields (when decoding into typed structs)
//
// Additional strict checks may be added in the future.
//
// Note that the strict checks do not change what is stored in v.
// For example, if duplicate fields are present, they will be parsed and stored in v,
// and errors about the duplicate fields will be returned in the strict error list.
func UnmarshalStrict(data []byte, v interface{}, strictOptions ...StrictOption) (strictErrors []error, err error) {
	if len(strictOptions) == 0 {
		err = internaljson.Unmarshal(data, v,
			// options matching UnmarshalCaseSensitivePreserveInts
			internaljson.CaseSensitive,
			internaljson.PreserveInts,
			// all strict options
			internaljson.DisallowDuplicateFields,
			internaljson.DisallowUnknownFields,
		)
	} else {
		opts := make([]internaljson.UnmarshalOpt, 0, 2+len(strictOptions))
		// options matching UnmarshalCaseSensitivePreserveInts
		opts = append(opts, internaljson.CaseSensitive, internaljson.PreserveInts)
		for _, strictOpt := range strictOptions {
			switch strictOpt {
			case DisallowDuplicateFields:
				opts = append(opts, internaljson.DisallowDuplicateFields)
			case DisallowUnknownFields:
				opts = append(opts, internaljson.DisallowUnknownFields)
			default:
				return nil, fmt.Errorf("unknown strict option %d", strictOpt)
			}
		}
		err = internaljson.Unmarshal(data, v, opts...)
	}

	if strictErr, ok := err.(*internaljson.UnmarshalStrictError); ok {
		return strictErr.Errors, nil
	}
	return nil, err
}

// SyntaxErrorOffset returns if the specified error is a syntax error produced by encoding/json or this package.
func SyntaxErrorOffset(err error) (isSyntaxError bool, offset int64) {
	switch err := err.(type) {
	case *gojson.SyntaxError:
		return true, err.Offset
	case *internaljson.SyntaxError:
		return true, err.Offset
	default:
		return false, 0
	}
}
