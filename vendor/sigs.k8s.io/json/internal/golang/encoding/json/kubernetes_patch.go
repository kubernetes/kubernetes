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
	"strings"
)

// Type-alias error and data types returned from decoding

type UnmarshalTypeError = gojson.UnmarshalTypeError
type UnmarshalFieldError = gojson.UnmarshalFieldError
type InvalidUnmarshalError = gojson.InvalidUnmarshalError
type Number = gojson.Number
type RawMessage = gojson.RawMessage
type Token = gojson.Token
type Delim = gojson.Delim

type UnmarshalOpt func(*decodeState)

func UseNumber(d *decodeState) {
	d.useNumber = true
}
func DisallowUnknownFields(d *decodeState) {
	d.disallowUnknownFields = true
}

// CaseSensitive requires json keys to exactly match specified json tags (for tagged struct fields)
// or struct field names (for untagged struct fields), or be treated as an unknown field.
func CaseSensitive(d *decodeState) {
	d.caseSensitive = true
}
func (d *Decoder) CaseSensitive() {
	d.d.caseSensitive = true
}

// PreserveInts decodes numbers as int64 when decoding to untyped fields,
// if the JSON data does not contain a "." character, parses as an integer successfully,
// and does not overflow int64. Otherwise, it falls back to default float64 decoding behavior.
//
// If UseNumber is also set, it takes precedence over PreserveInts.
func PreserveInts(d *decodeState) {
	d.preserveInts = true
}
func (d *Decoder) PreserveInts() {
	d.d.preserveInts = true
}

// DisallowDuplicateFields treats duplicate fields encountered while decoding as an error.
func DisallowDuplicateFields(d *decodeState) {
	d.disallowDuplicateFields = true
}
func (d *Decoder) DisallowDuplicateFields() {
	d.d.disallowDuplicateFields = true
}

// saveStrictError saves a strict decoding error,
// for reporting at the end of the unmarshal if no other errors occurred.
func (d *decodeState) saveStrictError(err error) {
	// prevent excessive numbers of accumulated errors
	if len(d.savedStrictErrors) >= 100 {
		return
	}
	// dedupe accumulated strict errors
	if d.seenStrictErrors == nil {
		d.seenStrictErrors = map[string]struct{}{}
	}
	msg := err.Error()
	if _, seen := d.seenStrictErrors[msg]; seen {
		return
	}

	// accumulate the error
	d.seenStrictErrors[msg] = struct{}{}
	d.savedStrictErrors = append(d.savedStrictErrors, err)
}

// UnmarshalStrictError holds errors resulting from use of strict disallow___ decoder directives.
// If this is returned from Unmarshal(), it means the decoding was successful in all other respects.
type UnmarshalStrictError struct {
	Errors []error
}

func (e *UnmarshalStrictError) Error() string {
	var b strings.Builder
	b.WriteString("json: ")
	for i, err := range e.Errors {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(err.Error())
	}
	return b.String()
}
