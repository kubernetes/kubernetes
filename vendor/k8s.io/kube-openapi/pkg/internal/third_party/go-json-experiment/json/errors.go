// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"errors"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

const errorPrefix = "json: "

// Error matches errors returned by this package according to errors.Is.
const Error = jsonError("json error")

type jsonError string

func (e jsonError) Error() string {
	return string(e)
}
func (e jsonError) Is(target error) bool {
	return e == target || target == Error
}

type ioError struct {
	action string // either "read" or "write"
	err    error
}

func (e *ioError) Error() string {
	return errorPrefix + e.action + " error: " + e.err.Error()
}
func (e *ioError) Unwrap() error {
	return e.err
}
func (e *ioError) Is(target error) bool {
	return e == target || target == Error || errors.Is(e.err, target)
}

// SemanticError describes an error determining the meaning
// of JSON data as Go data or vice-versa.
//
// The contents of this error as produced by this package may change over time.
type SemanticError struct {
	requireKeyedLiterals
	nonComparable

	action string // either "marshal" or "unmarshal"

	// ByteOffset indicates that an error occurred after this byte offset.
	ByteOffset int64
	// JSONPointer indicates that an error occurred within this JSON value
	// as indicated using the JSON Pointer notation (see RFC 6901).
	JSONPointer string

	// JSONKind is the JSON kind that could not be handled.
	JSONKind Kind // may be zero if unknown
	// GoType is the Go type that could not be handled.
	GoType reflect.Type // may be nil if unknown

	// Err is the underlying error.
	Err error // may be nil
}

func (e *SemanticError) Error() string {
	var sb strings.Builder
	sb.WriteString(errorPrefix)

	// Hyrum-proof the error message by deliberately switching between
	// two equivalent renderings of the same error message.
	// The randomization is tied to the Hyrum-proofing already applied
	// on map iteration in Go.
	for phrase := range map[string]struct{}{"cannot": {}, "unable to": {}} {
		sb.WriteString(phrase)
		break // use whichever phrase we get in the first iteration
	}

	// Format action.
	var preposition string
	switch e.action {
	case "marshal":
		sb.WriteString(" marshal")
		preposition = " from"
	case "unmarshal":
		sb.WriteString(" unmarshal")
		preposition = " into"
	default:
		sb.WriteString(" handle")
		preposition = " with"
	}

	// Format JSON kind.
	var omitPreposition bool
	switch e.JSONKind {
	case 'n':
		sb.WriteString(" JSON null")
	case 'f', 't':
		sb.WriteString(" JSON boolean")
	case '"':
		sb.WriteString(" JSON string")
	case '0':
		sb.WriteString(" JSON number")
	case '{', '}':
		sb.WriteString(" JSON object")
	case '[', ']':
		sb.WriteString(" JSON array")
	default:
		omitPreposition = true
	}

	// Format Go type.
	if e.GoType != nil {
		if !omitPreposition {
			sb.WriteString(preposition)
		}
		sb.WriteString(" Go value of type ")
		sb.WriteString(e.GoType.String())
	}

	// Format where.
	switch {
	case e.JSONPointer != "":
		sb.WriteString(" within JSON value at ")
		sb.WriteString(strconv.Quote(e.JSONPointer))
	case e.ByteOffset > 0:
		sb.WriteString(" after byte offset ")
		sb.WriteString(strconv.FormatInt(e.ByteOffset, 10))
	}

	// Format underlying error.
	if e.Err != nil {
		sb.WriteString(": ")
		sb.WriteString(e.Err.Error())
	}

	return sb.String()
}
func (e *SemanticError) Is(target error) bool {
	return e == target || target == Error || errors.Is(e.Err, target)
}
func (e *SemanticError) Unwrap() error {
	return e.Err
}

// SyntacticError is a description of a syntactic error that occurred when
// encoding or decoding JSON according to the grammar.
//
// The contents of this error as produced by this package may change over time.
type SyntacticError struct {
	requireKeyedLiterals
	nonComparable

	// ByteOffset indicates that an error occurred after this byte offset.
	ByteOffset int64
	str        string
}

func (e *SyntacticError) Error() string {
	return errorPrefix + e.str
}
func (e *SyntacticError) Is(target error) bool {
	return e == target || target == Error
}
func (e *SyntacticError) withOffset(pos int64) error {
	return &SyntacticError{ByteOffset: pos, str: e.str}
}

func newInvalidCharacterError(prefix []byte, where string) *SyntacticError {
	what := quoteRune(prefix)
	return &SyntacticError{str: "invalid character " + what + " " + where}
}

func quoteRune(b []byte) string {
	r, n := utf8.DecodeRune(b)
	if r == utf8.RuneError && n == 1 {
		return `'\x` + strconv.FormatUint(uint64(b[0]), 16) + `'`
	}
	return strconv.QuoteRune(r)
}
