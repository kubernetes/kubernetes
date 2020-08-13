// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"errors"
	"fmt"
	"io"

	"go.mongodb.org/mongo-driver/bson/bsontype"
)

const maxNestingDepth = 200

// ErrInvalidJSON indicates the JSON input is invalid
var ErrInvalidJSON = errors.New("invalid JSON input")

type jsonParseState byte

const (
	jpsStartState jsonParseState = iota
	jpsSawBeginObject
	jpsSawEndObject
	jpsSawBeginArray
	jpsSawEndArray
	jpsSawColon
	jpsSawComma
	jpsSawKey
	jpsSawValue
	jpsDoneState
	jpsInvalidState
)

type jsonParseMode byte

const (
	jpmInvalidMode jsonParseMode = iota
	jpmObjectMode
	jpmArrayMode
)

type extJSONValue struct {
	t bsontype.Type
	v interface{}
}

type extJSONObject struct {
	keys   []string
	values []*extJSONValue
}

type extJSONParser struct {
	js *jsonScanner
	s  jsonParseState
	m  []jsonParseMode
	k  string
	v  *extJSONValue

	err       error
	canonical bool
	depth     int
	maxDepth  int

	emptyObject bool
}

// newExtJSONParser returns a new extended JSON parser, ready to to begin
// parsing from the first character of the argued json input. It will not
// perform any read-ahead and will therefore not report any errors about
// malformed JSON at this point.
func newExtJSONParser(r io.Reader, canonical bool) *extJSONParser {
	return &extJSONParser{
		js:        &jsonScanner{r: r},
		s:         jpsStartState,
		m:         []jsonParseMode{},
		canonical: canonical,
		maxDepth:  maxNestingDepth,
	}
}

// peekType examines the next value and returns its BSON Type
func (ejp *extJSONParser) peekType() (bsontype.Type, error) {
	var t bsontype.Type
	var err error

	ejp.advanceState()
	switch ejp.s {
	case jpsSawValue:
		t = ejp.v.t
	case jpsSawBeginArray:
		t = bsontype.Array
	case jpsInvalidState:
		err = ejp.err
	case jpsSawComma:
		// in array mode, seeing a comma means we need to progress again to actually observe a type
		if ejp.peekMode() == jpmArrayMode {
			return ejp.peekType()
		}
	case jpsSawEndArray:
		// this would only be a valid state if we were in array mode, so return end-of-array error
		err = ErrEOA
	case jpsSawBeginObject:
		// peek key to determine type
		ejp.advanceState()
		switch ejp.s {
		case jpsSawEndObject: // empty embedded document
			t = bsontype.EmbeddedDocument
			ejp.emptyObject = true
		case jpsInvalidState:
			err = ejp.err
		case jpsSawKey:
			t = wrapperKeyBSONType(ejp.k)

			if t == bsontype.JavaScript {
				// just saw $code, need to check for $scope at same level
				_, err := ejp.readValue(bsontype.JavaScript)

				if err != nil {
					break
				}

				switch ejp.s {
				case jpsSawEndObject: // type is TypeJavaScript
				case jpsSawComma:
					ejp.advanceState()
					if ejp.s == jpsSawKey && ejp.k == "$scope" {
						t = bsontype.CodeWithScope
					} else {
						err = fmt.Errorf("invalid extended JSON: unexpected key %s in CodeWithScope object", ejp.k)
					}
				case jpsInvalidState:
					err = ejp.err
				default:
					err = ErrInvalidJSON
				}
			}
		}
	}

	return t, err
}

// readKey parses the next key and its type and returns them
func (ejp *extJSONParser) readKey() (string, bsontype.Type, error) {
	if ejp.emptyObject {
		ejp.emptyObject = false
		return "", 0, ErrEOD
	}

	// advance to key (or return with error)
	switch ejp.s {
	case jpsStartState:
		ejp.advanceState()
		if ejp.s == jpsSawBeginObject {
			ejp.advanceState()
		}
	case jpsSawBeginObject:
		ejp.advanceState()
	case jpsSawValue, jpsSawEndObject, jpsSawEndArray:
		ejp.advanceState()
		switch ejp.s {
		case jpsSawBeginObject, jpsSawComma:
			ejp.advanceState()
		case jpsSawEndObject:
			return "", 0, ErrEOD
		case jpsDoneState:
			return "", 0, io.EOF
		case jpsInvalidState:
			return "", 0, ejp.err
		default:
			return "", 0, ErrInvalidJSON
		}
	case jpsSawKey: // do nothing (key was peeked before)
	default:
		return "", 0, invalidRequestError("key")
	}

	// read key
	var key string

	switch ejp.s {
	case jpsSawKey:
		key = ejp.k
	case jpsSawEndObject:
		return "", 0, ErrEOD
	case jpsInvalidState:
		return "", 0, ejp.err
	default:
		return "", 0, invalidRequestError("key")
	}

	// check for colon
	ejp.advanceState()
	if err := ensureColon(ejp.s, key); err != nil {
		return "", 0, err
	}

	// peek at the value to determine type
	t, err := ejp.peekType()
	if err != nil {
		return "", 0, err
	}

	return key, t, nil
}

// readValue returns the value corresponding to the Type returned by peekType
func (ejp *extJSONParser) readValue(t bsontype.Type) (*extJSONValue, error) {
	if ejp.s == jpsInvalidState {
		return nil, ejp.err
	}

	var v *extJSONValue

	switch t {
	case bsontype.Null, bsontype.Boolean, bsontype.String:
		if ejp.s != jpsSawValue {
			return nil, invalidRequestError(t.String())
		}
		v = ejp.v
	case bsontype.Int32, bsontype.Int64, bsontype.Double:
		// relaxed version allows these to be literal number values
		if ejp.s == jpsSawValue {
			v = ejp.v
			break
		}
		fallthrough
	case bsontype.Decimal128, bsontype.Symbol, bsontype.ObjectID, bsontype.MinKey, bsontype.MaxKey, bsontype.Undefined:
		switch ejp.s {
		case jpsSawKey:
			// read colon
			ejp.advanceState()
			if err := ensureColon(ejp.s, ejp.k); err != nil {
				return nil, err
			}

			// read value
			ejp.advanceState()
			if ejp.s != jpsSawValue || !ejp.ensureExtValueType(t) {
				return nil, invalidJSONErrorForType("value", t)
			}

			v = ejp.v

			// read end object
			ejp.advanceState()
			if ejp.s != jpsSawEndObject {
				return nil, invalidJSONErrorForType("} after value", t)
			}
		default:
			return nil, invalidRequestError(t.String())
		}
	case bsontype.Binary, bsontype.Regex, bsontype.Timestamp, bsontype.DBPointer:
		if ejp.s != jpsSawKey {
			return nil, invalidRequestError(t.String())
		}
		// read colon
		ejp.advanceState()
		if err := ensureColon(ejp.s, ejp.k); err != nil {
			return nil, err
		}

		ejp.advanceState()
		if t == bsontype.Binary && ejp.s == jpsSawValue {
			// convert legacy $binary format
			base64 := ejp.v

			ejp.advanceState()
			if ejp.s != jpsSawComma {
				return nil, invalidJSONErrorForType(",", bsontype.Binary)
			}

			ejp.advanceState()
			key, t, err := ejp.readKey()
			if err != nil {
				return nil, err
			}
			if key != "$type" {
				return nil, invalidJSONErrorForType("$type", bsontype.Binary)
			}

			subType, err := ejp.readValue(t)
			if err != nil {
				return nil, err
			}

			ejp.advanceState()
			if ejp.s != jpsSawEndObject {
				return nil, invalidJSONErrorForType("2 key-value pairs and then }", bsontype.Binary)
			}

			v = &extJSONValue{
				t: bsontype.EmbeddedDocument,
				v: &extJSONObject{
					keys:   []string{"base64", "subType"},
					values: []*extJSONValue{base64, subType},
				},
			}
			break
		}

		// read KV pairs
		if ejp.s != jpsSawBeginObject {
			return nil, invalidJSONErrorForType("{", t)
		}

		keys, vals, err := ejp.readObject(2, true)
		if err != nil {
			return nil, err
		}

		ejp.advanceState()
		if ejp.s != jpsSawEndObject {
			return nil, invalidJSONErrorForType("2 key-value pairs and then }", t)
		}

		v = &extJSONValue{t: bsontype.EmbeddedDocument, v: &extJSONObject{keys: keys, values: vals}}

	case bsontype.DateTime:
		switch ejp.s {
		case jpsSawValue:
			v = ejp.v
		case jpsSawKey:
			// read colon
			ejp.advanceState()
			if err := ensureColon(ejp.s, ejp.k); err != nil {
				return nil, err
			}

			ejp.advanceState()
			switch ejp.s {
			case jpsSawBeginObject:
				keys, vals, err := ejp.readObject(1, true)
				if err != nil {
					return nil, err
				}
				v = &extJSONValue{t: bsontype.EmbeddedDocument, v: &extJSONObject{keys: keys, values: vals}}
			case jpsSawValue:
				if ejp.canonical {
					return nil, invalidJSONError("{")
				}
				v = ejp.v
			default:
				if ejp.canonical {
					return nil, invalidJSONErrorForType("object", t)
				}
				return nil, invalidJSONErrorForType("ISO-8601 Internet Date/Time Format as decribed in RFC-3339", t)
			}

			ejp.advanceState()
			if ejp.s != jpsSawEndObject {
				return nil, invalidJSONErrorForType("value and then }", t)
			}
		default:
			return nil, invalidRequestError(t.String())
		}
	case bsontype.JavaScript:
		switch ejp.s {
		case jpsSawKey:
			// read colon
			ejp.advanceState()
			if err := ensureColon(ejp.s, ejp.k); err != nil {
				return nil, err
			}

			// read value
			ejp.advanceState()
			if ejp.s != jpsSawValue {
				return nil, invalidJSONErrorForType("value", t)
			}
			v = ejp.v

			// read end object or comma and just return
			ejp.advanceState()
		case jpsSawEndObject:
			v = ejp.v
		default:
			return nil, invalidRequestError(t.String())
		}
	case bsontype.CodeWithScope:
		if ejp.s == jpsSawKey && ejp.k == "$scope" {
			v = ejp.v // this is the $code string from earlier

			// read colon
			ejp.advanceState()
			if err := ensureColon(ejp.s, ejp.k); err != nil {
				return nil, err
			}

			// read {
			ejp.advanceState()
			if ejp.s != jpsSawBeginObject {
				return nil, invalidJSONError("$scope to be embedded document")
			}
		} else {
			return nil, invalidRequestError(t.String())
		}
	case bsontype.EmbeddedDocument, bsontype.Array:
		return nil, invalidRequestError(t.String())
	}

	return v, nil
}

// readObject is a utility method for reading full objects of known (or expected) size
// it is useful for extended JSON types such as binary, datetime, regex, and timestamp
func (ejp *extJSONParser) readObject(numKeys int, started bool) ([]string, []*extJSONValue, error) {
	keys := make([]string, numKeys)
	vals := make([]*extJSONValue, numKeys)

	if !started {
		ejp.advanceState()
		if ejp.s != jpsSawBeginObject {
			return nil, nil, invalidJSONError("{")
		}
	}

	for i := 0; i < numKeys; i++ {
		key, t, err := ejp.readKey()
		if err != nil {
			return nil, nil, err
		}

		switch ejp.s {
		case jpsSawKey:
			v, err := ejp.readValue(t)
			if err != nil {
				return nil, nil, err
			}

			keys[i] = key
			vals[i] = v
		case jpsSawValue:
			keys[i] = key
			vals[i] = ejp.v
		default:
			return nil, nil, invalidJSONError("value")
		}
	}

	ejp.advanceState()
	if ejp.s != jpsSawEndObject {
		return nil, nil, invalidJSONError("}")
	}

	return keys, vals, nil
}

// advanceState reads the next JSON token from the scanner and transitions
// from the current state based on that token's type
func (ejp *extJSONParser) advanceState() {
	if ejp.s == jpsDoneState || ejp.s == jpsInvalidState {
		return
	}

	jt, err := ejp.js.nextToken()

	if err != nil {
		ejp.err = err
		ejp.s = jpsInvalidState
		return
	}

	valid := ejp.validateToken(jt.t)
	if !valid {
		ejp.err = unexpectedTokenError(jt)
		ejp.s = jpsInvalidState
		return
	}

	switch jt.t {
	case jttBeginObject:
		ejp.s = jpsSawBeginObject
		ejp.pushMode(jpmObjectMode)
		ejp.depth++

		if ejp.depth > ejp.maxDepth {
			ejp.err = nestingDepthError(jt.p, ejp.depth)
			ejp.s = jpsInvalidState
		}
	case jttEndObject:
		ejp.s = jpsSawEndObject
		ejp.depth--

		if ejp.popMode() != jpmObjectMode {
			ejp.err = unexpectedTokenError(jt)
			ejp.s = jpsInvalidState
		}
	case jttBeginArray:
		ejp.s = jpsSawBeginArray
		ejp.pushMode(jpmArrayMode)
	case jttEndArray:
		ejp.s = jpsSawEndArray

		if ejp.popMode() != jpmArrayMode {
			ejp.err = unexpectedTokenError(jt)
			ejp.s = jpsInvalidState
		}
	case jttColon:
		ejp.s = jpsSawColon
	case jttComma:
		ejp.s = jpsSawComma
	case jttEOF:
		ejp.s = jpsDoneState
		if len(ejp.m) != 0 {
			ejp.err = unexpectedTokenError(jt)
			ejp.s = jpsInvalidState
		}
	case jttString:
		switch ejp.s {
		case jpsSawComma:
			if ejp.peekMode() == jpmArrayMode {
				ejp.s = jpsSawValue
				ejp.v = extendJSONToken(jt)
				return
			}
			fallthrough
		case jpsSawBeginObject:
			ejp.s = jpsSawKey
			ejp.k = jt.v.(string)
			return
		}
		fallthrough
	default:
		ejp.s = jpsSawValue
		ejp.v = extendJSONToken(jt)
	}
}

var jpsValidTransitionTokens = map[jsonParseState]map[jsonTokenType]bool{
	jpsStartState: {
		jttBeginObject: true,
		jttBeginArray:  true,
		jttInt32:       true,
		jttInt64:       true,
		jttDouble:      true,
		jttString:      true,
		jttBool:        true,
		jttNull:        true,
		jttEOF:         true,
	},
	jpsSawBeginObject: {
		jttEndObject: true,
		jttString:    true,
	},
	jpsSawEndObject: {
		jttEndObject: true,
		jttEndArray:  true,
		jttComma:     true,
		jttEOF:       true,
	},
	jpsSawBeginArray: {
		jttBeginObject: true,
		jttBeginArray:  true,
		jttEndArray:    true,
		jttInt32:       true,
		jttInt64:       true,
		jttDouble:      true,
		jttString:      true,
		jttBool:        true,
		jttNull:        true,
	},
	jpsSawEndArray: {
		jttEndObject: true,
		jttEndArray:  true,
		jttComma:     true,
		jttEOF:       true,
	},
	jpsSawColon: {
		jttBeginObject: true,
		jttBeginArray:  true,
		jttInt32:       true,
		jttInt64:       true,
		jttDouble:      true,
		jttString:      true,
		jttBool:        true,
		jttNull:        true,
	},
	jpsSawComma: {
		jttBeginObject: true,
		jttBeginArray:  true,
		jttInt32:       true,
		jttInt64:       true,
		jttDouble:      true,
		jttString:      true,
		jttBool:        true,
		jttNull:        true,
	},
	jpsSawKey: {
		jttColon: true,
	},
	jpsSawValue: {
		jttEndObject: true,
		jttEndArray:  true,
		jttComma:     true,
		jttEOF:       true,
	},
	jpsDoneState:    {},
	jpsInvalidState: {},
}

func (ejp *extJSONParser) validateToken(jtt jsonTokenType) bool {
	switch ejp.s {
	case jpsSawEndObject:
		// if we are at depth zero and the next token is a '{',
		// we can consider it valid only if we are not in array mode.
		if jtt == jttBeginObject && ejp.depth == 0 {
			return ejp.peekMode() != jpmArrayMode
		}
	case jpsSawComma:
		switch ejp.peekMode() {
		// the only valid next token after a comma inside a document is a string (a key)
		case jpmObjectMode:
			return jtt == jttString
		case jpmInvalidMode:
			return false
		}
	}

	_, ok := jpsValidTransitionTokens[ejp.s][jtt]
	return ok
}

// ensureExtValueType returns true if the current value has the expected
// value type for single-key extended JSON types. For example,
// {"$numberInt": v} v must be TypeString
func (ejp *extJSONParser) ensureExtValueType(t bsontype.Type) bool {
	switch t {
	case bsontype.MinKey, bsontype.MaxKey:
		return ejp.v.t == bsontype.Int32
	case bsontype.Undefined:
		return ejp.v.t == bsontype.Boolean
	case bsontype.Int32, bsontype.Int64, bsontype.Double, bsontype.Decimal128, bsontype.Symbol, bsontype.ObjectID:
		return ejp.v.t == bsontype.String
	default:
		return false
	}
}

func (ejp *extJSONParser) pushMode(m jsonParseMode) {
	ejp.m = append(ejp.m, m)
}

func (ejp *extJSONParser) popMode() jsonParseMode {
	l := len(ejp.m)
	if l == 0 {
		return jpmInvalidMode
	}

	m := ejp.m[l-1]
	ejp.m = ejp.m[:l-1]

	return m
}

func (ejp *extJSONParser) peekMode() jsonParseMode {
	l := len(ejp.m)
	if l == 0 {
		return jpmInvalidMode
	}

	return ejp.m[l-1]
}

func extendJSONToken(jt *jsonToken) *extJSONValue {
	var t bsontype.Type

	switch jt.t {
	case jttInt32:
		t = bsontype.Int32
	case jttInt64:
		t = bsontype.Int64
	case jttDouble:
		t = bsontype.Double
	case jttString:
		t = bsontype.String
	case jttBool:
		t = bsontype.Boolean
	case jttNull:
		t = bsontype.Null
	default:
		return nil
	}

	return &extJSONValue{t: t, v: jt.v}
}

func ensureColon(s jsonParseState, key string) error {
	if s != jpsSawColon {
		return fmt.Errorf("invalid JSON input: missing colon after key \"%s\"", key)
	}

	return nil
}

func invalidRequestError(s string) error {
	return fmt.Errorf("invalid request to read %s", s)
}

func invalidJSONError(expected string) error {
	return fmt.Errorf("invalid JSON input; expected %s", expected)
}

func invalidJSONErrorForType(expected string, t bsontype.Type) error {
	return fmt.Errorf("invalid JSON input; expected %s for %s", expected, t)
}

func unexpectedTokenError(jt *jsonToken) error {
	switch jt.t {
	case jttInt32, jttInt64, jttDouble:
		return fmt.Errorf("invalid JSON input; unexpected number (%v) at position %d", jt.v, jt.p)
	case jttString:
		return fmt.Errorf("invalid JSON input; unexpected string (\"%v\") at position %d", jt.v, jt.p)
	case jttBool:
		return fmt.Errorf("invalid JSON input; unexpected boolean literal (%v) at position %d", jt.v, jt.p)
	case jttNull:
		return fmt.Errorf("invalid JSON input; unexpected null literal at position %d", jt.p)
	case jttEOF:
		return fmt.Errorf("invalid JSON input; unexpected end of input at position %d", jt.p)
	default:
		return fmt.Errorf("invalid JSON input; unexpected %c at position %d", jt.v.(byte), jt.p)
	}
}

func nestingDepthError(p, depth int) error {
	return fmt.Errorf("invalid JSON input; nesting too deep (%d levels) at position %d", depth, p)
}
