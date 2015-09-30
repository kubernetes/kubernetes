/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package tff

import (
	fflib "github.com/pquerna/ffjson/fflib/v1"

	"testing"
)

// Test data from https://github.com/akheron/jansson/tree/master/test/suites
// jansson, Copyright (c) 2009-2014 Petri Lehtinen <petri@digip.org>
// (MIT Licensed)

func TestInvalidBareKey(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{X:"foo"}`,
		&Xobj{})
}

func TestInvalidNoValue(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X":}`,
		&Xobj{})
}

func TestInvalidTrailingComma(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X":"foo",}`,
		&Xobj{})
}

func TestInvalidRougeComma(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{,}`,
		&Xobj{})
}

func TestInvalidRougeColon(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{:}`,
		&Xobj{})
}

func TestInvalidMissingColon(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X""foo"}`,
		&Xobj{})
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X" "foo"}`,
		&Xobj{})
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X","foo"}`,
		&Xobj{})
}

func TestInvalidUnmatchedBrace(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`[`,
		&Xobj{})
}

func TestInvalidUnmatchedBracket(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{`,
		&Xobj{})
}

func TestInvalidExpectedObjGotArray(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`[]`,
		&Xobj{})
}

func TestInvalidUnterminatedValue(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{"X": "foo`,
		&Xobj{})
}
