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

	_ "encoding/json"
	"testing"
)

// Test data from https://github.com/akheron/jansson/tree/master/test/suites/invalid
// jansson, Copyright (c) 2009-2014 Petri Lehtinen <petri@digip.org>
// (MIT Licensed)

func TestInvalidApostrophe(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`'`,
		&Xstring{})
}

func TestInvalidASCIIUnicodeIdentifier(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`aå`,
		&Xstring{})
}

func TestInvalidBraceComma(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{,}`,
		&Xstring{})
}

func TestInvalidBracketComma(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`[,]`,
		&Xarray{})
}

func TestInvalidBracketValueComma(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`[1,`,
		&Xarray{})
}

func TestInvalidEmptyValue(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		``,
		&Xarray{})
}

func TestInvalidGarbageAfterNewline(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		"[1,2,3]\nfoo",
		&Xarray{})
}

func TestInvalidGarbageAtEnd(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		"[1,2,3]foo",
		&Xarray{})
}

func TestInvalidIntStartingWithZero(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		"012",
		&Xint64{})
}

func TestInvalidEscape(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`"\a <-- invalid escape"`,
		&Xstring{})
}

func TestInvalidIdentifier(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`troo`,
		&Xbool{})
}

func TestInvalidNegativeInt(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`-123foo`,
		&Xint{})
}

func TestInvalidNegativeFloat(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`-124.123foo`,
		&Xfloat64{})
}

func TestInvalidSecondSurrogate(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`"\uD888\u3210 (first surrogate and invalid second surrogate)"`,
		&Xstring{})
}

func TestInvalidLoneOpenBrace(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`{`,
		&Xstring{})
}

func TestInvalidLoneOpenBracket(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`[`,
		&Xarray{})
}

func TestInvalidLoneCloseBrace(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`}`,
		&Xstring{})
}

func TestInvalidHighBytes(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		string('\xFF'),
		&Xstring{})
}

func TestInvalidLoneCloseBracket(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`]`,
		&Xarray{})
}

func TestInvalidMinusSignWithoutNumber(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`-`,
		&Xint{})
}

func TestInvalidNullByte(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		"\u0000",
		&Xstring{})
}

func TestInvalidNullByteInString(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		"\"\u0000 <- null byte\"",
		&Xstring{})
}

func TestInvalidFloatGarbageAfterE(t *testing.T) {
	testExpectedError(t,
		&fflib.LexerError{},
		`1ea`,
		&Xfloat64{})
}
