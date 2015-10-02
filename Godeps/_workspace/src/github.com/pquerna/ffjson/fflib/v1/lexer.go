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

/* Portions of this file are on derived from yajl: <https://github.com/lloyd/yajl> */
/*
 * Copyright (c) 2007-2014, Lloyd Hilaiel <me@lloyd.io>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package v1

import (
	"errors"
	"fmt"
	"io"
)

type FFParseState int

const (
	FFParse_map_start FFParseState = iota
	FFParse_want_key
	FFParse_want_colon
	FFParse_want_value
	FFParse_after_value
)

type FFTok int

const (
	FFTok_init          FFTok = iota
	FFTok_bool          FFTok = iota
	FFTok_colon         FFTok = iota
	FFTok_comma         FFTok = iota
	FFTok_eof           FFTok = iota
	FFTok_error         FFTok = iota
	FFTok_left_brace    FFTok = iota
	FFTok_left_bracket  FFTok = iota
	FFTok_null          FFTok = iota
	FFTok_right_brace   FFTok = iota
	FFTok_right_bracket FFTok = iota

	/* we differentiate between integers and doubles to allow the
	 * parser to interpret the number without re-scanning */
	FFTok_integer FFTok = iota
	FFTok_double  FFTok = iota

	FFTok_string FFTok = iota

	/* comment tokens are not currently returned to the parser, ever */
	FFTok_comment FFTok = iota
)

type FFErr int

const (
	FFErr_e_ok                           FFErr = iota
	FFErr_io                             FFErr = iota
	FFErr_string_invalid_utf8            FFErr = iota
	FFErr_string_invalid_escaped_char    FFErr = iota
	FFErr_string_invalid_json_char       FFErr = iota
	FFErr_string_invalid_hex_char        FFErr = iota
	FFErr_invalid_char                   FFErr = iota
	FFErr_invalid_string                 FFErr = iota
	FFErr_missing_integer_after_decimal  FFErr = iota
	FFErr_missing_integer_after_exponent FFErr = iota
	FFErr_missing_integer_after_minus    FFErr = iota
	FFErr_unallowed_comment              FFErr = iota
	FFErr_incomplete_comment             FFErr = iota
	FFErr_unexpected_token_type          FFErr = iota // TODO: improve this error
)

type FFLexer struct {
	reader   *ffReader
	Output   DecodingBuffer
	Token    FFTok
	Error    FFErr
	BigError error
	// TODO: convert all of this to an interface
	lastCurrentChar int
	captureAll      bool
	buf             Buffer
}

func NewFFLexer(input []byte) *FFLexer {
	fl := &FFLexer{
		Token:  FFTok_init,
		Error:  FFErr_e_ok,
		reader: newffReader(input),
		Output: &Buffer{},
	}
	// TODO: guess size?
	//fl.Output.Grow(64)
	return fl
}

type LexerError struct {
	offset int
	line   int
	char   int
	err    error
}

// Reset the Lexer and add new input.
func (ffl *FFLexer) Reset(input []byte) {
	ffl.Token = FFTok_init
	ffl.Error = FFErr_e_ok
	ffl.BigError = nil
	ffl.reader.Reset(input)
	ffl.lastCurrentChar = 0
	ffl.Output.Reset()
}

func (le *LexerError) Error() string {
	return fmt.Sprintf(`ffjson error: (%T)%s offset=%d line=%d char=%d`,
		le.err, le.err.Error(),
		le.offset, le.line, le.char)
}

func (ffl *FFLexer) WrapErr(err error) error {
	line, char := ffl.reader.PosWithLine()
	// TOOD: calcualte lines/characters based on offset
	return &LexerError{
		offset: ffl.reader.Pos(),
		line:   line,
		char:   char,
		err:    err,
	}
}

func (ffl *FFLexer) scanReadByte() (byte, error) {
	var c byte
	var err error
	if ffl.captureAll {
		c, err = ffl.reader.ReadByte()
	} else {
		c, err = ffl.reader.ReadByteNoWS()
	}

	if err != nil {
		ffl.Error = FFErr_io
		ffl.BigError = err
		return 0, err
	}

	return c, nil
}

func (ffl *FFLexer) readByte() (byte, error) {

	c, err := ffl.reader.ReadByte()
	if err != nil {
		ffl.Error = FFErr_io
		ffl.BigError = err
		return 0, err
	}

	return c, nil
}

func (ffl *FFLexer) unreadByte() {
	ffl.reader.UnreadByte()
}

func (ffl *FFLexer) wantBytes(want []byte, iftrue FFTok) FFTok {
	for _, b := range want {
		c, err := ffl.readByte()

		if err != nil {
			return FFTok_error
		}

		if c != b {
			ffl.unreadByte()
			// fmt.Printf("wanted bytes: %s\n", string(want))
			// TODO(pquerna): thsi is a bad error message
			ffl.Error = FFErr_invalid_string
			return FFTok_error
		}

		ffl.Output.WriteByte(c)
	}

	return iftrue
}

func (ffl *FFLexer) lexComment() FFTok {
	c, err := ffl.readByte()
	if err != nil {
		return FFTok_error
	}

	if c == '/' {
		// a // comment, scan until line ends.
		for {
			c, err := ffl.readByte()
			if err != nil {
				return FFTok_error
			}

			if c == '\n' {
				return FFTok_comment
			}
		}
	} else if c == '*' {
		// a /* */ comment, scan */
		for {
			c, err := ffl.readByte()
			if err != nil {
				return FFTok_error
			}

			if c == '*' {
				c, err := ffl.readByte()

				if err != nil {
					return FFTok_error
				}

				if c == '/' {
					return FFTok_comment
				}

				ffl.Error = FFErr_incomplete_comment
				return FFTok_error
			}
		}
	} else {
		ffl.Error = FFErr_incomplete_comment
		return FFTok_error
	}
}

func (ffl *FFLexer) lexString() FFTok {
	if ffl.captureAll {
		ffl.buf.Reset()
		err := ffl.reader.SliceString(&ffl.buf)

		if err != nil {
			ffl.BigError = err
			return FFTok_error
		}

		WriteJson(ffl.Output, ffl.buf.Bytes())

		return FFTok_string
	} else {
		err := ffl.reader.SliceString(ffl.Output)

		if err != nil {
			ffl.BigError = err
			return FFTok_error
		}

		return FFTok_string
	}
}

func (ffl *FFLexer) lexNumber() FFTok {
	var numRead int = 0
	tok := FFTok_integer

	c, err := ffl.readByte()
	if err != nil {
		return FFTok_error
	}

	/* optional leading minus */
	if c == '-' {
		ffl.Output.WriteByte(c)
		c, err = ffl.readByte()
		if err != nil {
			return FFTok_error
		}
	}

	/* a single zero, or a series of integers */
	if c == '0' {
		ffl.Output.WriteByte(c)
		c, err = ffl.readByte()
		if err != nil {
			return FFTok_error
		}
	} else if c >= '1' && c <= '9' {
		for c >= '0' && c <= '9' {
			ffl.Output.WriteByte(c)
			c, err = ffl.readByte()
			if err != nil {
				return FFTok_error
			}
		}
	} else {
		ffl.unreadByte()
		ffl.Error = FFErr_missing_integer_after_minus
		return FFTok_error
	}

	if c == '.' {
		numRead = 0
		ffl.Output.WriteByte(c)
		c, err = ffl.readByte()
		if err != nil {
			return FFTok_error
		}

		for c >= '0' && c <= '9' {
			ffl.Output.WriteByte(c)
			numRead++
			c, err = ffl.readByte()
			if err != nil {
				return FFTok_error
			}
		}

		if numRead == 0 {
			ffl.unreadByte()

			ffl.Error = FFErr_missing_integer_after_decimal
			return FFTok_error
		}

		tok = FFTok_double
	}

	/* optional exponent (indicates this is floating point) */
	if c == 'e' || c == 'E' {
		numRead = 0
		ffl.Output.WriteByte(c)

		c, err = ffl.readByte()
		if err != nil {
			return FFTok_error
		}

		/* optional sign */
		if c == '+' || c == '-' {
			ffl.Output.WriteByte(c)
			c, err = ffl.readByte()
			if err != nil {
				return FFTok_error
			}
		}

		for c >= '0' && c <= '9' {
			ffl.Output.WriteByte(c)
			numRead++
			c, err = ffl.readByte()
			if err != nil {
				return FFTok_error
			}
		}

		if numRead == 0 {
			ffl.Error = FFErr_missing_integer_after_exponent
			return FFTok_error
		}

		tok = FFTok_double
	}

	ffl.unreadByte()

	return tok
}

var true_bytes = []byte{'r', 'u', 'e'}
var false_bytes = []byte{'a', 'l', 's', 'e'}
var null_bytes = []byte{'u', 'l', 'l'}

func (ffl *FFLexer) Scan() FFTok {
	tok := FFTok_error
	if ffl.captureAll == false {
		ffl.Output.Reset()
	}
	ffl.Token = FFTok_init

	for {
		c, err := ffl.scanReadByte()
		if err != nil {
			if err == io.EOF {
				return FFTok_eof
			} else {
				return FFTok_error
			}
		}

		switch c {
		case '{':
			tok = FFTok_left_bracket
			if ffl.captureAll {
				ffl.Output.WriteByte('{')
			}
			goto lexed
		case '}':
			tok = FFTok_right_bracket
			if ffl.captureAll {
				ffl.Output.WriteByte('}')
			}
			goto lexed
		case '[':
			tok = FFTok_left_brace
			if ffl.captureAll {
				ffl.Output.WriteByte('[')
			}
			goto lexed
		case ']':
			tok = FFTok_right_brace
			if ffl.captureAll {
				ffl.Output.WriteByte(']')
			}
			goto lexed
		case ',':
			tok = FFTok_comma
			if ffl.captureAll {
				ffl.Output.WriteByte(',')
			}
			goto lexed
		case ':':
			tok = FFTok_colon
			if ffl.captureAll {
				ffl.Output.WriteByte(':')
			}
			goto lexed
		case '\t', '\n', '\v', '\f', '\r', ' ':
			if ffl.captureAll {
				ffl.Output.WriteByte(c)
			}
			break
		case 't':
			ffl.Output.WriteByte('t')
			tok = ffl.wantBytes(true_bytes, FFTok_bool)
			goto lexed
		case 'f':
			ffl.Output.WriteByte('f')
			tok = ffl.wantBytes(false_bytes, FFTok_bool)
			goto lexed
		case 'n':
			ffl.Output.WriteByte('n')
			tok = ffl.wantBytes(null_bytes, FFTok_null)
			goto lexed
		case '"':
			tok = ffl.lexString()
			goto lexed
		case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			ffl.unreadByte()
			tok = ffl.lexNumber()
			goto lexed
		case '/':
			tok = ffl.lexComment()
			goto lexed
		default:
			tok = FFTok_error
			ffl.Error = FFErr_invalid_char
		}
	}

lexed:
	ffl.Token = tok
	return tok
}

func (ffl *FFLexer) scanField(start FFTok, capture bool) ([]byte, error) {
	switch start {
	case FFTok_left_brace,
		FFTok_left_bracket:
		{
			end := FFTok_right_brace
			if start == FFTok_left_bracket {
				end = FFTok_right_bracket
				if capture {
					ffl.Output.WriteByte('{')
				}
			} else {
				if capture {
					ffl.Output.WriteByte('[')
				}
			}

			depth := 1
			if capture {
				ffl.captureAll = true
			}
			// TODO: work.
		scanloop:
			for {
				tok := ffl.Scan()
				//fmt.Printf("capture-token: %v end: %v depth: %v\n", tok, end, depth)
				switch tok {
				case FFTok_eof:
					return nil, errors.New("ffjson: unexpected EOF")
				case FFTok_error:
					if ffl.BigError != nil {
						return nil, ffl.BigError
					}
					return nil, ffl.Error.ToError()
				case end:
					depth--
					if depth == 0 {
						break scanloop
					}
				case start:
					depth++
				}
			}

			if capture {
				ffl.captureAll = false
			}

			if capture {
				return ffl.Output.Bytes(), nil
			} else {
				return nil, nil
			}
		}
	case FFTok_bool,
		FFTok_integer,
		FFTok_null,
		FFTok_double:
		// simple value, return it.
		if capture {
			return ffl.Output.Bytes(), nil
		} else {
			return nil, nil
		}

	case FFTok_string:
		//TODO(pquerna): so, other users expect this to be a quoted string :(
		if capture {
			ffl.buf.Reset()
			WriteJson(&ffl.buf, ffl.Output.Bytes())
			return ffl.buf.Bytes(), nil
		} else {
			return nil, nil
		}

	default:
		return nil, fmt.Errorf("ffjson: invalid capture type: %v", start)
	}
	panic("not reached")
}

// Captures an entire field value, including recursive objects,
// and converts them to a []byte suitable to pass to a sub-object's
// UnmarshalJSON
func (ffl *FFLexer) CaptureField(start FFTok) ([]byte, error) {
	return ffl.scanField(start, true)
}

func (ffl *FFLexer) SkipField(start FFTok) error {
	_, err := ffl.scanField(start, false)
	return err
}

// TODO(pquerna): return line number and offset.
func (err FFErr) ToError() error {
	switch err {
	case FFErr_e_ok:
		return nil
	case FFErr_io:
		return errors.New("ffjson: IO error")
	case FFErr_string_invalid_utf8:
		return errors.New("ffjson: string with invalid UTF-8 sequence")
	case FFErr_string_invalid_escaped_char:
		return errors.New("ffjson: string with invalid escaped character")
	case FFErr_string_invalid_json_char:
		return errors.New("ffjson: string with invalid JSON character")
	case FFErr_string_invalid_hex_char:
		return errors.New("ffjson: string with invalid hex character")
	case FFErr_invalid_char:
		return errors.New("ffjson: invalid character")
	case FFErr_invalid_string:
		return errors.New("ffjson: invalid string")
	case FFErr_missing_integer_after_decimal:
		return errors.New("ffjson: missing integer after decimal")
	case FFErr_missing_integer_after_exponent:
		return errors.New("ffjson: missing integer after exponent")
	case FFErr_missing_integer_after_minus:
		return errors.New("ffjson: missing integer after minus")
	case FFErr_unallowed_comment:
		return errors.New("ffjson: unallowed comment")
	case FFErr_incomplete_comment:
		return errors.New("ffjson: incomplete comment")
	case FFErr_unexpected_token_type:
		return errors.New("ffjson: unexpected token sequence")
	}

	panic(fmt.Sprintf("unknown error type: %v ", err))
}

func (state FFParseState) String() string {
	switch state {
	case FFParse_map_start:
		return "map:start"
	case FFParse_want_key:
		return "want_key"
	case FFParse_want_colon:
		return "want_colon"
	case FFParse_want_value:
		return "want_value"
	case FFParse_after_value:
		return "after_value"
	}

	panic(fmt.Sprintf("unknown parse state: %d", int(state)))
}

func (tok FFTok) String() string {
	switch tok {
	case FFTok_init:
		return "tok:init"
	case FFTok_bool:
		return "tok:bool"
	case FFTok_colon:
		return "tok:colon"
	case FFTok_comma:
		return "tok:comma"
	case FFTok_eof:
		return "tok:eof"
	case FFTok_error:
		return "tok:error"
	case FFTok_left_brace:
		return "tok:left_brace"
	case FFTok_left_bracket:
		return "tok:left_bracket"
	case FFTok_null:
		return "tok:null"
	case FFTok_right_brace:
		return "tok:right_brace"
	case FFTok_right_bracket:
		return "tok:right_bracket"
	case FFTok_integer:
		return "tok:integer"
	case FFTok_double:
		return "tok:double"
	case FFTok_string:
		return "tok:string"
	case FFTok_comment:
		return "comment"
	}

	panic(fmt.Sprintf("unknown token: %d", int(tok)))
}

/* a lookup table which lets us quickly determine three things:
 * cVEC - valid escaped control char
 * note.  the solidus '/' may be escaped or not.
 * cIJC - invalid json char
 * cVHC - valid hex char
 * cNFP - needs further processing (from a string scanning perspective)
 * cNUC - needs utf8 checking when enabled (from a string scanning perspective)
 */

const (
	cVEC int8 = 0x01
	cIJC int8 = 0x02
	cVHC int8 = 0x04
	cNFP int8 = 0x08
	cNUC int8 = 0x10
)

var byteLookupTable [256]int8 = [256]int8{
	cIJC,               /* 0 */
	cIJC,               /* 1 */
	cIJC,               /* 2 */
	cIJC,               /* 3 */
	cIJC,               /* 4 */
	cIJC,               /* 5 */
	cIJC,               /* 6 */
	cIJC,               /* 7 */
	cIJC,               /* 8 */
	cIJC,               /* 9 */
	cIJC,               /* 10 */
	cIJC,               /* 11 */
	cIJC,               /* 12 */
	cIJC,               /* 13 */
	cIJC,               /* 14 */
	cIJC,               /* 15 */
	cIJC,               /* 16 */
	cIJC,               /* 17 */
	cIJC,               /* 18 */
	cIJC,               /* 19 */
	cIJC,               /* 20 */
	cIJC,               /* 21 */
	cIJC,               /* 22 */
	cIJC,               /* 23 */
	cIJC,               /* 24 */
	cIJC,               /* 25 */
	cIJC,               /* 26 */
	cIJC,               /* 27 */
	cIJC,               /* 28 */
	cIJC,               /* 29 */
	cIJC,               /* 30 */
	cIJC,               /* 31 */
	0,                  /* 32 */
	0,                  /* 33 */
	cVEC | cIJC | cNFP, /* 34 */
	0,                  /* 35 */
	0,                  /* 36 */
	0,                  /* 37 */
	0,                  /* 38 */
	0,                  /* 39 */
	0,                  /* 40 */
	0,                  /* 41 */
	0,                  /* 42 */
	0,                  /* 43 */
	0,                  /* 44 */
	0,                  /* 45 */
	0,                  /* 46 */
	cVEC,               /* 47 */
	cVHC,               /* 48 */
	cVHC,               /* 49 */
	cVHC,               /* 50 */
	cVHC,               /* 51 */
	cVHC,               /* 52 */
	cVHC,               /* 53 */
	cVHC,               /* 54 */
	cVHC,               /* 55 */
	cVHC,               /* 56 */
	cVHC,               /* 57 */
	0,                  /* 58 */
	0,                  /* 59 */
	0,                  /* 60 */
	0,                  /* 61 */
	0,                  /* 62 */
	0,                  /* 63 */
	0,                  /* 64 */
	cVHC,               /* 65 */
	cVHC,               /* 66 */
	cVHC,               /* 67 */
	cVHC,               /* 68 */
	cVHC,               /* 69 */
	cVHC,               /* 70 */
	0,                  /* 71 */
	0,                  /* 72 */
	0,                  /* 73 */
	0,                  /* 74 */
	0,                  /* 75 */
	0,                  /* 76 */
	0,                  /* 77 */
	0,                  /* 78 */
	0,                  /* 79 */
	0,                  /* 80 */
	0,                  /* 81 */
	0,                  /* 82 */
	0,                  /* 83 */
	0,                  /* 84 */
	0,                  /* 85 */
	0,                  /* 86 */
	0,                  /* 87 */
	0,                  /* 88 */
	0,                  /* 89 */
	0,                  /* 90 */
	0,                  /* 91 */
	cVEC | cIJC | cNFP, /* 92 */
	0,                  /* 93 */
	0,                  /* 94 */
	0,                  /* 95 */
	0,                  /* 96 */
	cVHC,               /* 97 */
	cVEC | cVHC,        /* 98 */
	cVHC,               /* 99 */
	cVHC,               /* 100 */
	cVHC,               /* 101 */
	cVEC | cVHC,        /* 102 */
	0,                  /* 103 */
	0,                  /* 104 */
	0,                  /* 105 */
	0,                  /* 106 */
	0,                  /* 107 */
	0,                  /* 108 */
	0,                  /* 109 */
	cVEC,               /* 110 */
	0,                  /* 111 */
	0,                  /* 112 */
	0,                  /* 113 */
	cVEC,               /* 114 */
	0,                  /* 115 */
	cVEC,               /* 116 */
	0,                  /* 117 */
	0,                  /* 118 */
	0,                  /* 119 */
	0,                  /* 120 */
	0,                  /* 121 */
	0,                  /* 122 */
	0,                  /* 123 */
	0,                  /* 124 */
	0,                  /* 125 */
	0,                  /* 126 */
	0,                  /* 127 */
	cNUC,               /* 128 */
	cNUC,               /* 129 */
	cNUC,               /* 130 */
	cNUC,               /* 131 */
	cNUC,               /* 132 */
	cNUC,               /* 133 */
	cNUC,               /* 134 */
	cNUC,               /* 135 */
	cNUC,               /* 136 */
	cNUC,               /* 137 */
	cNUC,               /* 138 */
	cNUC,               /* 139 */
	cNUC,               /* 140 */
	cNUC,               /* 141 */
	cNUC,               /* 142 */
	cNUC,               /* 143 */
	cNUC,               /* 144 */
	cNUC,               /* 145 */
	cNUC,               /* 146 */
	cNUC,               /* 147 */
	cNUC,               /* 148 */
	cNUC,               /* 149 */
	cNUC,               /* 150 */
	cNUC,               /* 151 */
	cNUC,               /* 152 */
	cNUC,               /* 153 */
	cNUC,               /* 154 */
	cNUC,               /* 155 */
	cNUC,               /* 156 */
	cNUC,               /* 157 */
	cNUC,               /* 158 */
	cNUC,               /* 159 */
	cNUC,               /* 160 */
	cNUC,               /* 161 */
	cNUC,               /* 162 */
	cNUC,               /* 163 */
	cNUC,               /* 164 */
	cNUC,               /* 165 */
	cNUC,               /* 166 */
	cNUC,               /* 167 */
	cNUC,               /* 168 */
	cNUC,               /* 169 */
	cNUC,               /* 170 */
	cNUC,               /* 171 */
	cNUC,               /* 172 */
	cNUC,               /* 173 */
	cNUC,               /* 174 */
	cNUC,               /* 175 */
	cNUC,               /* 176 */
	cNUC,               /* 177 */
	cNUC,               /* 178 */
	cNUC,               /* 179 */
	cNUC,               /* 180 */
	cNUC,               /* 181 */
	cNUC,               /* 182 */
	cNUC,               /* 183 */
	cNUC,               /* 184 */
	cNUC,               /* 185 */
	cNUC,               /* 186 */
	cNUC,               /* 187 */
	cNUC,               /* 188 */
	cNUC,               /* 189 */
	cNUC,               /* 190 */
	cNUC,               /* 191 */
	cNUC,               /* 192 */
	cNUC,               /* 193 */
	cNUC,               /* 194 */
	cNUC,               /* 195 */
	cNUC,               /* 196 */
	cNUC,               /* 197 */
	cNUC,               /* 198 */
	cNUC,               /* 199 */
	cNUC,               /* 200 */
	cNUC,               /* 201 */
	cNUC,               /* 202 */
	cNUC,               /* 203 */
	cNUC,               /* 204 */
	cNUC,               /* 205 */
	cNUC,               /* 206 */
	cNUC,               /* 207 */
	cNUC,               /* 208 */
	cNUC,               /* 209 */
	cNUC,               /* 210 */
	cNUC,               /* 211 */
	cNUC,               /* 212 */
	cNUC,               /* 213 */
	cNUC,               /* 214 */
	cNUC,               /* 215 */
	cNUC,               /* 216 */
	cNUC,               /* 217 */
	cNUC,               /* 218 */
	cNUC,               /* 219 */
	cNUC,               /* 220 */
	cNUC,               /* 221 */
	cNUC,               /* 222 */
	cNUC,               /* 223 */
	cNUC,               /* 224 */
	cNUC,               /* 225 */
	cNUC,               /* 226 */
	cNUC,               /* 227 */
	cNUC,               /* 228 */
	cNUC,               /* 229 */
	cNUC,               /* 230 */
	cNUC,               /* 231 */
	cNUC,               /* 232 */
	cNUC,               /* 233 */
	cNUC,               /* 234 */
	cNUC,               /* 235 */
	cNUC,               /* 236 */
	cNUC,               /* 237 */
	cNUC,               /* 238 */
	cNUC,               /* 239 */
	cNUC,               /* 240 */
	cNUC,               /* 241 */
	cNUC,               /* 242 */
	cNUC,               /* 243 */
	cNUC,               /* 244 */
	cNUC,               /* 245 */
	cNUC,               /* 246 */
	cNUC,               /* 247 */
	cNUC,               /* 248 */
	cNUC,               /* 249 */
	cNUC,               /* 250 */
	cNUC,               /* 251 */
	cNUC,               /* 252 */
	cNUC,               /* 253 */
	cNUC,               /* 254 */
	cNUC,               /* 255 */
}
