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

/* Portions of this file are on Go stdlib's encoding/json/encode.go */
// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1

import (
	"io"
	"unicode/utf8"
	"strconv"
	"unicode/utf16"
	"unicode"
)

const hex = "0123456789abcdef"

type JsonStringWriter interface {
	io.Writer
	io.ByteWriter
	stringWriter
}

func WriteJsonString(buf JsonStringWriter, s string) {
	WriteJson(buf, []byte(s))
}

/**
 * Function ported from encoding/json: func (e *encodeState) string(s string) (int, error)
 */
func WriteJson(buf JsonStringWriter, s []byte) {
	buf.WriteByte('"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			/*
				if 0x20 <= b && b != '\\' && b != '"' && b != '<' && b != '>' && b != '&' {
					i++
					continue
				}
			*/
			if lt[b] == true {
				i++
				continue
			}

			if start < i {
				buf.Write(s[start:i])
			}
			switch b {
			case '\\', '"':
				buf.WriteByte('\\')
				buf.WriteByte(b)
			case '\n':
				buf.WriteByte('\\')
				buf.WriteByte('n')
			case '\r':
				buf.WriteByte('\\')
				buf.WriteByte('r')
			default:
				// This encodes bytes < 0x20 except for \n and \r,
				// as well as < and >. The latter are escaped because they
				// can lead to security holes when user-controlled strings
				// are rendered into JSON and served to some browsers.
				buf.WriteString(`\u00`)
				buf.WriteByte(hex[b>>4])
				buf.WriteByte(hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRune(s[i:])
		if c == utf8.RuneError && size == 1 {
			if start < i {
				buf.Write(s[start:i])
			}
			buf.WriteString(`\ufffd`)
			i += size
			start = i
			continue
		}
		// U+2028 is LINE SEPARATOR.
		// U+2029 is PARAGRAPH SEPARATOR.
		// They are both technically valid characters in JSON strings,
		// but don't work in JSONP, which has to be evaluated as JavaScript,
		// and can lead to security holes there. It is valid JSON to
		// escape them, so we do so unconditionally.
		// See http://timelessrepo.com/json-isnt-a-javascript-subset for discussion.
		if c == '\u2028' || c == '\u2029' {
			if start < i {
				buf.Write(s[start:i])
			}
			buf.WriteString(`\u202`)
			buf.WriteByte(hex[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		buf.Write(s[start:])
	}
	buf.WriteByte('"')
}

// UnquoteBytes will decode []byte containing json string to go string
// ported from encoding/json/decode.go
func UnquoteBytes(s []byte) (t []byte, ok bool) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return
	}
	s = s[1 : len(s)-1]

	// Check for unusual characters. If there are none,
	// then no unquoting is needed, so return a slice of the
	// original bytes.
	r := 0
	for r < len(s) {
		c := s[r]
		if c == '\\' || c == '"' || c < ' ' {
			break
		}
		if c < utf8.RuneSelf {
			r++
			continue
		}
		rr, size := utf8.DecodeRune(s[r:])
		if rr == utf8.RuneError && size == 1 {
			break
		}
		r += size
	}
	if r == len(s) {
		return s, true
	}

	b := make([]byte, len(s)+2*utf8.UTFMax)
	w := copy(b, s[0:r])
	for r < len(s) {
		// Out of room?  Can only happen if s is full of
		// malformed UTF-8 and we're replacing each
		// byte with RuneError.
		if w >= len(b)-2*utf8.UTFMax {
			nb := make([]byte, (len(b)+utf8.UTFMax)*2)
			copy(nb, b[0:w])
			b = nb
		}
		switch c := s[r]; {
		case c == '\\':
			r++
			if r >= len(s) {
				return
			}
			switch s[r] {
			default:
				return
			case '"', '\\', '/', '\'':
				b[w] = s[r]
				r++
				w++
			case 'b':
				b[w] = '\b'
				r++
				w++
			case 'f':
				b[w] = '\f'
				r++
				w++
			case 'n':
				b[w] = '\n'
				r++
				w++
			case 'r':
				b[w] = '\r'
				r++
				w++
			case 't':
				b[w] = '\t'
				r++
				w++
			case 'u':
				r--
				rr := getu4(s[r:])
				if rr < 0 {
					return
				}
				r += 6
				if utf16.IsSurrogate(rr) {
					rr1 := getu4(s[r:])
					if dec := utf16.DecodeRune(rr, rr1); dec != unicode.ReplacementChar {
						// A valid pair; consume.
						r += 6
						w += utf8.EncodeRune(b[w:], dec)
						break
					}
					// Invalid surrogate; fall back to replacement rune.
					rr = unicode.ReplacementChar
				}
				w += utf8.EncodeRune(b[w:], rr)
			}

		// Quote, control characters are invalid.
		case c == '"', c < ' ':
			return

		// ASCII
		case c < utf8.RuneSelf:
			b[w] = c
			r++
			w++

		// Coerce to well-formed UTF-8.
		default:
			rr, size := utf8.DecodeRune(s[r:])
			r += size
			w += utf8.EncodeRune(b[w:], rr)
		}
	}
	return b[0:w], true
}

// getu4 decodes \uXXXX from the beginning of s, returning the hex value,
// or it returns -1.
func getu4(s []byte) rune {
	if len(s) < 6 || s[0] != '\\' || s[1] != 'u' {
		return -1
	}
	r, err := strconv.ParseUint(string(s[2:6]), 16, 64)
	if err != nil {
		return -1
	}
	return rune(r)
}

// TODO(pquerna): consider combining wibth the normal byte mask.
var lt [256]bool = [256]bool{
	false, /* 0 */
	false, /* 1 */
	false, /* 2 */
	false, /* 3 */
	false, /* 4 */
	false, /* 5 */
	false, /* 6 */
	false, /* 7 */
	false, /* 8 */
	false, /* 9 */
	false, /* 10 */
	false, /* 11 */
	false, /* 12 */
	false, /* 13 */
	false, /* 14 */
	false, /* 15 */
	false, /* 16 */
	false, /* 17 */
	false, /* 18 */
	false, /* 19 */
	false, /* 20 */
	false, /* 21 */
	false, /* 22 */
	false, /* 23 */
	false, /* 24 */
	false, /* 25 */
	false, /* 26 */
	false, /* 27 */
	false, /* 28 */
	false, /* 29 */
	false, /* 30 */
	false, /* 31 */
	true,  /* 32 */
	true,  /* 33 */
	false, /* 34 */
	true,  /* 35 */
	true,  /* 36 */
	true,  /* 37 */
	false, /* 38 */
	true,  /* 39 */
	true,  /* 40 */
	true,  /* 41 */
	true,  /* 42 */
	true,  /* 43 */
	true,  /* 44 */
	true,  /* 45 */
	true,  /* 46 */
	true,  /* 47 */
	true,  /* 48 */
	true,  /* 49 */
	true,  /* 50 */
	true,  /* 51 */
	true,  /* 52 */
	true,  /* 53 */
	true,  /* 54 */
	true,  /* 55 */
	true,  /* 56 */
	true,  /* 57 */
	true,  /* 58 */
	true,  /* 59 */
	false, /* 60 */
	true,  /* 61 */
	false, /* 62 */
	true,  /* 63 */
	true,  /* 64 */
	true,  /* 65 */
	true,  /* 66 */
	true,  /* 67 */
	true,  /* 68 */
	true,  /* 69 */
	true,  /* 70 */
	true,  /* 71 */
	true,  /* 72 */
	true,  /* 73 */
	true,  /* 74 */
	true,  /* 75 */
	true,  /* 76 */
	true,  /* 77 */
	true,  /* 78 */
	true,  /* 79 */
	true,  /* 80 */
	true,  /* 81 */
	true,  /* 82 */
	true,  /* 83 */
	true,  /* 84 */
	true,  /* 85 */
	true,  /* 86 */
	true,  /* 87 */
	true,  /* 88 */
	true,  /* 89 */
	true,  /* 90 */
	true,  /* 91 */
	false, /* 92 */
	true,  /* 93 */
	true,  /* 94 */
	true,  /* 95 */
	true,  /* 96 */
	true,  /* 97 */
	true,  /* 98 */
	true,  /* 99 */
	true,  /* 100 */
	true,  /* 101 */
	true,  /* 102 */
	true,  /* 103 */
	true,  /* 104 */
	true,  /* 105 */
	true,  /* 106 */
	true,  /* 107 */
	true,  /* 108 */
	true,  /* 109 */
	true,  /* 110 */
	true,  /* 111 */
	true,  /* 112 */
	true,  /* 113 */
	true,  /* 114 */
	true,  /* 115 */
	true,  /* 116 */
	true,  /* 117 */
	true,  /* 118 */
	true,  /* 119 */
	true,  /* 120 */
	true,  /* 121 */
	true,  /* 122 */
	true,  /* 123 */
	true,  /* 124 */
	true,  /* 125 */
	true,  /* 126 */
	true,  /* 127 */
	true,  /* 128 */
	true,  /* 129 */
	true,  /* 130 */
	true,  /* 131 */
	true,  /* 132 */
	true,  /* 133 */
	true,  /* 134 */
	true,  /* 135 */
	true,  /* 136 */
	true,  /* 137 */
	true,  /* 138 */
	true,  /* 139 */
	true,  /* 140 */
	true,  /* 141 */
	true,  /* 142 */
	true,  /* 143 */
	true,  /* 144 */
	true,  /* 145 */
	true,  /* 146 */
	true,  /* 147 */
	true,  /* 148 */
	true,  /* 149 */
	true,  /* 150 */
	true,  /* 151 */
	true,  /* 152 */
	true,  /* 153 */
	true,  /* 154 */
	true,  /* 155 */
	true,  /* 156 */
	true,  /* 157 */
	true,  /* 158 */
	true,  /* 159 */
	true,  /* 160 */
	true,  /* 161 */
	true,  /* 162 */
	true,  /* 163 */
	true,  /* 164 */
	true,  /* 165 */
	true,  /* 166 */
	true,  /* 167 */
	true,  /* 168 */
	true,  /* 169 */
	true,  /* 170 */
	true,  /* 171 */
	true,  /* 172 */
	true,  /* 173 */
	true,  /* 174 */
	true,  /* 175 */
	true,  /* 176 */
	true,  /* 177 */
	true,  /* 178 */
	true,  /* 179 */
	true,  /* 180 */
	true,  /* 181 */
	true,  /* 182 */
	true,  /* 183 */
	true,  /* 184 */
	true,  /* 185 */
	true,  /* 186 */
	true,  /* 187 */
	true,  /* 188 */
	true,  /* 189 */
	true,  /* 190 */
	true,  /* 191 */
	true,  /* 192 */
	true,  /* 193 */
	true,  /* 194 */
	true,  /* 195 */
	true,  /* 196 */
	true,  /* 197 */
	true,  /* 198 */
	true,  /* 199 */
	true,  /* 200 */
	true,  /* 201 */
	true,  /* 202 */
	true,  /* 203 */
	true,  /* 204 */
	true,  /* 205 */
	true,  /* 206 */
	true,  /* 207 */
	true,  /* 208 */
	true,  /* 209 */
	true,  /* 210 */
	true,  /* 211 */
	true,  /* 212 */
	true,  /* 213 */
	true,  /* 214 */
	true,  /* 215 */
	true,  /* 216 */
	true,  /* 217 */
	true,  /* 218 */
	true,  /* 219 */
	true,  /* 220 */
	true,  /* 221 */
	true,  /* 222 */
	true,  /* 223 */
	true,  /* 224 */
	true,  /* 225 */
	true,  /* 226 */
	true,  /* 227 */
	true,  /* 228 */
	true,  /* 229 */
	true,  /* 230 */
	true,  /* 231 */
	true,  /* 232 */
	true,  /* 233 */
	true,  /* 234 */
	true,  /* 235 */
	true,  /* 236 */
	true,  /* 237 */
	true,  /* 238 */
	true,  /* 239 */
	true,  /* 240 */
	true,  /* 241 */
	true,  /* 242 */
	true,  /* 243 */
	true,  /* 244 */
	true,  /* 245 */
	true,  /* 246 */
	true,  /* 247 */
	true,  /* 248 */
	true,  /* 249 */
	true,  /* 250 */
	true,  /* 251 */
	true,  /* 252 */
	true,  /* 253 */
	true,  /* 254 */
	true,  /* 255 */
}
