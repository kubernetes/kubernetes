// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cacheobject

// This file deals with lexical matters of HTTP

func isSeparator(c byte) bool {
	switch c {
	case '(', ')', '<', '>', '@', ',', ';', ':', '\\', '"', '/', '[', ']', '?', '=', '{', '}', ' ', '\t':
		return true
	}
	return false
}

func isCtl(c byte) bool { return (0 <= c && c <= 31) || c == 127 }

func isChar(c byte) bool { return 0 <= c && c <= 127 }

func isAnyText(c byte) bool { return !isCtl(c) }

func isQdText(c byte) bool { return isAnyText(c) && c != '"' }

func isToken(c byte) bool { return isChar(c) && !isCtl(c) && !isSeparator(c) }

// Valid escaped sequences are not specified in RFC 2616, so for now, we assume
// that they coincide with the common sense ones used by GO. Malformed
// characters should probably not be treated as errors by a robust (forgiving)
// parser, so we replace them with the '?' character.
func httpUnquotePair(b byte) byte {
	// skip the first byte, which should always be '\'
	switch b {
	case 'a':
		return '\a'
	case 'b':
		return '\b'
	case 'f':
		return '\f'
	case 'n':
		return '\n'
	case 'r':
		return '\r'
	case 't':
		return '\t'
	case 'v':
		return '\v'
	case '\\':
		return '\\'
	case '\'':
		return '\''
	case '"':
		return '"'
	}
	return '?'
}

// raw must begin with a valid quoted string. Only the first quoted string is
// parsed and is unquoted in result. eaten is the number of bytes parsed, or -1
// upon failure.
func httpUnquote(raw string) (eaten int, result string) {
	buf := make([]byte, len(raw))
	if raw[0] != '"' {
		return -1, ""
	}
	eaten = 1
	j := 0 // # of bytes written in buf
	for i := 1; i < len(raw); i++ {
		switch b := raw[i]; b {
		case '"':
			eaten++
			buf = buf[0:j]
			return i + 1, string(buf)
		case '\\':
			if len(raw) < i+2 {
				return -1, ""
			}
			buf[j] = httpUnquotePair(raw[i+1])
			eaten += 2
			j++
			i++
		default:
			if isQdText(b) {
				buf[j] = b
			} else {
				buf[j] = '?'
			}
			eaten++
			j++
		}
	}
	return -1, ""
}
