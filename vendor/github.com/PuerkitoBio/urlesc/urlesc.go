// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package urlesc implements query escaping as per RFC 3986.
// It contains some parts of the net/url package, modified so as to allow
// some reserved characters incorrectly escaped by net/url.
// See https://github.com/golang/go/issues/5684
package urlesc

import (
	"bytes"
	"net/url"
	"strings"
)

type encoding int

const (
	encodePath encoding = 1 + iota
	encodeUserPassword
	encodeQueryComponent
	encodeFragment
)

// Return true if the specified character should be escaped when
// appearing in a URL string, according to RFC 3986.
func shouldEscape(c byte, mode encoding) bool {
	// §2.3 Unreserved characters (alphanum)
	if 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' {
		return false
	}

	switch c {
	case '-', '.', '_', '~': // §2.3 Unreserved characters (mark)
		return false

	// §2.2 Reserved characters (reserved)
	case ':', '/', '?', '#', '[', ']', '@', // gen-delims
		'!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=': // sub-delims
		// Different sections of the URL allow a few of
		// the reserved characters to appear unescaped.
		switch mode {
		case encodePath: // §3.3
			// The RFC allows sub-delims and : @.
			// '/', '[' and ']' can be used to assign meaning to individual path
			// segments.  This package only manipulates the path as a whole,
			// so we allow those as well.  That leaves only ? and # to escape.
			return c == '?' || c == '#'

		case encodeUserPassword: // §3.2.1
			// The RFC allows : and sub-delims in
			// userinfo.  The parsing of userinfo treats ':' as special so we must escape
			// all the gen-delims.
			return c == ':' || c == '/' || c == '?' || c == '#' || c == '[' || c == ']' || c == '@'

		case encodeQueryComponent: // §3.4
			// The RFC allows / and ?.
			return c != '/' && c != '?'

		case encodeFragment: // §4.1
			// The RFC text is silent but the grammar allows
			// everything, so escape nothing but #
			return c == '#'
		}
	}

	// Everything else must be escaped.
	return true
}

// QueryEscape escapes the string so it can be safely placed
// inside a URL query.
func QueryEscape(s string) string {
	return escape(s, encodeQueryComponent)
}

func escape(s string, mode encoding) string {
	spaceCount, hexCount := 0, 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if shouldEscape(c, mode) {
			if c == ' ' && mode == encodeQueryComponent {
				spaceCount++
			} else {
				hexCount++
			}
		}
	}

	if spaceCount == 0 && hexCount == 0 {
		return s
	}

	t := make([]byte, len(s)+2*hexCount)
	j := 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == ' ' && mode == encodeQueryComponent:
			t[j] = '+'
			j++
		case shouldEscape(c, mode):
			t[j] = '%'
			t[j+1] = "0123456789ABCDEF"[c>>4]
			t[j+2] = "0123456789ABCDEF"[c&15]
			j += 3
		default:
			t[j] = s[i]
			j++
		}
	}
	return string(t)
}

var uiReplacer = strings.NewReplacer(
	"%21", "!",
	"%27", "'",
	"%28", "(",
	"%29", ")",
	"%2A", "*",
)

// unescapeUserinfo unescapes some characters that need not to be escaped as per RFC3986.
func unescapeUserinfo(s string) string {
	return uiReplacer.Replace(s)
}

// Escape reassembles the URL into a valid URL string.
// The general form of the result is one of:
//
//	scheme:opaque
//	scheme://userinfo@host/path?query#fragment
//
// If u.Opaque is non-empty, String uses the first form;
// otherwise it uses the second form.
//
// In the second form, the following rules apply:
//	- if u.Scheme is empty, scheme: is omitted.
//	- if u.User is nil, userinfo@ is omitted.
//	- if u.Host is empty, host/ is omitted.
//	- if u.Scheme and u.Host are empty and u.User is nil,
//	   the entire scheme://userinfo@host/ is omitted.
//	- if u.Host is non-empty and u.Path begins with a /,
//	   the form host/path does not add its own /.
//	- if u.RawQuery is empty, ?query is omitted.
//	- if u.Fragment is empty, #fragment is omitted.
func Escape(u *url.URL) string {
	var buf bytes.Buffer
	if u.Scheme != "" {
		buf.WriteString(u.Scheme)
		buf.WriteByte(':')
	}
	if u.Opaque != "" {
		buf.WriteString(u.Opaque)
	} else {
		if u.Scheme != "" || u.Host != "" || u.User != nil {
			buf.WriteString("//")
			if ui := u.User; ui != nil {
				buf.WriteString(unescapeUserinfo(ui.String()))
				buf.WriteByte('@')
			}
			if h := u.Host; h != "" {
				buf.WriteString(h)
			}
		}
		if u.Path != "" && u.Path[0] != '/' && u.Host != "" {
			buf.WriteByte('/')
		}
		buf.WriteString(escape(u.Path, encodePath))
	}
	if u.RawQuery != "" {
		buf.WriteByte('?')
		buf.WriteString(u.RawQuery)
	}
	if u.Fragment != "" {
		buf.WriteByte('#')
		buf.WriteString(escape(u.Fragment, encodeFragment))
	}
	return buf.String()
}
