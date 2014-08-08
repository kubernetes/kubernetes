package util

import (
	"net/http"
	"net/url"
	"strings"
)

// RawURL is a wrapper http.Handler for setting URL.Path to the unescaped
// version of the URI to allow clients to handle encoded slashes.
func RawURL(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s := r.RequestURI
		u, _ := split(s, "#", true)
		rest, _ := split(u, "?", true)
		r.URL.Path = rest
		h.ServeHTTP(w, r)
	})
}

// SplitRawPath is a helper method for breaking an escaped URL path into
// segments according to RFC3986, which Go 1.X does not correctly handle
// (see https://code.google.com/p/go/issues/detail?id=3659).  This allows
// URL handlers to properly handle encoded path segments which might be
// part of an escaped URL path like:
//
//    /api/resource/encoded%2Fpath%2Fsegment
//
// This method will return "api", "resource", and "encoded/path/segment".
func SplitRawPath(path string) ([]string, error) {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}, nil
	}
	parts := strings.Split(path, "/")
	for i := range parts {
		part, err := unescape(parts[i], encodePath)
		if err != nil {
			return nil, err
		}
		parts[i] = part
	}
	return parts, nil
}

// The following code is copied from Go 1.2.2 directly - no modifications have been made.
// These methods are private in net/url and allow Go compatible splitting and unescaping
// of urls.

// Maybe s is of the form t c u.
// If so, return t, c u (or t, u if cutc == true).
// If not, return s, "".
// copied from net/url/url.go#split
func split(s string, c string, cutc bool) (string, string) {
	i := strings.Index(s, c)
	if i < 0 {
		return s, ""
	}
	if cutc {
		return s[0:i], s[i+len(c):]
	}
	return s[0:i], s[i:]
}

// copied from net/url/url.go
type encoding int

// copied from net/url/url.go
const (
	encodePath encoding = 1 + iota
	encodeUserPassword
	encodeQueryComponent
	encodeFragment
)

// copied from net/url/url.go#unhex
func ishex(c byte) bool {
	switch {
	case '0' <= c && c <= '9':
		return true
	case 'a' <= c && c <= 'f':
		return true
	case 'A' <= c && c <= 'F':
		return true
	}
	return false
}

// copied from net/url/url.go#unhex
func unhex(c byte) byte {
	switch {
	case '0' <= c && c <= '9':
		return c - '0'
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10
	}
	return 0
}

// unescape unescapes a string; the mode specifies
// which section of the URL string is being unescaped.
// copied from net/url/url.go#unescape
func unescape(s string, mode encoding) (string, error) {
	// Count %, check that they're well-formed.
	n := 0
	hasPlus := false
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			n++
			if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
				s = s[i:]
				if len(s) > 3 {
					s = s[0:3]
				}
				return "", url.EscapeError(s)
			}
			i += 3
		case '+':
			hasPlus = mode == encodeQueryComponent
			i++
		default:
			i++
		}
	}

	if n == 0 && !hasPlus {
		return s, nil
	}

	t := make([]byte, len(s)-2*n)
	j := 0
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			t[j] = unhex(s[i+1])<<4 | unhex(s[i+2])
			j++
			i += 3
		case '+':
			if mode == encodeQueryComponent {
				t[j] = ' '
			} else {
				t[j] = '+'
			}
			j++
			i++
		default:
			t[j] = s[i]
			j++
			i++
		}
	}
	return string(t), nil
}
