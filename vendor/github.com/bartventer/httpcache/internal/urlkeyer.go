// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/url"
	"strings"
	"unicode"
)

// URLKeyer describes the interface implemented by types that can generate a
// normalized cache key from a URL, following rules specified in RFC 3986 §6.
type URLKeyer interface {
	URLKey(u *url.URL) string
}

type URLKeyerFunc func(u *url.URL) string

func (f URLKeyerFunc) URLKey(u *url.URL) string {
	return f(u)
}

func NewURLKeyer() URLKeyer { return URLKeyerFunc(makeURLKey) }

// makeURLKey returns a normalized URL string suitable for use as a cache key.
// This helps ensure that URLs which are semantically the same but differ in minor ways
// (such as case, percent-encoding, or default ports) will map to the same cache entry.
//
// Reasons for normalization:
//   - Go's http.Request.URL is parsed but not fully normalized.
//   - URLs may use uppercase percent-encodings (%7E vs %7e).
//   - Default ports (:80 for HTTP, :443 for HTTPS) may be included unnecessarily.
//   - Hostnames may use different case (EXAMPLE.com).
//   - Paths may be empty.
//   - Dot-segments like /a/./b/../c may appear in paths.
//   - The Go standard library does not guarantee normalization for all these cases.
//
// References:
//   - RFC 3986 §6.2: https://datatracker.ietf.org/doc/html/rfc3986#section-6.2
//   - RFC 7230 §2.7.3: https://datatracker.ietf.org/doc/html/rfc7230#section-2.7.3
func makeURLKey(u *url.URL) string {
	if u.Opaque != "" {
		return u.Opaque
	}
	// RFC 3986 §6.2.2.3: Path normalization (dot-segment removal) is handled by
	// [url.URL.ResolveReference], which uses the RFC 3986 §5.2.4 algorithm.
	base, _ := url.Parse(u.Scheme + "://" + u.Host)
	normalized := base.ResolveReference(u)

	// RFC 3986 §6.2.2.1: Scheme is lowercased (already done by [url.Parse]).
	scheme := normalized.Scheme

	host, port := splitHostPort(normalized.Host)
	defaultP := defaultPort(scheme)
	if port == "" {
		port = defaultP
	}
	// RFC 3986 §6.2.2.1: Host is lowercased.
	hostPort := strings.ToLower(host)

	// RFC 3986 §6.2.3: Only include port if it is non-default for the scheme.
	if port != "" && port != defaultP {
		hostPort = hostPort + ":" + port
	}

	// RFC 3986 §6.2.3: An empty path for http/https is normalized to "/".
	// Also see https://datatracker.ietf.org/doc/html/rfc7230#section-2.7.3
	path := normalized.EscapedPath()
	if path == "" && (scheme == "http" || scheme == "https") {
		path = "/"
	}

	// RFC 3986 §6.2.2.2: Normalize percent-encoding in path.
	path = normalizePercentEncoding(path)
	result := scheme + "://" + hostPort + path

	// RFC 3986 §6.2.2.2: Normalize percent-encoding in query, if present.
	if normalized.RawQuery != "" {
		result += "?" + normalizePercentEncoding(normalized.RawQuery)
	}

	// RFC 3986 §6.1 Equivalence: "fragment components (if any) should be excluded from
	// the comparison"
	return result
}

// normalizePercentEncoding rewrites percent-encoded characters in a URL path or query
// so that unreserved characters are decoded, and all hex digits are uppercase.
// Follows RFC 3986 §6.2.2.2.
func normalizePercentEncoding(s string) string {
	var b strings.Builder
	i := 0
	for i < len(s) {
		if s[i] == '%' && i+2 < len(s) &&
			isHexDigit(s[i+1]) && isHexDigit(s[i+2]) {
			hexVal := fromHex(s[i+1])<<4 | fromHex(s[i+2])
			r := rune(hexVal)
			if isUnreserved(r) {
				b.WriteRune(r)
			} else {
				b.WriteString(percentEncodeUpper(hexVal))
			}
			i += 3
		} else {
			b.WriteByte(s[i])
			i++
		}
	}
	return b.String()
}

func isHexDigit(c byte) bool {
	return ('0' <= c && c <= '9') ||
		('A' <= c && c <= 'F') ||
		('a' <= c && c <= 'f')
}

func fromHex(c byte) byte {
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

// isUnreserved reports whether r is an unreserved character per RFC 3986 §2.3.
func isUnreserved(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) ||
		r == '-' || r == '.' || r == '_' || r == '~'
}

const hex = "0123456789ABCDEF"

// percentEncodeUpper returns the percent-encoded form of b using uppercase
// hex digits as specified in RFC 3986 §2.1.
func percentEncodeUpper(b byte) string {
	return "%" + string(hex[b>>4]) + string(hex[b&0x0F])
}
