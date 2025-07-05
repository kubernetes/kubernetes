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
	"iter"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
)

func defaultPort(scheme string) string {
	switch scheme {
	case "http":
		return "80"
	case "https":
		return "443"
	default:
		return ""
	}
}

// sameOrigin checks if two URIs have the same origin (scheme, host, port).
func sameOrigin(a, b *url.URL) bool {
	aPort := a.Port()
	if aPort == "" {
		aPort = defaultPort(a.Scheme)
	}
	bPort := b.Port()
	if bPort == "" {
		bPort = defaultPort(b.Scheme)
	}
	return strings.EqualFold(a.Scheme, b.Scheme) &&
		strings.EqualFold(a.Hostname(), b.Hostname()) &&
		aPort == bPort
}

// SetAgeHeader sets the Age header in the response based on the Age value.
// It assumes a non-nil Age pointer is provided.
func SetAgeHeader(resp *http.Response, clock Clock, age *Age) {
	adjusted := max(age.Value+clock.Since(age.Timestamp), 0)
	resp.Header.Set("Age", strconv.Itoa(int(adjusted.Seconds())))
}

// hopByHopHeaders returns a map of hop-by-hop headers that should be removed
// from the response before caching or forwarding it (RFC 9111 §3.1).
func hopByHopHeaders(respHeader http.Header) map[string]struct{} {
	m := map[string]struct{}{
		// As per RFC 9111 §7.6.1 (https://www.rfc-editor.org/rfc/rfc9110#section-7.6.1)
		"Connection":        {},
		"Proxy-Connection":  {},
		"Keep-Alive":        {},
		"TE":                {},
		"Transfer-Encoding": {},
		"Upgrade":           {},
		// RFC 9111 §3.1 proxy headers
		"Proxy-Authenticate":        {},
		"Proxy-Authentication-Info": {},
		"Proxy-Authorization":       {},
		// Also see net/http/response.go "respExcludeHeader" for additional excluded headers.
	}
	// Fields listed in the Connection header field
	for field := range TrimmedCSVCanonicalSeq(respHeader.Get("Connection")) {
		m[field] = struct{}{}
	}
	return m
}

// removeHopByHopHeaders removes hop-by-hop headers so that only
// end-to-end headers remain in the response as per RFC 9111 §3.1.
func removeHopByHopHeaders(resp *http.Response) {
	for hdr := range hopByHopHeaders(resp.Header) {
		delete(resp.Header, hdr)
	}
}

// updateStoredHeaders updates the stored response headers with the
// headers from the revalidated response, excluding hop-by-hop headers
// and the Content-Length header, as per RFC 9111 §3.2.
func updateStoredHeaders(storedResp, resp *http.Response) {
	omitted := hopByHopHeaders(resp.Header)
	omitted["Content-Length"] = struct{}{}
	for hdr, val := range resp.Header {
		if _, ok := omitted[hdr]; ok {
			continue
		}
		storedResp.Header[hdr] = val
	}
}

// isStaleErrorAllowed reports whether the cache is allowed to serve a stale response
// for the given error status code, according to RFC5861 §4.
func isStaleErrorAllowed(code int) bool {
	switch code {
	case http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

// From Go's net/url package.
// Copyright 2009 The Go Authors. All rights reserved.
//
// splitHostPort separates host and port. If the port is not valid, it returns
// the entire input as host, and it doesn't check the validity of the host.
// Unlike net.SplitHostPort, but per RFC 3986, it requires ports to be numeric.
func splitHostPort(hostPort string) (host, port string) {
	host = hostPort
	colon := strings.LastIndexByte(host, ':')
	if colon != -1 && validOptionalPort(host[colon:]) {
		host, port = host[:colon], host[colon+1:]
	}
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = host[1 : len(host)-1]
	}
	return
}

// From Go's net/url package.
// Copyright 2009 The Go Authors. All rights reserved.
//
// validOptionalPort reports whether port is either an empty string or matches "/^:\d*$/".
func validOptionalPort(port string) bool {
	if port == "" {
		return true
	}
	if port[0] != ':' {
		return false
	}
	for _, b := range port[1:] {
		if b < '0' || b > '9' {
			return false
		}
	}
	return true
}

func IsUnsafeMethod(method string) bool {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch:
		return true
	default:
		return false
	}
}

func IsNonErrorStatus(status int) bool {
	return (status >= 200 && status < 400)
}

// TrimmedCSVSeq returns an iterator over the raw comma-separated string.
// It yields each part of the string, trimmed of whitespace, and does not split inside quoted strings.
func TrimmedCSVSeq(s string) iter.Seq[string] {
	return func(yield func(string) bool) {
		var part strings.Builder
		inQuotes := false
		escape := false
		for i := range len(s) {
			c := s[i]
			switch {
			case escape:
				part.WriteByte(c)
				escape = false
			case c == '\\':
				part.WriteByte(c)
				escape = true
			case c == '"':
				part.WriteByte(c)
				inQuotes = !inQuotes
			case c == ',' && !inQuotes:
				p := textproto.TrimString(part.String())
				if len(p) > 0 {
					if !yield(p) {
						return
					}
				}
				part.Reset()
			default:
				part.WriteByte(c)
			}
		}
		if part.Len() > 0 {
			p := textproto.TrimString(part.String())
			if len(p) > 0 {
				_ = yield(p)
			}
		}
	}
}

// TrimmedCSVCanonicalSeq is the same as [TrimmedCSVSeq], but it yields each part
// in canonical form.
func TrimmedCSVCanonicalSeq(s string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for part := range TrimmedCSVSeq(s) {
			if !yield(http.CanonicalHeaderKey(part)) {
				return
			}
		}
	}
}
