// Copyright 2016 The CMux Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

package cmux

import (
	"bufio"
	"crypto/tls"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

// Any is a Matcher that matches any connection.
func Any() Matcher {
	return func(r io.Reader) bool { return true }
}

// PrefixMatcher returns a matcher that matches a connection if it
// starts with any of the strings in strs.
func PrefixMatcher(strs ...string) Matcher {
	pt := newPatriciaTreeString(strs...)
	return pt.matchPrefix
}

func prefixByteMatcher(list ...[]byte) Matcher {
	pt := newPatriciaTree(list...)
	return pt.matchPrefix
}

var defaultHTTPMethods = []string{
	"OPTIONS",
	"GET",
	"HEAD",
	"POST",
	"PUT",
	"DELETE",
	"TRACE",
	"CONNECT",
}

// HTTP1Fast only matches the methods in the HTTP request.
//
// This matcher is very optimistic: if it returns true, it does not mean that
// the request is a valid HTTP response. If you want a correct but slower HTTP1
// matcher, use HTTP1 instead.
func HTTP1Fast(extMethods ...string) Matcher {
	return PrefixMatcher(append(defaultHTTPMethods, extMethods...)...)
}

// TLS matches HTTPS requests.
//
// By default, any TLS handshake packet is matched. An optional whitelist
// of versions can be passed in to restrict the matcher, for example:
//  TLS(tls.VersionTLS11, tls.VersionTLS12)
func TLS(versions ...int) Matcher {
	if len(versions) == 0 {
		versions = []int{
			tls.VersionSSL30,
			tls.VersionTLS10,
			tls.VersionTLS11,
			tls.VersionTLS12,
		}
	}
	prefixes := [][]byte{}
	for _, v := range versions {
		prefixes = append(prefixes, []byte{22, byte(v >> 8 & 0xff), byte(v & 0xff)})
	}
	return prefixByteMatcher(prefixes...)
}

const maxHTTPRead = 4096

// HTTP1 parses the first line or upto 4096 bytes of the request to see if
// the conection contains an HTTP request.
func HTTP1() Matcher {
	return func(r io.Reader) bool {
		br := bufio.NewReader(&io.LimitedReader{R: r, N: maxHTTPRead})
		l, part, err := br.ReadLine()
		if err != nil || part {
			return false
		}

		_, _, proto, ok := parseRequestLine(string(l))
		if !ok {
			return false
		}

		v, _, ok := http.ParseHTTPVersion(proto)
		return ok && v == 1
	}
}

// grabbed from net/http.
func parseRequestLine(line string) (method, uri, proto string, ok bool) {
	s1 := strings.Index(line, " ")
	s2 := strings.Index(line[s1+1:], " ")
	if s1 < 0 || s2 < 0 {
		return
	}
	s2 += s1 + 1
	return line[:s1], line[s1+1 : s2], line[s2+1:], true
}

// HTTP2 parses the frame header of the first frame to detect whether the
// connection is an HTTP2 connection.
func HTTP2() Matcher {
	return hasHTTP2Preface
}

// HTTP1HeaderField returns a matcher matching the header fields of the first
// request of an HTTP 1 connection.
func HTTP1HeaderField(name, value string) Matcher {
	return func(r io.Reader) bool {
		return matchHTTP1Field(r, name, func(gotValue string) bool {
			return gotValue == value
		})
	}
}

// HTTP1HeaderFieldPrefix returns a matcher matching the header fields of the
// first request of an HTTP 1 connection. If the header with key name has a
// value prefixed with valuePrefix, this will match.
func HTTP1HeaderFieldPrefix(name, valuePrefix string) Matcher {
	return func(r io.Reader) bool {
		return matchHTTP1Field(r, name, func(gotValue string) bool {
			return strings.HasPrefix(gotValue, valuePrefix)
		})
	}
}

// HTTP2HeaderField returns a matcher matching the header fields of the first
// headers frame.
func HTTP2HeaderField(name, value string) Matcher {
	return func(r io.Reader) bool {
		return matchHTTP2Field(ioutil.Discard, r, name, func(gotValue string) bool {
			return gotValue == value
		})
	}
}

// HTTP2HeaderFieldPrefix returns a matcher matching the header fields of the
// first headers frame. If the header with key name has a value prefixed with
// valuePrefix, this will match.
func HTTP2HeaderFieldPrefix(name, valuePrefix string) Matcher {
	return func(r io.Reader) bool {
		return matchHTTP2Field(ioutil.Discard, r, name, func(gotValue string) bool {
			return strings.HasPrefix(gotValue, valuePrefix)
		})
	}
}

// HTTP2MatchHeaderFieldSendSettings matches the header field and writes the
// settings to the server. Prefer HTTP2HeaderField over this one, if the client
// does not block on receiving a SETTING frame.
func HTTP2MatchHeaderFieldSendSettings(name, value string) MatchWriter {
	return func(w io.Writer, r io.Reader) bool {
		return matchHTTP2Field(w, r, name, func(gotValue string) bool {
			return gotValue == value
		})
	}
}

// HTTP2MatchHeaderFieldPrefixSendSettings matches the header field prefix
// and writes the settings to the server. Prefer HTTP2HeaderFieldPrefix over
// this one, if the client does not block on receiving a SETTING frame.
func HTTP2MatchHeaderFieldPrefixSendSettings(name, valuePrefix string) MatchWriter {
	return func(w io.Writer, r io.Reader) bool {
		return matchHTTP2Field(w, r, name, func(gotValue string) bool {
			return strings.HasPrefix(gotValue, valuePrefix)
		})
	}
}

func hasHTTP2Preface(r io.Reader) bool {
	var b [len(http2.ClientPreface)]byte
	last := 0

	for {
		n, err := r.Read(b[last:])
		if err != nil {
			return false
		}

		last += n
		eq := string(b[:last]) == http2.ClientPreface[:last]
		if last == len(http2.ClientPreface) {
			return eq
		}
		if !eq {
			return false
		}
	}
}

func matchHTTP1Field(r io.Reader, name string, matches func(string) bool) (matched bool) {
	req, err := http.ReadRequest(bufio.NewReader(r))
	if err != nil {
		return false
	}

	return matches(req.Header.Get(name))
}

func matchHTTP2Field(w io.Writer, r io.Reader, name string, matches func(string) bool) (matched bool) {
	if !hasHTTP2Preface(r) {
		return false
	}

	done := false
	framer := http2.NewFramer(w, r)
	hdec := hpack.NewDecoder(uint32(4<<10), func(hf hpack.HeaderField) {
		if hf.Name == name {
			done = true
			if matches(hf.Value) {
				matched = true
			}
		}
	})
	for {
		f, err := framer.ReadFrame()
		if err != nil {
			return false
		}

		switch f := f.(type) {
		case *http2.SettingsFrame:
			// Sender acknoweldged the SETTINGS frame. No need to write
			// SETTINGS again.
			if f.IsAck() {
				break
			}
			if err := framer.WriteSettings(); err != nil {
				return false
			}
		case *http2.ContinuationFrame:
			if _, err := hdec.Write(f.HeaderBlockFragment()); err != nil {
				return false
			}
			done = done || f.FrameHeader.Flags&http2.FlagHeadersEndHeaders != 0
		case *http2.HeadersFrame:
			if _, err := hdec.Write(f.HeaderBlockFragment()); err != nil {
				return false
			}
			done = done || f.FrameHeader.Flags&http2.FlagHeadersEndHeaders != 0
		}

		if done {
			return matched
		}
	}
}
