package cmux

import (
	"bufio"
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
		return matchHTTP1Field(r, name, value)
	}
}

// HTTP2HeaderField resturns a matcher matching the header fields of the first
// headers frame.
func HTTP2HeaderField(name, value string) Matcher {
	return func(r io.Reader) bool {
		return matchHTTP2Field(r, name, value)
	}
}

func hasHTTP2Preface(r io.Reader) bool {
	var b [len(http2.ClientPreface)]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return false
	}

	return string(b[:]) == http2.ClientPreface
}

func matchHTTP1Field(r io.Reader, name, value string) (matched bool) {
	req, err := http.ReadRequest(bufio.NewReader(r))
	if err != nil {
		return false
	}

	return req.Header.Get(name) == value
}

func matchHTTP2Field(r io.Reader, name, value string) (matched bool) {
	if !hasHTTP2Preface(r) {
		return false
	}

	framer := http2.NewFramer(ioutil.Discard, r)
	hdec := hpack.NewDecoder(uint32(4<<10), func(hf hpack.HeaderField) {
		if hf.Name == name && hf.Value == value {
			matched = true
		}
	})
	for {
		f, err := framer.ReadFrame()
		if err != nil {
			return false
		}

		switch f := f.(type) {
		case *http2.HeadersFrame:
			if _, err := hdec.Write(f.HeaderBlockFragment()); err != nil {
				return false
			}
			if matched {
				return true
			}

			if f.FrameHeader.Flags&http2.FlagHeadersEndHeaders != 0 {
				return false
			}
		}
	}
}
