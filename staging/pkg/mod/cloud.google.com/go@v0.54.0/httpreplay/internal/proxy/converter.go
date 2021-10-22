// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"bytes"
	"io"
	"io/ioutil"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

// A Converter converts HTTP requests and responses to the Request and Response types
// of this package, while removing or redacting information.
type Converter struct {
	// These all apply to both headers and trailers.
	ClearHeaders          []tRegexp // replace matching headers with "CLEARED"
	RemoveRequestHeaders  []tRegexp // remove matching headers in requests
	RemoveResponseHeaders []tRegexp // remove matching headers in responses
	ClearParams           []tRegexp // replace matching query params with "CLEARED"
	RemoveParams          []tRegexp // remove matching query params
}

// A regexp that can be marshaled to and from text.
type tRegexp struct {
	*regexp.Regexp
}

func (r tRegexp) MarshalText() ([]byte, error) {
	return []byte(r.String()), nil
}

func (r *tRegexp) UnmarshalText(b []byte) error {
	var err error
	r.Regexp, err = regexp.Compile(string(b))
	return err
}

func (c *Converter) registerRemoveRequestHeaders(pat string) {
	c.RemoveRequestHeaders = append(c.RemoveRequestHeaders, pattern(pat))
}

func (c *Converter) registerClearHeaders(pat string) {
	c.ClearHeaders = append(c.ClearHeaders, pattern(pat))
}

func (c *Converter) registerRemoveParams(pat string) {
	c.RemoveParams = append(c.RemoveParams, pattern(pat))
}

func (c *Converter) registerClearParams(pat string) {
	c.ClearParams = append(c.ClearParams, pattern(pat))
}

var (
	defaultRemoveRequestHeaders = []string{
		"Authorization", // not only is it secret, but it is probably missing on replay
		"Proxy-Authorization",
		"Connection",
		"Content-Type", // because it may contain a random multipart boundary
		"Date",
		"Host",
		"Transfer-Encoding",
		"Via",
		"X-Forwarded-*",
		// Google-specific
		"X-Cloud-Trace-Context", // OpenCensus traces have a random ID
		"X-Goog-Api-Client",     // can differ for, e.g., different Go versions
	}

	defaultRemoveBothHeaders = []string{
		// Google-specific
		// GFEs scrub X-Google- and X-GFE- headers from requests and responses.
		// Drop them from recordings made by users inside Google.
		// http://g3doc/gfe/g3doc/gfe3/design/http_filters/google_header_filter
		// (internal Google documentation).
		"X-Google-*",
		"X-Gfe-*",
	}

	defaultClearHeaders = []string{
		// Google-specific
		// Used by Cloud Storage for customer-supplied encryption.
		"X-Goog-*Encryption-Key",
	}
)

func defaultConverter() *Converter {
	c := &Converter{}
	for _, h := range defaultClearHeaders {
		c.registerClearHeaders(h)
	}
	for _, h := range defaultRemoveRequestHeaders {
		c.registerRemoveRequestHeaders(h)
	}
	for _, h := range defaultRemoveBothHeaders {
		c.registerRemoveRequestHeaders(h)
		c.RemoveResponseHeaders = append(c.RemoveResponseHeaders, pattern(h))
	}
	return c
}

// Convert a pattern into a regexp.
// A pattern is like a literal regexp anchored on both ends, with only one
// non-literal character: "*", which matches zero or more characters.
func pattern(p string) tRegexp {
	q := regexp.QuoteMeta(p)
	q = "^" + strings.Replace(q, `\*`, `.*`, -1) + "$"
	// q must be a legal regexp.
	return tRegexp{regexp.MustCompile(q)}
}

func (c *Converter) convertRequest(req *http.Request) (*Request, error) {
	body, err := snapshotBody(&req.Body)
	if err != nil {
		return nil, err
	}
	// If the body is empty, set it to nil to make sure the proxy sends a
	// Content-Length header.
	if len(body) == 0 {
		req.Body = nil
	}
	mediaType, parts, err := parseRequestBody(req.Header.Get("Content-Type"), body)
	if err != nil {
		return nil, err
	}
	url2 := *req.URL
	url2.RawQuery = scrubQuery(url2.RawQuery, c.ClearParams, c.RemoveParams)
	return &Request{
		Method:    req.Method,
		URL:       url2.String(),
		Header:    scrubHeaders(req.Header, c.ClearHeaders, c.RemoveRequestHeaders),
		MediaType: mediaType,
		BodyParts: parts,
		Trailer:   scrubHeaders(req.Trailer, c.ClearHeaders, c.RemoveRequestHeaders),
	}, nil
}

// parseRequestBody parses the Content-Type header, reads the body, and splits it into
// parts if necessary. It returns the media type and the body parts.
func parseRequestBody(contentType string, body []byte) (string, [][]byte, error) {
	if contentType == "" {
		// No content-type header. Treat the body as a single part.
		return "", [][]byte{body}, nil
	}
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return "", nil, err
	}
	var parts [][]byte
	if strings.HasPrefix(mediaType, "multipart/") {
		mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
		for {
			p, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", nil, err
			}
			part, err := ioutil.ReadAll(p)
			if err != nil {
				return "", nil, err
			}
			// TODO(jba): care about part headers?
			parts = append(parts, part)
		}
	} else {
		parts = [][]byte{body}
	}
	return mediaType, parts, nil
}

func (c *Converter) convertResponse(res *http.Response) (*Response, error) {
	data, err := snapshotBody(&res.Body)
	if err != nil {
		return nil, err
	}
	return &Response{
		StatusCode: res.StatusCode,
		Proto:      res.Proto,
		ProtoMajor: res.ProtoMajor,
		ProtoMinor: res.ProtoMinor,
		Header:     scrubHeaders(res.Header, c.ClearHeaders, c.RemoveResponseHeaders),
		Body:       data,
		Trailer:    scrubHeaders(res.Trailer, c.ClearHeaders, c.RemoveResponseHeaders),
	}, nil
}

func snapshotBody(body *io.ReadCloser) ([]byte, error) {
	data, err := ioutil.ReadAll(*body)
	if err != nil {
		return nil, err
	}
	(*body).Close()
	*body = ioutil.NopCloser(bytes.NewReader(data))
	return data, nil
}

// Copy headers, clearing some and removing others.
func scrubHeaders(hs http.Header, clear, remove []tRegexp) http.Header {
	rh := http.Header{}
	for k, v := range hs {
		switch {
		case match(k, clear):
			rh.Set(k, "CLEARED")
		case match(k, remove):
			// skip
		default:
			rh[k] = v
		}
	}
	return rh
}

// Copy the query string, clearing some query params and removing others.
// Preserve the order of the string.
func scrubQuery(query string, clear, remove []tRegexp) string {
	// We can't use url.ParseQuery because it doesn't preserve order.
	var buf bytes.Buffer
	for {
		if i := strings.IndexAny(query, "&;"); i >= 0 {
			scrubParam(&buf, query[:i], query[i], clear, remove)
			query = query[i+1:]
		} else {
			scrubParam(&buf, query, 0, clear, remove)
			break
		}
	}
	s := buf.String()
	if strings.HasSuffix(s, "&") {
		return s[:len(s)-1]
	}
	return s
}

func scrubParam(buf *bytes.Buffer, param string, sep byte, clear, remove []tRegexp) {
	if param == "" {
		return
	}
	key := param
	value := ""
	if i := strings.Index(param, "="); i >= 0 {
		key, value = key[:i], key[i+1:]
	}
	ukey, err := url.QueryUnescape(key)
	// If the key is bad, just pass it and the value through.
	if err != nil {
		buf.WriteString(param)
		if sep != 0 {
			buf.WriteByte(sep)
		}
		return
	}
	if match(ukey, remove) {
		return
	}
	if match(ukey, clear) && value != "" {
		value = "CLEARED"
	}
	buf.WriteString(key)
	buf.WriteByte('=')
	buf.WriteString(value)
	if sep != 0 {
		buf.WriteByte(sep)
	}
}

func match(s string, res []tRegexp) bool {
	for _, re := range res {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}
