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

package httpcache

import (
	"bufio"
	"bytes"
	"net/http"

	"github.com/bartventer/httpcache/internal"
)

func make504Response(req *http.Request) (*http.Response, error) {
	var buf bytes.Buffer
	buf.WriteString("HTTP/1.1 504 Gateway Timeout\r\n")
	buf.WriteString("Cache-Control: no-cache\r\n")
	buf.WriteString("Content-Length: 0\r\n")
	buf.WriteString(
		internal.CacheStatusHeader + ": " + internal.CacheStatusBypass.Value + "\r\n",
	)
	buf.WriteString("Connection: close\r\n")
	buf.WriteString("\r\n")
	return http.ReadResponse(bufio.NewReader(&buf), req)
}

func cloneRequest(req *http.Request) *http.Request {
	req2 := new(http.Request)
	*req2 = *req
	req2.Header = req.Header.Clone()
	return req2
}

// withConditionalHeaders sets the conditional headers on the request based on the
// stored response headers as specified in RFC 9111 §4.3.1.
func withConditionalHeaders(req *http.Request, storedHdr http.Header) *http.Request {
	var req2 *http.Request
	if etag := storedHdr.Get("ETag"); etag != "" {
		req2 = cloneRequest(req)
		req2.Header.Set("If-None-Match", etag)
	}
	if lastModified := storedHdr.Get("Last-Modified"); lastModified != "" {
		if req2 == nil {
			req2 = cloneRequest(req)
		}
		req2.Header.Set("If-Modified-Since", lastModified)
	}
	if req2 != nil {
		req = req2
	}
	return req
}
