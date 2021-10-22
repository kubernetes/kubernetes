// Copyright 2018 Google LLC
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
	"log"
	"net/http"
)

// Useful things for when we need to figure out what's actually going on under the hood.

type debugTransport struct {
	prefix string
	t      *http.Transport
}

func (d debugTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	log.Printf("proxy %s: %s %s", d.prefix, req.Method, req.URL)
	logHeaders(req.Header)
	res, err := d.t.RoundTrip(req)
	if err != nil {
		log.Printf("proxy %s: error %v", d.prefix, err)
	} else {
		log.Printf("proxy %s: %s", d.prefix, res.Status)
		log.Printf("Uncompressed = %v", res.Uncompressed)
		log.Printf("ContentLength = %d", res.ContentLength)
		logHeaders(res.Header)
		log.Printf("Trailers:")
		logHeaders(res.Trailer)
	}
	return res, err
}

func logHeaders(hs http.Header) {
	for k, v := range hs {
		log.Printf("    %s: %s", k, v)
	}
}
