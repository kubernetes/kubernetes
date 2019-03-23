// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package cors handles cross-origin HTTP requests (CORS).
package cors

import (
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

type CORSInfo map[string]bool

// Set implements the flag.Value interface to allow users to define a list of CORS origins
func (ci *CORSInfo) Set(s string) error {
	m := make(map[string]bool)
	for _, v := range strings.Split(s, ",") {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if v != "*" {
			if _, err := url.Parse(v); err != nil {
				return fmt.Errorf("Invalid CORS origin: %s", err)
			}
		}
		m[v] = true

	}
	*ci = CORSInfo(m)
	return nil
}

func (ci *CORSInfo) String() string {
	o := make([]string, 0)
	for k := range *ci {
		o = append(o, k)
	}
	sort.StringSlice(o).Sort()
	return strings.Join(o, ",")
}

// OriginAllowed determines whether the server will allow a given CORS origin.
func (c CORSInfo) OriginAllowed(origin string) bool {
	return c["*"] || c[origin]
}

type CORSHandler struct {
	Handler http.Handler
	Info    *CORSInfo
}

// addHeader adds the correct cors headers given an origin
func (h *CORSHandler) addHeader(w http.ResponseWriter, origin string) {
	w.Header().Add("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Add("Access-Control-Allow-Origin", origin)
	w.Header().Add("Access-Control-Allow-Headers", "accept, content-type, authorization")
}

// ServeHTTP adds the correct CORS headers based on the origin and returns immediately
// with a 200 OK if the method is OPTIONS.
func (h *CORSHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Write CORS header.
	if h.Info.OriginAllowed("*") {
		h.addHeader(w, "*")
	} else if origin := req.Header.Get("Origin"); h.Info.OriginAllowed(origin) {
		h.addHeader(w, origin)
	}

	if req.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	h.Handler.ServeHTTP(w, req)
}
