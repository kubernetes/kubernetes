/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package genericapiserver

import (
	"net/http"
)

// intercept404ResponseHandler behaves like the underlying ResponseWriter, but
// keeps it untouched on 404 responses.
type intercept404ResponseHandler struct {
	w      http.ResponseWriter
	code   int
	header http.Header
}

// Code returns 0 until a code was written.
func (aw *intercept404ResponseHandler) Code() int {
	return aw.code
}

// NewAbortableResponseWriter creates an abortable ResponseWriter.
func NewIntercept404ResponseWriter(w http.ResponseWriter) *intercept404ResponseHandler {
	aw := intercept404ResponseHandler{
		w:      w,
		header: http.Header{},
	}
	copyHeader(w.Header(), aw.header)
	return &aw
}

func (aw *intercept404ResponseHandler) Write(bs []byte) (int, error) {
	switch aw.code {
	case 0:
		copyHeader(aw.header, aw.w.Header())
		aw.code = 200
		return aw.w.Write(bs)
	case 404:
		// ignore
		return len(bs), nil
	default:
		return aw.w.Write(bs)
	}
}

func (aw *intercept404ResponseHandler) Header() http.Header {
	if aw.code == 0 || aw.code == 404 {
		return aw.header
	} else {
		return aw.w.Header()
	}
}

func (aw *intercept404ResponseHandler) WriteHeader(code int) {
	if aw.code == 0 {
		aw.code = code
	}
	if code != 404 {
		copyHeader(aw.header, aw.w.Header())
		aw.w.WriteHeader(code)
	}
}

func copyHeader(from http.Header, to http.Header) {
	for k := range to {
		delete(to, k)
	}
	for k, vv := range from {
		to[k] = vv
	}
}

// MergeHandlers composes a and b such that first a is called, and in case of a 404 response b is called.
func MergeHandlers(a, b http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		aw := NewIntercept404ResponseWriter(w)
		a.ServeHTTP(aw, r)
		if aw.Code() == 404 {
			b.ServeHTTP(w, r)
			return
		}
	})
}
