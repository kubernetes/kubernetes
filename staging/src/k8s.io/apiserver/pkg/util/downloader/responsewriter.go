/*
Copyright 2018 The Kubernetes Authors.

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

package downloader

import (
	"fmt"
	"net/http"
)

// InMemoryResponseWriter is a http.ResponseWriter that keep the response in memory.
type InMemoryResponseWriter struct {
	writeHeaderCalled bool
	header            http.Header

	RespCode int
	Data     []byte
}

// NewInMemoryResponseWriter creates a new InMemoryResponseWriter
func NewInMemoryResponseWriter() *InMemoryResponseWriter {
	return &InMemoryResponseWriter{header: http.Header{}}
}

// Header implements http.ResponseWriter
func (r *InMemoryResponseWriter) Header() http.Header {
	return r.header
}

// WriteHeader implements http.ResponseWriter
func (r *InMemoryResponseWriter) WriteHeader(code int) {
	r.writeHeaderCalled = true
	r.RespCode = code
}

// Write implements http.ResponseWriter
func (r *InMemoryResponseWriter) Write(in []byte) (int, error) {
	if !r.writeHeaderCalled {
		r.WriteHeader(http.StatusOK)
	}
	r.Data = append(r.Data, in...)
	return len(in), nil
}

// String converts the response to a human readable string
func (r *InMemoryResponseWriter) String() string {
	s := fmt.Sprintf("ResponseCode: %d", r.RespCode)
	if r.Data != nil {
		s += fmt.Sprintf(", Body: %s", string(r.Data))
	}
	if r.header != nil {
		s += fmt.Sprintf(", Header: %s", r.header)
	}
	return s
}
