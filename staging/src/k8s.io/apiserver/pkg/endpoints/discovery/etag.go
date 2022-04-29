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

package discovery

import (
	"crypto/sha512"
	"fmt"
	"net/http"
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This file exposes helper functions used for calculating the E-Tag header
// used in discovery endpoint responses

func CalculateETag(resurces *metav1.APIResourceList) (string, error) {
	serialized, err := resurces.Marshal()
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%X", sha512.Sum512(serialized)), nil
}

func GetGroupVersionHash(path string, handler http.Handler) (string, error) {
	req, err := http.NewRequest("GET", path, nil)
	if err != nil {
		return "", err
	}
	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)
	if writer.respCode != http.StatusOK {
		return "", fmt.Errorf("Error requesting hash for group-version %v, %v", path, writer.String())
	}
	etag := writer.Header().Get("Etag")
	etag, err = strconv.Unquote(etag)
	if err != nil {
		return "", err
	}
	return etag, nil
}

// inMemoryResponseWriter is a http.Writer that keep the response in memory.
type inMemoryResponseWriter struct {
	writeHeaderCalled bool
	header            http.Header
	respCode          int
	data              []byte
}

func newInMemoryResponseWriter() *inMemoryResponseWriter {
	return &inMemoryResponseWriter{header: http.Header{}}
}

func (r *inMemoryResponseWriter) Header() http.Header {
	return r.header
}

func (r *inMemoryResponseWriter) WriteHeader(code int) {
	r.writeHeaderCalled = true
	r.respCode = code
}

func (r *inMemoryResponseWriter) Write(in []byte) (int, error) {
	if !r.writeHeaderCalled {
		r.WriteHeader(http.StatusOK)
	}
	r.data = append(r.data, in...)
	return len(in), nil
}

func (r *inMemoryResponseWriter) String() string {
	s := fmt.Sprintf("ResponseCode: %d", r.respCode)
	if r.data != nil {
		s += fmt.Sprintf(", Body: %s", string(r.data))
	}
	if r.header != nil {
		s += fmt.Sprintf(", Header: %s", r.header)
	}
	return s
}
