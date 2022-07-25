/*
Copyright 2021 The Kubernetes Authors.

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

package aggregator

import (
	"encoding/json"
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kube-openapi/pkg/handler3"
)

type NotFoundError struct {
}

func (e *NotFoundError) Error() string {
	return ""
}

// Downloader is the OpenAPI downloader type. It will try to download spec from /openapi/v3 and /openap/v3/<group>/<version> endpoints.
type Downloader struct {
}

// NewDownloader creates a new OpenAPI Downloader.
func NewDownloader() Downloader {
	return Downloader{}
}

func (s *Downloader) handlerWithUser(handler http.Handler, info user.Info) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		req = req.WithContext(request.WithUser(req.Context(), info))
		handler.ServeHTTP(w, req)
	})
}

// OpenAPIV3Root downloads the OpenAPI V3 root document from an APIService
func (s *Downloader) OpenAPIV3Root(handler http.Handler) (*handler3.OpenAPIV3Discovery, int, error) {
	handler = s.handlerWithUser(handler, &user.DefaultInfo{Name: aggregatorUser})
	handler = http.TimeoutHandler(handler, specDownloadTimeout, "request timed out")

	req, err := http.NewRequest("GET", "/openapi/v3", nil)
	if err != nil {
		return nil, 0, err
	}
	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.respCode {
	case http.StatusNotFound:
		return nil, writer.respCode, nil
	case http.StatusOK:
		groups := handler3.OpenAPIV3Discovery{}
		if err := json.Unmarshal(writer.data, &groups); err != nil {
			return nil, writer.respCode, err
		}
		return &groups, writer.respCode, nil
	}
	return nil, writer.respCode, fmt.Errorf("Error, could not get list of group versions for APIService")
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
