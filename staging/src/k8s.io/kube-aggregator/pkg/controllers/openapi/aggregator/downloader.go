/*
Copyright 2017 The Kubernetes Authors.

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
	"crypto/sha512"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type CacheableDownloader interface {
	UpdateHandler(http.Handler)
	Get() (*spec.Swagger, string, error)
}

// cacheableDownloader is a downloader that will always return the data
// and the etag.
type cacheableDownloader struct {
	name       string
	downloader *Downloader
	// handler is the http Handler for the apiservice that can be replaced
	handler atomic.Pointer[http.Handler]
	etag    string
	spec    *spec.Swagger
}

// NewCacheableDownloader creates a downloader that also returns the etag, making it useful to use as a cached dependency.
func NewCacheableDownloader(apiServiceName string, downloader *Downloader, handler http.Handler) CacheableDownloader {
	c := &cacheableDownloader{
		name:       apiServiceName,
		downloader: downloader,
	}
	c.handler.Store(&handler)
	return c
}
func (d *cacheableDownloader) UpdateHandler(handler http.Handler) {
	d.handler.Store(&handler)
}

func (d *cacheableDownloader) Get() (*spec.Swagger, string, error) {
	spec, etag, err := d.get()
	if err != nil {
		return spec, etag, fmt.Errorf("failed to download %v: %v", d.name, err)
	}
	return spec, etag, err
}

func (d *cacheableDownloader) get() (*spec.Swagger, string, error) {
	h := *d.handler.Load()
	swagger, etag, status, err := d.downloader.Download(h, d.etag)
	if err != nil {
		return nil, "", err
	}
	switch status {
	case http.StatusNotModified:
		// Nothing has changed, do nothing.
	case http.StatusOK:
		if swagger != nil {
			d.etag = etag
			d.spec = swagger
			break
		}
		fallthrough
	case http.StatusNotFound:
		return nil, "", ErrAPIServiceNotFound
	default:
		return nil, "", fmt.Errorf("invalid status code: %v", status)
	}
	return d.spec, d.etag, nil
}

// Downloader is the OpenAPI downloader type. It will try to download spec from /openapi/v2 or /swagger.json endpoint.
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

func etagFor(data []byte) string {
	return fmt.Sprintf("%s%X\"", locallyGeneratedEtagPrefix, sha512.Sum512(data))
}

// Download downloads openAPI spec from /openapi/v2 endpoint of the given handler.
// httpStatus is only valid if err == nil
func (s *Downloader) Download(handler http.Handler, etag string) (returnSpec *spec.Swagger, newEtag string, httpStatus int, err error) {
	handler = s.handlerWithUser(handler, &user.DefaultInfo{Name: aggregatorUser})
	handler = http.TimeoutHandler(handler, specDownloadTimeout, "request timed out")

	req, err := http.NewRequest("GET", "/openapi/v2", nil)
	if err != nil {
		return nil, "", 0, err
	}
	req.Header.Add("Accept", "application/json")

	// Only pass eTag if it is not generated locally
	if len(etag) > 0 && !strings.HasPrefix(etag, locallyGeneratedEtagPrefix) {
		req.Header.Add("If-None-Match", etag)
	}

	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.respCode {
	case http.StatusNotModified:
		if len(etag) == 0 {
			return nil, etag, http.StatusNotModified, fmt.Errorf("http.StatusNotModified is not allowed in absence of etag")
		}
		return nil, etag, http.StatusNotModified, nil
	case http.StatusNotFound:
		// Gracefully skip 404, assuming the server won't provide any spec
		return nil, "", http.StatusNotFound, nil
	case http.StatusOK:
		openAPISpec := &spec.Swagger{}
		if err := openAPISpec.UnmarshalJSON(writer.data); err != nil {
			return nil, "", 0, err
		}
		newEtag = writer.Header().Get("Etag")
		if len(newEtag) == 0 {
			newEtag = etagFor(writer.data)
			if len(etag) > 0 && strings.HasPrefix(etag, locallyGeneratedEtagPrefix) {
				// The function call with an etag and server does not report an etag.
				// That means this server does not support etag and the etag that passed
				// to the function generated previously by us. Just compare etags and
				// return StatusNotModified if they are the same.
				if etag == newEtag {
					return nil, etag, http.StatusNotModified, nil
				}
			}
		}
		return openAPISpec, newEtag, http.StatusOK, nil
	default:
		return nil, "", 0, fmt.Errorf("failed to retrieve openAPI spec, http error: %s", writer.String())
	}
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
