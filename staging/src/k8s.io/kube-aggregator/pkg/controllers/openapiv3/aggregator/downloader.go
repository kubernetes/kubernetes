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
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/spec3"
)

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

// gvList is a struct for the response of the /openapi/v3 endpoint to unmarshal into
type gvList struct {
	Paths []string `json:"Paths"`
}

// SpecETag is a OpenAPI v3 spec and etag pair for the endpoint of each OpenAPI group/version
type SpecETag struct {
	spec *spec3.OpenAPI
	etag string
}

// Download downloads OpenAPI v3 for all groups of a given handler
func (s *Downloader) Download(handler http.Handler, etagList map[string]string) (returnSpec map[string]*SpecETag, err error) {
	// TODO(jefftree): https://github.com/kubernetes/kubernetes/pull/105945#issuecomment-966455034
	// Move to proxy request in the aggregator and let the APIServices serve the OpenAPI directly
	handler = s.handlerWithUser(handler, &user.DefaultInfo{Name: aggregatorUser})
	handler = http.TimeoutHandler(handler, specDownloadTimeout, "request timed out")

	req, err := http.NewRequest("GET", "/openapi/v3", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Accept", "application/json")

	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.respCode {
	case http.StatusNotFound:
		// Gracefully skip 404, assuming the server won't provide any spec
		return nil, nil
	case http.StatusOK:
		groups := gvList{}
		aggregated := make(map[string]*SpecETag)

		if err := json.Unmarshal(writer.data, &groups); err != nil {
			return nil, err
		}
		for _, path := range groups.Paths {
			reqPath := fmt.Sprintf("/openapi/v3/%s", path)
			req, err := http.NewRequest("GET", reqPath, nil)
			if err != nil {
				return nil, err
			}
			req.Header.Add("Accept", "application/json")
			oldEtag, ok := etagList[path]
			if ok {
				req.Header.Add("If-None-Match", oldEtag)
			}
			openAPIWriter := newInMemoryResponseWriter()
			handler.ServeHTTP(openAPIWriter, req)

			switch openAPIWriter.respCode {
			case http.StatusNotFound:
				continue
			case http.StatusNotModified:
				aggregated[path] = &SpecETag{
					etag: oldEtag,
				}
			case http.StatusOK:
				var spec spec3.OpenAPI
				// TODO|jefftree: For OpenAPI v3 Beta, if the v3 spec is empty then
				// we should request the v2 endpoint and convert it to v3
				if len(openAPIWriter.data) > 0 {
					err = json.Unmarshal(openAPIWriter.data, &spec)
					if err != nil {
						return nil, err
					}
					etag := openAPIWriter.Header().Get("Etag")
					aggregated[path] = &SpecETag{
						spec: &spec,
						etag: etag,
					}
				}
			default:
				klog.Errorf("Error: unknown status %v", openAPIWriter.respCode)
			}
		}

		return aggregated, nil
	default:
		return nil, fmt.Errorf("failed to retrieve openAPI spec, http error: %s", writer.String())
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
