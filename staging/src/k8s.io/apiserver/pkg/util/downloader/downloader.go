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

	"k8s.io/apimachinery/pkg/runtime/openapi"
)

type handlerDownloader struct {
	handler     http.Handler
	path        string
	contentType string
}

var _ openapi.SpecDownloader = &handlerDownloader{}

// NewHandlerDownloader creates a new downloader for a handler, path, and content type
func NewHandlerDownloader(handler http.Handler, path, contentType string) openapi.SpecDownloader {
	return &handlerDownloader{
		handler:     handler,
		path:        path,
		contentType: contentType,
	}
}

// Download gets bytes from a handler by requesting a specific content type from a path and providing an etag.
// If the handler returns http.StatusNotModified then Download will not return the bytes, if it returns StatusOK
// then the bytes will be returned along with the new etag. Otherwise it will return an error.
func (d *handlerDownloader) Download(lastEtag string) (specBytes []byte, newEtag string, httpStatus int, err error) {
	req, err := http.NewRequest("GET", d.path, nil)
	if err != nil {
		return nil, "", 0, err
	}
	req.Header.Add("Accept", d.contentType)
	if len(lastEtag) > 0 {
		req.Header.Add("If-None-Match", lastEtag)
	}

	writer := NewInMemoryResponseWriter()
	d.handler.ServeHTTP(writer, req)

	switch writer.RespCode {
	case http.StatusNotModified:
		if len(lastEtag) == 0 {
			return nil, "", 0, fmt.Errorf("http.StatusNotModified is not allowed in absence of etag")
		}
		return nil, lastEtag, http.StatusNotModified, nil
	case http.StatusOK:
		specBytes = writer.Data
		newEtag = writer.Header().Get("Etag")
		if len(newEtag) == 0 {
			return nil, "", 0, fmt.Errorf("An etag must be present in the response if http.StatusOK is returned")
		}
		return specBytes, newEtag, http.StatusOK, nil
	default:
		return nil, "", 0, fmt.Errorf("failed to retrieve openAPI spec, http error: %s", writer.String())
	}
}
