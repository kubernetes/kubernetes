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
	"k8s.io/apiserver/pkg/util/responsewriter"
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
	writer := responsewriter.NewInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.RespCode() {
	case http.StatusNotFound:
		return nil, writer.RespCode(), nil
	case http.StatusOK:
		groups := handler3.OpenAPIV3Discovery{}
		if err := json.Unmarshal(writer.Data(), &groups); err != nil {
			return nil, writer.RespCode(), err
		}
		return &groups, writer.RespCode(), nil
	}
	return nil, writer.RespCode(), fmt.Errorf("Error, could not get list of group versions for APIService")
}
