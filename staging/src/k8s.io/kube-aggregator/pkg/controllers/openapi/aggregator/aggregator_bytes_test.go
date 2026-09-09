/*
Copyright The Kubernetes Authors.

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
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func buildAndRegisterSpecAggregatorBytes(t *testing.T, delegationHandlers []http.Handler, mux common.PathHandler) *specAggregatorBytes {
	t.Helper()
	downloader := NewDownloader()
	aggregatorSpec := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/apis/apiregistration.k8s.io/v1/": {},
				},
			},
		},
	}
	aggregatorSpecJSON, err := aggregatorSpec.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	return buildAndRegisterSpecAggregatorBytesForLocalServices(&downloader, aggregatorSpecJSON, delegationHandlers, mux)
}

func fetchOpenAPIRaw(mux *http.ServeMux) (body []byte, etag string, err error) {
	server := httptest.NewServer(mux)
	defer server.Close()
	client := server.Client()

	req, err := http.NewRequest(http.MethodGet, server.URL+"/openapi/v2", nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	body, err = io.ReadAll(resp.Body)
	if closeErr := resp.Body.Close(); closeErr != nil && err == nil {
		err = closeErr
	}
	return body, resp.Header.Get("Etag"), err
}

func newTestDelegateHandlers() []http.Handler {
	return []http.Handler{
		&openAPIHandler{
			openapi: &spec.Swagger{
				SwaggerProps: spec.SwaggerProps{
					Paths: &spec.Paths{
						Paths: map[string]spec.PathItem{
							"/apis/foo/v1/": {},
						},
					},
				},
			},
		},
	}
}

// TestBytesAggregatorMatchesClassic verifies that the bytes-mode aggregator
// serves byte-identical JSON and the same ETag as the classic aggregator.
func TestBytesAggregatorMatchesClassic(t *testing.T) {
	classicMux := http.NewServeMux()
	buildAndRegisterSpecAggregator(newTestDelegateHandlers(), classicMux)
	classicJSON, classicEtag, err := fetchOpenAPIRaw(classicMux)
	if err != nil {
		t.Fatal(err)
	}

	bytesMux := http.NewServeMux()
	buildAndRegisterSpecAggregatorBytes(t, newTestDelegateHandlers(), bytesMux)
	bytesJSON, bytesEtag, err := fetchOpenAPIRaw(bytesMux)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(classicJSON, bytesJSON) {
		t.Errorf("bytes-mode aggregator served different JSON than the classic aggregator:\nclassic: %s\nbytes:   %s", classicJSON, bytesJSON)
	}
	if classicEtag == "" || classicEtag != bytesEtag {
		t.Errorf("bytes-mode aggregator served a different ETag than the classic aggregator: classic %q, bytes %q", classicEtag, bytesEtag)
	}
}

// TestBytesAddRemoveAPIService mirrors TestAddRemoveAPIService against the
// bytes-mode aggregator.
func TestBytesAddRemoveAPIService(t *testing.T) {
	mux := http.NewServeMux()
	s := buildAndRegisterSpecAggregatorBytes(t, newTestDelegateHandlers(), mux)

	apiService := &v1.APIService{
		Spec: v1.APIServiceSpec{
			Group:   "apiservicegroup",
			Version: "v1",
			Service: &v1.ServiceReference{Name: "dummy"},
		},
	}
	apiService.Name = "apiservice"

	handler := &openAPIHandler{openapi: &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/apis/apiservicegroup/v1/": {},
				},
			},
		},
	}}

	if err := s.AddUpdateAPIService(apiService, handler); err != nil {
		t.Error(err)
	}
	if err := s.UpdateAPIServiceSpec(apiService.Name); err != nil {
		t.Error(err)
	}

	swagger, err := fetchOpenAPI(mux)
	if err != nil {
		t.Error(err)
	}
	expectPath(t, swagger, "/apis/foo/v1/")
	expectPath(t, swagger, "/apis/apiservicegroup/v1/")
	expectPath(t, swagger, "/apis/apiregistration.k8s.io/v1/")

	s.RemoveAPIService(apiService.Name)

	swagger, err = fetchOpenAPI(mux)
	if err != nil {
		t.Error(err)
	}
	expectNoPath(t, swagger, "/apis/apiservicegroup/v1/")
	expectPath(t, swagger, "/apis/foo/v1/")
	expectPath(t, swagger, "/apis/apiregistration.k8s.io/v1/")
}

// TestBytesFailingAPIServiceSkippedAggregation verifies a failing APIService
// is skipped by the bytes-mode merge, mirroring the classic behavior.
func TestBytesFailingAPIServiceSkippedAggregation(t *testing.T) {
	mux := http.NewServeMux()
	s := buildAndRegisterSpecAggregatorBytes(t, newTestDelegateHandlers(), mux)

	apiService := &v1.APIService{
		Spec: v1.APIServiceSpec{
			Group:   "failed",
			Version: "v1",
			Service: &v1.ServiceReference{Name: "dummy"},
		},
	}
	apiService.Name = "failed"

	if err := s.AddUpdateAPIService(apiService, &openAPIHandler{returnErr: true}); err != nil {
		t.Error(err)
	}
	if err := s.UpdateAPIServiceSpec(apiService.Name); err == nil {
		t.Error("expected UpdateAPIServiceSpec to return an error for a failing APIService")
	}

	// The aggregated spec must still be served without the failing
	// APIService's paths.
	swagger, err := fetchOpenAPI(mux)
	if err != nil {
		t.Error(err)
	}
	expectPath(t, swagger, "/apis/foo/v1/")
	expectNoPath(t, swagger, "/apis/failed/v1/")
}
