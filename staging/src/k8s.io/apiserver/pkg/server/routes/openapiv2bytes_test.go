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

package routes

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func testSpec(title string) *spec.Swagger {
	return &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger: "2.0",
			Info:    &spec.Info{InfoProps: spec.InfoProps{Title: title, Version: "v1.0"}},
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/apis/" + title + "/v1/": {},
				},
			},
		},
	}
}

func mustMarshal(t *testing.T, s *spec.Swagger) []byte {
	t.Helper()
	json, err := s.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	return json
}

func fetchV2(t *testing.T, mux *http.ServeMux, accept, ifNoneMatch string) (body []byte, etag string, status int, contentType string) {
	t.Helper()
	server := httptest.NewServer(mux)
	defer server.Close()
	req, err := http.NewRequest(http.MethodGet, server.URL+"/openapi/v2", nil)
	if err != nil {
		t.Fatal(err)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if ifNoneMatch != "" {
		req.Header.Set("If-None-Match", ifNoneMatch)
	}
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, err = io.ReadAll(resp.Body)
	if closeErr := resp.Body.Close(); closeErr != nil && err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	return body, resp.Header.Get("Etag"), resp.StatusCode, resp.Header.Get("Content-Type")
}

// TestOpenAPIV2BytesServiceMatchesHandler verifies that OpenAPIV2BytesService
// serves byte-identical bodies, ETags and content types to kube-openapi's
// handler.OpenAPIService for every accepted media type.
func TestOpenAPIV2BytesServiceMatchesHandler(t *testing.T) {
	s := testSpec("match")

	classicMux := http.NewServeMux()
	handler.NewOpenAPIService(s).RegisterOpenAPIVersionedService("/openapi/v2", classicMux)

	bytesMux := http.NewServeMux()
	NewOpenAPIV2BytesService(mustMarshal(t, s)).RegisterOpenAPIVersionedService("/openapi/v2", bytesMux)

	for _, accept := range []string{
		"application/json",
		"application/com.github.proto-openapi.spec.v2.v1.0+protobuf",
		"application/com.github.proto-openapi.spec.v2@v1.0+protobuf",
		"*/*",
	} {
		t.Run(accept, func(t *testing.T) {
			classicBody, classicEtag, classicStatus, classicCT := fetchV2(t, classicMux, accept, "")
			bytesBody, bytesEtag, bytesStatus, bytesCT := fetchV2(t, bytesMux, accept, "")
			if classicStatus != http.StatusOK || bytesStatus != http.StatusOK {
				t.Fatalf("expected 200/200, got %d/%d", classicStatus, bytesStatus)
			}
			if !bytes.Equal(classicBody, bytesBody) {
				t.Errorf("bodies differ for Accept %q", accept)
			}
			if classicEtag == "" || classicEtag != bytesEtag {
				t.Errorf("ETags differ for Accept %q: classic %q, bytes %q", accept, classicEtag, bytesEtag)
			}
			if classicCT != bytesCT {
				t.Errorf("content types differ for Accept %q: classic %q, bytes %q", accept, classicCT, bytesCT)
			}
		})
	}
}

// TestOpenAPIV2BytesServiceUpdates verifies the update methods swap the
// served content and that ETags are deterministic content hashes.
func TestOpenAPIV2BytesServiceUpdates(t *testing.T) {
	spec1, spec2 := testSpec("one"), testSpec("two")
	json1 := mustMarshal(t, spec1)

	svc := NewOpenAPIV2BytesService(json1)
	mux := http.NewServeMux()
	svc.RegisterOpenAPIVersionedService("/openapi/v2", mux)

	body1, etag1, _, _ := fetchV2(t, mux, "application/json", "")
	if !bytes.Equal(body1, json1) {
		t.Error("served body does not match the provided bytes")
	}

	// UpdateSpec marshals and serves the new spec; a stale ETag gets a 200.
	if err := svc.UpdateSpec(spec2); err != nil {
		t.Fatal(err)
	}
	body2, etag2, status2, _ := fetchV2(t, mux, "application/json", etag1)
	if status2 != http.StatusOK {
		t.Fatalf("expected 200 with fresh content for stale ETag, got %d", status2)
	}
	if !bytes.Equal(body2, mustMarshal(t, spec2)) {
		t.Error("UpdateSpec did not serve the marshaled new spec")
	}
	if etag2 == etag1 {
		t.Error("ETag did not change after UpdateSpec")
	}

	// UpdateSpecFromBytes back to the original content restores the original
	// ETag (content-hash determinism).
	svc.UpdateSpecFromBytes(json1)
	_, etag3, _, _ := fetchV2(t, mux, "application/json", "")
	if etag3 != etag1 {
		t.Errorf("ETag is not a deterministic content hash: got %q, want %q", etag3, etag1)
	}

	// UpdateSpecLazy serves the marshaled spec from a lazy parsed source.
	svc.UpdateSpecLazy(cached.Static(spec2, "some-etag"))
	body4, _, _, _ := fetchV2(t, mux, "application/json", "")
	if !bytes.Equal(body4, mustMarshal(t, spec2)) {
		t.Error("UpdateSpecLazy did not serve the marshaled lazy spec")
	}
}

// TestOpenAPIV2BytesServiceConditionalAndErrors verifies 304, 406 and 503
// behavior.
func TestOpenAPIV2BytesServiceConditionalAndErrors(t *testing.T) {
	svc := NewOpenAPIV2BytesService(mustMarshal(t, testSpec("cond")))
	mux := http.NewServeMux()
	svc.RegisterOpenAPIVersionedService("/openapi/v2", mux)

	_, etag, _, _ := fetchV2(t, mux, "application/json", "")
	if _, _, status, _ := fetchV2(t, mux, "application/json", etag); status != http.StatusNotModified {
		t.Errorf("expected 304 for current ETag, got %d", status)
	}
	if _, _, status, _ := fetchV2(t, mux, "text/html", ""); status != http.StatusNotAcceptable {
		t.Errorf("expected 406 for unacceptable Accept, got %d", status)
	}

	// A lazy source that fails before any content was served returns 503.
	failing := NewOpenAPIV2BytesServiceLazy(cached.Func(func() ([]byte, string, error) {
		return nil, "", fmt.Errorf("source unavailable")
	}))
	failMux := http.NewServeMux()
	failing.RegisterOpenAPIVersionedService("/openapi/v2", failMux)
	if _, _, status, _ := fetchV2(t, failMux, "application/json", ""); status != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for failing source with no cached data, got %d", status)
	}
	// Recovery: providing bytes starts serving 200s.
	failing.UpdateSpecFromBytes(mustMarshal(t, testSpec("recovered")))
	if _, _, status, _ := fetchV2(t, failMux, "application/json", ""); status != http.StatusOK {
		t.Errorf("expected 200 after recovery, got %d", status)
	}
}
