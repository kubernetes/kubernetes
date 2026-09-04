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
	"sync"
	"testing"

	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func lazyTestSpec(title string) *spec.Swagger {
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

func fetchLazyV2(t *testing.T, mux *http.ServeMux, ifNoneMatch string) (body []byte, etag string, status int) {
	t.Helper()
	server := httptest.NewServer(mux)
	defer server.Close()
	req, err := http.NewRequest(http.MethodGet, server.URL+"/openapi/v2", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/json")
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
	return body, resp.Header.Get("Etag"), resp.StatusCode
}

// TestLazyOpenAPIV2SpecSourceBuildsOnceOnDemand verifies the spec is not
// built until first use, is built exactly once, and that the served content
// and ETag match what the non-lazy service serves for the same spec.
func TestLazyOpenAPIV2SpecSourceBuildsOnceOnDemand(t *testing.T) {
	s := lazyTestSpec("lazy")
	var builds int
	source := NewLazyOpenAPIV2SpecSource(func() (*spec.Swagger, error) {
		builds++
		return s, nil
	})

	lazyMux := http.NewServeMux()
	handler.NewOpenAPIServiceLazy(source).RegisterOpenAPIVersionedService("/openapi/v2", lazyMux)
	if builds != 0 {
		t.Fatalf("expected no build before the first request, got %d", builds)
	}

	lazyBody, lazyEtag, status := fetchLazyV2(t, lazyMux, "")
	if status != http.StatusOK {
		t.Fatalf("expected 200, got %d", status)
	}
	if builds != 1 {
		t.Fatalf("expected exactly one build after the first request, got %d", builds)
	}

	// Repeated requests do not rebuild, and conditional requests get 304s.
	_, _, _ = fetchLazyV2(t, lazyMux, "")
	if _, _, status := fetchLazyV2(t, lazyMux, lazyEtag); status != http.StatusNotModified {
		t.Errorf("expected 304 for current ETag, got %d", status)
	}
	if builds != 1 {
		t.Errorf("expected no rebuilds on subsequent requests, got %d builds", builds)
	}

	// The served bytes and ETag are identical to the eager service.
	eagerMux := http.NewServeMux()
	handler.NewOpenAPIService(lazyTestSpec("lazy")).RegisterOpenAPIVersionedService("/openapi/v2", eagerMux)
	eagerBody, eagerEtag, _ := fetchLazyV2(t, eagerMux, "")
	if !bytes.Equal(lazyBody, eagerBody) {
		t.Error("lazy service served different bytes than the eager service")
	}
	if lazyEtag == "" || lazyEtag != eagerEtag {
		t.Errorf("lazy service served a different ETag: lazy %q, eager %q", lazyEtag, eagerEtag)
	}
}

// TestLazyOpenAPIV2SpecSourceRetriesOnError verifies that build errors are
// not cached: the endpoint returns 503 and the build is retried until it
// succeeds.
func TestLazyOpenAPIV2SpecSourceRetriesOnError(t *testing.T) {
	var builds int
	source := NewLazyOpenAPIV2SpecSource(func() (*spec.Swagger, error) {
		builds++
		if builds < 3 {
			return nil, fmt.Errorf("transient build failure %d", builds)
		}
		return lazyTestSpec("recovered"), nil
	})
	mux := http.NewServeMux()
	handler.NewOpenAPIServiceLazy(source).RegisterOpenAPIVersionedService("/openapi/v2", mux)

	for range 2 {
		if _, _, status := fetchLazyV2(t, mux, ""); status != http.StatusServiceUnavailable {
			t.Fatalf("expected 503 while build fails, got %d", status)
		}
	}
	if _, _, status := fetchLazyV2(t, mux, ""); status != http.StatusOK {
		t.Fatalf("expected 200 once build succeeds, got %d", status)
	}
	if builds != 3 {
		t.Errorf("expected 3 build attempts, got %d", builds)
	}
	if _, _, status := fetchLazyV2(t, mux, ""); status != http.StatusOK {
		t.Errorf("expected 200 after success, got %d", status)
	}
	if builds != 3 {
		t.Errorf("expected no rebuild after success, got %d attempts", builds)
	}
}

// TestLazyOpenAPIV2SpecSourceConcurrentFirstUse verifies concurrent first
// requests share a single build.
func TestLazyOpenAPIV2SpecSourceConcurrentFirstUse(t *testing.T) {
	var mu sync.Mutex
	builds := 0
	source := NewLazyOpenAPIV2SpecSource(func() (*spec.Swagger, error) {
		mu.Lock()
		builds++
		mu.Unlock()
		return lazyTestSpec("concurrent"), nil
	})
	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := source.Get(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if builds != 1 {
		t.Errorf("expected one build under concurrency, got %d", builds)
	}
}
