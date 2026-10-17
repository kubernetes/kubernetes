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
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	restful "github.com/emicklei/go-restful/v3"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type lazyTestObject struct {
	Name string `json:"name"`
}

// lazyTestContainer returns a container with one route that serves
// lazyTestObject, so that building the spec needs its definition.
func lazyTestContainer() *restful.Container {
	c := restful.NewContainer()
	ws := new(restful.WebService)
	ws.Path("/apis/lazytest.example.com/v1").Produces(restful.MIME_JSON)
	ws.Route(ws.GET("/objects").
		To(func(*restful.Request, *restful.Response) {}).
		Operation("listLazyTestObjects").
		Writes(lazyTestObject{}).
		Returns(http.StatusOK, "OK", lazyTestObject{}))
	c.Add(ws)
	return c
}

// lazyTestConfig returns an OpenAPI config whose GetDefinitions counts how
// often the spec is built. With withDefinitions false the build fails, since
// the route's type has no definition.
func lazyTestConfig(builds *int, mu *sync.Mutex, withDefinitions bool) *common.Config {
	return &common.Config{
		Info: &spec.Info{InfoProps: spec.InfoProps{Title: "lazytest", Version: "v1"}},
		GetDefinitions: func(common.ReferenceCallback) map[string]common.OpenAPIDefinition {
			mu.Lock()
			*builds++
			mu.Unlock()
			if !withDefinitions {
				return nil
			}
			return map[string]common.OpenAPIDefinition{
				"k8s.io/apiserver/pkg/server/routes.lazyTestObject": {
					Schema: spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}}},
				},
			}
		},
	}
}

func fetchLazyV2(t *testing.T, h http.Handler, ifNoneMatch string) (body []byte, etag string, status int) {
	t.Helper()
	server := httptest.NewServer(h)
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

// TestInstallV2BuildsOnceOnDemand verifies the spec is not built until
// first use, is built exactly once, and that the served content and ETag
// match what the eager service serves for the same spec.
func TestInstallV2BuildsOnceOnDemand(t *testing.T) {
	var builds int
	var mu sync.Mutex
	lazyMux := mux.NewPathRecorderMux("test")
	_, source := OpenAPI{Config: lazyTestConfig(&builds, &mu, true)}.InstallV2(lazyTestContainer(), lazyMux)
	if builds != 0 {
		t.Fatalf("expected no build before the first request, got %d", builds)
	}

	lazyBody, lazyEtag, status := fetchLazyV2(t, lazyMux, "")
	if status != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", status, lazyBody)
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

	// The served bytes and ETag are identical to the eager service for the
	// same spec.
	built, _, err := source.Get()
	if err != nil {
		t.Fatal(err)
	}
	eagerMux := http.NewServeMux()
	handler.NewOpenAPIService(built).RegisterOpenAPIVersionedService("/openapi/v2", eagerMux)
	eagerBody, eagerEtag, _ := fetchLazyV2(t, eagerMux, "")
	if !bytes.Equal(lazyBody, eagerBody) {
		t.Error("lazy service served different bytes than the eager service")
	}
	if lazyEtag == "" || lazyEtag != eagerEtag {
		t.Errorf("lazy service served a different ETag: lazy %q, eager %q", lazyEtag, eagerEtag)
	}
}

// TestInstallV2CachesBuildError verifies that a failed build is not
// retried: the endpoint serves 503 and the build runs once.
func TestInstallV2CachesBuildError(t *testing.T) {
	var builds int
	var mu sync.Mutex
	m := mux.NewPathRecorderMux("test")
	_, source := OpenAPI{Config: lazyTestConfig(&builds, &mu, false)}.InstallV2(lazyTestContainer(), m)

	for range 3 {
		if _, _, status := fetchLazyV2(t, m, ""); status != http.StatusServiceUnavailable {
			t.Fatalf("expected 503 for a broken spec, got %d", status)
		}
	}
	if builds != 1 {
		t.Errorf("expected the failed build to run once, got %d attempts", builds)
	}
	if _, _, err := source.Get(); err == nil {
		t.Error("expected Get to return the cached build error")
	}
}

// TestInstallV2ConcurrentFirstUse verifies concurrent first requests
// share a single build.
func TestInstallV2ConcurrentFirstUse(t *testing.T) {
	var builds int
	var mu sync.Mutex
	_, source := OpenAPI{Config: lazyTestConfig(&builds, &mu, true)}.InstallV2(lazyTestContainer(), mux.NewPathRecorderMux("test"))
	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() {
			if _, _, err := source.Get(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if builds != 1 {
		t.Errorf("expected one build under concurrency, got %d", builds)
	}
}
