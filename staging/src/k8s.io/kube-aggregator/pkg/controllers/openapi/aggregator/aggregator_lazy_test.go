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
	"net/http"
	"testing"

	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// TestLazyAggregatorSpecBuildsOnDemand verifies that when the aggregator's
// own spec is provided through a lazy source, it is not built until the
// aggregated spec is actually served, and is built only once.
func TestLazyAggregatorSpecBuildsOnDemand(t *testing.T) {
	mux := http.NewServeMux()
	downloader := NewDownloader()

	var builds int
	lazySpec := routes.NewLazyOpenAPIV2SpecSource(func() (*spec.Swagger, error) {
		builds++
		return &spec.Swagger{
			SwaggerProps: spec.SwaggerProps{
				Paths: &spec.Paths{
					Paths: map[string]spec.PathItem{
						"/apis/apiregistration.k8s.io/v1/": {},
					},
				},
			},
		}, nil
	})

	delegationHandlers := []http.Handler{
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

	buildAndRegisterSpecAggregatorForLocalServices(&downloader, lazySpec, delegationHandlers, mux)
	if builds != 0 {
		t.Fatalf("expected no build before the aggregated spec is served, got %d", builds)
	}

	swagger, err := fetchOpenAPI(mux)
	if err != nil {
		t.Fatal(err)
	}
	if builds != 1 {
		t.Fatalf("expected exactly one build after serving, got %d", builds)
	}
	expectPath(t, swagger, "/apis/apiregistration.k8s.io/v1/")
	expectPath(t, swagger, "/apis/foo/v1/")

	if _, err := fetchOpenAPI(mux); err != nil {
		t.Fatal(err)
	}
	if builds != 1 {
		t.Errorf("expected no rebuild on subsequent serving, got %d builds", builds)
	}
}
