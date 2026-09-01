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
	"sync"

	restful "github.com/emicklei/go-restful/v3"
	"k8s.io/klog/v2"

	"k8s.io/apiserver/pkg/server/mux"
	builder2 "k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// lazySpecSource is a cached.Value[*spec.Swagger] that builds the spec on
// first use. A successful build is cached for the lifetime of the source and
// the builder function is released so anything it captured (route containers,
// OpenAPI config) can be garbage collected. A failed build is NOT cached: the
// error is returned and the build is retried on the next Get, so a transient
// failure does not poison the endpoint. The mutex doubles as singleflight:
// concurrent first requests share one build.
type lazySpecSource struct {
	mu    sync.Mutex
	build func() (*spec.Swagger, error)
	spec  *spec.Swagger
}

// NewLazyOpenAPIV2SpecSource returns a cached spec source that invokes build
// on first use, with the semantics described on lazySpecSource.
func NewLazyOpenAPIV2SpecSource(build func() (*spec.Swagger, error)) cached.Value[*spec.Swagger] {
	return &lazySpecSource{build: build}
}

func (s *lazySpecSource) Get() (*spec.Swagger, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.spec != nil {
		return s.spec, "lazy-openapi-v2-built", nil
	}
	klog.V(2).Info("Building OpenAPI v2 spec on first use")
	built, err := s.build()
	if err != nil {
		return nil, "", err
	}
	s.spec = built
	// Release the builder so captured references (route container, OpenAPI
	// config) become collectable.
	s.build = nil
	return s.spec, "lazy-openapi-v2-built", nil
}

// InstallV2Lazy is the lazy-build variant of InstallV2, used when the
// OpenAPIV2LazyBuild feature gate is enabled. The /openapi/v2 endpoint is
// registered immediately, but the spec is only built on the first request
// (or the first programmatic use of the returned service). Once built, the
// served content, ETags and update semantics are identical to InstallV2.
// Build errors surface as 503s on the endpoint and are retried on subsequent
// requests instead of terminating the server. No parsed base spec is
// returned; callers needing one must build their own copy.
func (oa OpenAPI) InstallV2Lazy(c *restful.Container, mux *mux.PathRecorderMux) *handler.OpenAPIService {
	source := NewLazyOpenAPIV2SpecSource(func() (*spec.Swagger, error) {
		spec, err := builder2.BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(c.RegisteredWebServices()), oa.Config)
		if err != nil {
			return nil, err
		}
		spec.Definitions = handler.PruneDefaults(spec.Definitions)
		return spec, nil
	})
	openAPIVersionedService := handler.NewOpenAPIServiceLazy(source)
	openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", mux)

	return openAPIVersionedService
}
