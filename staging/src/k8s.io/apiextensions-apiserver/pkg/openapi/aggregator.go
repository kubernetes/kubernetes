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

package openapi

import (
	"fmt"
	"sync"

	"github.com/go-openapi/spec"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/handler"
)

// AggregationManager is the interface between OpenAPI Aggregator service and a controller
// that manages CRD openapi spec aggregation
type AggregationManager interface {
	// AddUpdateLocalAPIService allows adding/updating local API service with nil handler and
	// nil Spec.Service. This function can be used for local dynamic OpenAPI spec aggregation
	// management (e.g. CRD)
	AddUpdateLocalAPIServiceSpec(name string, spec *spec.Swagger, etag string) error
	RemoveAPIServiceSpec(apiServiceName string) error
}

type specAggregator struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// Map of API Services' OpenAPI specs by their name
	openAPISpecs map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIService          *handler.OpenAPIService
	openAPIVersionedService *handler.OpenAPIService
}

var _ AggregationManager = &specAggregator{}

// NewAggregationManager constructs a specAggregator from input openAPIService, openAPIVersionedService and
// recorded static OpenAPI spec. The function returns an AggregationManager interface.
func NewAggregationManager(openAPIService, openAPIVersionedService *handler.OpenAPIService, staticSpec *spec.Swagger) (AggregationManager, error) {
	// openAPIVersionedService and deprecated openAPIService should be initialized together
	if (openAPIService == nil) != (openAPIVersionedService == nil) {
		return nil, fmt.Errorf("unexpected openapi service initialization error")
	}
	return &specAggregator{
		openAPISpecs: map[string]*openAPISpecInfo{
			"initial_static_spec": {
				spec: staticSpec,
			},
		},
		openAPIService:          openAPIService,
		openAPIVersionedService: openAPIVersionedService,
	}, nil
}

// openAPISpecInfo is used to store OpenAPI spec with its priority.
// It can be used to sort specs with their priorities.
type openAPISpecInfo struct {
	// Name of a registered ApiService
	name string

	// Specification of this API Service. If null then the spec is not loaded yet.
	spec *spec.Swagger
	etag string
}

// buildOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe. The caller is responsible to hold proper locks.
func (s *specAggregator) buildOpenAPISpec() (specToReturn *spec.Swagger, err error) {
	specs := []openAPISpecInfo{}
	for _, specInfo := range s.openAPISpecs {
		if specInfo.spec == nil {
			continue
		}
		specs = append(specs, *specInfo)
	}
	if len(specs) == 0 {
		return &spec.Swagger{}, nil
	}
	for _, specInfo := range specs {
		if specToReturn == nil {
			specToReturn, err = aggregator.CloneSpec(specInfo.spec)
			if err != nil {
				return nil, err
			}
			continue
		}
		mergeSpecs(specToReturn, specInfo.spec)
	}
	// Add minimum required keys if missing, to properly serve the OpenAPI spec
	// through apiextensions-apiserver HTTP handler. These keys will not be
	// aggregated to top-level OpenAPI spec (only paths and definitions will).
	// However these keys make the OpenAPI->proto serialization happy.
	if specToReturn.Info == nil {
		specToReturn.Info = &spec.Info{
			InfoProps: spec.InfoProps{
				Title: "Kubernetes",
			},
		}
	}
	if len(specToReturn.Swagger) == 0 {
		specToReturn.Swagger = "2.0"
	}
	return specToReturn, nil
}

// updateOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe. The caller is responsible to hold proper locks.
func (s *specAggregator) updateOpenAPISpec() error {
	if s.openAPIService == nil || s.openAPIVersionedService == nil {
		// openAPIVersionedService and deprecated openAPIService should be initialized together
		if !(s.openAPIService == nil && s.openAPIVersionedService == nil) {
			return fmt.Errorf("unexpected openapi service initialization error")
		}
		return nil
	}
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return err
	}
	// openAPIService.UpdateSpec and openAPIVersionedService.UpdateSpec read the same swagger spec
	// serially and update their local caches separately. Both endpoints will have same spec in
	// their caches if the caller is holding proper locks.
	err = s.openAPIService.UpdateSpec(specToServe)
	if err != nil {
		return err
	}
	return s.openAPIVersionedService.UpdateSpec(specToServe)
}

// tryUpdatingServiceSpecs tries updating openAPISpecs map with specified specInfo, and keeps the map intact
// if the update fails.
func (s *specAggregator) tryUpdatingServiceSpecs(specInfo *openAPISpecInfo) error {
	orgSpecInfo, exists := s.openAPISpecs[specInfo.name]
	s.openAPISpecs[specInfo.name] = specInfo
	if err := s.updateOpenAPISpec(); err != nil {
		if exists {
			s.openAPISpecs[specInfo.name] = orgSpecInfo
		} else {
			delete(s.openAPISpecs, specInfo.name)
		}
		return err
	}
	return nil
}

// tryDeleteServiceSpecs tries delete specified specInfo from openAPISpecs map, and keeps the map intact
// if the update fails.
func (s *specAggregator) tryDeleteServiceSpecs(apiServiceName string) error {
	orgSpecInfo, exists := s.openAPISpecs[apiServiceName]
	if !exists {
		return nil
	}
	delete(s.openAPISpecs, apiServiceName)
	if err := s.updateOpenAPISpec(); err != nil {
		s.openAPISpecs[apiServiceName] = orgSpecInfo
		return err
	}
	return nil
}

// AddUpdateLocalAPIService allows adding/updating local API service with nil handler and
// nil Spec.Service. This function can be used for local dynamic OpenAPI spec aggregation
// management (e.g. CRD)
func (s *specAggregator) AddUpdateLocalAPIServiceSpec(name string, spec *spec.Swagger, etag string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	return s.tryUpdatingServiceSpecs(&openAPISpecInfo{
		name: name,
		spec: spec,
		etag: etag,
	})
}

// RemoveAPIServiceSpec removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveAPIServiceSpec(apiServiceName string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	if _, existingService := s.openAPISpecs[apiServiceName]; !existingService {
		return nil
	}

	return s.tryDeleteServiceSpecs(apiServiceName)
}

// mergeSpecs simply adds source openapi spec to dest and ignores any path/definition
// conflicts because CRD openapi spec should not have conflict
func mergeSpecs(dest, source *spec.Swagger) {
	// Paths may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
	if source.Paths == nil {
		// If Path is nil, none of the model defined in Definitions is used and we
		// should not do anything.
		// NOTE: this should not happen for CRD specs, because we automatically construct
		// the Paths for CRD specs. We use utilruntime.HandleError to log this impossible
		// case
		utilruntime.HandleError(fmt.Errorf("unexpected CRD spec with empty Path: %v", *source))
		return
	}
	if dest.Paths == nil {
		dest.Paths = &spec.Paths{}
	}
	for k, v := range source.Definitions {
		if dest.Definitions == nil {
			dest.Definitions = spec.Definitions{}
		}
		dest.Definitions[k] = v
	}
	for k, v := range source.Paths.Paths {
		// PathItem may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
		if dest.Paths.Paths == nil {
			dest.Paths.Paths = map[string]spec.PathItem{}
		}
		dest.Paths.Paths[k] = v
	}
}
