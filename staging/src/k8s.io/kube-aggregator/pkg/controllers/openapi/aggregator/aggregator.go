/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"net/http"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
)

// SpecAggregator calls out to http handlers of APIServices and merges specs. It keeps state of the last
// known specs including the http etag.
type SpecAggregator interface {
	AddUpdateService(name string, spec *spec.Swagger, services ...*apiregistration.APIService) error
	RemoveService(name string, services ...schema.GroupVersion) error

	UpdateSpec(name string, spec *spec.Swagger, etag string) error
	Spec(name string) (spec *spec.Swagger, etag string, exists bool)
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = 60 * time.Second
	localDelegateChainNamePattern = "k8s_internal_local_delegation_chain_%010d"

	// A randomly generated UUID to differentiate local and remote eTags.
	locallyGeneratedEtagPrefix = "\"6E8F849B434D4B98A569B9D7718876E9-"
)

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader *Downloader, delegationTarget server.DelegationTarget, webServices []*restful.WebService,
	config *common.Config, pathHandler common.PathHandler) (SpecAggregator, error) {
	s := &specAggregator{
		specs: map[string]*openAPISpecInfo{},
	}

	i := 0
	// Build Aggregator's spec
	aggregatorOpenAPISpec, err := builder.BuildOpenAPISpec(webServices, config)
	if err != nil {
		return nil, err
	}

	// Reserving non-name spec for aggregator's Spec.
	s.addLocalSpec(aggregatorOpenAPISpec, nil, fmt.Sprintf(localDelegateChainNamePattern, i), "")
	i++
	for delegate := delegationTarget; delegate != nil; delegate = delegate.NextDelegate() {
		handler := delegate.UnprotectedHandler()
		if handler == nil {
			continue
		}
		delegateSpec, etag, _, err := downloader.Download(handler, "")
		if err != nil {
			return nil, err
		}
		if delegateSpec == nil {
			continue
		}
		s.addLocalSpec(delegateSpec, handler, fmt.Sprintf(localDelegateChainNamePattern, i), etag)
		i++
	}

	// Build initial spec to serve.
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return nil, err
	}

	// Install handler
	s.openAPIVersionedService, err = handler.RegisterOpenAPIVersionedService(
		specToServe, "/openapi/v2", pathHandler)
	if err != nil {
		return nil, err
	}

	return s, nil
}

type specAggregator struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// specs shared by APIServices indexed by a unique name
	specs map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *handler.OpenAPIService
}

var _ SpecAggregator = &specAggregator{}

// This function is not thread safe as it only being called on startup.
func (s *specAggregator) addLocalSpec(spec *spec.Swagger, localHandler http.Handler, name, etag string) {
	localAPIService := apiregistration.APIService{}
	localAPIService.Name = name
	s.specs[name] = &openAPISpecInfo{
		etag:       etag,
		apiService: localAPIService,
		handler:    localHandler,
		spec:       spec,
	}
}

// openAPISpecInfo is used to store OpenAPI specs and the corresponding APIServices specified in the spec.
type openAPISpecInfo struct {
	// those APIServices which share the spec. An APIService can only be part of one openAPISpecInfo.
	apiServices []*apiregistration.APIService

	// specification of these API services. If null then the spec is not loaded yet.
	spec *spec.Swagger
	etag string
}

// buildOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe. The caller is responsible to hold proper locks.
func (s *specAggregator) buildOpenAPISpec() (specToReturn *spec.Swagger, err error) {
	specs := []openAPISpecInfo{}
	for _, specInfo := range s.specs {
		if specInfo.spec == nil {
			continue
		}
		specs = append(specs, *specInfo)
	}
	if len(specs) == 0 {
		return &spec.Swagger{}, nil
	}
	sortByPriority(specs)
	for _, specInfo := range specs {
		if specToReturn == nil {
			specToReturn = &spec.Swagger{}
			*specToReturn = *specInfo.spec
			// Paths and Definitions are set by MergeSpecsIgnorePathConflict
			specToReturn.Paths = nil
			specToReturn.Definitions = nil
		}
		if err := aggregator.MergeSpecsIgnorePathConflict(specToReturn, specInfo.spec); err != nil {
			return nil, err
		}
	}
	return specToReturn, nil
}

// updateOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe. The caller is responsible to hold proper locks.
func (s *specAggregator) updateOpenAPISpec() error {
	if s.openAPIVersionedService == nil {
		return nil
	}
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return err
	}
	return s.openAPIVersionedService.UpdateSpec(specToServe)
}

// tryUpdatingSpecInfo tries updating the specInfo under the given name. It restores the old
// info if the update fails.
func (s *specAggregator) tryUpdatingSpecInfo(name string, specInfo *openAPISpecInfo) error {
	orgSpecInfo, exists := s.specs[name]
	s.specs[name] = specInfo
	if err := s.updateOpenAPISpec(); err != nil {
		if exists {
			s.specs[name] = orgSpecInfo
		} else {
			delete(s.specs, name)
		}
		return err
	}
	return nil
}

// tryDeletingSpecInfo tries delete the specInfo with the given name. It restores the old value if the update fails.
func (s *specAggregator) tryDeletingSpecInfo(name string) error {
	orgSpecInfo, exists := s.specs[name]
	if !exists {
		return nil
	}
	delete(s.specs, name)
	if err := s.updateOpenAPISpec(); err != nil {
		s.specs[name] = orgSpecInfo
		return err
	}
	return nil
}

// UpdateSpec updates the OpenAPI spec for the given name. It is thread safe.
func (s *specAggregator) UpdateSpec(name string, spec *spec.Swagger, etag string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	specInfo, found := s.specs[name]
	if !found {
		return fmt.Errorf("OpenAPI spec for %q does not exists", name)
	}

	// For APIServices (non-local) specs, only merge their /apis/ prefixed endpoint as it is the only paths
	// proxy handler delegates.
	if specInfo.apiService.Spec.Service != nil {
		spec = aggregator.FilterSpecByPathsWithoutSideEffects(spec, []string{"/apis/"})
	}
	// TODO: use similar filtering to split the spec per APIService

	return s.tryUpdatingSpecInfo(name, &openAPISpecInfo{
		apiService: specInfo.apiServices,
		spec:       spec,
		etag:       etag,
	})
}

// AddUpdateSpec adds or updates the APIService to belong to the given name. If it was assigned
// to another name before, it is removed from there, potentially removing the whole spec.
// It is thread safe.
func (s *specAggregator) AddUpdateService(name string, spec *spec.Swagger, services ...*apiregistration.APIService) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	if apiService.Spec.Service == nil {
		// All local specs should be already aggregated using local delegate chain
		return nil
	}

	asdf

	newSpec := &openAPISpecInfo{
		apiServices: services,
	}
	if specInfo, existingService := s.specs[name]; existingService {
		newSpec.etag = specInfo.etag
		newSpec.spec = specInfo.spec
	}
	// TODOODODODODOODODODO remove from old
	return s.tryUpdatingServiceSpecs(name, newSpec)
}

// RemoveAPIServiceSpec removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveSpec(name string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	if _, existingService := s.specs[name]; !existingService {
		return nil
	}

	return s.tryDeletingSpecInfo(apiServiceName)
}

// Spec returns last known spec and etag.
func (s *specAggregator) Spec(name string) (spec *spec.Swagger, etag string, exists bool) {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	if info, existingService := s.specs[apiServiceName]; existingService {
		return info.spec, info.etag, true
	}
	return nil, "", false
}
