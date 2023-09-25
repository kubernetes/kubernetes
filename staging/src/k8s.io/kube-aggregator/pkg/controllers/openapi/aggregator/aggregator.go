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
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful/v3"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/klog/v2"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var ErrAPIServiceNotFound = errors.New("resource not found")

// SpecAggregator calls out to http handlers of APIServices and merges specs. It keeps state of the last
// known specs including the http etag.
type SpecAggregator interface {
	AddUpdateAPIService(apiService *v1.APIService, handler http.Handler) error
	// UpdateAPIServiceSpec updates the APIService. It returns ErrAPIServiceNotFound if the APIService doesn't exist.
	UpdateAPIServiceSpec(apiServiceName string) error
	RemoveAPIService(apiServiceName string)
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = time.Minute
	localDelegateChainNamePattern = "k8s_internal_local_delegation_chain_%010d"

	// A randomly generated UUID to differentiate local and remote eTags.
	locallyGeneratedEtagPrefix = "\"6E8F849B434D4B98A569B9D7718876E9-"
)

// openAPISpecInfo is used to store OpenAPI specs.
// The apiService object is used to sort specs with their priorities.
type openAPISpecInfo struct {
	apiService v1.APIService
	// spec is the cached OpenAPI spec
	spec cached.LastSuccess[*spec.Swagger]

	// The downloader is used only for non-local apiservices to
	// re-update the spec every so often.
	// Calling Get() is not thread safe and should only be called by a single
	// thread via the openapi controller.
	downloader CacheableDownloader
}

type specAggregator struct {
	// mutex protects the specsByAPIServiceName map and its contents.
	mutex sync.Mutex

	// Map of API Services' OpenAPI specs by their name
	specsByAPIServiceName map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *handler.OpenAPIService

	downloader *Downloader
}

func buildAndRegisterSpecAggregatorForLocalServices(downloader *Downloader, aggregatorSpec *spec.Swagger, delegationHandlers []http.Handler, pathHandler common.PathHandler) *specAggregator {
	s := &specAggregator{
		downloader:            downloader,
		specsByAPIServiceName: map[string]*openAPISpecInfo{},
	}
	cachedAggregatorSpec := cached.Static(aggregatorSpec, "never-changes")
	s.addLocalSpec(fmt.Sprintf(localDelegateChainNamePattern, 0), cachedAggregatorSpec)
	for i, handler := range delegationHandlers {
		name := fmt.Sprintf(localDelegateChainNamePattern, i+1)

		spec := NewCacheableDownloader(name, downloader, handler)
		s.addLocalSpec(name, spec)
	}

	s.openAPIVersionedService = handler.NewOpenAPIServiceLazy(s.buildMergeSpecLocked())
	s.openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", pathHandler)
	return s
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader *Downloader, delegationTarget server.DelegationTarget, webServices []*restful.WebService,
	config *common.Config, pathHandler common.PathHandler) (SpecAggregator, error) {

	aggregatorOpenAPISpec, err := builder.BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(webServices), config)
	if err != nil {
		return nil, err
	}
	aggregatorOpenAPISpec.Definitions = handler.PruneDefaults(aggregatorOpenAPISpec.Definitions)

	var delegationHandlers []http.Handler

	for delegate := delegationTarget; delegate != nil; delegate = delegate.NextDelegate() {
		handler := delegate.UnprotectedHandler()
		if handler == nil {
			continue
		}
		// ignore errors for the empty delegate we attach at the end the chain
		// atm the empty delegate returns 503 when the server hasn't been fully initialized
		// and the spec downloader only silences 404s
		if len(delegate.ListedPaths()) == 0 && delegate.NextDelegate() == nil {
			continue
		}
		delegationHandlers = append(delegationHandlers, handler)
	}
	s := buildAndRegisterSpecAggregatorForLocalServices(downloader, aggregatorOpenAPISpec, delegationHandlers, pathHandler)
	return s, nil
}

func (s *specAggregator) addLocalSpec(name string, cachedSpec cached.Value[*spec.Swagger]) {
	service := v1.APIService{}
	service.Name = name
	info := &openAPISpecInfo{
		apiService: service,
	}
	info.spec.Store(cachedSpec)
	s.specsByAPIServiceName[name] = info
}

// buildMergeSpecLocked creates a new cached mergeSpec from the list of cached specs.
func (s *specAggregator) buildMergeSpecLocked() cached.Value[*spec.Swagger] {
	apiServices := make([]*v1.APIService, 0, len(s.specsByAPIServiceName))
	for k := range s.specsByAPIServiceName {
		apiServices = append(apiServices, &s.specsByAPIServiceName[k].apiService)
	}
	sortByPriority(apiServices)
	caches := make([]cached.Value[*spec.Swagger], len(apiServices))
	for i, apiService := range apiServices {
		caches[i] = &(s.specsByAPIServiceName[apiService.Name].spec)
	}

	return cached.MergeList(func(results []cached.Result[*spec.Swagger]) (*spec.Swagger, string, error) {
		var merged *spec.Swagger
		etags := make([]string, 0, len(results))
		for _, specInfo := range results {
			result, etag, err := specInfo.Get()
			if err != nil {
				// APIService name and err message will be included in
				// the error message as part of decorateError
				klog.Warning(err)
				continue
			}
			if merged == nil {
				merged = &spec.Swagger{}
				*merged = *result
				// Paths, Definitions and parameters are set by
				// MergeSpecsIgnorePathConflictRenamingDefinitionsAndParameters
				merged.Paths = nil
				merged.Definitions = nil
				merged.Parameters = nil
			}
			etags = append(etags, etag)
			if err := aggregator.MergeSpecsIgnorePathConflictRenamingDefinitionsAndParameters(merged, result); err != nil {
				return nil, "", fmt.Errorf("failed to build merge specs: %v", err)
			}
		}
		// Printing the etags list is stable because it is sorted.
		return merged, fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("%#v", etags)))), nil
	}, caches)
}

// updateServiceLocked updates the spec cache by downloading the latest
// version of the spec.
func (s *specAggregator) updateServiceLocked(name string) error {
	specInfo, exists := s.specsByAPIServiceName[name]
	if !exists {
		return ErrAPIServiceNotFound
	}
	result, etag, err := specInfo.downloader.Get()
	filteredResult := cached.Transform[*spec.Swagger](func(result *spec.Swagger, etag string, err error) (*spec.Swagger, string, error) {
		if err != nil {
			return nil, "", err
		}
		return aggregator.FilterSpecByPathsWithoutSideEffects(result, []string{"/apis/"}), etag, nil
	}, cached.Result[*spec.Swagger]{Value: result, Etag: etag, Err: err})
	specInfo.spec.Store(filteredResult)
	return err
}

// UpdateAPIServiceSpec updates the api service. It is thread safe.
func (s *specAggregator) UpdateAPIServiceSpec(apiServiceName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.updateServiceLocked(apiServiceName)
}

// AddUpdateAPIService adds the api service. It is thread safe. If the
// apiservice already exists, it will be updated.
func (s *specAggregator) AddUpdateAPIService(apiService *v1.APIService, handler http.Handler) error {
	if apiService.Spec.Service == nil {
		return nil
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()

	existingSpec, exists := s.specsByAPIServiceName[apiService.Name]
	if !exists {
		specInfo := &openAPISpecInfo{
			apiService: *apiService,
			downloader: NewCacheableDownloader(apiService.Name, s.downloader, handler),
		}
		specInfo.spec.Store(cached.Result[*spec.Swagger]{Err: fmt.Errorf("spec for apiservice %s is not yet available", apiService.Name)})
		s.specsByAPIServiceName[apiService.Name] = specInfo
		s.openAPIVersionedService.UpdateSpecLazy(s.buildMergeSpecLocked())
	} else {
		existingSpec.apiService = *apiService
		existingSpec.downloader.UpdateHandler(handler)
	}

	return nil
}

// RemoveAPIService removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveAPIService(apiServiceName string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if _, exists := s.specsByAPIServiceName[apiServiceName]; !exists {
		return
	}
	delete(s.specsByAPIServiceName, apiServiceName)
	// Re-create the mergeSpec for the new list of apiservices
	s.openAPIVersionedService.UpdateSpecLazy(s.buildMergeSpecLocked())
}
