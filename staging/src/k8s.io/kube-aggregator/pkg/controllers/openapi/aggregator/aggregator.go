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
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var ErrNotFound = errors.New("resource not found")

// SpecAggregator calls out to http handlers of APIServices and merges specs. It keeps state of the last
// known specs including the http etag.
type SpecAggregator interface {
	AddAPIService(apiService *v1.APIService, handler http.Handler) error
	UpdateAPIService(apiServiceName string) error
	RemoveAPIService(apiServiceName string) error
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = time.Minute
	localDelegateChainNamePattern = "k8s_internal_local_delegation_chain_%010d"

	// A randomly generated UUID to differentiate local and remote eTags.
	locallyGeneratedEtagPrefix = "\"6E8F849B434D4B98A569B9D7718876E9-"
)

// openAPISpecInfo is used to store OpenAPI spec with its priority.
// It can be used to sort specs with their priorities.
type openAPISpecInfo struct {
	apiService v1.APIService
	// This is the cache used to merge the specs.
	spec cached.Replaceable[*spec.Swagger]
	// The downloader is used only for non-local apiservices to
	// re-update the spec every so often.
	downloader cached.Data[*spec.Swagger]
}

type specAggregator struct {
	mutex sync.Mutex

	// Map of API Services' OpenAPI specs by their name
	openAPISpecs map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *handler.OpenAPIService

	downloader *Downloader
}

// Creates a new cache that wraps the error with useful information for debugging.
func newDecoratedCache(name string, cache cached.Data[*spec.Swagger]) cached.Data[*spec.Swagger] {
	return cached.NewTransformer(func(result cached.Result[*spec.Swagger]) cached.Result[*spec.Swagger] {
		if result.Err != nil {
			return cached.NewResultErr[*spec.Swagger](fmt.Errorf("failed to download %v: %v", name, result.Err))
		}
		return result
	}, cache)
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader *Downloader, delegationTarget server.DelegationTarget, webServices []*restful.WebService,
	config *common.Config, pathHandler common.PathHandler) (SpecAggregator, error) {
	s := &specAggregator{
		downloader:   downloader,
		openAPISpecs: map[string]*openAPISpecInfo{},
	}

	i := 0

	aggregatorSpec := cached.NewStaticSource(func() cached.Result[*spec.Swagger] {
		aggregatorOpenAPISpec, err := builder.BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(webServices), config)
		if err != nil {
			return cached.NewResultErr[*spec.Swagger](fmt.Errorf("Failed to build aggregator spec: %v", err))
		}
		return cached.NewResultOK(aggregatorOpenAPISpec, "never-changes")
	})

	s.addLocalSpec(fmt.Sprintf(localDelegateChainNamePattern, i), aggregatorSpec)

	i++
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
		s.addLocalSpec(fmt.Sprintf(localDelegateChainNamePattern, i), newDecoratedCache(fmt.Sprintf(localDelegateChainNamePattern, i), NewCacheableDownloader(downloader, handler)))
		i++
	}
	s.openAPIVersionedService = handler.NewOpenAPIServiceLazy(s.buildMergeSpecLocked())

	// Install handler
	err := s.openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", pathHandler)
	if err != nil {
		return nil, err
	}

	return s, nil
}

func (s *specAggregator) addLocalSpec(name string, spec cached.Data[*spec.Swagger]) {
	service := v1.APIService{}
	service.Name = name
	info := &openAPISpecInfo{
		apiService: service,
	}
	info.spec.Replace(spec)
	s.openAPISpecs[name] = info
}

// buildMergeSpecLocked creates a new cached mergeSpec from the list of cached.
func (s *specAggregator) buildMergeSpecLocked() cached.Data[*spec.Swagger] {
	caches := make(map[string]cached.Data[*spec.Swagger], len(s.openAPISpecs))
	for name, specInfo := range s.openAPISpecs {
		caches[name] = cached.NewTransformer[*spec.Swagger](func(result cached.Result[*spec.Swagger]) cached.Result[*spec.Swagger] {
			if result.Err != nil {
				return result
			}
			// Make a copy of the data so that we don't modify the cache.
			swagger := *result.Data
			swagger.Definitions = handler.PruneDefaults(swagger.Definitions)
			return cached.NewResultOK(&swagger, result.Etag)
		}, &specInfo.spec)
	}
	return cached.NewMerger(func(results map[string]cached.Result[*spec.Swagger]) cached.Result[*spec.Swagger] {
		s.mutex.Lock()
		defer s.mutex.Unlock()
		specs := make([]openAPISpecInfo, 0, len(s.openAPISpecs))
		for _, specInfo := range s.openAPISpecs {
			specs = append(specs, *specInfo)
		}
		sortByPriority(specs)
		var merged *spec.Swagger
		etags := make(map[string]string, len(specs))
		for _, specInfo := range specs {
			result := results[specInfo.apiService.Name].Get()
			if result.Err != nil {
				return result
			}
			if merged == nil {
				merged = &spec.Swagger{}
				*merged = *result.Data
				merged.Paths = nil
				merged.Definitions = nil
			}
			etags[specInfo.apiService.Name] = result.Etag
			if err := aggregator.MergeSpecsIgnorePathConflict(merged, result.Data); err != nil {
				return cached.NewResultErr[*spec.Swagger](fmt.Errorf("Failed to build merge specs: %v", err))
			}
		}
		// Printing the etags map is stable.
		return cached.NewResultOK(merged, fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("%#v", etags)))))
	}, caches)
}

// updateServiceLocked updates the spec cache by downloading the latest
// version of the spec.
func (s *specAggregator) updateServiceLocked(name string) error {
	specInfo, exists := s.openAPISpecs[name]
	if !exists {
		return ErrNotFound
	}
	result := specInfo.downloader.Get()
	specInfo.spec.Replace(cached.NewTransformer[*spec.Swagger](func(result cached.Result[*spec.Swagger]) cached.Result[*spec.Swagger] {
		if result.Err != nil {
			return result
		}
		return cached.NewResultOK(aggregator.FilterSpecByPathsWithoutSideEffects(result.Data, []string{"/apis/"}), result.Etag)
	}, result))
	return result.Err
}

// UpdateAPIService updates the api services.. It is thread safe.
func (s *specAggregator) UpdateAPIService(apiServiceName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.updateServiceLocked(apiServiceName)
}

// AddAPIService adds the api service. It is thread safe. If the
// apiservice already exists, it will be updated.
func (s *specAggregator) AddAPIService(apiService *v1.APIService, handler http.Handler) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	_, exists := s.openAPISpecs[apiService.Name]
	if !exists {
		s.openAPISpecs[apiService.Name] = &openAPISpecInfo{
			apiService: *apiService,
			downloader: newDecoratedCache(apiService.Name, NewCacheableDownloader(s.downloader, handler)),
		}
		// Re-create the mergeSpec for the new list of apiservices
		s.openAPIVersionedService.UpdateSpecLazy(s.buildMergeSpecLocked())
	}

	return s.updateServiceLocked(apiService.Name)
}

// RemoveAPIService removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveAPIService(apiServiceName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if _, exists := s.openAPISpecs[apiServiceName]; !exists {
		return ErrNotFound
	}
	delete(s.openAPISpecs, apiServiceName)
	// Re-create the mergeSpec for the new list of apiservices
	s.openAPIVersionedService.UpdateSpecLazy(s.buildMergeSpecLocked())
	return nil
}
