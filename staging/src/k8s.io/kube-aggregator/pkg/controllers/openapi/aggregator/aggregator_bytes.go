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
	"crypto/sha256"
	"fmt"
	"net/http"
	"sync"

	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/klog/v2"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// This file contains the bytes-mode implementation of SpecAggregator, used
// when the OpenAPIV2BytesCache feature is enabled. It mirrors the classic
// implementation in aggregator.go, with one difference: every cached value in
// the pipeline holds marshaled JSON spec bytes rather than parsed
// *spec.Swagger structures. Parsed specs exist only transiently, while
// filtering a downloaded spec or merging the sources, and become garbage
// once the result is marshaled. The served bytes and ETags are identical to
// the classic implementation.

// openAPIBytesSpecInfo is the bytes-mode counterpart of openAPISpecInfo.
type openAPIBytesSpecInfo struct {
	apiService v1.APIService
	// spec is the cached marshaled OpenAPI spec
	spec cached.LastSuccess[[]byte]

	// The downloader is used only for non-local apiservices to
	// re-update the spec every so often.
	// Calling Get() is not thread safe and should only be called by a single
	// thread via the openapi controller.
	downloader CacheableBytesDownloader
}

type specAggregatorBytes struct {
	// mutex protects the specsByAPIServiceName map and its contents.
	mutex sync.Mutex

	// Map of API Services' marshaled OpenAPI specs by their name
	specsByAPIServiceName map[string]*openAPIBytesSpecInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *routes.OpenAPIV2BytesService

	downloader *Downloader
}

func buildAndRegisterSpecAggregatorBytesForLocalServices(downloader *Downloader, aggregatorSpecJSON []byte, delegationHandlers []http.Handler, pathHandler common.PathHandler) *specAggregatorBytes {
	s := &specAggregatorBytes{
		downloader:            downloader,
		specsByAPIServiceName: map[string]*openAPIBytesSpecInfo{},
	}
	cachedAggregatorSpec := cached.Static(aggregatorSpecJSON, "never-changes")
	s.addLocalSpec(fmt.Sprintf(localDelegateChainNamePattern, 0), cachedAggregatorSpec)
	for i, handler := range delegationHandlers {
		name := fmt.Sprintf(localDelegateChainNamePattern, i+1)

		spec := NewCacheableBytesDownloader(name, downloader, handler)
		s.addLocalSpec(name, spec)
	}

	s.openAPIVersionedService = routes.NewOpenAPIV2BytesServiceLazy(s.buildMergeSpecLocked())
	s.openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", pathHandler)
	return s
}

func (s *specAggregatorBytes) addLocalSpec(name string, cachedSpec cached.Value[[]byte]) {
	service := v1.APIService{}
	service.Name = name
	info := &openAPIBytesSpecInfo{
		apiService: service,
	}
	info.spec.Store(cachedSpec)
	s.specsByAPIServiceName[name] = info
}

// buildMergeSpecLocked creates a new cached mergeSpec from the list of cached
// specs. Sources are parsed and the merged result marshaled inside the merge
// function, so no parsed spec outlives a (re-)merge.
func (s *specAggregatorBytes) buildMergeSpecLocked() cached.Value[[]byte] {
	apiServices := make([]*v1.APIService, 0, len(s.specsByAPIServiceName))
	for k := range s.specsByAPIServiceName {
		apiServices = append(apiServices, &s.specsByAPIServiceName[k].apiService)
	}
	sortByPriority(apiServices)
	caches := make([]cached.Value[[]byte], len(apiServices))
	for i, apiService := range apiServices {
		caches[i] = &(s.specsByAPIServiceName[apiService.Name].spec)
	}

	return cached.MergeList(func(results []cached.Result[[]byte]) ([]byte, string, error) {
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
			downloaded := &spec.Swagger{}
			if err := downloaded.UnmarshalJSON(result); err != nil {
				klog.Warningf("failed to unmarshal OpenAPI spec: %v", err)
				continue
			}
			if merged == nil {
				merged = &spec.Swagger{}
				*merged = *downloaded
				// Paths, Definitions and parameters are set by
				// MergeSpecsIgnorePathConflictRenamingDefinitionsAndParameters
				merged.Paths = nil
				merged.Definitions = nil
				merged.Parameters = nil
			}
			etags = append(etags, etag)
			if err := aggregator.MergeSpecsIgnorePathConflictRenamingDefinitionsAndParameters(merged, downloaded); err != nil {
				return nil, "", fmt.Errorf("failed to build merge specs: %w", err)
			}
		}
		// Printing the etags list is stable because it is sorted.
		mergedEtag := fmt.Sprintf("%x", sha256.Sum256(fmt.Appendf(nil, "%#v", etags)))
		if merged == nil {
			return nil, mergedEtag, nil
		}
		json, err := merged.MarshalJSON()
		if err != nil {
			return nil, "", fmt.Errorf("failed to marshal merged spec: %w", err)
		}
		return json, mergedEtag, nil
	}, caches)
}

// updateServiceLocked updates the spec cache by downloading the latest
// version of the spec.
func (s *specAggregatorBytes) updateServiceLocked(name string) error {
	specInfo, exists := s.specsByAPIServiceName[name]
	if !exists {
		return ErrAPIServiceNotFound
	}
	result, etag, err := specInfo.downloader.Get()
	filteredResult := cached.Transform[[]byte](func(result []byte, etag string, err error) ([]byte, string, error) {
		if err != nil {
			return nil, "", err
		}
		downloaded := &spec.Swagger{}
		if err := downloaded.UnmarshalJSON(result); err != nil {
			return nil, "", fmt.Errorf("failed to unmarshal OpenAPI spec: %w", err)
		}
		group := specInfo.apiService.Spec.Group
		version := specInfo.apiService.Spec.Version
		filtered := aggregator.FilterSpecByPathsWithoutSideEffects(downloaded, []string{"/apis/" + group + "/" + version + "/"})
		json, err := filtered.MarshalJSON()
		if err != nil {
			return nil, "", fmt.Errorf("failed to marshal filtered OpenAPI spec: %w", err)
		}
		return json, etag, nil
	}, cached.Result[[]byte]{Value: result, Etag: etag, Err: err})
	specInfo.spec.Store(filteredResult)
	return err
}

// UpdateAPIServiceSpec updates the api service. It is thread safe.
func (s *specAggregatorBytes) UpdateAPIServiceSpec(apiServiceName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.updateServiceLocked(apiServiceName)
}

// AddUpdateAPIService adds the api service. It is thread safe. If the
// apiservice already exists, it will be updated.
func (s *specAggregatorBytes) AddUpdateAPIService(apiService *v1.APIService, handler http.Handler) error {
	if apiService.Spec.Service == nil {
		return nil
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()

	existingSpec, exists := s.specsByAPIServiceName[apiService.Name]
	if !exists {
		specInfo := &openAPIBytesSpecInfo{
			apiService: *apiService,
			downloader: NewCacheableBytesDownloader(apiService.Name, s.downloader, handler),
		}
		specInfo.spec.Store(cached.Result[[]byte]{Err: fmt.Errorf("spec for apiservice %s is not yet available", apiService.Name)})
		s.specsByAPIServiceName[apiService.Name] = specInfo
		s.openAPIVersionedService.UpdateSpecLazyBytes(s.buildMergeSpecLocked())
	} else {
		existingSpec.apiService = *apiService
		existingSpec.downloader.UpdateHandler(handler)
	}

	return nil
}

// RemoveAPIService removes an api service from OpenAPI aggregation. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregatorBytes) RemoveAPIService(apiServiceName string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if _, exists := s.specsByAPIServiceName[apiServiceName]; !exists {
		return
	}
	delete(s.specsByAPIServiceName, apiServiceName)
	// Re-create the mergeSpec for the new list of apiservices
	s.openAPIVersionedService.UpdateSpecLazyBytes(s.buildMergeSpecLocked())
}
