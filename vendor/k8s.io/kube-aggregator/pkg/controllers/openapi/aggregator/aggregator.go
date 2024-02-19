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
	"strings"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful/v3"

	"k8s.io/klog/v2"

	"k8s.io/apiserver/pkg/server"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// SpecAggregator calls out to http handlers of APIServices and merges specs. It keeps state of the last
// known specs including the http etag.
type SpecAggregator interface {
	AddUpdateAPIService(handler http.Handler, apiService *v1.APIService) error
	UpdateAPIServiceSpec(apiServiceName string, spec *spec.Swagger, etag string) error
	RemoveAPIServiceSpec(apiServiceName string) error
	GetAPIServiceInfo(apiServiceName string) (handler http.Handler, etag string, exists bool)
	GetAPIServiceNames() []string
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = 60 * time.Second
	localDelegateChainNamePrefix  = "k8s_internal_local_delegation_chain_"
	localDelegateChainNamePattern = localDelegateChainNamePrefix + "%010d"

	// A randomly generated UUID to differentiate local and remote eTags.
	locallyGeneratedEtagPrefix = "\"6E8F849B434D4B98A569B9D7718876E9-"
)

// IsLocalAPIService returns true for local specs from delegates.
func IsLocalAPIService(apiServiceName string) bool {
	return strings.HasPrefix(apiServiceName, localDelegateChainNamePrefix)
}

// GetAPIServiceNames returns the names of APIServices recorded in specAggregator.openAPISpecs.
// We use this function to pass the names of local APIServices to the controller in this package,
// so that the controller can periodically sync the OpenAPI spec from delegation API servers.
func (s *specAggregator) GetAPIServiceNames() []string {
	names := make([]string, 0, len(s.openAPISpecs))
	for key := range s.openAPISpecs {
		names = append(names, key)
	}
	return names
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader *Downloader, delegationTarget server.DelegationTarget, webServices []*restful.WebService,
	config *common.Config, pathHandler common.PathHandler) (SpecAggregator, error) {
	s := &specAggregator{
		openAPISpecs: map[string]*openAPISpecInfo{},
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
			// ignore errors for the empty delegate we attach at the end the chain
			// atm the empty delegate returns 503 when the server hasn't been fully initialized
			// and the spec downloader only silences 404s
			if len(delegate.ListedPaths()) == 0 && delegate.NextDelegate() == nil {
				continue
			}
			return nil, err
		}
		if delegateSpec == nil {
			continue
		}
		s.addLocalSpec(delegateSpec, handler, fmt.Sprintf(localDelegateChainNamePattern, i), etag)
		i++
	}

	// Build initial spec to serve.
	klog.V(2).Infof("Building initial OpenAPI spec")
	defer func(start time.Time) {
		duration := time.Since(start)
		klog.V(2).Infof("Finished initial OpenAPI spec generation after %v", duration)

		regenerationCounter.With(map[string]string{"apiservice": "*", "reason": "startup"})
		regenerationDurationGauge.With(map[string]string{"reason": "startup"}).Set(duration.Seconds())
	}(time.Now())
	specToServe, err := s.buildOpenAPISpec()
	if err != nil {
		return nil, err
	}

	// Install handler
	s.openAPIVersionedService, err = handler.NewOpenAPIService(specToServe)
	if err != nil {
		return nil, err
	}
	err = s.openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", pathHandler)
	if err != nil {
		return nil, err
	}

	return s, nil
}

type specAggregator struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// Map of API Services' OpenAPI specs by their name
	openAPISpecs map[string]*openAPISpecInfo

	// provided for dynamic OpenAPI spec
	openAPIVersionedService *handler.OpenAPIService
}

var _ SpecAggregator = &specAggregator{}

// This function is not thread safe as it only being called on startup.
func (s *specAggregator) addLocalSpec(spec *spec.Swagger, localHandler http.Handler, name, etag string) {
	localAPIService := v1.APIService{}
	localAPIService.Name = name
	s.openAPISpecs[name] = &openAPISpecInfo{
		etag:       etag,
		apiService: localAPIService,
		handler:    localHandler,
		spec:       spec,
	}
}

// openAPISpecInfo is used to store OpenAPI spec with its priority.
// It can be used to sort specs with their priorities.
type openAPISpecInfo struct {
	apiService v1.APIService

	// Specification of this API Service. If null then the spec is not loaded yet.
	spec    *spec.Swagger
	handler http.Handler
	etag    string
}

// buildOpenAPISpec aggregates all OpenAPI specs.  It is not thread-safe. The caller is responsible to hold proper locks.
func (s *specAggregator) buildOpenAPISpec() (specToReturn *spec.Swagger, err error) {
	specs := []openAPISpecInfo{}
	for _, specInfo := range s.openAPISpecs {
		if specInfo.spec == nil {
			continue
		}
		// Copy the spec before removing the defaults.
		localSpec := *specInfo.spec
		localSpecInfo := *specInfo
		localSpecInfo.spec = &localSpec
		localSpecInfo.spec.Definitions = handler.PruneDefaults(specInfo.spec.Definitions)
		specs = append(specs, localSpecInfo)
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

// tryUpdatingServiceSpecs tries updating openAPISpecs map with specified specInfo, and keeps the map intact
// if the update fails.
func (s *specAggregator) tryUpdatingServiceSpecs(specInfo *openAPISpecInfo) error {
	if specInfo == nil {
		return fmt.Errorf("invalid input: specInfo must be non-nil")
	}
	_, updated := s.openAPISpecs[specInfo.apiService.Name]
	origSpecInfo, existedBefore := s.openAPISpecs[specInfo.apiService.Name]
	s.openAPISpecs[specInfo.apiService.Name] = specInfo

	// Skip aggregation if OpenAPI spec didn't change
	if existedBefore && origSpecInfo != nil && origSpecInfo.etag == specInfo.etag {
		return nil
	}
	klog.V(2).Infof("Updating OpenAPI spec because %s is updated", specInfo.apiService.Name)
	defer func(start time.Time) {
		duration := time.Since(start)
		klog.V(2).Infof("Finished OpenAPI spec generation after %v", duration)

		reason := "add"
		if updated {
			reason = "update"
		}

		regenerationCounter.With(map[string]string{"apiservice": specInfo.apiService.Name, "reason": reason})
		regenerationDurationGauge.With(map[string]string{"reason": reason}).Set(duration.Seconds())
	}(time.Now())
	if err := s.updateOpenAPISpec(); err != nil {
		if existedBefore {
			s.openAPISpecs[specInfo.apiService.Name] = origSpecInfo
		} else {
			delete(s.openAPISpecs, specInfo.apiService.Name)
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
	klog.V(2).Infof("Updating OpenAPI spec because %s is removed", apiServiceName)
	defer func(start time.Time) {
		duration := time.Since(start)
		klog.V(2).Infof("Finished OpenAPI spec generation after %v", duration)

		regenerationCounter.With(map[string]string{"apiservice": apiServiceName, "reason": "delete"})
		regenerationDurationGauge.With(map[string]string{"reason": "delete"}).Set(duration.Seconds())
	}(time.Now())
	if err := s.updateOpenAPISpec(); err != nil {
		s.openAPISpecs[apiServiceName] = orgSpecInfo
		return err
	}
	return nil
}

// UpdateAPIServiceSpec updates the api service's OpenAPI spec. It is thread safe.
func (s *specAggregator) UpdateAPIServiceSpec(apiServiceName string, spec *spec.Swagger, etag string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	specInfo, existingService := s.openAPISpecs[apiServiceName]
	if !existingService {
		return fmt.Errorf("APIService %q does not exists", apiServiceName)
	}

	// For APIServices (non-local) specs, only merge their /apis/ prefixed endpoint as it is the only paths
	// proxy handler delegates.
	if specInfo.apiService.Spec.Service != nil {
		spec = aggregator.FilterSpecByPathsWithoutSideEffects(spec, []string{"/apis/"})
	}

	return s.tryUpdatingServiceSpecs(&openAPISpecInfo{
		apiService: specInfo.apiService,
		spec:       spec,
		handler:    specInfo.handler,
		etag:       etag,
	})
}

// AddUpdateAPIService adds or updates the api service. It is thread safe.
func (s *specAggregator) AddUpdateAPIService(handler http.Handler, apiService *v1.APIService) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	if apiService.Spec.Service == nil {
		// All local specs should be already aggregated using local delegate chain
		return nil
	}

	newSpec := &openAPISpecInfo{
		apiService: *apiService,
		handler:    handler,
	}
	if specInfo, existingService := s.openAPISpecs[apiService.Name]; existingService {
		newSpec.etag = specInfo.etag
		newSpec.spec = specInfo.spec
	}
	return s.tryUpdatingServiceSpecs(newSpec)
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

// GetAPIServiceSpec returns api service spec info
func (s *specAggregator) GetAPIServiceInfo(apiServiceName string) (handler http.Handler, etag string, exists bool) {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	if info, existingService := s.openAPISpecs[apiServiceName]; existingService {
		return info.handler, info.etag, true
	}
	return nil, "", false
}
