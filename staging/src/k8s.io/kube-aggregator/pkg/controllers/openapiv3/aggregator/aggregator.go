/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/server"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
)

// SpecAggregator calls out to http handlers of APIServices and caches specs. It keeps state of the last
// known specs including the http etag.
// TODO(jefftree): remove the downloading and caching and proxy directly to the APIServices. This is possible because we
// don't have to merge here, which is cpu intensive in v2
type SpecAggregator interface {
	AddUpdateAPIService(handler http.Handler, apiService *v1.APIService)
	UpdateAPIServiceSpec(apiServiceName string) error
	RemoveAPIServiceSpec(apiServiceName string)
	GetAPIServiceNames() []string
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = 60 * time.Second
	localDelegateChainNamePrefix  = "k8s_internal_local_delegation_chain_"
	localDelegateChainNamePattern = localDelegateChainNamePrefix + "%010d"
)

// IsLocalAPIService returns true for local specs from delegates.
func IsLocalAPIService(apiServiceName string) bool {
	return strings.HasPrefix(apiServiceName, localDelegateChainNamePrefix)
}

// GetAPIServicesName returns the names of APIServices recorded in openAPIV3Specs.
// We use this function to pass the names of local APIServices to the controller in this package,
// so that the controller can periodically sync the OpenAPI spec from delegation API servers.
func (s *specAggregator) GetAPIServiceNames() []string {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	names := make([]string, len(s.openAPIV3Specs))
	for key := range s.openAPIV3Specs {
		names = append(names, key)
	}
	return names
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader Downloader, delegationTarget server.DelegationTarget, pathHandler common.PathHandlerByGroupVersion) (SpecAggregator, error) {
	var err error
	s := &specAggregator{
		openAPIV3Specs: map[string]*openAPIV3APIServiceInfo{},
		downloader:     downloader,
	}

	s.openAPIV3VersionedService, err = handler3.NewOpenAPIService(nil)
	if err != nil {
		return nil, err
	}
	err = s.openAPIV3VersionedService.RegisterOpenAPIV3VersionedService("/openapi/v3", pathHandler)
	if err != nil {
		return nil, err
	}

	i := 1
	for delegate := delegationTarget; delegate != nil; delegate = delegate.NextDelegate() {
		handler := delegate.UnprotectedHandler()
		if handler == nil {
			continue
		}

		apiServiceName := fmt.Sprintf(localDelegateChainNamePattern, i)
		localAPIService := v1.APIService{}
		localAPIService.Name = apiServiceName
		s.AddUpdateAPIService(handler, &localAPIService)
		s.UpdateAPIServiceSpec(apiServiceName)
		i++
	}

	return s, nil
}

// AddUpdateAPIService adds or updates the api service. It is thread safe.
func (s *specAggregator) AddUpdateAPIService(handler http.Handler, apiservice *v1.APIService) {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	// If the APIService is being updated, use the existing struct.
	if apiServiceInfo, ok := s.openAPIV3Specs[apiservice.Name]; ok {
		apiServiceInfo.apiService = *apiservice
		apiServiceInfo.handler = handler
	}
	s.openAPIV3Specs[apiservice.Name] = &openAPIV3APIServiceInfo{
		apiService: *apiservice,
		handler:    handler,
		specs:      make(map[string]*openAPIV3SpecInfo),
	}
}

// UpdateAPIServiceSpec updates all the OpenAPI v3 specs that the APIService serves.
// It is thread safe.
func (s *specAggregator) UpdateAPIServiceSpec(apiServiceName string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	apiService, exists := s.openAPIV3Specs[apiServiceName]
	if !exists {
		return fmt.Errorf("APIService %s does not exist for update", apiServiceName)
	}

	// Pass a list of old etags to the Downloader to prevent transfers if etags match
	etagList := make(map[string]string)
	for gv, specInfo := range apiService.specs {
		etagList[gv] = specInfo.etag
	}
	groups, err := s.downloader.Download(apiService.handler, etagList)
	if err != nil {
		return err
	}

	// Remove any groups that do not exist anymore
	for group := range s.openAPIV3Specs[apiServiceName].specs {
		if _, exists := groups[group]; !exists {
			s.openAPIV3VersionedService.DeleteGroupVersion(group)
			delete(s.openAPIV3Specs[apiServiceName].specs, group)
		}
	}

	for group, info := range groups {
		if info.spec == nil {
			continue
		}

		// If ETag has not changed, no update is necessary
		oldInfo, exists := s.openAPIV3Specs[apiServiceName].specs[group]
		if exists && oldInfo.etag == info.etag {
			continue
		}
		s.openAPIV3Specs[apiServiceName].specs[group] = &openAPIV3SpecInfo{
			spec: info.spec,
			etag: info.etag,
		}
		s.openAPIV3VersionedService.UpdateGroupVersion(group, info.spec)
	}
	return nil
}

type specAggregator struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// OpenAPI V3 specs by APIService name
	openAPIV3Specs map[string]*openAPIV3APIServiceInfo
	// provided for dynamic OpenAPI spec
	openAPIV3VersionedService *handler3.OpenAPIService

	// For downloading the OpenAPI v3 specs from apiservices
	downloader Downloader
}

var _ SpecAggregator = &specAggregator{}

type openAPIV3APIServiceInfo struct {
	apiService v1.APIService
	handler    http.Handler
	specs      map[string]*openAPIV3SpecInfo
}

type openAPIV3SpecInfo struct {
	spec *spec3.OpenAPI
	etag string
}

// RemoveAPIServiceSpec removes an api service from the OpenAPI map. If it does not exist, no error is returned.
// It is thread safe.
func (s *specAggregator) RemoveAPIServiceSpec(apiServiceName string) {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	if apiServiceInfo, ok := s.openAPIV3Specs[apiServiceName]; ok {
		for gv := range apiServiceInfo.specs {
			s.openAPIV3VersionedService.DeleteGroupVersion(gv)
		}
		delete(s.openAPIV3Specs, apiServiceName)
	}
}
