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
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"k8s.io/apiserver/pkg/server"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/common"
)

// SpecProxier proxies OpenAPI V3 requests to their respective APIService
type SpecProxier interface {
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

// GetAPIServicesName returns the names of APIServices recorded in apiServiceInfo.
// We use this function to pass the names of local APIServices to the controller in this package,
// so that the controller can periodically sync the OpenAPI spec from delegation API servers.
func (s *specProxier) GetAPIServiceNames() []string {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	names := make([]string, len(s.apiServiceInfo))
	for key := range s.apiServiceInfo {
		names = append(names, key)
	}
	return names
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader Downloader, delegationTarget server.DelegationTarget, pathHandler common.PathHandlerByGroupVersion) (SpecProxier, error) {
	s := &specProxier{
		apiServiceInfo: map[string]*openAPIV3APIServiceInfo{},
		downloader:     downloader,
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
	s.register(pathHandler)

	return s, nil
}

// AddUpdateAPIService adds or updates the api service. It is thread safe.
func (s *specProxier) AddUpdateAPIService(handler http.Handler, apiservice *v1.APIService) {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	// If the APIService is being updated, use the existing struct.
	if apiServiceInfo, ok := s.apiServiceInfo[apiservice.Name]; ok {
		apiServiceInfo.apiService = *apiservice
		apiServiceInfo.handler = handler
	}
	s.apiServiceInfo[apiservice.Name] = &openAPIV3APIServiceInfo{
		apiService: *apiservice,
		handler:    handler,
	}
}

// UpdateAPIServiceSpec updates all the OpenAPI v3 specs that the APIService serves.
// It is thread safe.
func (s *specProxier) UpdateAPIServiceSpec(apiServiceName string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	apiService, exists := s.apiServiceInfo[apiServiceName]
	if !exists {
		return fmt.Errorf("APIService %s does not exist for update", apiServiceName)
	}

	gv, err := s.downloader.OpenAPIV3Root(apiService.handler)
	if err != nil {
		return err
	}
	s.apiServiceInfo[apiServiceName].gvList = gv
	return nil
}

type specProxier struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// OpenAPI V3 specs by APIService name
	apiServiceInfo map[string]*openAPIV3APIServiceInfo

	// For downloading the OpenAPI v3 specs from apiservices
	downloader Downloader
}

var _ SpecProxier = &specProxier{}

type openAPIV3APIServiceInfo struct {
	apiService v1.APIService
	handler    http.Handler
	gvList     []string
}

// RemoveAPIServiceSpec removes an api service from the OpenAPI map. If it does not exist, no error is returned.
// It is thread safe.
func (s *specProxier) RemoveAPIServiceSpec(apiServiceName string) {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	if _, ok := s.apiServiceInfo[apiServiceName]; ok {
		delete(s.apiServiceInfo, apiServiceName)
	}
}

// handleDiscovery is the handler for OpenAPI V3 Discovery
func (s *specProxier) handleDiscovery(w http.ResponseWriter, r *http.Request) {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	gvList := make(map[string]bool)
	for _, apiServiceInfo := range s.apiServiceInfo {
		for _, gv := range apiServiceInfo.gvList {
			gvList[gv] = true
		}
	}

	keys := make([]string, 0, len(gvList))
	for k := range gvList {
		keys = append(keys, k)
	}

	sort.Strings(keys)
	output := map[string][]string{"Paths": keys}
	j, err := json.Marshal(output)
	if err != nil {
		return
	}

	http.ServeContent(w, r, "/openapi/v3", time.Now(), bytes.NewReader(j))
}

// handleGroupVersion is the OpenAPI V3 handler for a specified group/version
func (s *specProxier) handleGroupVersion(w http.ResponseWriter, r *http.Request) {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	// TODO: Import this logic from kube-openapi instead of duplicating
	// URLs for OpenAPI V3 have the format /openapi/v3/<groupversionpath>
	// SplitAfterN with 4 yields ["", "openapi", "v3", <groupversionpath>]
	url := strings.SplitAfterN(r.URL.Path, "/", 4)
	targetGV := url[3]

	for _, apiServiceInfo := range s.apiServiceInfo {
		for _, gv := range apiServiceInfo.gvList {
			if targetGV == gv {
				apiServiceInfo.handler.ServeHTTP(w, r)
				return
			}
		}
	}
	// No group-versions match the desired request
	w.WriteHeader(404)
}

// Register registers the OpenAPI V3 Discovery and GroupVersion handlers
func (s *specProxier) register(handler common.PathHandlerByGroupVersion) {
	handler.Handle("/openapi/v3", http.HandlerFunc(s.handleDiscovery))
	handler.HandlePrefix("/openapi/v3/", http.HandlerFunc(s.handleGroupVersion))
}
