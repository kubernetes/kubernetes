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
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/emicklei/go-restful/v3"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/klog/v2"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/openapiconv"

	v2aggregator "k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator"
)

var ErrAPIServiceNotFound = errors.New("resource not found")

// SpecProxier proxies OpenAPI V3 requests to their respective APIService
type SpecProxier interface {
	AddUpdateAPIService(handler http.Handler, apiService *v1.APIService)
	// UpdateAPIServiceSpec updates the APIService. It returns ErrAPIServiceNotFound if the APIService doesn't exist.
	UpdateAPIServiceSpec(apiServiceName string) error
	RemoveAPIServiceSpec(apiServiceName string)
	GetAPIServiceNames() []string
}

const (
	aggregatorUser                = "system:aggregator"
	specDownloadTimeout           = 60 * time.Second
	localDelegateChainNamePrefix  = "k8s_internal_local_delegation_chain_"
	localDelegateChainNamePattern = localDelegateChainNamePrefix + "%010d"
	openAPIV2Converter            = "openapiv2converter"
)

// IsLocalAPIService returns true for local specs from delegates.
func IsLocalAPIService(apiServiceName string) bool {
	return strings.HasPrefix(apiServiceName, localDelegateChainNamePrefix)
}

// GetAPIServiceNames returns the names of APIServices recorded in apiServiceInfo.
// We use this function to pass the names of local APIServices to the controller in this package,
// so that the controller can periodically sync the OpenAPI spec from delegation API servers.
func (s *specProxier) GetAPIServiceNames() []string {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	names := make([]string, 0, len(s.apiServiceInfo))
	for key := range s.apiServiceInfo {
		names = append(names, key)
	}
	return names
}

// BuildAndRegisterAggregator registered OpenAPI aggregator handler. This function is not thread safe as it only being called on startup.
func BuildAndRegisterAggregator(downloader Downloader, delegationTarget server.DelegationTarget, aggregatorService *restful.Container, openAPIConfig *common.OpenAPIV3Config, pathHandler common.PathHandlerByGroupVersion) (SpecProxier, error) {
	s := &specProxier{
		apiServiceInfo: map[string]*openAPIV3APIServiceInfo{},
		downloader:     downloader,
	}

	if aggregatorService != nil && openAPIConfig != nil {
		// Make native types exposed by aggregator available to the aggregated
		// OpenAPI (normal handle is disabled by skipOpenAPIInstallation option)
		aggregatorLocalServiceName := "k8s_internal_local_kube_aggregator_types"
		v3Mux := mux.NewPathRecorderMux(aggregatorLocalServiceName)
		_ = routes.OpenAPI{
			V3Config: openAPIConfig,
		}.InstallV3(aggregatorService, v3Mux)

		s.AddUpdateAPIService(v3Mux, &v1.APIService{
			ObjectMeta: metav1.ObjectMeta{
				Name: aggregatorLocalServiceName,
			},
		})
		s.UpdateAPIServiceSpec(aggregatorLocalServiceName)
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

	handler := handler3.NewOpenAPIService()
	s.openAPIV2ConverterHandler = handler
	openAPIV2ConverterMux := mux.NewPathRecorderMux(openAPIV2Converter)
	s.openAPIV2ConverterHandler.RegisterOpenAPIV3VersionedService("/openapi/v3", openAPIV2ConverterMux)
	openAPIV2ConverterAPIService := v1.APIService{}
	openAPIV2ConverterAPIService.Name = openAPIV2Converter
	s.AddUpdateAPIService(openAPIV2ConverterMux, &openAPIV2ConverterAPIService)
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
		return
	}
	s.apiServiceInfo[apiservice.Name] = &openAPIV3APIServiceInfo{
		apiService: *apiservice,
		handler:    handler,
	}
}

func getGroupVersionStringFromAPIService(apiService v1.APIService) string {
	if apiService.Spec.Group == "" && apiService.Spec.Version == "" {
		return ""
	}
	return "apis/" + apiService.Spec.Group + "/" + apiService.Spec.Version
}

// UpdateAPIServiceSpec updates all the OpenAPI v3 specs that the APIService serves.
// It is thread safe.
func (s *specProxier) UpdateAPIServiceSpec(apiServiceName string) error {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	return s.updateAPIServiceSpecLocked(apiServiceName)
}

func (s *specProxier) updateAPIServiceSpecLocked(apiServiceName string) error {
	apiService, exists := s.apiServiceInfo[apiServiceName]
	if !exists {
		return ErrAPIServiceNotFound
	}

	if !apiService.isLegacyAPIService {
		gv, httpStatus, err := s.downloader.OpenAPIV3Root(apiService.handler)
		if err != nil {
			return err
		}
		if httpStatus == http.StatusNotFound {
			apiService.isLegacyAPIService = true
		} else {
			s.apiServiceInfo[apiServiceName].discovery = gv
			return nil
		}
	}

	newDownloader := v2aggregator.Downloader{}
	v2Spec, etag, httpStatus, err := newDownloader.Download(apiService.handler, apiService.etag)
	if err != nil {
		return err
	}
	apiService.etag = etag
	if httpStatus == http.StatusOK {
		v3Spec := openapiconv.ConvertV2ToV3(v2Spec)
		s.openAPIV2ConverterHandler.UpdateGroupVersion(getGroupVersionStringFromAPIService(apiService.apiService), v3Spec)
		s.updateAPIServiceSpecLocked(openAPIV2Converter)
	}
	return nil
}

type specProxier struct {
	// mutex protects all members of this struct.
	rwMutex sync.RWMutex

	// OpenAPI V3 specs by APIService name
	apiServiceInfo map[string]*openAPIV3APIServiceInfo

	// For downloading the OpenAPI v3 specs from apiservices
	downloader Downloader

	openAPIV2ConverterHandler *handler3.OpenAPIService
}

var _ SpecProxier = &specProxier{}

type openAPIV3APIServiceInfo struct {
	apiService v1.APIService
	handler    http.Handler
	discovery  *handler3.OpenAPIV3Discovery

	// These fields are only used if the /openapi/v3 endpoint is not served by an APIService
	// Legacy APIService indicates that an APIService does not support OpenAPI V3, and the OpenAPI V2
	// will be downloaded, converted to V3 (lossy), and served by the aggregator
	etag               string
	isLegacyAPIService bool
}

// RemoveAPIServiceSpec removes an api service from the OpenAPI map. If it does not exist, no error is returned.
// It is thread safe.
func (s *specProxier) RemoveAPIServiceSpec(apiServiceName string) {
	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()
	if apiServiceInfo, ok := s.apiServiceInfo[apiServiceName]; ok {
		s.openAPIV2ConverterHandler.DeleteGroupVersion(getGroupVersionStringFromAPIService(apiServiceInfo.apiService))
		_ = s.updateAPIServiceSpecLocked(openAPIV2Converter)
		delete(s.apiServiceInfo, apiServiceName)
	}
}

func (s *specProxier) getOpenAPIV3Root() handler3.OpenAPIV3Discovery {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()

	merged := handler3.OpenAPIV3Discovery{
		Paths: make(map[string]handler3.OpenAPIV3DiscoveryGroupVersion),
	}

	for _, apiServiceInfo := range s.apiServiceInfo {
		if apiServiceInfo.discovery == nil {
			continue
		}

		for key, item := range apiServiceInfo.discovery.Paths {
			merged.Paths[key] = item
		}
	}
	return merged
}

// handleDiscovery is the handler for OpenAPI V3 Discovery
func (s *specProxier) handleDiscovery(w http.ResponseWriter, r *http.Request) {
	merged := s.getOpenAPIV3Root()
	j, err := json.Marshal(&merged)
	if err != nil {
		w.WriteHeader(500)
		klog.Errorf("failed to created merged OpenAPIv3 discovery response: %s", err.Error())
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
		if apiServiceInfo.discovery == nil {
			continue
		}

		for key := range apiServiceInfo.discovery.Paths {
			if targetGV == key {
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
	handler.Handle("/openapi/v3", metrics.InstrumentHandlerFunc("GET",
		/* group = */ "",
		/* version = */ "",
		/* resource = */ "",
		/* subresource = */ "openapi/v3",
		/* scope = */ "",
		/* component = */ "",
		/* deprecated */ false,
		/* removedRelease */ "",
		http.HandlerFunc(s.handleDiscovery)))
	handler.HandlePrefix("/openapi/v3/", metrics.InstrumentHandlerFunc("GET",
		/* group = */ "",
		/* version = */ "",
		/* resource = */ "",
		/* subresource = */ "openapi/v3/",
		/* scope = */ "",
		/* component = */ "",
		/* deprecated */ false,
		/* removedRelease */ "",
		http.HandlerFunc(s.handleGroupVersion)))
}
