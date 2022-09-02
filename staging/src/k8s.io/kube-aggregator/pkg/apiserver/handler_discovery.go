/*
Copyright 2016 The Kubernetes Authors.

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

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/v2"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	scheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper"
)

// Given a list of APIServices and proxyHandlers for contacting them,
// DiscoveryManager caches a list of discovery documents for each server

type DiscoveryAggregationController interface {
	// Adds or Updates an APIService from the Aggregated Discovery Controller's
	// knowledge base
	// Thread-safe
	AddAPIService(apiService *apiregistrationv1.APIService, handler http.Handler)

	// Removes an APIService from the Aggregated Discovery Controller's Knowledge
	// bank
	// Thread-safe
	RemoveAPIService(apiServiceName string)

	// Adds or Updates a local APIService from the Aggregated Discovery Controller's
	// knowledge base
	// Thread-safe
	AddLocalDelegate(name string, target genericapiserver.DelegationTarget)

	// Spwans a worker which waits for added/updated apiservices and updates
	// the unified discovery document by contacting the aggregated api services
	//
	// Blocks until local apiservices are populated or returns an error
	Run(stopCh <-chan struct{}) error
}

type discoveryManager struct {
	// Locks `services`
	servicesLock sync.RWMutex

	// Map from APIService's name (or a unique string for local servers)
	// to information about contacting that API Service
	apiServices map[string]apiServiceInfo

	// Map of local delegate names to a handler which can contact their API
	// surface
	localDelegates map[string]genericapiserver.DelegationTarget

	// Locks cachedResults
	resultsLock sync.RWMutex

	// Map from APIService.Spec.Service to the previously fetched value
	// (Note that many APIServices might use the same APIService.Spec.Service)
	cachedResults map[serviceKey]cachedResult

	// Queue of dirty apiServiceKey which need to be refreshed
	// It is important that the reconciler for this queue does not excessively
	// contact the apiserver if a key was enqueued before the server was last
	// contacted.
	dirtyAPIServiceQueue workqueue.RateLimitingInterface

	// Merged handler which stores all known groupversions
	mergedDiscoveryHandler discoveryendpoint.ResourceManager
}

type serviceKey struct {
	Namespace string
	Name      string
	Port      int32

	// If service is a local service, unique name identifying it
	// If LocalName is not empty, all other fields should be empty
	LocalName string
}

// Human-readable String representation used for logs
func (s serviceKey) String() string {
	if s.LocalName == "" {
		return fmt.Sprintf("%v/%v:%v", s.Namespace, s.Name, s.Port)
	}
	return s.LocalName
}

func newServiceKey(service apiregistrationv1.ServiceReference) serviceKey {
	// Docs say. Defaults to 443 for compatibility reasons.
	// BETA: Should this be a shared constant to avoid drifting with the
	// implementation?
	port := int32(443)
	if service.Port != nil {
		port = *service.Port
	}

	return serviceKey{
		Name:      service.Name,
		Namespace: service.Namespace,
		Port:      port,
	}
}

type cachedResult struct {
	// Currently cached discovery document for this service
	// Map from group name to version name to
	discovery map[metav1.GroupVersion]metav1.APIVersionDiscovery

	// ETag hash of the cached discoveryDocument
	etag string

	lastUpdated time.Time

	// results are stale if server was attempted to be contacted and it failed
	// if lastUpdated is beyond a certain threshold, and stale is true, the result
	// will be purged
	stale bool
}

// Information about a specific APIService/GroupVersion
type apiServiceInfo struct {
	// Date this APIService was marked dirty. Used for request deduplication
	lastMarkedDirty time.Time

	// ServiceReference for this APIService
	service serviceKey

	// Method for contacting the service
	handler http.Handler
}

var _ DiscoveryAggregationController = &discoveryManager{}

func NewDiscoveryManager(
	target discoveryendpoint.ResourceManager,
) DiscoveryAggregationController {
	return &discoveryManager{
		mergedDiscoveryHandler: target,
		apiServices:            make(map[string]apiServiceInfo),
		cachedResults:          make(map[serviceKey]cachedResult),
		localDelegates:         make(map[string]genericapiserver.DelegationTarget),
		dirtyAPIServiceQueue:   workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "discovery-manager"),
	}
}

// Try to sync a single APIService.
func (dm *discoveryManager) syncAPIService(apiServiceName string) error {
	dm.servicesLock.RLock()
	apiServiceInfo, exists := dm.apiServices[apiServiceName]
	dm.servicesLock.RUnlock()

	gv := helper.APIServiceNameToGroupVersion(apiServiceName)
	mgv := metav1.GroupVersion{Group: gv.Group, Version: gv.Version}

	if !exists {
		// apiservice was removed. remove it from merged discovery
		dm.mergedDiscoveryHandler.RemoveGroupVersion(mgv)
		return nil
	}

	cacheKey := apiServiceInfo.service

	// Lookup last cached result for this apiservice's service.
	dm.resultsLock.RLock()
	cached, exists := dm.cachedResults[cacheKey]
	dm.resultsLock.RUnlock()

	// De-deduplicate this request if the APIService was last marked dirty
	// before the cache entry of its service was last updated, then assume the
	// result would be the same (as a form of request de-duplication)
	if exists && !cached.stale && cached.lastUpdated.After(apiServiceInfo.lastMarkedDirty) {
		// Check cache entry for this groupversion and insert it into the
		// document if present
		if entry, exists := cached.discovery[mgv]; exists {
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
		} else {
			// Use empty GV, since there is an APIService for it
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
				Version: gv.Version,
			})
		}
		return nil
	}

	// If we have a handler to contact the server for this APIService, and
	// the cache entry is too old to use, refresh the cache entry now.
	handler := apiServiceInfo.handler
	handler = http.TimeoutHandler(handler, 5*time.Second, "request timed out")

	req, err := http.NewRequest("GET", "/discovery/v2", nil)
	if err != nil {
		// NewRequest should not fail, but if it does for some reason,
		// log it and continue
		return fmt.Errorf("failed to create http.Request: %v", err)

	}

	// Apply aggregator user proxy header to request
	// transport.SetAuthProxyHeaders(req, "system:kube-aggregator", []string{"system:masters"}, nil)
	req = req.WithContext(request.WithUser(req.Context(), &user.DefaultInfo{Name: "system:kube-aggregator"}))

	// req.Header.Add("Accept", runtime.ContentTypeProtobuf)
	req.Header.Add("Accept", runtime.ContentTypeJSON)

	if exists && len(cached.etag) > 0 {
		req.Header.Add("If-None-Match", cached.etag)
	}

	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	dm.resultsLock.Lock()
	defer dm.resultsLock.Unlock()

	switch writer.respCode {
	case http.StatusNotModified:
		// Keep old entry, update timestamp
		dm.cachedResults[cacheKey] = cachedResult{
			discovery:   cached.discovery,
			etag:        cached.etag,
			lastUpdated: time.Now(),
		}

		if entry, exists := cached.discovery[mgv]; exists {
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
		} else {
			// Use empty GV, since there is an APIService for it
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
				Version: gv.Version,
			})
		}

		return nil
	case http.StatusNotFound:
		// Discovery Document is not being served at all.
		// Fall back to legacy discovery information
		return errors.New("not found")

	case http.StatusOK:
		parsed := &metav1.APIGroupDiscoveryList{}
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), writer.data, parsed); err != nil {
			return err
		}

		klog.Infof("DiscoveryManager: Successfully downloaded discovery for %s", cacheKey.String())

		// Convert discovery info into a map for convenient lookup later
		discoMap := map[metav1.GroupVersion]metav1.APIVersionDiscovery{}
		for _, g := range parsed.Groups {
			for _, v := range g.Versions {
				discoMap[metav1.GroupVersion{Group: g.Name, Version: v.Version}] = v
			}
		}

		if entry, exists := discoMap[mgv]; exists {
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
		} else {
			// Use empty GV, since there is an APIService for it
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
				Version: gv.Version,
			})
		}

		// Save cached result
		dm.cachedResults[cacheKey] = cachedResult{
			discovery:   discoMap,
			etag:        writer.Header().Get("Etag"),
			lastUpdated: time.Now(),
		}
		return nil

	default:
		// Unhandled response. Mark information as stale.
		// Try again later.
		//!TODO: After a few tries, just wipe it out?
		// 	or after certain time?
		if !cached.stale {
			dm.cachedResults[cacheKey] = cachedResult{
				discovery:   cached.discovery,
				etag:        cached.etag,
				lastUpdated: time.Now(),
				stale:       true,
			}
		}

		// Re-use old entry for this GV
		if entry, exists := cached.discovery[mgv]; exists {
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
		} else {
			// Use empty GV, since there is an APIService for it
			dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
				Version: gv.Version,
			})
		}

		klog.Infof("DiscoveryManager: Failed to download discovery for %v: %v %s", cacheKey.String(), writer.respCode, writer.data)
		return fmt.Errorf("service %s returned non-success response code: %v", cacheKey.String(), writer.respCode)
	}
}

// Spwans a goroutune which waits for added/updated apiservices and updates
// the discovery document accordingly
func (dm *discoveryManager) Run(stopCh <-chan struct{}) error {
	klog.Info("Starting ResourceDiscoveryManager")

	var result error

	// Spawn workers
	for i := 0; i < 2; i++ {
		go func() {
			for {
				next, shutdown := dm.dirtyAPIServiceQueue.Get()
				if shutdown {
					return
				}

				if err := dm.syncAPIService(next.(string)); err != nil {
					dm.dirtyAPIServiceQueue.AddRateLimited(next)
				} else {
					dm.dirtyAPIServiceQueue.Forget(next)
				}
				dm.dirtyAPIServiceQueue.Done(next)
			}
		}()
	}

	// Refresh external Services every minute
	wait.PollImmediateUntil(1*time.Minute, func() (done bool, err error) {
		dm.servicesLock.Lock()
		defer dm.servicesLock.Unlock()

		now := time.Now()
		for key, info := range dm.apiServices {
			//!TODO: filter for external apiservices only
			info.lastMarkedDirty = now
			dm.apiServices[key] = info

			dm.dirtyAPIServiceQueue.Add(key)
		}

		return true, nil
	}, stopCh)

	return result
}

// Adds an APIService to be tracked by the discovery manager. If the APIService
// is already known
func (dm *discoveryManager) AddAPIService(apiService *apiregistrationv1.APIService, handler http.Handler) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	// If service is nil then its information is contained by a local APIService
	// but we can not know which.
	if apiService.Spec.Service == nil {
		gv := helper.APIServiceNameToGroupVersion(apiService.Name)

		// If the local delegate responds to /apis/<group>/version
		//	or /api/<version> (for empty group)
		// then we will use that as a service
		searchPath := "/apis/" + gv.Group + "/" + gv.Version
		if len(gv.Group) == 0 {
			searchPath = "/api/" + gv.Version
		}

		for name, delegate := range dm.localDelegates {
			listedPaths := delegate.ListedPaths()
			for _, path := range listedPaths {
				if path == searchPath {
					dm.apiServices[apiService.Name] = apiServiceInfo{
						handler:         delegate.UnprotectedHandler(),
						lastMarkedDirty: time.Now(),
						service: serviceKey{
							LocalName: name,
						},
					}
					dm.dirtyAPIServiceQueue.Add(apiService.Name)
					return
				}
			}
		}

		klog.Infof("Failed to find local service for apiservice: %s", apiService.Name)
	} else {
		// Add or update APIService record and mark it as dirty
		dm.apiServices[apiService.Name] = apiServiceInfo{
			handler:         handler,
			lastMarkedDirty: time.Now(),
			service:         newServiceKey(*apiService.Spec.Service),
		}
		dm.dirtyAPIServiceQueue.Add(apiService.Name)
	}
}

func (dm *discoveryManager) AddLocalDelegate(name string, target genericapiserver.DelegationTarget) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	dm.localDelegates[name] = target
}

func (dm *discoveryManager) RemoveAPIService(apiServiceName string) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	// Delete record of this groupversion if it exists
	delete(dm.apiServices, apiServiceName)

	dm.dirtyAPIServiceQueue.Add(apiServiceName)
}

// !TODO: This was copied from staging/src/k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator/downloader.go
// which was copied from staging/src/k8s.io/kube-aggregator/pkg/controllers/openapiv3/aggregator/downloader.go
// so we should find a home for this
// inMemoryResponseWriter is a http.Writer that keep the response in memory.
type inMemoryResponseWriter struct {
	writeHeaderCalled bool
	header            http.Header
	respCode          int
	data              []byte
}

func newInMemoryResponseWriter() *inMemoryResponseWriter {
	return &inMemoryResponseWriter{header: http.Header{}}
}

func (r *inMemoryResponseWriter) Header() http.Header {
	return r.header
}

func (r *inMemoryResponseWriter) WriteHeader(code int) {
	r.writeHeaderCalled = true
	r.respCode = code
}

func (r *inMemoryResponseWriter) Write(in []byte) (int, error) {
	if !r.writeHeaderCalled {
		r.WriteHeader(http.StatusOK)
	}
	r.data = append(r.data, in...)
	return len(in), nil
}

func (r *inMemoryResponseWriter) String() string {
	s := fmt.Sprintf("ResponseCode: %d", r.respCode)
	if r.data != nil {
		s += fmt.Sprintf(", Body: %s", string(r.data))
	}
	if r.header != nil {
		s += fmt.Sprintf(", Header: %s", r.header)
	}
	return s
}
