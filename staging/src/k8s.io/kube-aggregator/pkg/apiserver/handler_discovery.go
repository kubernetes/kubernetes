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
	"k8s.io/apiserver/pkg/endpoints"
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
	Run(stopCh <-chan struct{})

	// Returns true if all GVs of local APIServices have been added as an
	// APIService and synced to the discovery document
	LocalServicesSynced() bool

	// Returns true if all non-local APIServices that have been added
	// are synced at least once to the discovery document
	ExternalServicesSynced() bool
}

type discoveryManager struct {
	// Locks `services`
	servicesLock sync.RWMutex

	// Map from APIService's name (or a unique string for local servers)
	// to information about contacting that API Service
	apiServices map[string]groupVersionInfo

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

	// Guaranteed to be a time less than the time the server responded with the
	// discovery data.
	lastUpdated time.Time

	// results are stale if server was attempted to be contacted and it failed
	// if lastUpdated is beyond a certain threshold, and stale is true, the result
	// will be purged
	stale bool
}

// Information about a specific APIService/GroupVersion
type groupVersionInfo struct {
	// Date this APIService was marked dirty.
	// Guaranteed to be a time greater than the most recent time the APIService
	// was known to be modified.
	//
	// Used for request deduplication to ensure the data used to reconcile each
	// apiservice was retrieved after the time of the APIService change:
	// real_apiservice_change_time < groupVersionInfo.lastMarkedDirty < cachedResult.lastUpdated < real_document_fresh_time
	//
	// This ensures that if the apiservice was changed after the last cached entry
	// was stored, the discovery document will always be re-fetched.
	lastMarkedDirty time.Time

	// Last time sync funciton was run for this GV.
	lastReconciled time.Time

	// ServiceReference of this GroupVersion. This identifies the Service which
	// describes how to contact the server responsible for this GroupVersion.
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
		apiServices:            make(map[string]groupVersionInfo),
		cachedResults:          make(map[serviceKey]cachedResult),
		localDelegates:         make(map[string]genericapiserver.DelegationTarget),
		dirtyAPIServiceQueue:   workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "discovery-manager"),
	}
}

// Returns discovery data for the given apiservice.
// Caches the result.
// Returns the cached result if it is retrieved after the apiservice was last
// marked dirty
func (dm *discoveryManager) fetchFreshDiscoveryForService(gv metav1.GroupVersion, info groupVersionInfo) (cachedResult, error) {
	// Lookup last cached result for this apiservice's service.
	dm.resultsLock.RLock()
	cached, exists := dm.cachedResults[info.service]
	dm.resultsLock.RUnlock()

	// If entry exists and was updated after the given time, just stop now
	if exists && !cached.stale && cached.lastUpdated.After(info.lastMarkedDirty) {
		return cached, nil
	}

	// If we have a handler to contact the server for this APIService, and
	// the cache entry is too old to use, refresh the cache entry now.
	handler := http.TimeoutHandler(info.handler, 5*time.Second, "request timed out")
	req, err := http.NewRequest("GET", "/discovery/v2", nil)
	if err != nil {
		// NewRequest should not fail, but if it does for some reason,
		// log it and continue
		return cached, fmt.Errorf("failed to create http.Request: %v", err)
	}

	// Apply aggregator user to request
	req = req.WithContext(
		request.WithUser(
			req.Context(), &user.DefaultInfo{Name: "system:kube-aggregator"}))

	// req.Header.Add("Accept", runtime.ContentTypeProtobuf)
	req.Header.Add("Accept", runtime.ContentTypeJSON)

	if exists && len(cached.etag) > 0 {
		req.Header.Add("If-None-Match", cached.etag)
	}

	// Important that the time recorded in the data's "lastUpdated" is conservatively
	// from BEFORE the request is dispatched so that lastUpdated can be used to
	// de-duplicate requests.
	now := time.Now()
	writer := newInMemoryResponseWriter()
	handler.ServeHTTP(writer, req)

	switch writer.respCode {
	case http.StatusNotModified:
		dm.resultsLock.Lock()
		defer dm.resultsLock.Unlock()

		// Keep old entry, update timestamp
		cached = cachedResult{
			discovery:   cached.discovery,
			etag:        cached.etag,
			lastUpdated: now,
		}
		dm.cachedResults[info.service] = cached

		return cached, nil
	case http.StatusNotFound:
		// Discovery Document is not being served at all.
		// Fall back to legacy discovery information
		if len(gv.Version) == 0 {
			return cached, errors.New("not found")
		}

		var path string
		if len(gv.Group) == 0 {
			path = "/api/" + gv.Version
		} else {
			path = "/apis/" + gv.Group + "/" + gv.Version
		}

		req, err := http.NewRequest("GET", path, nil)
		if err != nil {
			// NewRequest should not fail, but if it does for some reason,
			// log it and continue
			return cached, fmt.Errorf("failed to create http.Request: %v", err)
		}

		// Apply aggregator user to request
		req = req.WithContext(
			request.WithUser(
				req.Context(), &user.DefaultInfo{Name: "system:kube-aggregator"}))

		// req.Header.Add("Accept", runtime.ContentTypeProtobuf)
		req.Header.Add("Accept", runtime.ContentTypeJSON)

		if exists && len(cached.etag) > 0 {
			req.Header.Add("If-None-Match", cached.etag)
		}

		writer := newInMemoryResponseWriter()
		handler.ServeHTTP(writer, req)

		if writer.respCode != http.StatusOK {
			return cached, fmt.Errorf("failed to download discovery for %s: %v", path, writer.String())
		}

		parsed := &metav1.APIResourceList{}
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), writer.data, parsed); err != nil {
			return cached, err
		}

		// Create a discomap with single group-version
		discoMap := map[metav1.GroupVersion]metav1.APIVersionDiscovery{
			// Convert old-style APIGroupList to new information
			gv: {
				Version:   gv.Version,
				Resources: endpoints.ConvertGroupVersionIntoToDiscovery(parsed.APIResources),
			},
		}

		cached = cachedResult{
			discovery:   discoMap,
			lastUpdated: now,
		}

		// Don't bother saving result, there is no ETag support on this endpoint
		dm.cachedResults[info.service] = cached

		return cached, nil

	case http.StatusOK:

		parsed := &metav1.APIGroupDiscoveryList{}
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), writer.data, parsed); err != nil {
			return cached, err
		}

		klog.Infof("DiscoveryManager: Successfully downloaded discovery for %s", info.service.String())

		// Convert discovery info into a map for convenient lookup later
		discoMap := map[metav1.GroupVersion]metav1.APIVersionDiscovery{}
		for _, g := range parsed.Groups {
			for _, v := range g.Versions {
				discoMap[metav1.GroupVersion{Group: g.Name, Version: v.Version}] = v
			}
		}

		dm.resultsLock.Lock()
		defer dm.resultsLock.Unlock()

		// Save cached result
		cached = cachedResult{
			discovery:   discoMap,
			etag:        writer.Header().Get("Etag"),
			lastUpdated: now,
		}
		dm.cachedResults[info.service] = cached
		return cached, nil

	default:
		dm.resultsLock.Lock()
		defer dm.resultsLock.Unlock()

		// Unhandled response. Mark information as stale.
		// Try again later.
		//!TODO: After a few tries, just wipe it out?
		// 	or after certain time?
		if !cached.stale {
			cached = cachedResult{
				discovery:   cached.discovery,
				etag:        cached.etag,
				lastUpdated: now,
				stale:       true,
			}
			dm.cachedResults[info.service] = cached
		}

		klog.Infof("DiscoveryManager: Failed to download discovery for %v: %v %s",
			info.service.String(), writer.respCode, writer.data)
		return cached, fmt.Errorf("service %s returned non-success response code: %v",
			info.service.String(), writer.respCode)
	}
}

// Try to sync a single APIService.
func (dm *discoveryManager) syncAPIService(apiServiceName string) error {
	dm.servicesLock.RLock()
	info, exists := dm.apiServices[apiServiceName]
	dm.servicesLock.RUnlock()

	gv := helper.APIServiceNameToGroupVersion(apiServiceName)
	mgv := metav1.GroupVersion{Group: gv.Group, Version: gv.Version}

	if !exists {
		// apiservice was removed. remove it from merged discovery
		dm.mergedDiscoveryHandler.RemoveGroupVersion(mgv)
		return nil
	}

	// Lookup last cached result for this apiservice's service.
	now := time.Now()
	cached, err := dm.fetchFreshDiscoveryForService(mgv, info)

	dm.servicesLock.Lock()
	info.lastReconciled = now
	dm.apiServices[apiServiceName] = info
	dm.servicesLock.Unlock()

	if err != nil {
		// There was an error fetching discovery for this APIService.
		// Just use empty GV to mark that GV exists, but no resources.
		//
		// TODO: Maybe also stick in a status for the version the error?
		dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
			Version: gv.Version,
		})
		return err
	}

	// Check cache entry for this groupversion and insert it into the
	// document if present
	if entry, exists := cached.discovery[mgv]; exists {
		//!TODO: mark with staleness of cache
		dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
	} else {
		// Use empty GV, since there is an APIService for it
		//!TODO: mark with staleness of cache
		dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, metav1.APIVersionDiscovery{
			Version: gv.Version,
		})
	}

	return nil
}

// Spwans a goroutune which waits for added/updated apiservices and updates
// the discovery document accordingly
func (dm *discoveryManager) Run(stopCh <-chan struct{}) {
	klog.Info("Starting ResourceDiscoveryManager")

	// Spawn workers
	// These workers wait for APIServices to be marked dirty.
	// Worker ensures the cached discovery document hosted by the ServiceReference of
	// the APIService is at least as fresh as the APIService, then includes the
	// APIService's groupversion into the merged document
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
	wait.PollUntil(1*time.Minute, func() (done bool, err error) {
		dm.servicesLock.Lock()
		defer dm.servicesLock.Unlock()

		now := time.Now()

		// Mark all non-local APIServices as dirty
		for key, info := range dm.apiServices {
			if len(info.service.LocalName) > 0 {
				continue
			}
			info.lastMarkedDirty = now
			dm.apiServices[key] = info

			dm.dirtyAPIServiceQueue.Add(key)
		}

		return true, nil
	}, stopCh)

	// Shutdown the queue since stopCh was signalled
	dm.dirtyAPIServiceQueue.ShutDown()
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
					dm.apiServices[apiService.Name] = groupVersionInfo{
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
		dm.apiServices[apiService.Name] = groupVersionInfo{
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

func (dm *discoveryManager) ExternalServicesSynced() bool {
	dm.servicesLock.RLock()
	defer dm.servicesLock.RUnlock()

	for _, info := range dm.apiServices {
		if info.lastMarkedDirty.After(info.lastReconciled) {
			return false
		}
	}

	return true
}

func (dm *discoveryManager) LocalServicesSynced() bool {
	dm.servicesLock.RLock()
	defer dm.servicesLock.RUnlock()

	// Make sure each local delegate has been contacted, and that each GroupVersion
	// in their discovery document has been added to the mergedResourceManager
	for key, target := range dm.localDelegates {
		cached, err := dm.fetchFreshDiscoveryForService(metav1.GroupVersion{},
			groupVersionInfo{
				lastMarkedDirty: time.Now(),
				service:         serviceKey{LocalName: key},
				handler:         target.UnprotectedHandler(),
			})
		if err != nil {
			return false
		}

		// Make sure each APIService pointing to local delegate has been contacted
		for gv := range cached.discovery {
			info, exists := dm.apiServices[gv.Version+"."+gv.Group]
			if !exists {
				return false
			} else if len(info.service.LocalName) == 0 {
				// Skip external APIServices
				continue
			}

			if info.lastMarkedDirty.After(info.lastReconciled) {
				return false
			}
		}
	}
	// If we reach this point all local services had reconciled at least once
	return true
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
