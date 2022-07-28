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

	"github.com/emicklei/go-restful/v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	discoveryv1 "k8s.io/apiserver/pkg/endpoints/discovery/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
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
	AddLocalAPIService(name string, handler http.Handler)

	// Spwans a worker which waits for added/updated apiservices and updates
	// the unified discovery document by contacting the aggregated api services
	//
	// Blocks until local apiservices are populated or returns an error
	Run(stopCh <-chan struct{}) error

	// Returns a restful webservice which responds to discovery requests
	// Thread-safe
	WebService() *restful.WebService
}

type discoveryManager struct {
	serializer runtime.NegotiatedSerializer

	// Channel used to indicate that the document needs to be refreshed
	// The Run() function starts a worker thread which waits for signals on this
	// channel to refetch new discovery documents.
	dirtyChannel chan struct{}

	// Locks `services`
	servicesLock sync.RWMutex

	// Map from APIService's Namespace/Name (or a unique string for local servers)
	// to information about contacting that API Service
	services map[serviceKey]*apiServiceInfo

	// Merged handler which stores all known groupversions
	mergedDiscoveryHandler discoveryv1.ResourceManager
}

// Either a ServiceReference or a Local Service Name for use as a key in a map
// Difference from ServiceReference: Port is no longer *int32 to avoid hashing
// port memory address
type serviceKey struct {
	Name      string
	Namespace string
	Port      int32

	// If service is a local service, unique name identifying it
	// If LocalName is not empty, all other fields should be empty
	LocalName string
}

// Human-readable String representation used for loggs
func (s serviceKey) String() string {
	if s.LocalName == "" {
		return fmt.Sprintf("%v/%v:%v", s.Namespace, s.Name, s.Port)
	}
	return s.LocalName
}

func (s serviceKey) isLocalService() bool {
	return s.LocalName != ""
}

func newLocalServiceKey(name string) serviceKey {
	return serviceKey{
		LocalName: name,
	}
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

type apiServiceInfo struct {
	// False if the APIService is known to possibly have updated discovery
	// information.
	fresh bool

	// Currently cached discovery document for this service
	// a nil discovery document indicates this service needs to be re-fetched
	discovery *metav1.DiscoveryAPIGroupList

	// ETag hash of the cached discoveryDocument
	etag string

	// Handler for this service key
	handler http.Handler

	// APIServices which are hosted by the represented ServiceReference
	dependentAPIServices sets.String
}

var _ DiscoveryAggregationController = &discoveryManager{}

func NewDiscoveryManager(
	codecs serializer.CodecFactory,
	serializer runtime.NegotiatedSerializer,
) DiscoveryAggregationController {
	return &discoveryManager{
		serializer:             serializer,
		mergedDiscoveryHandler: discoveryv1.NewResourceManager(serializer),
		services:               make(map[serviceKey]*apiServiceInfo),
		dirtyChannel:           make(chan struct{}),
	}
}

func handlerWithUser(handler http.Handler, info user.Info) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		req = req.WithContext(request.WithUser(req.Context(), info))
		handler.ServeHTTP(w, req)
	})
}

// Synchronously refreshes the discovery document by contacting all known
// APIServices. Waits until all respond or timeout.
func (dm *discoveryManager) refreshDocument(localOnly bool) error {
	klog.Info("Refreshing discovery information from apiservices")

	// information needed in the below loop to update a service
	type serviceUpdateInfo struct {
		// Service
		service serviceKey

		// Handler For the service which responds to /discovery/v1 requests
		handler http.Handler

		// ETag of the existing discovery information known about the service
		etag string
	}

	var servicesToUpdate chan serviceUpdateInfo

	// Collect all services which have no discovery document and then update them
	// Done in two steps like this to avoid holding the lock while fetching the
	// documents.
	func() {
		dm.servicesLock.Lock()
		defer dm.servicesLock.Unlock()

		servicesToUpdate = make(chan serviceUpdateInfo, len(dm.services))

		// Close the buffered channel once we have finished populating if
		// Note that even though it is closed it may still deliver any
		// enqueued items to the webworkers.
		defer close(servicesToUpdate)

		for key, apiServiceInfo := range dm.services {
			if !key.isLocalService() && len(apiServiceInfo.dependentAPIServices) == 0 {
				// Purge unused non-local services from knowledge base
				delete(dm.services, key)
			} else if localOnly && !key.isLocalService() {
				// If only local services are desired to be fetched, skip
				// non-local
				continue
			} else if apiServiceInfo.fresh {
				// Skip APIServices which are not marked dirty
				continue
			}

			// Should not block. Buffered channel is as large as services
			// slice
			servicesToUpdate <- serviceUpdateInfo{
				service: key,
				handler: apiServiceInfo.handler,
				etag:    apiServiceInfo.etag,
			}
		}
	}()

	// Download update discovery documents in parallel
	type resultItem struct {
		service   serviceKey
		discovery *metav1.DiscoveryAPIGroupList
		etag      string
		error     error
	}

	waitGroup := sync.WaitGroup{}
	results := make(chan resultItem, len(servicesToUpdate))

	webworker := func() {
		defer waitGroup.Done()

		// Send a GET request to /discovery/v1 for each service that needs to
		// be updated
		// If channel is already closed, but has buffered items, will still only
		// stop once all enqueued items have been processed
		for updateInfo := range servicesToUpdate {
			key := updateInfo.service
			handler := updateInfo.handler
			handler = handlerWithUser(handler, &user.DefaultInfo{Name: "system:kube-aggregator", Groups: []string{"system:masters"}})
			handler = http.TimeoutHandler(handler, 5*time.Second, "request timed out")

			req, err := http.NewRequest("GET", "/discovery/v1", nil)
			if err != nil {
				// NewRequest should not fail, but if it does for some reason,
				// log it and continue
				klog.Errorf("failed to create http.Request for /discovery/v1: %v", err)
				continue
			}
			req.Header.Add("Accept", "application/json")

			if updateInfo.etag != "" {
				req.Header.Add("If-None-Match", updateInfo.etag)
			}

			writer := newInMemoryResponseWriter()
			handler.ServeHTTP(writer, req)

			switch writer.respCode {
			case http.StatusNotModified:
				// Do nothing. Just keep the old entry
			case http.StatusNotFound:
				// Wipe out any data for this service
				results <- resultItem{
					service:   key,
					discovery: nil,
					error:     errors.New("not found"),
				}
			case http.StatusOK:
				parsed := &metav1.DiscoveryAPIGroupList{}
				if err := utiljson.Unmarshal(writer.data, parsed); err != nil {
					results <- resultItem{
						service: key,
						error:   err,
					}
					continue
				}

				results <- resultItem{
					service:   key,
					discovery: parsed,
					etag:      writer.Header().Get("Etag"),
				}
				klog.Infof("DiscoveryManager: Successfully downloaded discovery for %s", key.String())
			default:
				// Wipe out discovery information
				results <- resultItem{
					service:   key,
					discovery: nil,
					error:     fmt.Errorf("service %s returned unknown response code: %v", key.String(), writer.respCode),
				}
				klog.Infof("DiscoveryManager: Failed to download discovery for %s: %s %s", key.String(), writer.respCode, writer.data)
			}
		}
	}

	// Spawn 2 webworkers
	for i := 0; i < 2; i++ {
		waitGroup.Add(1)
		go webworker()
	}

	// For for all transfers to either finish or fail
	waitGroup.Wait()
	close(results)

	if len(results) == 0 {
		return nil
	}

	// Merge information back into services list and inform the endpoint handler
	//  of updated information
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	var errors []error
	for info := range results {
		service, exists := dm.services[info.service]
		if !exists {
			// If a service was in services list at the beginning of this
			// function call but not anymore, then it was removed in the meantime
			// so we just throw away this result.
			continue
		}

		service.fresh = true
		service.discovery = info.discovery
		service.etag = info.etag

		if info.error != nil {
			errors = append(errors, info.error)
		}
	}

	// After merging all the data back together, give it to the endpoint handler
	// to respond to HTTP requests
	var allGroups []metav1.DiscoveryAPIGroup
	for _, info := range dm.services {
		if info.discovery != nil {
			allGroups = append(allGroups, info.discovery.Groups...)
		}
	}
	dm.mergedDiscoveryHandler.SetGroups(allGroups)

	if len(errors) > 0 {
		return fmt.Errorf("%v", errors)
	}
	return nil
}

func (dm *discoveryManager) markAPIServicesDirty() {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()
	for key, info := range dm.services {
		if !key.isLocalService() {
			info.fresh = false
		}
	}
}

// Spwans a goroutune which waits for added/updated apiservices and updates
// the discovery document accordingly
func (dm *discoveryManager) Run(stopCh <-chan struct{}) error {
	klog.Info("Starting ResourceDiscoveryManager")

	readyChannel := make(chan struct{})
	var result error

	go func() {
		// Signal the ready channel once local APIServices have been fetched
		defer close(readyChannel)
		result = dm.refreshDocument(true)
	}()

	// Wait for eitehr local APIServices to populate, or the worker to be
	// cancelled. Whichever comes first.
	select {
	case <-stopCh:
		return errors.New("worker stopped")
	case <-readyChannel:
		// Local API services now populated
	}

	// Every time the dirty channel is signalled, refresh the document
	// debounce in 1s intervals so that successive updates don't keep causing
	// a refresh
	go func() {
		debounce(time.Second, dm.dirtyChannel, stopCh, func() {
			err := dm.refreshDocument(false)
			if err != nil {
				klog.Error(err)
			}

		})
	}()

	// TODO: This should be in a constant
	ticker := time.NewTicker(60 * time.Second)
	go func() {
		for {
			select {
			case <-ticker.C:
				dm.markAPIServicesDirty()
				dm.kickWorker()
			case <-stopCh:
				ticker.Stop()
				return
			}
		}
	}()

	return result
}

// Wakes worker thread to notice the change and update the discovery document
func (dm *discoveryManager) kickWorker() {
	select {
	case dm.dirtyChannel <- struct{}{}:
		// Flagged to the channel that the object is dirty
	default:
		// Don't wait/Do nothing if the channel is already flagged
	}
}

// Adds an APIService to be tracked by the discovery manager. If the APIService
// is already known
func (dm *discoveryManager) AddAPIService(apiService *apiregistrationv1.APIService, handler http.Handler) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	// If this APIService is associated with a different apiserver, it should
	// be removed as a dependency and the old server should be marked as dirty
	for _, info := range dm.services {
		if info.dependentAPIServices.Has(apiService.Name) {
			info.dependentAPIServices.Delete(apiService.Name)
			info.fresh = false
		}
	}

	// If service is nil then a local APIServer owns this APIService
	if apiService.Spec.Service == nil {
		// Mark all local services as dirty since we cannot disambiguate which
		// owns the given api group. This is not expensive due to usage of ETags
		// to detect if there were no changes and the local nature of the
		// HTTP handler
		for key, info := range dm.services {
			if key.isLocalService() {
				info.fresh = false
			}
		}
	} else {
		serviceKey := newServiceKey(*apiService.Spec.Service)

		if service, exists := dm.services[serviceKey]; exists {
			// Set the fresh flag to false
			service.fresh = false
		} else {
			// Create a copy of the provided handler when creating a new service.
			// This is to prevent the given pointer from shifting to a different
			// service out from under us.
			//
			if ph, ok := handler.(*proxyHandler); ok {
				proxyHandlerCopy := *ph
				handler = &proxyHandlerCopy
			}

			// APIService is new to us, so start tracking it
			dm.services[serviceKey] = &apiServiceInfo{
				dependentAPIServices: sets.NewString(),
				handler:              handler,
			}
		}

		dm.services[serviceKey].dependentAPIServices.Insert(apiService.Name)
	}

	dm.kickWorker()
}

func (dm *discoveryManager) AddLocalAPIService(name string, handler http.Handler) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	serviceKey := newLocalServiceKey(name)

	if _, exists := dm.services[serviceKey]; exists {
		klog.Errorf("Attempted to add local APIService %s but it already exists", name)
	} else {
		// APIService is new to us, so start tracking it
		dm.services[serviceKey] = &apiServiceInfo{
			// add the name itself to knownGroups so local service is never
			// removed
			handler: handler,
		}
	}

	dm.kickWorker()
}

func (dm *discoveryManager) RemoveAPIService(apiServiceName string) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	// Find record of a group with given name
	for _, info := range dm.services {
		if info.dependentAPIServices.Has(apiServiceName) {
			info.dependentAPIServices.Delete(apiServiceName)
			dm.kickWorker()
			return
		}
	}

	// If we reached this point, then no services had the name given.
	// Thus it is possible it is contained by one of the local apiservices.
	// Just refresh them all
	for key, info := range dm.services {
		if key.isLocalService() {
			info.fresh = false
		}
	}
	dm.kickWorker()
}

func (dm *discoveryManager) WebService() *restful.WebService {
	return dm.mergedDiscoveryHandler.WebService()
}

// Takes an input structP{} channel and quantizes the channel sends to the given
// interval
// Should be moved into util library somewhere?
func debounce(interval time.Duration, input <-chan struct{}, stopCh <-chan struct{}, cb func()) {
	var timer *time.Timer = time.NewTimer(interval)

	// Make sure the timer is initially empty
	if !timer.Stop() {
		// As documentation for Stop() instructions. Does not block.
		<-timer.C
	}
	for {
		select {
		case <-stopCh:
			return
		case <-input:
			timer.Reset(interval)
		case <-timer.C:
			cb()
		}
	}
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
