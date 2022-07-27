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
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

// Given a list of APIServices and proxyHandlers for contacting them,
// DiscoveryManager caches a list of discovery documents for each server

type DiscoveryAggregationController interface {
	// Adds or Updates an APIService from the Aggregated Discovery Controller's
	// knowledge base
	// Thread-safe
	AddAPIService(apiService *v1.APIService, handler http.Handler)

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
	servicesLock sync.Mutex

	// Map from v1.APIService.Spec.Service (or a unique string for local servers)
	// to currently stored discovery document retrieved from that service.
	services map[string]*apiServiceInfo

	// Merged handler which stores all known groupversions
	mergedDiscoveryHandler discoveryv1.ResourceManager
}

type apiServiceInfo struct {
	fresh bool

	// Currently cached discovery document for this apiservice
	// a nil discovery document indicates this service needs to be re-fetched
	discovery *metav1.DiscoveryAPIGroupList

	// ETag hash of the cached discoveryDocument
	etag string

	// Must be non-nil
	proxyHandler http.Handler

	// Groups handled by this apiService
	knownGroups sets.String

	// Whether this is a local APIServer
	local bool
}

var _ DiscoveryAggregationController = &discoveryManager{}

func NewDiscoveryManager(
	codecs serializer.CodecFactory,
	serializer runtime.NegotiatedSerializer,
) DiscoveryAggregationController {
	return &discoveryManager{
		serializer:             serializer,
		mergedDiscoveryHandler: discoveryv1.NewResourceManager(serializer),
		services:               make(map[string]*apiServiceInfo),
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
func (self *discoveryManager) refreshDocument(localOnly bool) error {
	klog.Info("Refreshing discovery information from apiservices")

	// information needed in the below loop to update a service
	type serviceUpdateInfo struct {
		// Name of service
		name string

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
		self.servicesLock.Lock()
		defer self.servicesLock.Unlock()

		servicesToUpdate = make(chan serviceUpdateInfo, len(self.services))

		// Close the buffered channel once we have finished populating if
		// Note that even though it is closed it may still deliver any
		// enqueued items to the webworkers.
		defer close(servicesToUpdate)

		for name, service := range self.services {
			if localOnly && !service.local {
				continue
			}

			if !service.local && service.knownGroups.Len() == 0 {
				// Clean up unused services
				delete(self.services, name)
			} else if !service.fresh {

				// Should not block. Buffered channel is as large as services
				// slice
				servicesToUpdate <- serviceUpdateInfo{
					name:    name,
					handler: service.proxyHandler,
					etag:    service.etag,
				}
			}
		}
	}()

	// Download update discovery documents in parallel
	type resultItem struct {
		name      string
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
			name := updateInfo.name
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
					name:      name,
					discovery: &metav1.DiscoveryAPIGroupList{},
					error:     errors.New("not found"),
				}
			case http.StatusOK:
				parsed := &metav1.DiscoveryAPIGroupList{}
				if err := utiljson.Unmarshal(writer.data, parsed); err != nil {
					results <- resultItem{
						name:  name,
						error: err,
					}
					continue
				}

				results <- resultItem{
					name:      name,
					discovery: parsed,
					etag:      writer.Header().Get("Etag"),
				}
				klog.Infof("DiscoveryManager: Successfully downloaded discovery for %s", name)
			default:
				results <- resultItem{
					name:      name,
					discovery: &metav1.DiscoveryAPIGroupList{},
					error:     fmt.Errorf("service %s returned unknown response code: %v", name, writer.respCode),
				}
				klog.Infof("DiscoveryManager: Failed to download discovery for %s: %s %s", name, writer.respCode, writer.data)
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
	self.servicesLock.Lock()
	defer self.servicesLock.Unlock()

	for info := range results {
		if service, exists := self.services[info.name]; exists {
			service.fresh = true
			service.discovery = info.discovery
			service.etag = info.etag
		} else {
			// If a service was in services list at the beginning of this
			// function call but not anymore, then it was removed in the meantime
			// so we just throw away this result.
			continue
		}

		// If there was an issue with fetching local types then throw an error
		if info.error != nil && self.services[info.name].local {
			return info.error
		}
	}

	// After merging all the data back together, give it to the endpoint handler
	// to respond to HTTP requests
	var allGroups []metav1.DiscoveryAPIGroup
	for _, info := range self.services {
		if info.discovery != nil {
			allGroups = append(allGroups, info.discovery.Groups...)
		}
	}
	self.mergedDiscoveryHandler.SetGroups(allGroups)

	return nil
}

func (self *discoveryManager) markAPIServicesDirty() {
	self.servicesLock.Lock()
	defer self.servicesLock.Unlock()
	for _, info := range self.services {
		if !info.local {
			info.fresh = false
		}
	}
}

// Spwans a goroutune which waits for added/updated apiservices and updates
// the discovery document accordingly
func (self *discoveryManager) Run(stopCh <-chan struct{}) error {
	klog.Info("Starting ResourceDiscoveryManager")

	readyChannel := make(chan struct{})
	var result error

	go func() {
		// Signal the ready channel once local APIServices have been fetched
		defer close(readyChannel)
		result = self.refreshDocument(true)
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
		debounce(time.Second, self.dirtyChannel, stopCh, func() {
			err := self.refreshDocument(false)
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
				self.markAPIServicesDirty()
				self.kickWorker()
			case <-stopCh:
				ticker.Stop()
				return
			}
		}
	}()

	return result
}

// Wakes worker thread to notice the change and update the discovery document
func (self *discoveryManager) kickWorker() {
	select {
	case self.dirtyChannel <- struct{}{}:
		// Flagged to the channel that the object is dirty
	default:
		// Don't wait/Do nothing if the channel is already flagged
	}
}

// Adds an APIService to be tracked by the discovery manager. If the APIService
// is already known
func (self *discoveryManager) AddAPIService(apiService *v1.APIService, handler http.Handler) {
	// If service is nil then a local APIServer owns this APIService
	// However, we have no way of disambiguating them, so in that case we mark
	// all as dirty
	serviceName := ""
	if apiService.Spec.Service != nil {
		serviceName = apiService.Spec.Service.String()
	}

	self.servicesLock.Lock()
	defer self.servicesLock.Unlock()

	// If this APIService is associated with a different server it should be
	// removed
	for otherServiceName, info := range self.services {
		if info.knownGroups.Has(apiService.Name) && otherServiceName == serviceName {
			info.knownGroups.Delete(apiService.Name)
			info.fresh = false
		}
	}

	if serviceName == "" {
		// Mark all local services as dirty since we cannot disambiguate which
		// owns the given api group. This is not expensive due to usage of ETags
		// to detect if there were no changes.
		for _, info := range self.services {
			if info.local {
				info.fresh = false
			}
		}
	} else {
		if service, exists := self.services[serviceName]; exists {
			// Set the fresh flag to false
			service.fresh = false
		} else {
			// APIService is new to us, so start tracking it
			self.services[serviceName] = &apiServiceInfo{
				knownGroups:  sets.NewString(),
				proxyHandler: handler,
			}
		}
		self.services[serviceName].knownGroups.Insert(apiService.Name)
	}

	self.kickWorker()
}

func (self *discoveryManager) AddLocalAPIService(name string, handler http.Handler) {
	self.servicesLock.Lock()
	defer self.servicesLock.Unlock()

	if _, exists := self.services[name]; exists {
		klog.Errorf("Attempted to add local APIService %s but it already exists", name)
	} else {
		// APIService is new to us, so start tracking it
		self.services[name] = &apiServiceInfo{
			// add the name itself to knownGroups so local service is never
			// removed
			knownGroups:  sets.NewString(name),
			proxyHandler: handler,
			local:        true,
		}
	}

	self.kickWorker()
}

func (self *discoveryManager) RemoveAPIService(apiServiceName string) {
	self.servicesLock.Lock()
	defer self.servicesLock.Unlock()

	// Find record of a group with given name
	for _, info := range self.services {
		if info.knownGroups.Has(apiServiceName) {
			info.knownGroups.Delete(apiServiceName)
			self.kickWorker()
			return
		}
	}

	// If we reached this point, then no services had the name given.
	// Thus it is possible it is contained by one of the local apiservices.
	// Just refresh them all
	for _, info := range self.services {
		if info.local {
			info.fresh = false
		}
	}
	self.kickWorker()
}

func (self *discoveryManager) WebService() *restful.WebService {
	return self.mergedDiscoveryHandler.WebService()
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
