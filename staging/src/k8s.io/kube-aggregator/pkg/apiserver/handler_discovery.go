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

	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	"k8s.io/apiserver/pkg/endpoints/request"
	scheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper"
)

var APIRegistrationGroupVersion metav1.GroupVersion = metav1.GroupVersion{Group: "apiregistration.k8s.io", Version: "v1"}

// Maximum is 20000. Set to higher than that so apiregistration always is listed
// first (mirrors v1 discovery behavior)
var APIRegistrationGroupPriority int = 20001

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

	// Spwans a worker which waits for added/updated apiservices and updates
	// the unified discovery document by contacting the aggregated api services
	Run(stopCh <-chan struct{})

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

// Version of Service/Spec with relevant fields for use as a cache key
type serviceKey struct {
	Namespace string
	Name      string
	Port      int32
}

// Human-readable String representation used for logs
func (s serviceKey) String() string {
	return fmt.Sprintf("%v/%v:%v", s.Namespace, s.Name, s.Port)
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
	discovery map[metav1.GroupVersion]apidiscoveryv2beta1.APIVersionDiscovery

	// ETag hash of the cached discoveryDocument
	etag string

	// Guaranteed to be a time less than the time the server responded with the
	// discovery data.
	lastUpdated time.Time
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

	// Last time sync function was run for this GV.
	lastReconciled time.Time

	// ServiceReference of this GroupVersion. This identifies the Service which
	// describes how to contact the server responsible for this GroupVersion.
	service serviceKey

	// groupPriority describes the priority of the APIService's group for sorting
	groupPriority int

	// groupPriority describes the priority of the APIService version for sorting
	versionPriority int

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
		dirtyAPIServiceQueue:   workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "discovery-manager"),
	}
}

// Returns discovery data for the given apiservice.
// Caches the result.
// Returns the cached result if it is retrieved after the apiservice was last
// marked dirty
// If there was an error in fetching, returns the stale cached result if it exists,
// and a non-nil error
// If the result is current, returns nil error and non-nil result
func (dm *discoveryManager) fetchFreshDiscoveryForService(gv metav1.GroupVersion, info groupVersionInfo) (*cachedResult, error) {
	// Lookup last cached result for this apiservice's service.
	cached, exists := dm.getCacheEntryForService(info.service)

	// If entry exists and was updated after the given time, just stop now
	if exists && cached.lastUpdated.After(info.lastMarkedDirty) {
		return &cached, nil
	}

	// If we have a handler to contact the server for this APIService, and
	// the cache entry is too old to use, refresh the cache entry now.
	handler := http.TimeoutHandler(info.handler, 5*time.Second, "request timed out")
	req, err := http.NewRequest("GET", "/apis", nil)
	if err != nil {
		// NewRequest should not fail, but if it does for some reason,
		// log it and continue
		return &cached, fmt.Errorf("failed to create http.Request: %v", err)
	}

	// Apply aggregator user to request
	req = req.WithContext(
		request.WithUser(
			req.Context(), &user.DefaultInfo{Name: "system:kube-aggregator", Groups: []string{"system:masters"}}))
	req = req.WithContext(request.WithRequestInfo(req.Context(), &request.RequestInfo{
		Path:              req.URL.Path,
		IsResourceRequest: false,
	}))
	req.Header.Add("Accept", runtime.ContentTypeJSON+";g=apidiscovery.k8s.io;v=v2beta1;as=APIGroupDiscoveryList")

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
		// Keep old entry, update timestamp
		cached = cachedResult{
			discovery:   cached.discovery,
			etag:        cached.etag,
			lastUpdated: now,
		}

		dm.setCacheEntryForService(info.service, cached)
		return &cached, nil
	case http.StatusNotAcceptable:
		// Discovery Document is not being served at all.
		// Fall back to legacy discovery information
		if len(gv.Version) == 0 {
			return nil, errors.New("not found")
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
			return nil, fmt.Errorf("failed to create http.Request: %v", err)
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
			return nil, fmt.Errorf("failed to download discovery for %s: %v", path, writer.String())
		}

		parsed := &metav1.APIResourceList{}
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), writer.data, parsed); err != nil {
			return nil, err
		}

		// Create a discomap with single group-version
		resources, err := endpoints.ConvertGroupVersionIntoToDiscovery(parsed.APIResources)
		if err != nil {
			return nil, err
		}

		discoMap := map[metav1.GroupVersion]apidiscoveryv2beta1.APIVersionDiscovery{
			// Convert old-style APIGroupList to new information
			gv: {
				Version:   gv.Version,
				Resources: resources,
			},
		}

		cached = cachedResult{
			discovery:   discoMap,
			lastUpdated: now,
		}

		// Save the resolve, because it is still useful in case other services
		// are already marked dirty. THey can use it without making http request
		dm.setCacheEntryForService(info.service, cached)
		return &cached, nil

	case http.StatusOK:
		parsed := &apidiscoveryv2beta1.APIGroupDiscoveryList{}
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), writer.data, parsed); err != nil {
			return nil, err
		}
		klog.V(3).Infof("DiscoveryManager: Successfully downloaded discovery for %s", info.service.String())

		// Convert discovery info into a map for convenient lookup later
		discoMap := map[metav1.GroupVersion]apidiscoveryv2beta1.APIVersionDiscovery{}
		for _, g := range parsed.Items {
			for _, v := range g.Versions {
				discoMap[metav1.GroupVersion{Group: g.Name, Version: v.Version}] = v
			}
		}

		// Save cached result
		cached = cachedResult{
			discovery:   discoMap,
			etag:        writer.Header().Get("Etag"),
			lastUpdated: now,
		}
		dm.setCacheEntryForService(info.service, cached)
		return &cached, nil

	default:
		klog.Infof("DiscoveryManager: Failed to download discovery for %v: %v %s",
			info.service.String(), writer.respCode, writer.data)
		return nil, fmt.Errorf("service %s returned non-success response code: %v",
			info.service.String(), writer.respCode)
	}
}

// Try to sync a single APIService.
func (dm *discoveryManager) syncAPIService(apiServiceName string) error {
	info, exists := dm.getInfoForAPIService(apiServiceName)

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

	info.lastReconciled = now
	dm.setInfoForAPIService(apiServiceName, &info)

	var entry apidiscoveryv2beta1.APIVersionDiscovery

	// Extract the APIService's specific resource information from the
	// groupversion
	if cached == nil {
		// There was an error fetching discovery for this APIService, and
		// there is nothing in the cache for this GV.
		//
		// Just use empty GV to mark that GV exists, but no resources.
		// Also mark that it is stale to indicate the fetch failed
		// TODO: Maybe also stick in a status for the version the error?
		entry = apidiscoveryv2beta1.APIVersionDiscovery{
			Version: gv.Version,
		}
	} else {
		// Find our specific groupversion within the discovery document
		entry, exists = cached.discovery[mgv]
		if exists {
			// The stale/fresh entry has our GV, so we can include it in the doc
		} else {
			// Successfully fetched discovery information from the server, but
			// the server did not include this groupversion?
			entry = apidiscoveryv2beta1.APIVersionDiscovery{
				Version: gv.Version,
			}
		}
	}

	// The entry's staleness depends upon if `fetchFreshDiscoveryForService`
	// returned an error or not.
	if err == nil {
		entry.Freshness = apidiscoveryv2beta1.DiscoveryFreshnessCurrent
	} else {
		entry.Freshness = apidiscoveryv2beta1.DiscoveryFreshnessStale
	}

	dm.mergedDiscoveryHandler.AddGroupVersion(gv.Group, entry)
	dm.mergedDiscoveryHandler.SetGroupVersionPriority(metav1.GroupVersion(gv), info.groupPriority, info.versionPriority)
	return nil
}

// Spwans a goroutune which waits for added/updated apiservices and updates
// the discovery document accordingly
func (dm *discoveryManager) Run(stopCh <-chan struct{}) {
	klog.Info("Starting ResourceDiscoveryManager")

	// Shutdown the queue since stopCh was signalled
	defer dm.dirtyAPIServiceQueue.ShutDown()

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

				func() {
					defer dm.dirtyAPIServiceQueue.Done(next)

					if err := dm.syncAPIService(next.(string)); err != nil {
						dm.dirtyAPIServiceQueue.AddRateLimited(next)
					} else {
						dm.dirtyAPIServiceQueue.Forget(next)
					}
				}()
			}
		}()
	}

	// Ensure that apiregistration.k8s.io is the first group in the discovery group.
	dm.mergedDiscoveryHandler.SetGroupVersionPriority(APIRegistrationGroupVersion, APIRegistrationGroupPriority, 0)

	wait.PollUntil(1*time.Minute, func() (done bool, err error) {
		dm.servicesLock.Lock()
		defer dm.servicesLock.Unlock()

		now := time.Now()

		// Mark all non-local APIServices as dirty
		for key, info := range dm.apiServices {
			info.lastMarkedDirty = now
			dm.apiServices[key] = info
			dm.dirtyAPIServiceQueue.Add(key)
		}
		return false, nil
	}, stopCh)
}

// Adds an APIService to be tracked by the discovery manager. If the APIService
// is already known
func (dm *discoveryManager) AddAPIService(apiService *apiregistrationv1.APIService, handler http.Handler) {
	// If service is nil then its information is contained by a local APIService
	// which is has already been added to the manager.
	if apiService.Spec.Service == nil {
		return
	}

	// Add or update APIService record and mark it as dirty
	dm.setInfoForAPIService(apiService.Name, &groupVersionInfo{
		groupPriority:   int(apiService.Spec.GroupPriorityMinimum),
		versionPriority: int(apiService.Spec.VersionPriority),
		handler:         handler,
		lastMarkedDirty: time.Now(),
		service:         newServiceKey(*apiService.Spec.Service),
	})
	dm.dirtyAPIServiceQueue.Add(apiService.Name)
}

func (dm *discoveryManager) RemoveAPIService(apiServiceName string) {
	if dm.setInfoForAPIService(apiServiceName, nil) != nil {
		// mark dirty if there was actually something deleted
		dm.dirtyAPIServiceQueue.Add(apiServiceName)
	}
}

func (dm *discoveryManager) ExternalServicesSynced() bool {
	dm.servicesLock.RLock()
	defer dm.servicesLock.RUnlock()
	for _, info := range dm.apiServices {
		if info.lastReconciled.IsZero() {
			return false
		}
	}

	return true
}

//
// Lock-protected accessors
//

func (dm *discoveryManager) getCacheEntryForService(key serviceKey) (cachedResult, bool) {
	dm.resultsLock.RLock()
	defer dm.resultsLock.RUnlock()

	result, ok := dm.cachedResults[key]
	return result, ok
}

func (dm *discoveryManager) setCacheEntryForService(key serviceKey, result cachedResult) {
	dm.resultsLock.Lock()
	defer dm.resultsLock.Unlock()

	dm.cachedResults[key] = result
}

func (dm *discoveryManager) getInfoForAPIService(name string) (groupVersionInfo, bool) {
	dm.servicesLock.RLock()
	defer dm.servicesLock.RUnlock()

	result, ok := dm.apiServices[name]
	return result, ok
}

func (dm *discoveryManager) setInfoForAPIService(name string, result *groupVersionInfo) (oldValueIfExisted *groupVersionInfo) {
	dm.servicesLock.Lock()
	defer dm.servicesLock.Unlock()

	if oldValue, exists := dm.apiServices[name]; exists {
		oldValueIfExisted = &oldValue
	}

	if result != nil {
		dm.apiServices[name] = *result
	} else {
		delete(dm.apiServices, name)
	}

	return oldValueIfExisted
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
