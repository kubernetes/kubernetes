/*
Copyright 2025 The Kubernetes Authors.

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

package aggregated

import (
	"net/http"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilsort "k8s.io/apimachinery/pkg/util/sort"
)

// PeerDiscoveryProvider defines an interface to get peer resources for merged discovery.
type PeerDiscoveryProvider interface {
	GetPeerResources() map[string][]apidiscoveryv2.APIGroupDiscovery
}

// PeerMergedResourceManager defines the interface for managing merged discovery resources
// that combines both local and peer server resources.
type PeerMergedResourceManager interface {
	// InvalidateCache invalidates the merged discovery caches
	// This should be called when peer discovery data changes.
	InvalidateCache()

	// ServeHTTP handles merged discovery HTTP requests.
	http.Handler
}

// peerMergedDiscoveryHandler handles merged discovery requests that include both local and peer resources.
type peerMergedDiscoveryHandler struct {
	localResourceManager  ResourceManager
	peerDiscoveryProvider PeerDiscoveryProvider
	serializer            runtime.NegotiatedSerializer
	cache                 atomic.Pointer[cachedGroupList]
	serveHTTPFunc         func(http.ResponseWriter, *http.Request)
}

// NewPeerMergedDiscoveryHandler creates a new handler for merged discovery.
func NewPeerMergedDiscoveryHandler(localDiscoveryProvider ResourceManager, peerDiscoveryProvider PeerDiscoveryProvider, path string) PeerMergedResourceManager {
	scheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme)

	pmd := &peerMergedDiscoveryHandler{
		localResourceManager:  localDiscoveryProvider,
		peerDiscoveryProvider: peerDiscoveryProvider,
		serializer:            codecs,
	}

	pmd.localResourceManager.AddInvalidationCallback(func() {
		pmd.InvalidateCache()
	})

	// Instrumentation wrapper for serveHTTP
	pmd.serveHTTPFunc = metrics.InstrumentHandlerFunc(request.MethodGet,
		"", "", "", path, "", metrics.APIServerComponent, false, "",
		func(resp http.ResponseWriter, req *http.Request) {
			pmd.serveHTTP(resp, req)
		})

	return pmd
}

// InvalidateCache invalidates the merged discovery caches.
// This should be called when peer discovery data changes.
func (h *peerMergedDiscoveryHandler) InvalidateCache() {
	h.cache.Store(nil)
	klog.V(4).Info("Invalidated merged discovery caches")
}

func (h *peerMergedDiscoveryHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	h.serveHTTPFunc(resp, req)
}

func (h *peerMergedDiscoveryHandler) serveHTTP(resp http.ResponseWriter, req *http.Request) {
	cache := h.fetchFromCache()
	response := cache.cachedResponse
	etag := cache.cachedResponseETag

	writeDiscoveryResponse(&response, etag, h.serializer, resp, req)
}

func (h *peerMergedDiscoveryHandler) fetchFromCache() *cachedGroupList {
	cacheLoad := h.cache.Load()
	if cacheLoad != nil {
		return cacheLoad
	}

	// Get local groups
	var localGroups []apidiscoveryv2.APIGroupDiscovery
	if rdm, ok := h.localResourceManager.(resourceManager); ok {
		localCache := rdm.fetchFromCache()
		if localCache != nil {
			localGroups = localCache.cachedResponse.Items
		}
	}

	// Get peer resources if provider is set
	var mergedGroups []apidiscoveryv2.APIGroupDiscovery
	if h.peerDiscoveryProvider != nil {
		peerResources := h.peerDiscoveryProvider.GetPeerResources()
		mergedGroups = h.mergeResources(localGroups, peerResources)
	} else {
		mergedGroups = localGroups
	}

	response := apidiscoveryv2.APIGroupDiscoveryList{
		Items: mergedGroups,
	}
	etag, err := calculateETag(response)
	if err != nil {
		klog.Errorf("failed to calculate etag for discovery document: %v", err)
		etag = ""
	}

	cached := &cachedGroupList{
		cachedResponse:     response,
		cachedResponseETag: etag,
	}
	h.cache.Store(cached)
	return cached
}

func (h *peerMergedDiscoveryHandler) mergeResources(
	localGroups []apidiscoveryv2.APIGroupDiscovery,
	peerGroupDiscovery map[string][]apidiscoveryv2.APIGroupDiscovery,
) []apidiscoveryv2.APIGroupDiscovery {
	// Return local groups if no peer resources exist.
	if len(peerGroupDiscovery) == 0 {
		return localGroups
	}

	groupMap := make(map[string]apidiscoveryv2.APIGroupDiscovery)
	allGroups := make([][]string, 0, 1+len(peerGroupDiscovery))

	// Add local groups first to preserve ordering.
	localGroupNames := make([]string, 0, len(localGroups))
	for _, group := range localGroups {
		localGroupNames = append(localGroupNames, group.Name)
		groupMap[group.Name] = group
	}
	allGroups = append(allGroups, localGroupNames)

	// Add peer groups.
	for _, peerGroupList := range peerGroupDiscovery {
		peerGroupNames := make([]string, 0, len(peerGroupList))
		for _, group := range peerGroupList {
			peerGroupNames = append(peerGroupNames, group.Name)
			groupMap[group.Name] = group
		}
		allGroups = append(allGroups, peerGroupNames)
	}

	MergedRequestCounter.Inc()
	return h.convertToSortedSliceTopo(groupMap, allGroups)
}

func (h *peerMergedDiscoveryHandler) convertToSortedSliceTopo(
	groupMap map[string]apidiscoveryv2.APIGroupDiscovery,
	groupOrderings [][]string,
) []apidiscoveryv2.APIGroupDiscovery {
	mergedOrderGroups := utilsort.SortDiscoveryGroupsTopo(groupOrderings)
	result := make([]apidiscoveryv2.APIGroupDiscovery, 0, len(mergedOrderGroups))
	for _, groupName := range mergedOrderGroups {
		if group, ok := groupMap[groupName]; ok {
			result = append(result, group)
		}
	}
	return result
}
