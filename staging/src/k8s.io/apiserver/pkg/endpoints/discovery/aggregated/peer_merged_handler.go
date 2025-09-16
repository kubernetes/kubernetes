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
	"sort"
	"sync/atomic"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

// PeerDiscoveryProvider defines an interface to get peer resources for merged discovery.
type PeerDiscoveryProvider interface {
	GetPeerResources() map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery
}

// PeerMergedResourceManager defines the interface for managing merged discovery resources
// that combines both local and peer server resources.
type PeerMergedResourceManager interface {
	// SetPeerDiscoveryProvider sets the peer discovery provider for merged discovery.
	SetPeerDiscoveryProvider(provider PeerDiscoveryProvider)

	// InvalidateClusterWideCaches invalidates the merged discovery caches
	// This should be called when peer discovery data changes.
	InvalidateClusterWideCaches()

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
func NewPeerMergedDiscoveryHandler(rdm ResourceManager, path string) PeerMergedResourceManager {
	scheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(scheme))
	utilruntime.Must(apidiscoveryv2beta1.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme)

	pmd := &peerMergedDiscoveryHandler{
		localResourceManager: rdm,
		serializer:           codecs,
	}

	pmd.serveHTTPFunc = metrics.InstrumentHandlerFunc(request.MethodGet,
		/* group = */ "",
		/* version = */ "",
		/* resource = */ "",
		/* subresource = */ path,
		/* scope = */ "",
		/* component = */ metrics.APIServerComponent,
		/* deprecated */ false,
		/* removedRelease */ "",
		pmd.serveHTTP)

	return pmd
}

func (m *peerMergedDiscoveryHandler) SetPeerDiscoveryProvider(provider PeerDiscoveryProvider) {
	m.peerDiscoveryProvider = provider
	m.cache.Store(nil)
}

// InvalidateClusterWideCaches invalidates the merged discovery caches.
// This should be called when peer discovery data changes.
func (m *peerMergedDiscoveryHandler) InvalidateClusterWideCaches() {
	m.cache.Store(nil)
	klog.V(4).Info("Invalidated merged discovery caches")
}

func (m *peerMergedDiscoveryHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	m.serveHTTPFunc(resp, req)
}

func (m *peerMergedDiscoveryHandler) serveHTTP(resp http.ResponseWriter, req *http.Request) {
	cache := m.fetchFromCache()
	response := cache.cachedResponse
	etag := cache.cachedResponseETag

	writeDiscoveryResponse(&response, etag, m.serializer, resp, req)
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
		mergedGroups = mergeResources(localGroups, peerResources)
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

func mergeResources(
	localGroups []apidiscoveryv2.APIGroupDiscovery,
	peerResources map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery,
) []apidiscoveryv2.APIGroupDiscovery {

	groupMap := make(map[string]*apidiscoveryv2.APIGroupDiscovery)
	localResourceSet := make(map[schema.GroupVersionResource]bool)

	// Process local groups first.
	for _, group := range localGroups {
		groupCopy := group.DeepCopy()
		for i := range groupCopy.Versions {
			for j := range groupCopy.Versions[i].Resources {
				gvr := schema.GroupVersionResource{
					Group:    group.Name,
					Version:  groupCopy.Versions[i].Version,
					Resource: groupCopy.Versions[i].Resources[j].Resource,
				}
				localResourceSet[gvr] = true
			}
		}
		groupMap[group.Name] = groupCopy
	}

	// Add peer resources, skip duplicates.
	for _, resources := range peerResources {
		for gvr, resource := range resources {
			if localResourceSet[gvr] {
				continue
			}

			addPeerResource(groupMap, gvr, resource)
		}
	}

	// Convert to sorted slice.
	return convertToSortedSlice(groupMap)
}

func addPeerResource(
	groupMap map[string]*apidiscoveryv2.APIGroupDiscovery,
	gvr schema.GroupVersionResource,
	resource *apidiscoveryv2.APIResourceDiscovery,
) {
	groupName := gvr.Group
	group, exists := groupMap[groupName]
	if !exists {
		group = &apidiscoveryv2.APIGroupDiscovery{
			ObjectMeta: metav1.ObjectMeta{Name: groupName},
			Versions:   []apidiscoveryv2.APIVersionDiscovery{},
		}
		groupMap[groupName] = group
	}

	versionIndex := -1
	for i, version := range group.Versions {
		if version.Version == gvr.Version {
			versionIndex = i
			break
		}
	}

	if versionIndex == -1 {
		group.Versions = append(group.Versions, apidiscoveryv2.APIVersionDiscovery{
			Version:   gvr.Version,
			Resources: []apidiscoveryv2.APIResourceDiscovery{},
		})
		versionIndex = len(group.Versions) - 1
	}

	resourceCopy := resource.DeepCopy()
	group.Versions[versionIndex].Resources = append(group.Versions[versionIndex].Resources, *resourceCopy)
}

func convertToSortedSlice(groupMap map[string]*apidiscoveryv2.APIGroupDiscovery) []apidiscoveryv2.APIGroupDiscovery {
	result := make([]apidiscoveryv2.APIGroupDiscovery, 0, len(groupMap))
	for _, group := range groupMap {
		result = append(result, *group)
	}

	// TODO: is this right?
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})

	return result
}
