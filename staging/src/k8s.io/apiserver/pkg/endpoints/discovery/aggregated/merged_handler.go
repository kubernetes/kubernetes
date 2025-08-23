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
	"fmt"
	"net/http"
	"sort"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// MergedResourceManager defines the interface for managing merged discovery resources
// that combines both local and peer server resources.
type MergedResourceManager interface {
	// SetPeerDiscoveryProvider sets the peer discovery provider for merged discovery.
	SetPeerDiscoveryProvider(provider PeerDiscoveryProvider)

	// InvalidateClusterWideCaches invalidates the merged discovery caches
	// This should be called when peer discovery data changes.
	InvalidateClusterWideCaches()

	// ServeHTTP handles merged discovery HTTP requests.
	http.Handler
}

// mergedDiscoveryHandler handles merged discovery requests that include both local and peer resources.
type mergedDiscoveryHandler struct {
	localResourceManager  ResourceManager
	peerDiscoveryProvider PeerDiscoveryProvider
	serializer            runtime.NegotiatedSerializer
	localServerID         string
	cache                 atomic.Pointer[mergedDiscoveryCache]
	cacheWithServerIDs    atomic.Pointer[mergedDiscoveryCache]
}

// mergedDiscoveryCache holds the cached merged discovery response.
type mergedDiscoveryCache struct {
	cachedResponse     apidiscoveryv2.APIGroupDiscoveryList
	cachedResponseETag string
}

func NewMergedDiscoveryHandler(localResourceManager ResourceManager, localServerID string) *mergedDiscoveryHandler {
	scheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(scheme))
	utilruntime.Must(apidiscoveryv2beta1.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme)

	return &mergedDiscoveryHandler{
		localResourceManager: localResourceManager,
		serializer:           codecs,
		localServerID:        localServerID,
	}
}

func (m *mergedDiscoveryHandler) SetPeerDiscoveryProvider(provider PeerDiscoveryProvider) {
	m.peerDiscoveryProvider = provider
	m.cache.Store(nil)
	m.cacheWithServerIDs.Store(nil)
}

// InvalidateClusterWideCaches invalidates the merged discovery caches.
// This should be called when peer discovery data changes.
func (m *mergedDiscoveryHandler) InvalidateClusterWideCaches() {
	m.cache.Store(nil)
	m.cacheWithServerIDs.Store(nil)
	klog.V(4).Info("Invalidated merged discovery caches")
}

func (m *mergedDiscoveryHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.UnknownVersionInteroperabilityProxy) {
		klog.V(4).Info("UnknownVersionInteroperabilityProxy feature gate disabled")
		utilruntime.HandleError(fmt.Errorf("merged discovery requires UnknownVersionInteroperabilityProxy feature gate"))
		w.WriteHeader(http.StatusNotImplemented)
		return
	}

	if m.peerDiscoveryProvider == nil {
		klog.V(4).Info("No peer discovery provider available")
		utilruntime.HandleError(fmt.Errorf("merged discovery not available - no peer discovery provider"))
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	includeServerIDs := req.URL.Query().Get("includeServerIds") == "true"
	response, etag := m.buildMergedResponse(includeServerIDs)
	writeDiscoveryResponse(response, etag, m.serializer, w, req)
}

func (m *mergedDiscoveryHandler) buildMergedResponse(includeServerIDs bool) (*apidiscoveryv2.APIGroupDiscoveryList, string) {
	var cached *mergedDiscoveryCache
	if includeServerIDs {
		cached = m.cacheWithServerIDs.Load()
	} else {
		cached = m.cache.Load()
	}

	if cached != nil {
		return &cached.cachedResponse, cached.cachedResponseETag
	}

	klog.V(4).Infof("Building new merged discovery response (includeServerIDs=%v)", includeServerIDs)

	// Access the underlying resourceDiscoveryManager to get local groups.
	localResourceManager, ok := m.localResourceManager.(resourceManager)
	if !ok {
		klog.Error("Unable to access underlying resource discovery manager")
		return &apidiscoveryv2.APIGroupDiscoveryList{}, ""
	}

	// Get local discovery data.
	localCache := localResourceManager.resourceDiscoveryManager.fetchFromCache()
	localGroups := localCache.cachedResponse.Items

	// Get peer resources.
	peerResources := m.peerDiscoveryProvider.GetPeerResources()

	// Merge local and peer resources.
	mergedGroups := m.mergeResources(localGroups, peerResources, includeServerIDs)
	response := &apidiscoveryv2.APIGroupDiscoveryList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIGroupDiscoveryList",
			APIVersion: apidiscoveryv2.SchemeGroupVersion.String(),
		},
		Items: mergedGroups,
	}

	// Calculate etag for merged response.
	etag, err := calculateETag(*response)
	if err != nil {
		klog.Errorf("Failed to calculate etag for merged discovery: %v", err)
		etag = localCache.cachedResponseETag // Fall back to local discovery etag.
	}

	// Cache the response in the appropriate cache.
	cacheEntry := &mergedDiscoveryCache{
		cachedResponse:     *response,
		cachedResponseETag: etag,
	}

	if includeServerIDs {
		m.cacheWithServerIDs.Store(cacheEntry)
	} else {
		m.cache.Store(cacheEntry)
	}

	return response, etag
}

func (m *mergedDiscoveryHandler) mergeResources(
	localGroups []apidiscoveryv2.APIGroupDiscovery,
	peerResources map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery,
	includeServerIDs bool,
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

				if includeServerIDs {
					groupCopy.Versions[i].Resources[j].ServerIDs = []string{m.localServerID}
				}
			}
		}
		groupMap[group.Name] = groupCopy
	}

	// Add peer resources, skip duplicates.
	for serverID, resources := range peerResources {
		for gvr, resource := range resources {
			// Only record peer serverID for resources already served locally.
			if localResourceSet[gvr] {
				if includeServerIDs {
					m.addPeerServerID(groupMap, gvr, serverID)
				}
				continue
			}

			m.addPeerResource(groupMap, gvr, resource, serverID, includeServerIDs)
		}
	}

	// Convert to sorted slice.
	return m.convertToSortedSlice(groupMap)
}

func (m *mergedDiscoveryHandler) addPeerServerID(
	groupMap map[string]*apidiscoveryv2.APIGroupDiscovery,
	gvr schema.GroupVersionResource,
	serverID string,
) {
	groupName := gvr.Group

	group, exists := groupMap[groupName]
	if !exists {
		return
	}

	for i := range group.Versions {
		if group.Versions[i].Version != gvr.Version {
			continue
		}

		for j := range group.Versions[i].Resources {
			if group.Versions[i].Resources[j].Resource != gvr.Resource {
				continue
			}

			// Add server ID if not already present.
			serverIDs := group.Versions[i].Resources[j].ServerIDs
			for _, existingID := range serverIDs {
				if existingID == serverID {
					// Already present.
					return
				}
			}

			serverIDs = append(serverIDs, serverID)
			sort.Strings(serverIDs)
			group.Versions[i].Resources[j].ServerIDs = serverIDs
			return
		}
	}
}

func (m *mergedDiscoveryHandler) addPeerResource(
	groupMap map[string]*apidiscoveryv2.APIGroupDiscovery,
	gvr schema.GroupVersionResource,
	resource *apidiscoveryv2.APIResourceDiscovery,
	serverID string,
	includeServerIDs bool,
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
	if includeServerIDs {
		resourceCopy.ServerIDs = []string{serverID}
	}

	group.Versions[versionIndex].Resources = append(group.Versions[versionIndex].Resources, *resourceCopy)
}

func (m *mergedDiscoveryHandler) convertToSortedSlice(groupMap map[string]*apidiscoveryv2.APIGroupDiscovery) []apidiscoveryv2.APIGroupDiscovery {
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
