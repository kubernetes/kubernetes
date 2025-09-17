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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
)

// PeerDiscoveryProvider defines an interface to get peer resources for merged discovery.
type PeerDiscoveryProvider interface {
	GetPeerResources() map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery
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
	apiServiceLister      listers.APIServiceLister
	serializer            runtime.NegotiatedSerializer
	cache                 atomic.Pointer[cachedGroupList]
	serveHTTPFunc         func(http.ResponseWriter, *http.Request)
}

// NewPeerMergedDiscoveryHandler creates a new handler for merged discovery.
func NewPeerMergedDiscoveryHandler(localDiscoveryProvider ResourceManager, peerDiscoveryProvider PeerDiscoveryProvider, apiServiceLister listers.APIServiceLister, path string) PeerMergedResourceManager {
	scheme := runtime.NewScheme()
	utilruntime.Must(apidiscoveryv2.AddToScheme(scheme))
	codecs := serializer.NewCodecFactory(scheme)

	pmd := &peerMergedDiscoveryHandler{
		localResourceManager:  localDiscoveryProvider,
		peerDiscoveryProvider: peerDiscoveryProvider,
		apiServiceLister:      apiServiceLister,
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
	peerResources map[string]map[schema.GroupVersionResource]*apidiscoveryv2.APIResourceDiscovery,
) []apidiscoveryv2.APIGroupDiscovery {
	groupMap := make(map[string]*apidiscoveryv2.APIGroupDiscovery)
	localResourceSet := make(map[schema.GroupVersionResource]bool)
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
	if len(peerResources) == 0 {
		return h.convertToSortedSlice(groupMap)
	}
	MergedRequestCounter.Inc()
	for _, resources := range peerResources {
		for gvr, resource := range resources {
			if localResourceSet[gvr] {
				continue
			}
			addPeerResource(groupMap, gvr, resource)
		}
	}

	return h.convertToSortedSlice(groupMap)
}

func (h *peerMergedDiscoveryHandler) convertToSortedSlice(groupMap map[string]*apidiscoveryv2.APIGroupDiscovery) []apidiscoveryv2.APIGroupDiscovery {
	groupList, priorities := h.collectPriorities(groupMap)
	sort.SliceStable(groupList, func(i, j int) bool {
		gi, gj := groupList[i], groupList[j]

		gpi, gpj := getGroupPriority(gi, priorities), getGroupPriority(gj, priorities)
		if gpi != gpj {
			return gpi > gpj // higher group priority first (descending)
		}

		vpi, vpj := getMaxVersionPriority(gi, priorities), getMaxVersionPriority(gj, priorities)
		if vpi != vpj {
			return vpi > vpj // higher version priority first (descending)
		}

		// Fallback to lexicographical sort by group name
		return gi.Name < gj.Name
	})

	return groupList
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

func (h *peerMergedDiscoveryHandler) collectPriorities(
	groupMap map[string]*apidiscoveryv2.APIGroupDiscovery,
) ([]apidiscoveryv2.APIGroupDiscovery, map[gvKey]groupVersionPriorities) {
	priorities := make(map[gvKey]groupVersionPriorities)
	// If no lister, return all groups/versions
	if h.apiServiceLister == nil {
		groupList := make([]apidiscoveryv2.APIGroupDiscovery, 0, len(groupMap))
		for _, group := range groupMap {
			groupList = append(groupList, *group)
		}
		return groupList, priorities
	}

	apiServices, err := h.apiServiceLister.List(labels.Everything())
	apiServiceMap := make(map[string]*v1.APIService)
	if err == nil {
		for _, apiSvc := range apiServices {
			apiServiceMap[apiSvc.Name] = apiSvc
		}
	}

	groupList := make([]apidiscoveryv2.APIGroupDiscovery, 0, len(groupMap))
	for _, group := range groupMap {
		versionList := make([]apidiscoveryv2.APIVersionDiscovery, 0, len(group.Versions))
		for _, version := range group.Versions {
			apiServiceName := version.Version + "." + group.Name
			if apiSvc, ok := apiServiceMap[apiServiceName]; ok {
				priorities[gvKey{group: group.Name, version: version.Version}] = groupVersionPriorities{
					groupPriority:   apiSvc.Spec.GroupPriorityMinimum,
					versionPriority: apiSvc.Spec.VersionPriority,
				}
			}
			versionList = append(versionList, version)
		}
		if len(versionList) > 0 {
			groupCopy := group.DeepCopy()
			groupCopy.Versions = versionList
			groupList = append(groupList, *groupCopy)
		}
	}
	return groupList, priorities
}

func getGroupPriority(group apidiscoveryv2.APIGroupDiscovery, priorities map[gvKey]groupVersionPriorities) int32 {
	if len(group.Versions) == 0 {
		return -1
	}
	key := gvKey{group: group.Name, version: group.Versions[0].Version}
	if p, ok := priorities[key]; ok {
		return p.groupPriority
	}
	return -1
}

func getMaxVersionPriority(group apidiscoveryv2.APIGroupDiscovery, priorities map[gvKey]groupVersionPriorities) int32 {
	max := int32(-1)
	for _, version := range group.Versions {
		key := gvKey{group: group.Name, version: version.Version}
		if p, ok := priorities[key]; ok {
			if p.versionPriority > max {
				max = p.versionPriority
			}
		}
	}
	return max
}

type groupVersionPriorities struct {
	groupPriority   int32
	versionPriority int32
}

type gvKey struct {
	group   string
	version string
}
