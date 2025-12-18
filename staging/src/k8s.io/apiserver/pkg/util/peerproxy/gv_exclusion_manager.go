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

package peerproxy

import (
	"context"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
)

// GVExtractor is a function that extracts group-versions from an object.
// It returns a slice of GroupVersions belonging to CRDs or aggregated APIs that
// should be excluded from peer-discovery to avoid advertising stale CRDs/aggregated APIs
// in peer-aggregated discovery that were deleted but still appear in a peer's discovery.
type GVExtractor func(obj interface{}) []schema.GroupVersion

// RegisterCRDInformerHandlers registers event handlers for CRD informer using a custom extractor.
// The extractor function is responsible for extracting GroupVersions from CRD objects.
func (h *peerProxyHandler) RegisterCRDInformerHandlers(crdInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	h.crdInformer = crdInformer
	h.crdExtractor = extractor
	_, err := h.crdInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			gvs := extractor(obj)
			if gvs == nil {
				return
			}
			h.addExcludedGVs(gvs)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			gvs := extractor(newObj)
			if gvs == nil {
				return
			}
			h.addExcludedGVs(gvs)
		},
		DeleteFunc: func(obj interface{}) {
			// Handle tombstone objects
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			gvs := extractor(obj)
			if gvs == nil {
				return
			}
			for _, gv := range gvs {
				h.gvDeletionQueue.Add(gv.String())
			}
		},
	})
	return err
}

// RegisterAPIServiceInformerHandlers registers event handlers for APIService informer using a custom extractor.
// The extractor function is responsible for extracting GroupVersions from APIService objects
// and determining if they represent aggregated APIs.
func (h *peerProxyHandler) RegisterAPIServiceInformerHandlers(apiServiceInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	h.apiServiceInformer = apiServiceInformer
	h.apiServiceExtractor = extractor
	_, err := h.apiServiceInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			gvs := extractor(obj)
			if gvs == nil {
				return
			}
			h.addExcludedGVs(gvs)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			gvs := extractor(newObj)
			if gvs == nil {
				return
			}
			h.addExcludedGVs(gvs)
		},
		DeleteFunc: func(obj interface{}) {
			// Handle tombstone objects
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}

			gvs := extractor(obj)
			if gvs == nil {
				return
			}
			for _, gv := range gvs {
				h.gvDeletionQueue.Add(gv.String())
			}
		},
	})
	return err
}

// addExcludedGVs adds group-versions to the exclusion set, filters them from the peer discovery cache,
// and triggers invalidation callbacks if the cache changed.
func (h *peerProxyHandler) addExcludedGVs(gvs []schema.GroupVersion) {
	if len(gvs) == 0 {
		return
	}

	h.excludedGVsMu.Lock()
	for _, gv := range gvs {
		h.excludedGVs[gv] = nil
	}
	h.excludedGVsMu.Unlock()

	// Load current peer cache and filter it
	cache := h.peerDiscoveryInfoCache.Load()
	if cache == nil {
		return
	}

	cacheMap, ok := cache.(map[string]PeerDiscoveryCacheEntry)
	if !ok {
		klog.Warning("Invalid cache type in peerDiscoveryInfoCache")
		return
	}

	// Always trigger filter and callbacks, to ensure late-appearing GVs are filtered.
	if peerDiscovery, peerDiscoveryChanged := h.filterPeerDiscoveryCache(cacheMap); peerDiscoveryChanged {
		h.storePeerDiscoveryCacheAndInvalidate(peerDiscovery)
	}
}

// filterPeerDiscoveryCache filters the provided peer discovery cache,
// excluding GVs in h.excludedGVs.
func (h *peerProxyHandler) filterPeerDiscoveryCache(cacheMap map[string]PeerDiscoveryCacheEntry) (map[string]PeerDiscoveryCacheEntry, bool) {
	h.excludedGVsMu.RLock()
	if len(h.excludedGVs) == 0 {
		// No exclusions, no filtering needed.
		h.excludedGVsMu.RUnlock()
		return cacheMap, false
	}

	excludedCloned := make(map[schema.GroupVersion]struct{}, len(h.excludedGVs))
	for gv := range h.excludedGVs {
		excludedCloned[gv] = struct{}{}
	}
	h.excludedGVsMu.RUnlock()

	filtered := make(map[string]PeerDiscoveryCacheEntry, len(cacheMap))
	peerDiscoveryChanged := false
	for peerID, entry := range cacheMap {
		newEntry, peerChanged := h.filterPeerCacheEntry(entry, excludedCloned)
		filtered[peerID] = newEntry
		if peerChanged {
			peerDiscoveryChanged = true
		}
	}

	return filtered, peerDiscoveryChanged
}

// filterPeerCacheEntry filters a single peer's cache entry for excluded groups.
// Returns the filtered entry and whether any excluded group was present.
func (h *peerProxyHandler) filterPeerCacheEntry(entry PeerDiscoveryCacheEntry, excluded map[schema.GroupVersion]struct{}) (PeerDiscoveryCacheEntry, bool) {
	filteredGVRs := make(map[schema.GroupVersionResource]bool, len(entry.GVRs))
	peerDiscoveryChanged := false

	for existingGVR, v := range entry.GVRs {
		gv := schema.GroupVersion{Group: existingGVR.Group, Version: existingGVR.Version}
		if _, skip := excluded[gv]; skip {
			peerDiscoveryChanged = true
			continue
		}
		filteredGVRs[existingGVR] = v
	}

	if !peerDiscoveryChanged {
		return entry, false
	}

	// Filter the group discovery list
	filteredGroups := h.filterGroupDiscovery(entry.GroupDiscovery, excluded)
	return PeerDiscoveryCacheEntry{
		GVRs:           filteredGVRs,
		GroupDiscovery: filteredGroups,
	}, true
}

func (h *peerProxyHandler) filterGroupDiscovery(groupDiscoveries []apidiscoveryv2.APIGroupDiscovery, excluded map[schema.GroupVersion]struct{}) []apidiscoveryv2.APIGroupDiscovery {
	var filteredDiscovery []apidiscoveryv2.APIGroupDiscovery
	for _, groupDiscovery := range groupDiscoveries {
		filteredGroup := apidiscoveryv2.APIGroupDiscovery{
			ObjectMeta: groupDiscovery.ObjectMeta,
		}

		for _, version := range groupDiscovery.Versions {
			gv := schema.GroupVersion{Group: groupDiscovery.Name, Version: version.Version}
			if _, found := excluded[gv]; found {
				// This version is excluded, skip it
				continue
			}
			filteredGroup.Versions = append(filteredGroup.Versions, version)
		}

		// Only add the group to the final list if it still has any versions left
		if len(filteredGroup.Versions) > 0 {
			filteredDiscovery = append(filteredDiscovery, filteredGroup)
		}
	}
	return filteredDiscovery
}

// RunGVDeletionWorkers starts workers to process GroupVersions from deleted CRDs and APIServices.
// When a GV is processed, it is checked for usage by other resources. If no longer in use,
// it is marked with a deletion timestamp, initiating a grace period. After this period,
// the RunExcludedGVsReaper is responsible for removing the GV from the exclusion set.
func (h *peerProxyHandler) RunGVDeletionWorkers(ctx context.Context, workers int) {
	defer func() {
		klog.Info("Shutting down GV deletion workers")
		h.gvDeletionQueue.ShutDown()
	}()
	klog.Infof("Starting %d GV deletion worker(s)", workers)
	for i := 0; i < workers; i++ {
		go func(workerID int) {
			for h.processNextGVDeletion() {
				select {
				case <-ctx.Done():
					klog.Infof("GV deletion worker %d shutting down", workerID)
					return
				default:
				}
			}
		}(i)
	}

	<-ctx.Done() // Wait till context is done to call deferred shutdown function
}

// processNextGVDeletion processes a single item from the GV deletion queue.
func (h *peerProxyHandler) processNextGVDeletion() bool {
	gvString, shutdown := h.gvDeletionQueue.Get()
	if shutdown {
		return false
	}
	defer h.gvDeletionQueue.Done(gvString)

	gv, err := schema.ParseGroupVersion(gvString)
	if err != nil {
		klog.Errorf("Failed to parse GroupVersion %q: %v", gvString, err)
		return true
	}

	h.markGVForDeletionIfUnused(gv)
	return true
}

// markGVForDeletionIfUnused checks if a group-version is still in use.
// If not, it marks it for deletion by setting a timestamp.
func (h *peerProxyHandler) markGVForDeletionIfUnused(gv schema.GroupVersion) {
	// Best-effort check if GV is still in use. This is racy - a new CRD/APIService could be
	// created after this check completes and returns false.
	if h.isGVUsed(gv) {
		return
	}

	h.excludedGVsMu.Lock()
	defer h.excludedGVsMu.Unlock()

	// Only mark if it's currently active (nil timestamp)
	if ts, exists := h.excludedGVs[gv]; exists && ts == nil {
		now := time.Now()
		h.excludedGVs[gv] = &now
		klog.V(4).Infof("Marking group-version %q for deletion, grace period started", gv)
	}
}

// isGVUsed checks the informer stores to see if any CRD or APIService
// is still using the given group-version.
//
// This is necessary because multiple CRDs or aggregated APIs can share the same group-version,
// but define different resources (kinds) or versions. Deleting one CRD or APIService
// for a group-version does not mean the group-version is entirely gone—other CRDs or APIs in that group-version
// may still exist. We must only remove a group-version from the exclusion set when there are
// no remaining CRDs or APIService objects for that group-version, to ensure we do not prematurely
// allow peer-aggregated discovery or proxying for APIs that are still served locally.
func (h *peerProxyHandler) isGVUsed(gv schema.GroupVersion) bool {
	// Check CRD informer store for the specific group-version
	if h.crdInformer != nil && h.crdExtractor != nil {
		crdList := h.crdInformer.GetStore().List()
		for _, item := range crdList {
			gvs := h.crdExtractor(item)
			if gvs == nil {
				continue
			}
			for _, extractedGV := range gvs {
				if extractedGV == gv {
					return true
				}
			}
		}
	}

	// Check APIService informer store for the specific group-version
	if h.apiServiceInformer != nil && h.apiServiceExtractor != nil {
		apiSvcList := h.apiServiceInformer.GetStore().List()
		for _, item := range apiSvcList {
			gvs := h.apiServiceExtractor(item)
			if gvs == nil {
				continue
			}
			for _, extractedGV := range gvs {
				if extractedGV == gv {
					return true
				}
			}
		}
	}

	return false
}

// RunExcludedGVsReaper starts a goroutine that periodically cleans up
// excluded group-versions that have passed their grace period.
func (h *peerProxyHandler) RunExcludedGVsReaper(stopCh <-chan struct{}) {
	klog.Infof("Starting excluded group-versions reaper with %s interval", h.reaperCheckInterval)
	ticker := time.NewTicker(h.reaperCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			h.reapExcludedGVs()
		case <-stopCh:
			klog.Info("Shutting down excluded group-versions reaper")
			return
		}
	}
}

// reapExcludedGVs is the garbage collector function.
// It removes group-versions from excludedGVs if their grace period has expired.
func (h *peerProxyHandler) reapExcludedGVs() {
	h.excludedGVsMu.Lock()
	defer h.excludedGVsMu.Unlock()

	now := time.Now()
	for gv, deletionTime := range h.excludedGVs {
		if deletionTime == nil {
			// Still actively excluded
			continue
		}

		if now.Sub(*deletionTime) > h.exclusionGracePeriod {
			klog.V(4).Infof("Reaping excluded group-version %q (grace period expired)", gv)
			delete(h.excludedGVs, gv)
		}
	}
}
