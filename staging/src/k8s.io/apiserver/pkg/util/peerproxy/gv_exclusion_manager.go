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
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
)

// GVExtractor is a function that extracts group-versions from an object.
// It returns a slice of GroupVersions belonging to CRDs or aggregated APIs that
// should be excluded from peer-discovery to avoid advertising stale CRDs/aggregated APIs
// in peer-aggregated discovery that were deleted but still appear in a peer's discovery.
type GVExtractor func(obj interface{}) []schema.GroupVersion

// GVExclusionManager manages the exclusion of group-versions from peer discovery.
// It maintains two atomic maps:
// - currentlyActiveGVs: All GVs currently served by CRDs and aggregated APIServices
// - recentlyDeletedGVs: GVs belonging to CRDs and aggregated APIServices that were recently deleted,
// tracked with deletion timestamp for grace period
//
// It runs three workers:
// 1. Active GV Tracker: Triggered on CRD/APIService events, rebuilds active GVs
// 2. Reaper: Periodically removes expired GVs from recentlyDeletedGVs
// 3. Peer Discovery Re-filter: Rate-limited worker that filters peer cache
type GVExclusionManager struct {
	// Atomic maps for lock-free access
	currentlyActiveGVs atomic.Value // map[schema.GroupVersion]struct{}
	recentlyDeletedGVs atomic.Value // map[schema.GroupVersion]time.Time

	// Informers for fetching active GVs
	crdInformer         cache.SharedIndexInformer
	crdExtractor        GVExtractor
	apiServiceInformer  cache.SharedIndexInformer
	apiServiceExtractor GVExtractor

	// Worker 1: triggered by CRD/APIService events
	activeGVQueue workqueue.TypedRateLimitingInterface[string]
	// Worker 2: periodic reaper's configuration
	exclusionGracePeriod time.Duration
	reaperCheckInterval  time.Duration
	// Worker 3: triggered by Active/Deleted GV changes
	refilterQueue workqueue.TypedRateLimitingInterface[string]

	peerDiscoveryCache   *atomic.Value // peerProxyHandler.peerDiscoveryInfoCache
	invalidationCallback *atomic.Pointer[func()]
}

// NewGVExclusionManager creates a new GV exclusion manager.
func NewGVExclusionManager(
	exclusionGracePeriod time.Duration,
	reaperCheckInterval time.Duration,
	peerDiscoveryCache *atomic.Value,
	invalidationCallback *atomic.Pointer[func()],
) *GVExclusionManager {
	mgr := &GVExclusionManager{
		exclusionGracePeriod: exclusionGracePeriod,
		reaperCheckInterval:  reaperCheckInterval,
		peerDiscoveryCache:   peerDiscoveryCache,
		invalidationCallback: invalidationCallback,
		activeGVQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "active-gv-tracker",
			}),
		refilterQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "peer-discovery-refilter",
			}),
	}

	mgr.currentlyActiveGVs.Store(map[schema.GroupVersion]struct{}{})
	mgr.recentlyDeletedGVs.Store(map[schema.GroupVersion]time.Time{})

	return mgr
}

// RegisterCRDInformerHandlers registers event handlers for CRD informer using a custom extractor.
// The extractor function is responsible for extracting GroupVersions from CRD objects.
func (m *GVExclusionManager) RegisterCRDInformerHandlers(crdInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	m.crdInformer = crdInformer
	m.crdExtractor = extractor
	_, err := m.crdInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			m.handleGVUpdate()
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			m.handleGVUpdate()
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
			m.onDeleteEvent(gvs)
		},
	})
	return err
}

// RegisterAPIServiceInformerHandlers registers event handlers for APIService informer using a custom extractor.
// The extractor function is responsible for extracting GroupVersions from APIService objects
// and determining if they represent aggregated APIs.
func (m *GVExclusionManager) RegisterAPIServiceInformerHandlers(apiServiceInformer cache.SharedIndexInformer, extractor GVExtractor) error {
	m.apiServiceInformer = apiServiceInformer
	m.apiServiceExtractor = extractor
	_, err := m.apiServiceInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			m.handleGVUpdate()
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			m.handleGVUpdate()
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
			m.onDeleteEvent(gvs)
		},
	})
	return err
}

// WaitForCacheSync waits for the informer caches to sync.
func (m *GVExclusionManager) WaitForCacheSync(stopCh <-chan struct{}) bool {
	synced := []cache.InformerSynced{}
	if m.crdInformer != nil {
		synced = append(synced, m.crdInformer.HasSynced)
	}
	if m.apiServiceInformer != nil {
		synced = append(synced, m.apiServiceInformer.HasSynced)
	}
	if len(synced) == 0 {
		return true
	}
	return cache.WaitForNamedCacheSync("gv-exclusion-manager", stopCh, synced...)
}

// getExclusionSet returns the combined exclusion set i.e., union(currentlyActiveGVs, recentlyDeletedGVs)
func (m *GVExclusionManager) getExclusionSet() map[schema.GroupVersion]struct{} {
	activeMap := m.loadCurrentlyActiveGVs()
	deletedMap := m.loadRecentlyDeletedGVs()

	exclusionSet := make(map[schema.GroupVersion]struct{}, len(activeMap)+len(deletedMap))
	for gv := range activeMap {
		exclusionSet[gv] = struct{}{}
	}
	for gv := range deletedMap {
		exclusionSet[gv] = struct{}{}
	}

	return exclusionSet
}

// handleGVUpdate is called when a CRD or APIService event occurs.
// This triggers Worker 1 to rebuild the active GV set.
func (m *GVExclusionManager) handleGVUpdate() {
	m.activeGVQueue.Add("sync")
}

// onDeleteEvent is called when a CRD or APIService is deleted.
// This removes the GVs from currentlyActiveGVs, adds them to recentlyDeletedGVs
// with current timestamp, and queues Worker 3 to refilter the peer discovery cache.
func (m *GVExclusionManager) onDeleteEvent(gvs []schema.GroupVersion) {
	if len(gvs) == 0 {
		return
	}

	// Remove from currentlyActiveGVs
	activeMap := m.loadCurrentlyActiveGVs()
	newActiveMap := make(map[schema.GroupVersion]struct{}, len(activeMap))
	for k, v := range activeMap {
		newActiveMap[k] = v
	}
	for _, gv := range gvs {
		delete(newActiveMap, gv)
	}
	m.currentlyActiveGVs.Store(newActiveMap)

	// Add to recentlyDeletedGVs
	deletedMap := m.loadRecentlyDeletedGVs()
	newDeletedMap := make(map[schema.GroupVersion]time.Time, len(deletedMap)+len(gvs))
	for k, v := range deletedMap {
		newDeletedMap[k] = v
	}
	now := time.Now()
	for _, gv := range gvs {
		newDeletedMap[gv] = now
		klog.V(4).Infof("GV %s deleted: moved to recentlyDeletedGVs", gv.String())
	}
	m.recentlyDeletedGVs.Store(newDeletedMap)

	// Queue Worker 3
	m.refilterQueue.Add("refilter")
}

// RunPeerDiscoveryActiveGVTracker runs Worker 1: Active GV Tracker
// This worker is triggered by CRD/APIService events and rebuilds the active GV set.
func (m *GVExclusionManager) RunPeerDiscoveryActiveGVTracker(ctx context.Context, workers int) {
	defer m.activeGVQueue.ShutDown()

	klog.Infof("Starting %d Active GV Tracker worker(s)", workers)
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, m.runActiveGVTrackerWorker, time.Second)
	}

	<-ctx.Done()
	klog.Info("Active GV Tracker workers stopped")
}

func (m *GVExclusionManager) runActiveGVTrackerWorker(ctx context.Context) {
	for m.processNextActiveGV(ctx) {
	}
}

func (m *GVExclusionManager) processNextActiveGV(ctx context.Context) bool {
	key, shutdown := m.activeGVQueue.Get()
	if shutdown {
		return false
	}
	defer m.activeGVQueue.Done(key)

	select {
	case <-ctx.Done():
		return false
	default:
	}

	m.reconcileActiveGVs()
	return true
}

// reconcileActiveGVs fetches all GVs from CRD and APIService informers,
// detects diffs with the previous state, and queues Worker 3 if changes detected.
func (m *GVExclusionManager) reconcileActiveGVs() {
	freshGVs := make(map[schema.GroupVersion]struct{})

	// Fetch GVs from CRD informer
	if m.crdInformer != nil && m.crdExtractor != nil {
		crdList := m.crdInformer.GetStore().List()
		for _, item := range crdList {
			gvs := m.crdExtractor(item)
			for _, gv := range gvs {
				freshGVs[gv] = struct{}{}
			}
		}
	}

	// Fetch GVs from APIService informer
	if m.apiServiceInformer != nil && m.apiServiceExtractor != nil {
		apiSvcList := m.apiServiceInformer.GetStore().List()
		for _, item := range apiSvcList {
			gvs := m.apiServiceExtractor(item)
			for _, gv := range gvs {
				freshGVs[gv] = struct{}{}
			}
		}
	}

	// Load previous active GVs and detect diff
	previousGVs := m.loadCurrentlyActiveGVs()
	diffDetected := m.diffGVs(previousGVs, freshGVs)

	if diffDetected {
		m.currentlyActiveGVs.Store(freshGVs)
		klog.V(4).Infof("Active GVs updated: %d GVs now active", len(freshGVs))

		// Queue Worker 3 for re-filtering
		m.refilterQueue.Add("refilter")
	} else {
		klog.V(4).Infof("No diff detected in active GVs")
	}
}

// diffGVs checks if there's a presence/absence difference between two GV maps
func (m *GVExclusionManager) diffGVs(old, new map[schema.GroupVersion]struct{}) bool {
	if len(old) != len(new) {
		return true
	}

	for gv := range old {
		if _, exists := new[gv]; !exists {
			return true
		}
	}

	for gv := range new {
		if _, exists := old[gv]; !exists {
			return true
		}
	}

	return false
}

// RunPeerDiscoveryReaper runs Worker 2: Reaper
// This worker periodically removes expired GVs from recentlyDeletedGVs.
func (m *GVExclusionManager) RunPeerDiscoveryReaper(ctx context.Context) {
	klog.Infof("Starting GV Reaper with %s interval", m.reaperCheckInterval)
	ticker := time.NewTicker(m.reaperCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.reapExpiredGVs()
		case <-ctx.Done():
			klog.Info("GV Reaper stopped")
			return
		}
	}
}

// reapExpiredGVs removes GVs from recentlyDeletedGVs that have exceeded the grace period.
func (m *GVExclusionManager) reapExpiredGVs() {
	deletedMap := m.loadRecentlyDeletedGVs()
	now := time.Now()
	newDeletedMap := make(map[schema.GroupVersion]time.Time)
	anyReaped := false

	for gv, deletionTime := range deletedMap {
		if now.Sub(deletionTime) > m.exclusionGracePeriod {
			klog.V(4).Infof("Reaping GV %s (grace period expired)", gv.String())
			anyReaped = true
			// Don't add to new map (effectively removing it)
		} else {
			newDeletedMap[gv] = deletionTime
		}
	}

	if anyReaped {
		// Atomic swap
		m.recentlyDeletedGVs.Store(newDeletedMap)
	}
}

// RunPeerDiscoveryRefilter runs Worker 3: Peer Discovery Re-filter
// This worker filters the peer discovery cache using the exclusion set.
func (m *GVExclusionManager) RunPeerDiscoveryRefilter(ctx context.Context, workers int) {
	defer m.refilterQueue.ShutDown()

	klog.Infof("Starting %d Peer Discovery Re-filter worker(s)", workers)
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, m.runRefilterWorker, time.Second)
	}

	<-ctx.Done()
	klog.Info("Peer Discovery Re-filter workers stopped")
}

func (m *GVExclusionManager) runRefilterWorker(ctx context.Context) {
	for m.processNextRefilter(ctx) {
	}
}

func (m *GVExclusionManager) processNextRefilter(ctx context.Context) bool {
	key, shutdown := m.refilterQueue.Get()
	if shutdown {
		return false
	}
	defer m.refilterQueue.Done(key)

	select {
	case <-ctx.Done():
		return false
	default:
	}

	m.refilterPeerDiscoveryCache()
	return true
}

// refilterPeerDiscoveryCache filters the peer discovery cache using the exclusion set.
// Only updates the cache and calls invalidation callback if filtering actually changed content.
func (m *GVExclusionManager) refilterPeerDiscoveryCache() {
	if m.peerDiscoveryCache == nil {
		klog.Warning("peerDiscoveryCache reference not set")
		return
	}

	cache := m.peerDiscoveryCache.Load()
	if cache == nil {
		klog.V(4).Infof("Peer discovery cache is empty, skipping re-filter")
		return
	}

	cacheMap, ok := cache.(map[string]PeerDiscoveryCacheEntry)
	if !ok {
		klog.Warning("Invalid cache type in peerDiscoveryInfoCache")
		return
	}

	if len(cacheMap) == 0 {
		klog.V(4).Infof("Peer discovery cache is empty, skipping re-filter")
		return
	}

	// Filter the cache
	filteredCache, changed := m.FilterPeerDiscoveryCache(cacheMap)

	if changed {
		// Atomic swap peer cache
		m.peerDiscoveryCache.Store(filteredCache)

		// Call invalidation callback
		if m.invalidationCallback != nil {
			if callback := m.invalidationCallback.Load(); callback != nil {
				(*callback)()
			}
		}

		klog.V(4).Infof("Peer discovery cache re-filtered, %d GVs excluded", len(m.getExclusionSet()))
	} else {
		klog.V(4).Infof("No changes after filtering peer discovery cache")
	}
}

// FilterPeerDiscoveryCache applies the current exclusion set to the provided peer cache.
// This should be called when new peers are added to ensure their discovery is filtered.
// Returns the filtered cache and a boolean indicating if any changes were made.
func (m *GVExclusionManager) FilterPeerDiscoveryCache(cacheMap map[string]PeerDiscoveryCacheEntry) (map[string]PeerDiscoveryCacheEntry, bool) {
	exclusionSet := m.getExclusionSet()
	if len(exclusionSet) == 0 {
		return cacheMap, false
	}

	filtered := make(map[string]PeerDiscoveryCacheEntry, len(cacheMap))
	anyChanged := false

	for peerID, entry := range cacheMap {
		filteredEntry, changed := m.filterPeerCacheEntry(entry, exclusionSet)
		filtered[peerID] = filteredEntry
		if changed {
			anyChanged = true
		}
	}

	return filtered, anyChanged
}

// filterPeerCacheEntry filters a single peer's cache entry for excluded GVs.
// Returns the filtered entry and whether any GVs were excluded.
func (m *GVExclusionManager) filterPeerCacheEntry(
	entry PeerDiscoveryCacheEntry,
	exclusionSet map[schema.GroupVersion]struct{},
) (PeerDiscoveryCacheEntry, bool) {
	filteredGVRs := make(map[schema.GroupVersionResource]bool, len(entry.GVRs))
	anyExcluded := false

	for existingGVR, v := range entry.GVRs {
		gv := schema.GroupVersion{Group: existingGVR.Group, Version: existingGVR.Version}
		if _, excluded := exclusionSet[gv]; excluded {
			anyExcluded = true
			continue
		}
		filteredGVRs[existingGVR] = v
	}

	if !anyExcluded {
		return entry, false
	}

	// Filter the group discovery list
	filteredGroups := m.filterGroupDiscovery(entry.GroupDiscovery, exclusionSet)

	return PeerDiscoveryCacheEntry{
		GVRs:           filteredGVRs,
		GroupDiscovery: filteredGroups,
	}, true
}

// filterGroupDiscovery filters group discovery entries, removing excluded GVs.
func (m *GVExclusionManager) filterGroupDiscovery(
	groupDiscoveries []apidiscoveryv2.APIGroupDiscovery,
	exclusionSet map[schema.GroupVersion]struct{},
) []apidiscoveryv2.APIGroupDiscovery {
	var filteredDiscovery []apidiscoveryv2.APIGroupDiscovery
	for _, groupDiscovery := range groupDiscoveries {
		filteredGroup := apidiscoveryv2.APIGroupDiscovery{
			ObjectMeta: groupDiscovery.ObjectMeta,
		}

		for _, version := range groupDiscovery.Versions {
			gv := schema.GroupVersion{Group: groupDiscovery.Name, Version: version.Version}
			if _, found := exclusionSet[gv]; found {
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

func (m *GVExclusionManager) loadCurrentlyActiveGVs() map[schema.GroupVersion]struct{} {
	if val := m.currentlyActiveGVs.Load(); val != nil {
		if gvMap, ok := val.(map[schema.GroupVersion]struct{}); ok {
			return gvMap
		}
	}
	return map[schema.GroupVersion]struct{}{}
}

func (m *GVExclusionManager) loadRecentlyDeletedGVs() map[schema.GroupVersion]time.Time {
	if val := m.recentlyDeletedGVs.Load(); val != nil {
		if gvMap, ok := val.(map[schema.GroupVersion]time.Time); ok {
			return gvMap
		}
	}
	return map[schema.GroupVersion]time.Time{}
}
