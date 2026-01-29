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
// It runs two workers and a periodic ticker:
//  1. Active GV Tracker: Triggered on CRD/APIService events or reaper ticks,
//     rebuilds active GVs and reaps expired deleted GVs
//  2. Peer Discovery Re-filter: Rate-limited worker that filters peer cache
//  3. Reaper Ticker: Periodically triggers the Active GV Tracker to reap expired GVs
type GVExclusionManager struct {
	// Atomic maps for lock-free access
	currentlyActiveGVs atomic.Value // map[schema.GroupVersion]struct{}
	recentlyDeletedGVs atomic.Value // map[schema.GroupVersion]time.Time

	// Informers for fetching active GVs
	crdInformer         cache.SharedIndexInformer
	crdExtractor        GVExtractor
	apiServiceInformer  cache.SharedIndexInformer
	apiServiceExtractor GVExtractor

	// Worker 1: triggered by CRD/APIService events or reaper ticks
	activeGVQueue workqueue.TypedRateLimitingInterface[string]
	// Reaper ticker configuration
	exclusionGracePeriod time.Duration
	reaperCheckInterval  time.Duration
	// Worker 2: triggered by Active/Deleted GV changes
	refilterQueue workqueue.TypedRateLimitingInterface[string]

	// rawPeerDiscoveryCache is written only by peerLeaseQueue worker
	// when peer leases change
	rawPeerDiscoveryCache *atomic.Value // map[string]PeerDiscoveryCacheEntry
	// filteredPeerDiscoveryCache is written only by the refilter worker
	// when raw cache or exclusion set changes.
	filteredPeerDiscoveryCache atomic.Value // map[string]PeerDiscoveryCacheEntry
	invalidationCallback       *atomic.Pointer[func()]
}

// NewGVExclusionManager creates a new GV exclusion manager.
func NewGVExclusionManager(
	exclusionGracePeriod time.Duration,
	reaperCheckInterval time.Duration,
	rawPeerDiscoveryCache *atomic.Value,
	invalidationCallback *atomic.Pointer[func()],
) *GVExclusionManager {
	mgr := &GVExclusionManager{
		exclusionGracePeriod:  exclusionGracePeriod,
		reaperCheckInterval:   reaperCheckInterval,
		rawPeerDiscoveryCache: rawPeerDiscoveryCache,
		invalidationCallback:  invalidationCallback,
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
	mgr.filteredPeerDiscoveryCache.Store(map[string]PeerDiscoveryCacheEntry{})

	return mgr
}

// GetFilteredPeerDiscoveryCache returns the filtered peer discovery cache.
func (m *GVExclusionManager) GetFilteredPeerDiscoveryCache() map[string]PeerDiscoveryCacheEntry {
	if cache := m.filteredPeerDiscoveryCache.Load(); cache != nil {
		if cacheMap, ok := cache.(map[string]PeerDiscoveryCacheEntry); ok {
			return cacheMap
		}
	}
	return map[string]PeerDiscoveryCacheEntry{}
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
			m.handleGVUpdate()
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
			m.handleGVUpdate()
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
// This triggers reconciliation which rebuilds the active GV set
// and also reaps expired GVs if indicated so.
func (m *GVExclusionManager) handleGVUpdate() {
	m.activeGVQueue.Add("sync")
}

// RunPeerDiscoveryActiveGVTracker runs the Active GV Tracker worker.
// This worker is triggered by CRD/APIService events and
// rebuilds the active GV set and reaps expired GVs.
// Only a single worker is used to avoid race conditions on atomic store operations.
func (m *GVExclusionManager) RunPeerDiscoveryActiveGVTracker(ctx context.Context) {
	defer m.activeGVQueue.ShutDown()

	klog.Info("Starting Active GV Tracker worker")
	go wait.UntilWithContext(ctx, m.runActiveGVTrackerWorker, time.Second)

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

// reconcileActiveGVs does the following
// 1. fetches all GVs from CRD and APIService informers
// 2. detects diffs with the previous state
// 3. adds deleted GVs to recentlyDeletedGVs
// 4. reaps expired GVs, and queues Worker 2 if changes detected.
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
	deletedGVs, activeGVsChanged := diffGVs(previousGVs, freshGVs)

	// Update recentlyDeletedGVs: add newly deleted GVs and reap expired ones
	recentlyDeletedChanged := m.updateRecentlyDeletedGVs(deletedGVs)

	if activeGVsChanged || recentlyDeletedChanged {
		if activeGVsChanged {
			m.currentlyActiveGVs.Store(freshGVs)
			klog.V(4).Infof("Active GVs updated: %d GVs now active", len(freshGVs))
		}
		m.TriggerRefilter()
	} else {
		klog.V(4).Infof("No changes detected in active or recently deleted GVs")
	}
}

// updateRecentlyDeletedGVs adds newly deleted GVs to recentlyDeletedGVs
// and reaps expired ones. Returns true if any changes were made.
func (m *GVExclusionManager) updateRecentlyDeletedGVs(deletedGVs []schema.GroupVersion) bool {
	deletedMap := m.loadRecentlyDeletedGVs()
	// Early return if nothing to do
	if len(deletedGVs) == 0 && len(deletedMap) == 0 {
		return false
	}

	now := time.Now()
	newDeletedMap := make(map[schema.GroupVersion]time.Time, len(deletedMap)+len(deletedGVs))
	changed := false

	// Copy existing entries, reaping expired ones
	for gv, deletionTime := range deletedMap {
		if now.Sub(deletionTime) > m.exclusionGracePeriod {
			klog.V(4).Infof("Reaping GV %s (grace period expired)", gv.String())
			changed = true
			// Don't add to new map (effectively removing it)
		} else {
			newDeletedMap[gv] = deletionTime
		}
	}

	// Add newly deleted GVs
	for _, gv := range deletedGVs {
		newDeletedMap[gv] = now
		klog.V(4).Infof("GV %s deleted: moved to recentlyDeletedGVs", gv.String())
		changed = true
	}

	if changed {
		m.recentlyDeletedGVs.Store(newDeletedMap)
	}

	return changed
}

// diffGVs compares old and new GV maps, returns GVs that were deleted (in old but not new)
// and a boolean indicating if there were any changes (additions or deletions).
func diffGVs(old, new map[schema.GroupVersion]struct{}) ([]schema.GroupVersion, bool) {
	var deletedGVs []schema.GroupVersion
	hasChanges := len(old) != len(new)

	// Find deleted GVs (in old but not in new)
	for gv := range old {
		if _, exists := new[gv]; !exists {
			deletedGVs = append(deletedGVs, gv)
			hasChanges = true
		}
	}

	return deletedGVs, hasChanges
}

// RunPeerDiscoveryReaper runs Worker 2: Reaper
// This worker periodically triggers reconciliation which also reaps expired GVs.
func (m *GVExclusionManager) RunPeerDiscoveryReaper(ctx context.Context) {
	klog.Infof("Starting GV Reaper with %s interval", m.reaperCheckInterval)
	ticker := time.NewTicker(m.reaperCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Trigger reconciliation which will also reap expired GVs
			m.activeGVQueue.Add("sync")
		case <-ctx.Done():
			klog.Info("GV Reaper stopped")
			return
		}
	}
}

// RunPeerDiscoveryRefilter runs the Peer Discovery Re-filter worker.
// This worker filters the peer discovery cache using the exclusion set.
func (m *GVExclusionManager) RunPeerDiscoveryRefilter(ctx context.Context) {
	defer m.refilterQueue.ShutDown()

	klog.Info("Starting Peer Discovery Re-filter worker")
	go wait.UntilWithContext(ctx, m.runRefilterWorker, time.Second)

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

// refilterPeerDiscoveryCache reads the raw peer discovery cache,
// applies exclusion filtering, and stores the result to the filtered cache.
func (m *GVExclusionManager) refilterPeerDiscoveryCache() {
	var cacheMap map[string]PeerDiscoveryCacheEntry
	if m.rawPeerDiscoveryCache != nil {
		if rawCache := m.rawPeerDiscoveryCache.Load(); rawCache != nil {
			cacheMap, _ = rawCache.(map[string]PeerDiscoveryCacheEntry)
		}
	}

	if len(cacheMap) == 0 {
		m.filteredPeerDiscoveryCache.Store(map[string]PeerDiscoveryCacheEntry{})
		klog.V(4).Infof("Raw peer discovery cache is empty or unavailable")
		return
	}

	filteredCache := m.filterPeerDiscoveryCache(cacheMap)
	m.filteredPeerDiscoveryCache.Store(filteredCache)

	if m.invalidationCallback != nil {
		if callback := m.invalidationCallback.Load(); callback != nil {
			(*callback)()
		}
	}

	klog.V(4).Infof("Peer discovery cache re-filtered, %d GVs in exclusion set", len(m.getExclusionSet()))

}

// filterPeerDiscoveryCache applies the current exclusion set to the provided peer cache.
func (m *GVExclusionManager) filterPeerDiscoveryCache(cacheMap map[string]PeerDiscoveryCacheEntry) map[string]PeerDiscoveryCacheEntry {
	exclusionSet := m.getExclusionSet()
	if len(exclusionSet) == 0 {
		return cacheMap
	}

	filtered := make(map[string]PeerDiscoveryCacheEntry, len(cacheMap))
	for peerID, entry := range cacheMap {
		filtered[peerID] = m.filterPeerCacheEntry(entry, exclusionSet)
	}

	return filtered
}

// filterPeerCacheEntry filters a single peer's cache entry for excluded GVs.
func (m *GVExclusionManager) filterPeerCacheEntry(
	entry PeerDiscoveryCacheEntry,
	exclusionSet map[schema.GroupVersion]struct{},
) PeerDiscoveryCacheEntry {
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

	// If no GVRs were excluded, the exclusion set doesn't intersect with this peer's GVs,
	// so we can return the entry unchanged without filtering GroupDiscovery.
	if !anyExcluded {
		return entry
	}
	filteredGroups := m.filterGroupDiscovery(entry.GroupDiscovery, exclusionSet)

	return PeerDiscoveryCacheEntry{
		GVRs:           filteredGVRs,
		GroupDiscovery: filteredGroups,
	}
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

// TriggerRefilter triggers the refilter worker to apply exclusions to the filtered cache.
// This should be called by peerLeaseQueue after updating the raw peer discovery cache.
func (m *GVExclusionManager) TriggerRefilter() {
	m.refilterQueue.Add("refilter")
}
