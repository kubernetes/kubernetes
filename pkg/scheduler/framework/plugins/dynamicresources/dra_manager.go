/*
Copyright 2024 The Kubernetes Authors.

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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"reflect"
	"slices"
	"sync"

	"github.com/go-logr/logr"

	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	resourcelisters "k8s.io/client-go/listers/resource/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

var _ fwk.SharedDRAManager = &DefaultDRAManager{}

// DefaultDRAManager is the default implementation of SharedDRAManager. It obtains the DRA objects
// from API informers, and uses an AssumeCache and a map of in-flight allocations in order
// to avoid race conditions when modifying ResourceClaims.
type DefaultDRAManager struct {
	resourceClaimTracker  *claimTracker
	resourceSliceLister   *resourceSliceLister
	deviceClassLister     *deviceClassLister
	extendedResourceCache *extendedresourcecache.ExtendedResourceCache
}

// metadataRegistrationSynced is a test seam: it lets unit tests hold the
// handler registration unsynced while the informer and tracker are synced.
var metadataRegistrationSynced = func(registration cache.ResourceEventHandlerRegistration) cache.InformerSynced {
	return registration.HasSynced
}

func NewDRAManager(ctx context.Context, claimsCache *assumecache.AssumeCache, resourceSliceTracker *resourceslicetracker.Tracker, informerFactory informers.SharedInformerFactory) *DefaultDRAManager {
	logger := klog.FromContext(ctx)
	var acct *affinityAccounting
	if utilfeature.DefaultFeatureGate.Enabled(features.DRASharingAffinity) && utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity) {
		acct = &affinityAccounting{
			extractor: sharingaffinity.NewExtractor(256),
			slices:    resourceSliceTracker,
		}
	}
	manager := &DefaultDRAManager{
		resourceClaimTracker: &claimTracker{
			cache:               claimsCache,
			inFlightAllocations: make(map[types.UID]inFlightAllocation),
			allocatedDevices:    newAllocatedDevices(logger, acct),
			logger:              logger,
			affinity:            acct,
		},
		resourceSliceLister: &resourceSliceLister{tracker: resourceSliceTracker},
		deviceClassLister:   &deviceClassLister{classLister: informerFactory.Resource().V1().DeviceClasses().Lister()},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
		manager.extendedResourceCache = extendedresourcecache.NewExtendedResourceCache(logger)
	}

	// Reacting to events is more efficient than iterating over the list
	// repeatedly in PreFilter.
	if acct != nil {
		c := manager.resourceClaimTracker
		c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
		c.metadataSlices = resourceSliceTracker
		c.allocatedDevices.metadata = c.metadata
		c.initReconciler(ctx)
		handler := cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj any) { c.publishMetadata(false, affectedPools(obj)) },
			UpdateFunc: func(oldObj, newObj any) { c.publishMetadata(false, affectedPools(oldObj, newObj)) },
			DeleteFunc: func(obj any) { c.publishMetadata(false, affectedPools(obj)) },
		}
		registration, err := resourceSliceTracker.AddEventHandler(handler)
		if err != nil {
			logger.Error(err, "Registering sharing affinity metadata handler")
		} else {
			go func() {
				if cache.WaitForCacheSync(ctx.Done(), metadataRegistrationSynced(registration)) {
					if c.metadataSyncedHook != nil {
						c.metadataSyncedHook()
					}
					c.publishMetadata(true, nil)
				}
			}()
		}
	}
	c := manager.resourceClaimTracker
	claimRegistration := c.cache.AddEventHandler(c.allocatedDevices.handlers())
	if acct != nil {
		go func() {
			if cache.WaitForCacheSync(ctx.Done(), claimRegistrationSynced(claimRegistration)) {
				a := c.allocatedDevices
				unlock := a.lockOwners()
				a.claimsSynced = true
				a.revision++
				for key := range a.owners {
					a.enqueuePoolLocked(key.pool)
				}
				wasReady := a.startupReady
				changed := a.completeStartupLocked(logger)
				startupDone := !wasReady && a.startupReady
				unlock()
				c.notifyExclusionChanges(logger, changed, startupDone)
			}
		}()
	}

	return manager
}

func (s *DefaultDRAManager) ResourceClaims() fwk.ResourceClaimTracker {
	return s.resourceClaimTracker
}

func (s *DefaultDRAManager) ResourceSlices() fwk.ResourceSliceLister {
	return s.resourceSliceLister
}

func (s *DefaultDRAManager) DeviceClasses() fwk.DeviceClassLister {
	return s.deviceClassLister
}

// DeviceClassResolver will always return a valid interface implementation. It
// wraps a nil extendedresourcecache.ExtendedResourceCache if the feature is
// disabled.
//
// That's okay, extendedresourcecache.ExtendedResourceCache.GetDeviceClass
// returns nil if called for nil.
func (s *DefaultDRAManager) DeviceClassResolver() fwk.DeviceClassResolver {
	return s.extendedResourceCache
}

var _ fwk.ResourceSliceLister = &resourceSliceLister{}

type resourceSliceLister struct {
	tracker *resourceslicetracker.Tracker
}

func (l *resourceSliceLister) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return l.tracker.ListPatchedResourceSlices()
}

var _ fwk.DeviceClassLister = &deviceClassLister{}

type deviceClassLister struct {
	classLister resourcelisters.DeviceClassLister
}

func (l *deviceClassLister) Get(className string) (*resourceapi.DeviceClass, error) {
	return l.classLister.Get(className)
}

func (l *deviceClassLister) List() ([]*resourceapi.DeviceClass, error) {
	return l.classLister.List(labels.Everything())
}

var _ fwk.ResourceClaimTracker = &claimTracker{}

type claimTracker struct {
	// cache enables temporarily storing a newer claim object
	// while the scheduler has allocated it and the corresponding object
	// update from the apiserver has not been processed by the claim
	// informer callbacks. ResourceClaimTracker get added here in PreBind and removed by
	// the informer callback (based on the "newer than" comparison in the
	// assume cache).
	//
	// It uses cache.MetaNamespaceKeyFunc to generate object names, which
	// therefore are "<namespace>/<name>".
	//
	// This is necessary to ensure that reconstructing the resource usage
	// at the start of a pod scheduling cycle doesn't reuse the resources
	// assigned to such a claim. Alternatively, claim allocation state
	// could also get tracked across pod scheduling cycles, but that
	// - adds complexity (need to carefully sync state with informer events
	//   for claims and ResourceSlices)
	// - would make integration with cluster autoscaler harder because it would need
	//   to trigger informer callbacks.
	cache *assumecache.AssumeCache
	// inFlightMutex syncs access to inFlightAllocations.
	inFlightMutex sync.RWMutex
	// inFlightAllocations is a map from claim UUIDs to claim objects for those claims
	// for which allocation was triggered during a scheduling cycle and the
	// corresponding claim status update call in PreBind has not been done
	// yet. It also includes a reference count tracking how many actively
	// scheduling Pods in a PodGroup are using that pending allocation. If
	// another pod outside the PodGroup needs the claim, the pod is treated as
	// "not schedulable yet". For those pods, the cluster event for the
	// claim status update will make them schedulable.
	//
	// This mechanism avoids the following problem:
	// - Pod A triggers allocation for claim X.
	// - Pod B shares access to that claim and gets scheduled because
	//   the claim is assumed to be allocated.
	// - PreBind for pod B is called first, tries to update reservedFor and
	//   fails because the claim is not really allocated yet.
	//
	// We could avoid the ordering problem by allowing either pod A or pod B
	// to set the allocation. But that is more complicated and leads to another
	// problem:
	// - Pod A and B get scheduled as above.
	// - PreBind for pod A gets called first, then fails with a temporary API error.
	//   It removes the updated claim from the in-flight claims because of that.
	// - PreBind for pod B gets called next and succeeds with adding the
	//   allocation and its own reservedFor entry.
	// - The assume cache is now not reflecting that the claim is allocated,
	//   which could lead to reusing the same resource for some other claim.
	//
	// For pods in a PodGroup, a pending allocation may be shared among several
	// pods in the group. In the PreBind phase, the allocation will be written
	// for the first pod that succeeds. The scenario above is prevented by
	// keeping the pending allocation in-flight as long as it has not been
	// unreserved for every pod in the group, as tracked by
	// inFlightAllocationSharers.
	//
	// A sync.Map is used because in practice sharing of a claim between
	// pods is expected to be rare compared to per-pod claim, so we end up
	// hitting the "multiple goroutines read, write, and overwrite entries
	// for disjoint sets of keys" case that sync.Map is optimized for.
	inFlightAllocations       map[types.UID]inFlightAllocation
	allocatedDevices          *allocatedDevices
	logger                    klog.Logger
	affinity                  *affinityAccounting
	metadata                  *affinityMetadata
	metadataSlices            poolSliceLister
	publishMutex              sync.Mutex
	metadataSyncedHook        func()
	publishAfterListHook      func(string)
	publishBeforeCommitHook   func(string, string)
	reserveAfterListHook      func()
	reserveAfterCommitHook    func()
	reconcilerDone            chan struct{}
	reconcileProcessHook      func(poolName)
	reconcileAfterComputeHook func(poolName, types.UID)
	reconcileAfterCommitHook  func(poolName)
	listenersMu               sync.Mutex
	listeners                 map[uint64]exclusionListener
	nextListener              uint64
}

type inFlightAllocation struct {
	claim      *resourceapi.ResourceClaim
	sharers    int
	affinity   claimAffinity
	provenance []structured.DeviceAffinity
	tokens     sets.Set[reservationToken]
}

type reservationToken = types.UID

type planSource int

const (
	planNew planSource = iota
	planPending
)

type poolName struct {
	driver string
	pool   string
}

type poolMeta struct {
	complete           bool
	generation         int64
	digest             string
	lastCompleteDigest string
	epoch              uint64
}

type affinityMetadata struct {
	synced bool
	pools  map[poolName]poolMeta
}

type metaSnap = affinityMetadata

type affinityReservation struct {
	claimUID    types.UID
	claimRV     string
	claim       *resourceapi.ResourceClaim
	source      planSource
	provenance  []structured.DeviceAffinity
	fenceEpochs map[poolName]uint64
	token       reservationToken
}

type pendingAffinity struct {
	claimUID   types.UID
	allocation *resourceapi.AllocationResult
	provenance []structured.DeviceAffinity
}

type affinityPendingAllocator interface {
	ReserveClaimWithAffinity(context.Context, affinityReservation) error
	ReleaseClaimPendingAllocation(types.UID, reservationToken) bool
}

type affinityPendingReader interface {
	GetPendingAffinity(types.UID) (pendingAffinity, bool)
}

var (
	errAffinityStale               = errors.New("sharing affinity reservation is stale")
	errAffinityMetadataNotCoherent = errors.New("sharing affinity metadata not yet coherent")
	errAffinityNotReady            = errors.New("sharing affinity owner reconciliation not yet ready")
	// Wrapped together with errAffinityStale when only the claim's
	// ResourceVersion moved (labels, finalizers, ...). The claim queueing hint
	// skips such updates, so Reserve must request a backoff retry instead of
	// parking the pod as unschedulable.
	errAffinityClaimVersionChanged = errors.New("claim resource version changed")
)

func metadataView(all []*resourceapi.ResourceSlice, key poolName) poolMeta {
	var slices []*resourceapi.ResourceSlice
	for _, slice := range all {
		if slice.Spec.Driver == key.driver && slice.Spec.Pool.Name == key.pool {
			slices = append(slices, slice)
		}
	}
	return groupedMetadataView(slices, key)
}

// groupedMetadataView expects slices already filtered to key; grouping once per
// list keeps full-pool scans linear instead of quadratic in the number of pools.
func groupedMetadataView(slices []*resourceapi.ResourceSlice, key poolName) poolMeta {
	view := structured.ComputePoolMetadataView(slices, key.driver, key.pool)
	return poolMeta{complete: view.Complete, generation: view.Generation, digest: view.Digest}
}

func groupSlicesByPool(all []*resourceapi.ResourceSlice) map[poolName][]*resourceapi.ResourceSlice {
	byPool := make(map[poolName][]*resourceapi.ResourceSlice)
	for _, slice := range all {
		key := poolName{slice.Spec.Driver, slice.Spec.Pool.Name}
		byPool[key] = append(byPool[key], slice)
	}
	return byPool
}

// affectedPools returns the pools touched by an event. A nil result (unknown
// object) makes publishMetadata fall back to recomputing every pool.
func affectedPools(objs ...any) sets.Set[poolName] {
	keys := sets.New[poolName]()
	for _, obj := range objs {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		slice, ok := obj.(*resourceapi.ResourceSlice)
		if !ok || slice == nil {
			return nil
		}
		keys.Insert(poolName{slice.Spec.Driver, slice.Spec.Pool.Name})
	}
	return keys
}

func sameMetadata(a, b poolMeta) bool {
	return a.complete == b.complete && a.generation == b.generation && a.digest == b.digest
}

// Publication is level-triggered: callbacks may arrive out of order, but each
// lists the latest slices while holding the publisher lock through commit.
// A nil affected set recomputes every known pool.
func (c *claimTracker) publishMetadata(synced bool, affected sets.Set[poolName]) {
	c.publishMutex.Lock()
	changed := sets.New[poolName]()
	startupDone := false
	defer func() {
		c.publishMutex.Unlock()
		c.notifyExclusionChanges(c.logger, changed, startupDone)
	}()
	all, err := c.metadataSlices.ListPatchedResourceSlices()
	if err != nil {
		c.logger.Error(err, "Listing sharing affinity pool metadata")
		return
	}
	source := "event"
	if synced {
		source = "init"
	}
	if c.publishAfterListHook != nil {
		c.publishAfterListHook(source)
	}
	byPool := groupSlicesByPool(all)
	keys := affected
	if synced || keys == nil {
		keys = sets.New[poolName]()
		for key := range byPool {
			keys.Insert(key)
		}
		c.allocatedDevices.mutex.RLock()
		for key := range c.metadata.pools {
			keys.Insert(key)
		}
		c.allocatedDevices.mutex.RUnlock()
	}
	views := make(map[poolName]poolMeta, len(keys))
	for key := range keys {
		views[key] = groupedMetadataView(byPool[key], key)
		if c.publishBeforeCommitHook != nil {
			c.publishBeforeCommitHook(key.driver, key.pool)
		}
	}
	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()
	a := c.allocatedDevices
	a.mutex.Lock()
	defer a.mutex.Unlock()
	for key, view := range views {
		previous, found := c.metadata.pools[key]
		if found && sameMetadata(previous, view) {
			continue
		}
		view.epoch = previous.epoch + 1
		view.lastCompleteDigest = previous.lastCompleteDigest
		if view.lastCompleteDigest == "" {
			view.lastCompleteDigest = structured.EmptyExtractorDigest
		}
		if view.complete {
			view.lastCompleteDigest = view.digest
		}
		c.metadata.pools[key] = view
		changed.Insert(key)
		a.enqueuePoolLocked(key)
		a.revision++
	}
	if synced && !c.metadata.synced {
		c.metadata.synced = true
		for key := range a.owners {
			a.enqueuePoolLocked(key.pool)
		}
		a.revision++
	}
	wasReady := a.startupReady
	changed = changed.Union(a.completeStartupLocked(c.logger))
	startupDone = !wasReady && a.startupReady
	a.clearDrainedLocked()
}

func (c *claimTracker) gatherAllocatedStateWithMetadata() (*structured.AllocatedState, metaSnap, error) {
	a := c.allocatedDevices
	a.mutex.RLock()
	var metadata metaSnap
	if c.metadata != nil {
		metadata = metaSnap{synced: a.startupReady, pools: maps.Clone(c.metadata.pools)}
	}
	revision := a.revision
	a.mutex.RUnlock()
	state, err := c.GatherAllocatedState()
	if err != nil {
		return nil, metadata, err
	}
	if revision != a.Revision() {
		return nil, metadata, errClaimTrackerConcurrentModification
	}
	return state, metadata, nil
}

func (m metaSnap) coherent(all []*resourceapi.ResourceSlice) bool {
	if !m.synced {
		return false
	}
	_, ok := m.incoherentPool(all)
	return !ok
}

// incoherentPool reports a pool whose published metadata lags the lister, so
// the scheduling status can identify it without exposing extractor content.
func (m metaSnap) incoherentPool(all []*resourceapi.ResourceSlice) (poolName, bool) {
	byPool := groupSlicesByPool(all)
	keys := sets.New[poolName]()
	for key := range m.pools {
		keys.Insert(key)
	}
	for key := range byPool {
		keys.Insert(key)
	}
	for key := range keys {
		if !sameMetadata(m.pools[key], groupedMetadataView(byPool[key], key)) {
			return key, true
		}
	}
	return poolName{}, false
}

func cloneAffinityProvenance(in []structured.DeviceAffinity) []structured.DeviceAffinity {
	out := slices.Clone(in)
	for i := range out {
		out[i].Values = maps.Clone(in[i].Values)
	}
	return out
}

func (c *claimTracker) GetPendingAffinity(uid types.UID) (pendingAffinity, bool) {
	c.inFlightMutex.RLock()
	defer c.inFlightMutex.RUnlock()
	entry, found := c.inFlightAllocations[uid]
	if !found || entry.tokens == nil {
		return pendingAffinity{}, false
	}
	return pendingAffinity{claimUID: uid, allocation: entry.claim.Status.Allocation.DeepCopy(), provenance: cloneAffinityProvenance(entry.provenance)}, true
}

func (c *claimTracker) ReleaseClaimPendingAllocation(uid types.UID, token reservationToken) bool {
	c.inFlightMutex.Lock()
	var changed sets.Set[poolName]
	defer func() {
		c.inFlightMutex.Unlock()
		c.notifyExclusionChanges(c.logger, changed, false)
	}()
	entry, found := c.inFlightAllocations[uid]
	if !found || !entry.tokens.Has(token) {
		return false
	}
	entry.tokens.Delete(token)
	if len(entry.tokens) > 0 || entry.sharers > 0 {
		return false
	}
	changed = sets.New[poolName]()
	for pool := range claimPoolDevices(entry.claim) {
		changed.Insert(pool)
	}
	c.removeInFlightLocked(uid)
	return true
}

func (c *claimTracker) reservationIdentity(r affinityReservation) error {
	if r.claim == nil || r.claim.UID != r.claimUID || r.claim.Status.Allocation == nil || r.token == "" {
		return fmt.Errorf("invalid sharing affinity reservation: %w", errAffinityStale)
	}
	if isSpecialClaimName(r.claim.Name) {
		return nil
	}
	current, err := c.Get(r.claim.Namespace, r.claim.Name)
	if err == nil && current.UID == r.claimUID && current.Status.Allocation == nil && current.ResourceVersion != r.claimRV {
		return fmt.Errorf("claim %s/%s changed since Filter: %w: %w", r.claim.Namespace, r.claim.Name, errAffinityClaimVersionChanged, errAffinityStale)
	}
	// An equal resourceVersion cannot explain a different persisted
	// allocation; that is a competing owner, not a lagging cache.
	if err == nil && current.UID == r.claimUID && current.ResourceVersion == r.claimRV && current.Status.Allocation != nil &&
		!apiequality.Semantic.DeepEqual(current.Status.Allocation, r.claim.Status.Allocation) {
		return fmt.Errorf("claim %s/%s has a different persisted allocation: %w", r.claim.Namespace, r.claim.Name, errAffinityConflict)
	}
	if err != nil || current.UID != r.claimUID || current.ResourceVersion != r.claimRV || current.Status.Allocation != nil {
		return fmt.Errorf("claim %s/%s changed since Filter: %w", r.claim.Namespace, r.claim.Name, errAffinityStale)
	}
	return nil
}

func (c *claimTracker) reservationViews(r affinityReservation) (map[poolName]poolMeta, error) {
	all, err := c.metadataSlices.ListPatchedResourceSlices()
	if err != nil {
		return nil, err
	}
	views := make(map[poolName]poolMeta)
	for _, device := range r.provenance {
		key := poolName{device.Pool.Driver, device.Pool.Name}
		view := metadataView(all, key)
		if !view.complete || view.generation != device.Pool.Generation || view.digest != device.Digest {
			return nil, fmt.Errorf("pool %s/%s changed since Filter: %w", key.driver, key.pool, errAffinityStale)
		}
		views[key] = view
	}
	return views, nil
}

func (c *claimTracker) ReserveClaimWithAffinity(ctx context.Context, r affinityReservation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.metadata == nil || c.metadataSlices == nil {
		return errAffinityMetadataNotCoherent
	}
	if err := c.reservationIdentity(r); err != nil {
		return err
	}
	views, err := c.reservationViews(r)
	if err != nil {
		return err
	}
	if c.reserveAfterListHook != nil {
		c.reserveAfterListHook()
	}
	if err := c.commitAffinityReservation(r, views); err != nil {
		return err
	}
	if c.reserveAfterCommitHook != nil {
		c.reserveAfterCommitHook()
	}
	after, err := c.reservationViews(r)
	if err == nil && !reflect.DeepEqual(views, after) {
		err = errAffinityStale
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		c.ReleaseClaimPendingAllocation(r.claimUID, r.token)
	}
	return err
}

func (c *claimTracker) commitAffinityReservation(r affinityReservation, views map[poolName]poolMeta) error {
	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()
	a := c.allocatedDevices
	a.mutex.Lock()
	defer a.mutex.Unlock()
	// AssumeCache emits its callbacks after releasing its mutex, making its
	// Get safe as a leaf operation under the accounting locks.
	if err := c.reservationIdentity(r); err != nil {
		return err
	}
	inactive := sets.New[poolName]()
	for key, view := range views {
		current, found := c.metadata.pools[key]
		epoch, captured := r.fenceEpochs[key]
		if !hasExtractors(view) && !a.affinityRelevantLocked(key, view, true) {
			inactive.Insert(key)
			continue
		}
		if !found || !captured || !sameMetadata(current, view) || current.epoch != epoch {
			return fmt.Errorf("pool %s/%s publication changed since Filter: %w", key.driver, key.pool, errAffinityStale)
		}
	}
	if entry, found := c.inFlightAllocations[r.claimUID]; found && entry.tokens == nil {
		return fmt.Errorf("claim %s has a legacy pending allocation: %w", r.claimUID, errAffinityStale)
	}
	results := r.claim.Status.Allocation.Devices.Results
	if len(r.provenance) != len(results) {
		return fmt.Errorf("affinity provenance does not match claim allocation: %w", errAffinityStale)
	}
	seen := sets.New[int]()
	for _, device := range r.provenance {
		i := device.ResultIndex
		if i < 0 || i >= len(results) || seen.Has(i) {
			return fmt.Errorf("invalid affinity result index %d: %w", i, errAffinityStale)
		}
		seen.Insert(i)
		result := results[i]
		id := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
		if device.Device != id || device.Request != result.Request || device.Pool.Driver != result.Driver || device.Pool.Name != result.Pool {
			return fmt.Errorf("affinity provenance does not match result %d: %w", i, errAffinityStale)
		}
		if device.Contributes != (device.Constrained && result.ShareID != nil && (result.AdminAccess == nil || !*result.AdminAccess)) || device.Constrained && len(device.Values) == 0 {
			return fmt.Errorf("invalid affinity ownership for result %d: %w", i, errAffinityStale)
		}
	}
	for key := range views {
		if !a.poolReadyLocked(key) {
			if !c.metadata.synced {
				return fmt.Errorf("%w: %w", errAffinityNotReady, errAffinityMetadataNotCoherent)
			}
			return fmt.Errorf("pool %s/%s: %w", key.driver, key.pool, errAffinityNotReady)
		}
	}
	var contributions claimAffinity
	for _, device := range r.provenance {
		result := results[device.ResultIndex]
		id := device.Device
		if a.fenced.Has(id) {
			return fmt.Errorf("device %s requires drain: %w", id, errAffinityConflict)
		}
		record := &deviceAffinity{}
		if observed := a.affinityDevices[id]; observed != nil {
			record = observed.clone()
		}
		for _, contribution := range contributions {
			if contribution.device == id {
				record.insert(c.logger, contribution)
			}
		}
		state, present := foldDeviceAffinity(record)
		if present && (state.Status == structured.AffinityStatusUnreconstructable || device.Constrained && !maps.Equal(state.LockedAffinity, device.Values)) {
			return fmt.Errorf("device %s has incompatible sharing affinity: %w", id, errAffinityConflict)
		}
		if device.Contributes {
			contributions = append(contributions, affinityContribution{
				device: id, id: affinityContributionID{claimUID: r.claimUID, device: id, shareID: *result.ShareID},
				pool: device.Pool, request: result.Request, status: structured.AffinityStatusLocked, values: maps.Clone(device.Values),
			})
		}
	}
	if entry, found := c.inFlightAllocations[r.claimUID]; found {
		if !apiequality.Semantic.DeepEqual(entry.claim.Status.Allocation, r.claim.Status.Allocation) || !reflect.DeepEqual(entry.provenance, r.provenance) {
			return fmt.Errorf("claim %s has a different pending allocation: %w", r.claimUID, errAffinityConflict)
		}
		entry.tokens.Insert(r.token)
		a.revision++
		return nil
	}
	c.inFlightAllocations[r.claimUID] = inFlightAllocation{
		claim: r.claim.DeepCopy(), affinity: normalizeClaimAffinity(c.logger, r.claimUID, contributions),
		provenance: cloneAffinityProvenance(r.provenance), tokens: sets.New(r.token),
	}
	a.changeInFlightOwnersLocked(r.claim, 1)
	perPool := make([]poolExtraction, 0, len(views))
	for pool, devices := range claimPoolDevices(r.claim) {
		result := poolExtraction{pool: pool, view: views[pool], devices: devices.UnsortedList()}
		for _, contribution := range contributions {
			if contribution.pool.Driver == pool.driver && contribution.pool.Name == pool.pool {
				result.contributions = append(result.contributions, contribution)
			}
		}
		perPool = append(perPool, result)
	}
	a.createInFlightRecordsLocked(c.inFlightAllocations[r.claimUID].claim, perPool)
	for pool := range inactive {
		rec := a.owners[ownerPoolKey{uid: r.claimUID, inFlight: true, pool: pool}]
		if rec == nil {
			// Admin-only pools have provenance, but no affinity owner record.
			continue
		}
		rec.view, rec.epoch, rec.pending, rec.reason, rec.inactive = views[pool], 0, false, "", true
	}
	return nil
}

// Tests use this seam to invalidate a snapshot deterministically.
var gatherAllocatedStateTestHook func()

func (c *claimTracker) GetPendingAllocation(claimUID types.UID) *resourceapi.AllocationResult {
	c.inFlightMutex.RLock()
	defer c.inFlightMutex.RUnlock()

	inFlight, found := c.inFlightAllocations[claimUID]
	if !found || inFlight.claim == nil {
		return nil
	}
	return inFlight.claim.Status.Allocation
}

func (c *claimTracker) SignalClaimPendingAllocation(claimUID types.UID, allocatedClaim *resourceapi.ResourceClaim) error {
	// CEL must not hold up unrelated in-flight accounting while it runs.
	var results []poolExtraction
	if c.affinity != nil {
		results = c.affinity.extractByPool(klog.NewContext(context.Background(), c.logger), allocatedClaim)
	}
	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()

	inFlight, found := c.inFlightAllocations[claimUID]
	if found {
		if inFlight.tokens != nil {
			return fmt.Errorf("claim %s has a tokenized sharing-affinity reservation", claimUID)
		}
		inFlight.sharers++
		c.inFlightAllocations[claimUID] = inFlight

		claim := inFlight.claim
		c.logger.V(5).Info("Added share for in-flight claim", "claim", klog.KObj(claim), "uid", claimUID, "version", claim.ResourceVersion, "sharers", inFlight.sharers)
		return nil
	}

	// Check that the claim really is unallocated. The Pod's CycleState may be
	// stale and this claim may have been allocated since that was calculated.
	//
	// Extended resources claims cannot be shared, so skip checking the assume
	// cache because we already know it's not allocated.
	if !isSpecialClaimName(allocatedClaim.Name) {
		assumedClaim, err := c.Get(allocatedClaim.Namespace, allocatedClaim.Name)
		if err != nil {
			return fmt.Errorf("look up assumed claim %s/%s, UID=%s: %w", allocatedClaim.Namespace, allocatedClaim.Name, claimUID, err)
		}
		if assumedClaim.UID == claimUID && assumedClaim.Status.Allocation != nil {
			c.logger.V(6).Info("Claim is already allocated, not creating in-flight", "claim", klog.KObj(assumedClaim), "uid", claimUID, "version", assumedClaim.ResourceVersion)
			return nil
		}
	}

	c.inFlightAllocations[claimUID] = inFlightAllocation{
		claim:   allocatedClaim,
		sharers: 1,
	}
	if c.affinity != nil {
		c.allocatedDevices.mutex.Lock()
		c.allocatedDevices.changeInFlightOwnersLocked(allocatedClaim, 1)
		c.allocatedDevices.createInFlightRecordsLocked(allocatedClaim, results)
		c.allocatedDevices.mutex.Unlock()
	}
	// This is the same verbosity as the corresponding log in the assume cache.
	c.logger.V(5).Info("Added in-flight claim", "claim", klog.KObj(allocatedClaim), "uid", claimUID, "version", allocatedClaim.ResourceVersion)
	// There's no reason to return an error in this implementation, but the error is helpful for other implementations.
	// For example, implementations that have to deal with fake claims might want to return an error if the allocation
	// is for an invalid claim.
	return nil
}

func (c *claimTracker) MaybeRemoveClaimPendingAllocation(claimUID types.UID, forceRemove bool) (deleted bool) {
	c.inFlightMutex.Lock()
	var changed sets.Set[poolName]
	defer func() {
		c.inFlightMutex.Unlock()
		c.notifyExclusionChanges(c.logger, changed, false)
	}()

	inFlight, found := c.inFlightAllocations[claimUID]
	// The assume cache doesn't log this, but maybe it should.
	if !found {
		c.logger.V(5).Info("Redundant remove of in-flight claim, not found", "uid", claimUID)
		return false
	}
	claim := inFlight.claim
	if len(inFlight.tokens) != 0 && !forceRemove {
		return false
	}

	if forceRemove || inFlight.sharers == 1 {
		changed = sets.New[poolName]()
		for pool := range claimPoolDevices(claim) {
			changed.Insert(pool)
		}
		c.removeInFlightLocked(claimUID)
		c.logger.V(5).Info("Removed in-flight claim", "claim", klog.KObj(claim), "uid", claimUID, "version", claim.ResourceVersion)
		return true
	}

	inFlight.sharers--
	c.inFlightAllocations[claimUID] = inFlight
	c.logger.V(5).Info("Claim is still shared by other pods, not removing in-flight claim", "claim", klog.KObj(claim), "uid", claimUID, "version", claim.ResourceVersion, "sharers", inFlight.sharers)
	return false
}

// The caller holds inFlightMutex. All accounting follows the same lock order:
// inFlightMutex before allocatedDevices.mutex.
func (c *claimTracker) removeInFlightLocked(uid types.UID) {
	entry, found := c.inFlightAllocations[uid]
	if !found {
		return
	}
	if c.affinity == nil {
		delete(c.inFlightAllocations, uid)
		return
	}
	c.allocatedDevices.mutex.Lock()
	defer c.allocatedDevices.mutex.Unlock()
	delete(c.inFlightAllocations, uid)
	c.allocatedDevices.changeInFlightOwnersLocked(entry.claim, -1)
	c.allocatedDevices.deleteInFlightRecordsLocked(uid)
}

func (c *claimTracker) checkInFlightMirrorLocked() bool {
	owners := make(map[structured.DeviceID]int)
	for _, entry := range c.inFlightAllocations {
		if entry.claim == nil || entry.claim.Status.Allocation == nil {
			continue
		}
		for _, result := range entry.claim.Status.Allocation.Devices.Results {
			if result.AdminAccess != nil && *result.AdminAccess {
				continue
			}
			owners[structured.MakeDeviceID(result.Driver, result.Pool, result.Device)]++
		}
	}
	return maps.Equal(owners, c.allocatedDevices.inFlightOwners)
}

func (c *claimTracker) Get(namespace, claimName string) (*resourceapi.ResourceClaim, error) {
	obj, err := c.cache.Get(namespace + "/" + claimName)
	if err != nil {
		return nil, err
	}
	claim, ok := obj.(*resourceapi.ResourceClaim)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T for assumed object %s/%s", obj, namespace, claimName)
	}
	return claim, nil
}

func (c *claimTracker) List() ([]*resourceapi.ResourceClaim, error) {
	var result []*resourceapi.ResourceClaim
	// Probably not worth adding an index for?
	objs := c.cache.List(nil)
	for _, obj := range objs {
		claim, ok := obj.(*resourceapi.ResourceClaim)
		if ok {
			result = append(result, claim)
		}
	}
	return result, nil
}

// errClaimTrackerConcurrentModification gets returned if ListAllAllocatedDevices
// or GatherAllocatedState need to be retried.
//
// There is a rare race when a claim is initially in-flight:
// - allocated is created from cache (claim not there)
// - someone removes from the in-flight claims and adds to the cache
// - we start checking in-flight claims (claim not there anymore)
// => claim ignored
//
// A proper fix would be to rewrite the assume cache, allocatedDevices,
// and the in-flight map so that they are under a single lock. But that's
// a pretty big change and prevents reusing the assume cache. So instead
// we check for changes in the set of allocated devices and keep trying
// until we get an attempt with no concurrent changes.
//
// A claim being first in the cache, then only in-flight cannot happen,
// so we don't need to re-check the in-flight claims.
var errClaimTrackerConcurrentModification = errors.New("conflicting concurrent modification")

func (c *claimTracker) ListAllAllocatedDevices() (a sets.Set[structured.DeviceID], err error) {
	c.logger.V(6).Info("Starting ListAllAllocatedDevices")
	defer func() {
		c.logger.V(6).Info("Finished ListAllAllocatedDevices", "allocatedDevices", logAllocatedDevices(c.logger, a), "err", err)
	}()

	// Start with a fresh set that matches the current known state of the
	// world according to the informers.
	allocated, revision := c.allocatedDevices.Get()

	// Whatever is in flight also has to be checked.
	for _, inFlight := range c.allInFlightAllocationsRLocked() {
		claim := inFlight.claim
		foreachAllocatedDevice(claim, func(deviceID structured.DeviceID) {
			c.logger.V(6).Info("Device is in flight for allocation", "device", deviceID, "claim", klog.KObj(claim))
			allocated.Insert(deviceID)
		}, false, func(structured.SharedDeviceID) {}, func(structured.DeviceConsumedCapacity) {})
	}

	if revision == c.allocatedDevices.Revision() {
		// Our current result is valid, nothing changed in the meantime.
		return allocated, nil
	}

	return nil, errClaimTrackerConcurrentModification
}

// GatherAllocatedState collects and returns the current allocation state of all devices
// across the cluster. This includes:
// - AllocatedDevices: Set of device IDs that are fully allocated (dedicated mode)
// - AllocatedSharedDeviceIDs: Set of device IDs with shared allocations when consumable capacity is enabled
// - AggregatedCapacity: Consumed capacity across all devices when consumable capacity is enabled
//
// The function handles two allocation models:
//  1. Legacy dedicated mode (DRAConsumableCapacity disabled): Devices are allocated exclusively
//     to a single claim. Shared devices are converted to their base device IDs.
//  2. Consumable capacity mode (DRAConsumableCapacity enabled): Devices can be shared across
//     multiple claims with capacity tracking.
//
// The function ensures consistency by:
// - Reading allocation state from informer-backed cache
// - Including in-flight allocations that haven't been persisted yet
// - Using revision numbers to detect concurrent modifications and retry if needed
//
// Returns errClaimTrackerConcurrentModification if the state changed during collection,
// indicating the caller should retry.
func (c *claimTracker) GatherAllocatedState() (s *structured.AllocatedState, err error) {
	c.logger.V(6).Info("Starting GatherAllocatedState")
	defer func() {
		var a sets.Set[structured.DeviceID]
		var affinityDevices int
		if s != nil {
			a = s.AllocatedDevices
			affinityDevices = len(s.DeviceSharingAffinity.DeviceStates)
		}
		c.logger.V(6).Info("Finished GatherAllocatedState", "allocatedDevices", logAllocatedDevices(c.logger, a), "affinityDevices", affinityDevices, "err", err)
	}()
	if c.affinity != nil {
		c.inFlightMutex.RLock()
		c.allocatedDevices.mutex.RLock()
		state := c.gatherAffinityStateLocked()
		revision := c.allocatedDevices.revision
		c.allocatedDevices.mutex.RUnlock()
		c.inFlightMutex.RUnlock()
		if gatherAllocatedStateTestHook != nil {
			gatherAllocatedStateTestHook()
			if revision != c.allocatedDevices.Revision() {
				return nil, errClaimTrackerConcurrentModification
			}
		}
		return state, nil
	}

	// Start with a fresh set that matches the current known state of the
	// world according to the informers.
	enabledConsumableCapacity := utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity)

	allocated, revision1 := c.allocatedDevices.Get()
	allocatedSharedDeviceIDs, revision2 := c.allocatedDevices.GetSharedDeviceIDs()
	aggregatedCapacity, revision3 := c.allocatedDevices.Capacities()
	owners, revision4 := c.allocatedDevices.affinitySnapshot()
	c.allocatedDevices.mutex.RLock()
	fenced := c.allocatedDevices.fenced.Clone()
	var stalePools sets.Set[schedulerapi.PoolID]
	if c.affinity != nil {
		stalePools = sets.New[schedulerapi.PoolID]()
		for key := range c.allocatedDevices.owners {
			if !c.allocatedDevices.poolReadyLocked(key.pool) {
				stalePools.Insert(schedulerapi.PoolID{Driver: key.pool.driver, Pool: key.pool.pool})
			}
		}
	}
	revision5 := c.allocatedDevices.revision
	c.allocatedDevices.mutex.RUnlock()

	if revision1 != revision2 || revision2 != revision3 || revision3 != revision4 || revision4 != revision5 {
		// Already not consistent. Try again.
		return nil, errClaimTrackerConcurrentModification
	}

	if !enabledConsumableCapacity {
		// When the DRAConsumableCapacity feature is disabled, we fall back to the legacy
		// dedicated device allocation model.
		// This ensures backward compatibility with the original DRA behavior where devices
		// could only be allocated exclusively to a single claim.
		for deviceID := range allocatedSharedDeviceIDs {
			allocated.Insert(deviceID)
		}
		// Reset allocatedSharedDeviceIDs and aggregatedCapacity
		allocatedSharedDeviceIDs = sets.New[structured.DeviceID]()
		aggregatedCapacity = make(schedulerapi.ConsumedCapacityCollection)
	}

	// Whatever is in flight also has to be checked.
	for _, inFlight := range c.allInFlightAllocationsRLocked() {
		claim := inFlight.claim
		foreachAllocatedDevice(claim,
			func(deviceID structured.DeviceID) { // dedicatedDeviceCallback
				c.logger.V(6).Info("Device is in flight for allocation", "device", deviceID, "claim", klog.KObj(claim))
				allocated.Insert(deviceID)
			},
			enabledConsumableCapacity,
			func(sharedDeviceID structured.SharedDeviceID) { // sharedDeviceCallback
				c.logger.V(6).Info("Device is in flight for allocation", "shared device", sharedDeviceID, "claim", klog.KObj(claim))
				allocatedSharedDeviceIDs.Insert(sharedDeviceID.GetDeviceID())
			},
			func(capacity structured.DeviceConsumedCapacity) { // consumedCapacityCallback
				c.logger.V(6).Info("Device is in flight for allocation", "consumed capacity", capacity, "claim", klog.KObj(claim))
				aggregatedCapacity.Insert(capacity)
			})
	}
	var affinityStates structured.AffinityStateCollection
	if c.affinity != nil {
		affinityStates = make(structured.AffinityStateCollection, len(owners))
		for device, record := range owners {
			if state, present := foldDeviceAffinity(record); present {
				affinityStates[device] = state
			}
		}
		for device := range fenced {
			affinityStates[device] = structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}
		}
	}
	if gatherAllocatedStateTestHook != nil {
		gatherAllocatedStateTestHook()
	}
	if revision1 == c.allocatedDevices.Revision() {
		// Our current result is valid, nothing changed in the meantime.
		return &structured.AllocatedState{
			AllocatedDevices:         allocated,
			AllocatedSharedDeviceIDs: allocatedSharedDeviceIDs,
			AggregatedCapacity:       aggregatedCapacity,
			DeviceSharingAffinity: structured.SharingAffinityState{
				DeviceStates: affinityStates,
				StalePools:   stalePools,
			},
		}, nil
	}

	return nil, errClaimTrackerConcurrentModification
}

func (c *claimTracker) allInFlightAllocationsRLocked() iter.Seq2[types.UID, inFlightAllocation] {
	return func(yield func(types.UID, inFlightAllocation) bool) {
		c.inFlightMutex.RLock()
		defer c.inFlightMutex.RUnlock()
		for uid, inFlight := range c.inFlightAllocations {
			if !yield(uid, inFlight) {
				return
			}
		}
	}
}

func (c *claimTracker) AssumeClaimAfterAPICall(claim *resourceapi.ResourceClaim) error {
	return c.cache.Assume(claim)
}

func (c *claimTracker) AssumedClaimRestore(namespace, claimName string) {
	c.cache.Restore(namespace + "/" + claimName)
}

// At V(6), log only a limited number of devices to avoid blowing up logs. For
// many E2E tests, 10 devices is enough for all devices without having to
// truncate, at least when running the tests sequentially.
const maxDevicesLevel6 = 10

// logAllocatedDevices returns a handle for the value in a structured log call which
// includes varying amounts of information about the allocated devices, depending on
// the verbosity of the logger.
func logAllocatedDevices(logger klog.Logger, allocatedDevices sets.Set[structured.DeviceID]) any {
	// We need to check verbosity here because our caller's source code
	// location may be relevant (-vmodule !).
	helper, logger := logger.WithCallStackHelper()
	helper()

	// We always produce the same output at V <= 5. 6 adds all IDs.
	verbosity := 5
	for i := 7; i > verbosity; i-- {
		if loggerV := logger.V(i); loggerV.Enabled() {
			verbosity = i
			break
		}
	}

	return &allocatedDevicesLogger{verbosity, allocatedDevices}
}

type allocatedDevicesLogger struct {
	verbosity int
	devices   sets.Set[structured.DeviceID]
}

var _ logr.Marshaler = &allocatedDevicesLogger{}

func (a *allocatedDevicesLogger) MarshalLog() any {
	if a.verbosity < 6 {
		return nil
	}

	info := struct {
		Count   int      `json:"count"`
		Devices []string `json:"devices"`
	}{
		Count: len(a.devices),
	}
	ids := make([]string, 0, len(a.devices))
	for id := range a.devices {
		ids = append(ids, id.String())
	}
	slices.Sort(ids)
	if a.verbosity == 6 && len(ids) > maxDevicesLevel6 {
		truncated := make([]string, 0, maxDevicesLevel6+1)
		truncated = append(truncated, ids[:maxDevicesLevel6/2]...)
		truncated = append(truncated, "...")
		truncated = append(truncated, ids[len(ids)-maxDevicesLevel6/2:]...)
		ids = truncated
	}
	info.Devices = ids

	return info
}
