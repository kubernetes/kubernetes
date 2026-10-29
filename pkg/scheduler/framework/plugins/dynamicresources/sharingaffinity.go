/*
Copyright The Kubernetes Authors.

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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
)

const sharingAffinityCacheSize = 1024

var (
	errSharingAffinityConflict = errors.New("sharing affinity conflict on device")
	errSharingAffinityStale    = errors.New("sharing affinity pool metadata changed or incomplete")
)

func newSharingAffinityExtractor() *sharingaffinity.Extractor {
	extractor := sharingaffinity.NewExtractor(sharingAffinityCacheSize)
	extractor.SetMetricsRecorder(sharingAffinityMetricsRecorder())
	return extractor
}

// sharingAffinityTracker maintains sharing affinity lock state across scheduling
// cycles solely for devices that have active shared allocations (allocated
// ResourceClaims or in-flight reservations with a non-nil ShareID and non-admin
// access), reconciling those claims against their referenced ResourceSlice pool
// metadata. Unallocated devices and pools without active shared claims are not
// tracked here; they are evaluated during PreFilter/Filter by the allocator.
type sharingAffinityTracker struct {
	claimTracker           *claimTracker
	ctx                    context.Context
	resourceSliceLister    fwk.ResourceSliceLister
	hasSliceTracker        bool
	extractor              *sharingaffinity.Extractor
	establishedDeviceLocks map[structured.DeviceID]*deviceLockBaseline
	incompleteDevices      sets.Set[structured.DeviceID]
	claimExtractionCache   map[claimExtractionKey]claimExtractionEntry
	state                  structured.DeviceAffinityStateCollection
	claimsRevision         int64
	stateValid             bool

	claimsMutex             sync.RWMutex
	allocatedClaims         map[types.UID]*resourceapi.ResourceClaim
	allocatedClaimsRevision int64
}

// deviceLockBaseline tracks the sticky sharing affinity lock established on a device.
// Once ambiguous is set, the device remains Unreconstructable until all active shared
// allocations on that device drain.
type deviceLockBaseline struct {
	established map[string]string
	ambiguous   bool
}

// claimExtractionKey uniquely identifies a cached CEL extraction result for a
// specific claim, request, pool generation/digest, and claim configuration.
type claimExtractionKey struct {
	claimUID          types.UID
	resourceVersion   string
	request           string
	driver            string
	pool              string
	poolGeneration    int64
	poolDigest        string
	claimConfigDigest [32]byte
}

// claimExtractionEntry holds the cached extracted affinity map and whether
// extraction succeeded for all required pool extractors.
type claimExtractionEntry struct {
	values map[string]string
	valid  bool
}

// activeClaimEntry pairs an allocated or in-flight ResourceClaim with any
// pre-computed ClaimAffinity captured during Filter/Reserve.
type activeClaimEntry struct {
	claim    *resourceapi.ResourceClaim
	affinity *structured.ClaimAffinity
}

// activeDeviceOwner represents a single non-admin shared allocation result on a
// device, including the owning claim, optional pre-computed affinity, and request name.
type activeDeviceOwner struct {
	claim    *resourceapi.ResourceClaim
	affinity *structured.ClaimAffinity
	request  string
}

// sharingAffinityMetricsRecorder returns a MetricsRecorder wired to the
// kube-scheduler sharing affinity Prometheus counters.
func sharingAffinityMetricsRecorder() sharingaffinity.MetricsRecorder {
	return sharingaffinity.MetricsRecorder{
		IncExtractorEvaluations: func() {
			if schedmetrics.SharingAffinityExtractorEvaluationsTotal != nil {
				schedmetrics.SharingAffinityExtractorEvaluationsTotal.Inc()
			}
		},
		IncLockConflicts: func() {
			if schedmetrics.SharingAffinityLockConflictTotal != nil {
				schedmetrics.SharingAffinityLockConflictTotal.Inc()
			}
		},
		IncNoKeysExtracted: func() {
			if schedmetrics.SharingAffinityNoKeysExtractedTotal != nil {
				schedmetrics.SharingAffinityNoKeysExtractedTotal.Inc()
			}
		},
	}
}

// claimMapKey returns a unique key for the given claim, using its UID if available,
// or falling back to a combination of namespace and name.
func claimMapKey(claim *resourceapi.ResourceClaim) types.UID {
	if claim.UID != "" {
		return claim.UID
	}
	return types.UID(claim.Namespace + "/" + claim.Name)
}

// addAllocatedClaim records an allocated shared ResourceClaim and increments the
// shared-claims revision counter.
func (sa *sharingAffinityTracker) addAllocatedClaim(claim *resourceapi.ResourceClaim) {
	sa.claimsMutex.Lock()
	defer sa.claimsMutex.Unlock()

	sa.allocatedClaims[claimMapKey(claim)] = claim
	sa.allocatedClaimsRevision++
}

// removeAllocatedClaim removes an allocated shared ResourceClaim and increments the
// shared-claims revision counter.
func (sa *sharingAffinityTracker) removeAllocatedClaim(claim *resourceapi.ResourceClaim) {
	sa.claimsMutex.Lock()
	defer sa.claimsMutex.Unlock()

	delete(sa.allocatedClaims, claimMapKey(claim))
	sa.allocatedClaimsRevision++
}

// getAllocatedClaims returns a copy of the current shared claims map and its revision.
func (sa *sharingAffinityTracker) getAllocatedClaims() (map[types.UID]*resourceapi.ResourceClaim, int64) {
	sa.claimsMutex.RLock()
	defer sa.claimsMutex.RUnlock()

	return maps.Clone(sa.allocatedClaims), sa.allocatedClaimsRevision
}

// getAllocatedClaimsRevision returns the current revision of allocated shared claims.
func (sa *sharingAffinityTracker) getAllocatedClaimsRevision() int64 {
	sa.claimsMutex.RLock()
	defer sa.claimsMutex.RUnlock()

	return sa.allocatedClaimsRevision
}

// enableSharingAffinity initializes sharing affinity tracking on the claimTracker,
// registers callbacks on allocatedDevices and the ResourceSlice tracker, and performs
// an initial synchronization of device affinity state.
func (c *claimTracker) enableSharingAffinity(ctx context.Context, sliceLister fwk.ResourceSliceLister, sliceTracker *resourceslicetracker.Tracker) {
	sa := &sharingAffinityTracker{
		claimTracker:           c,
		ctx:                    ctx,
		resourceSliceLister:    sliceLister,
		hasSliceTracker:        sliceTracker != nil,
		extractor:              newSharingAffinityExtractor(),
		establishedDeviceLocks: make(map[structured.DeviceID]*deviceLockBaseline),
		incompleteDevices:      sets.New[structured.DeviceID](),
		claimExtractionCache:   make(map[claimExtractionKey]claimExtractionEntry),
		allocatedClaims:        make(map[types.UID]*resourceapi.ResourceClaim),
	}
	c.inFlightMutex.Lock()
	c.sharingAffinity = sa
	c.inFlightMutex.Unlock()

	c.allocatedDevices.sharingAffinity = sa
	if sliceTracker != nil {
		sliceTracker.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    func(_ any) { sa.onResourceClaimOrSliceChanged() },
			UpdateFunc: func(_, _ any) { sa.onResourceClaimOrSliceChanged() },
			DeleteFunc: func(_ any) { sa.onResourceClaimOrSliceChanged() },
		})
	}
	sa.onResourceClaimOrSliceChanged()
}

// updateDeviceMetricsLocked updates the Prometheus gauges for currently locked
// and unreconstructable devices. The caller must hold the owning claimTracker's
// inFlightMutex.
func (sa *sharingAffinityTracker) updateDeviceMetricsLocked() {
	if sa == nil {
		return
	}
	var locked, unreconstructable int
	for deviceID, baseline := range sa.establishedDeviceLocks {
		if sa.incompleteDevices.Has(deviceID) || (baseline != nil && baseline.ambiguous) {
			unreconstructable++
		} else if baseline != nil && len(baseline.established) > 0 {
			locked++
		}
	}
	for deviceID := range sa.incompleteDevices {
		if _, ok := sa.establishedDeviceLocks[deviceID]; !ok {
			unreconstructable++
		}
	}
	if schedmetrics.SharingAffinityLockedDevices != nil {
		schedmetrics.SharingAffinityLockedDevices.Set(float64(locked))
	}
	if schedmetrics.SharingAffinityUnreconstructableDevices != nil {
		schedmetrics.SharingAffinityUnreconstructableDevices.Set(float64(unreconstructable))
	}
}

// onResourceClaimOrSliceChanged resynchronizes device sharing affinity state
// when an allocated ResourceClaim or ResourceSlice changes.
func (sa *sharingAffinityTracker) onResourceClaimOrSliceChanged() {
	if sa == nil || sa.claimTracker == nil {
		return
	}
	sa.claimTracker.inFlightMutex.Lock()
	defer sa.claimTracker.inFlightMutex.Unlock()
	_, _ = sa.syncDeviceAffinityLocked()
}

// pruneDrainedDeviceLocksLocked removes established locks and incomplete-device
// markers for devices that no longer have any active or in-flight shared allocations.
// The caller must hold the owning claimTracker's inFlightMutex.
func (sa *sharingAffinityTracker) pruneDrainedDeviceLocksLocked() {
	if sa == nil {
		return
	}
	sa.stateValid = false
	if len(sa.establishedDeviceLocks) == 0 && len(sa.incompleteDevices) == 0 {
		return
	}
	observedClaims, _ := sa.getAllocatedClaims()
	activeDevices := sets.New[structured.DeviceID]()
	recordClaimDevices := func(claim *resourceapi.ResourceClaim) {
		if claim == nil || claim.Status.Allocation == nil {
			return
		}
		for _, res := range claim.Status.Allocation.Devices.Results {
			if res.ShareID == nil || (res.AdminAccess != nil && *res.AdminAccess) {
				continue
			}
			activeDevices.Insert(structured.MakeDeviceID(res.Driver, res.Pool, res.Device))
		}
	}
	for _, claim := range observedClaims {
		recordClaimDevices(claim)
	}
	if sa.claimTracker != nil {
		for _, inFlight := range sa.claimTracker.inFlightAllocations {
			recordClaimDevices(inFlight.claim)
		}
	}
	for deviceID := range sa.establishedDeviceLocks {
		if !activeDevices.Has(deviceID) {
			delete(sa.establishedDeviceLocks, deviceID)
		}
	}
	for deviceID := range sa.incompleteDevices {
		if !activeDevices.Has(deviceID) {
			sa.incompleteDevices.Delete(deviceID)
		}
	}
	sa.updateDeviceMetricsLocked()
}

// gatherDeviceSharingAffinity returns a snapshot of the current device sharing
// affinity state. It returns a clone of the cached sa.state when valid, or
// recomputes establishedDeviceLocks, incompleteDevices, and sa.state from
// current allocated claims, in-flight reservations, and ResourceSlices when the
// cache has been invalidated (e.g., by an in-flight reservation change, an
// unprocessed claim revision, a prior slice list error, or when no sliceTracker
// event handler is wired).
func (sa *sharingAffinityTracker) gatherDeviceSharingAffinity() (structured.DeviceAffinityStateCollection, error) {
	if sa == nil || sa.claimTracker == nil {
		return nil, nil
	}
	c := sa.claimTracker
	c.inFlightMutex.RLock()
	claimsRevision := sa.getAllocatedClaimsRevision()
	// claimsRevision only tracks informer-observed claims; stateValid also
	// covers in-flight allocation changes and prior sync errors, and
	// hasSliceTracker ensures slice changes are pushed via event callbacks.
	if sa.hasSliceTracker && sa.stateValid && sa.claimsRevision == claimsRevision {
		state := sa.state.Clone()
		c.inFlightMutex.RUnlock()
		return state, nil
	}
	c.inFlightMutex.RUnlock()

	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()
	// Re-check under the exclusive lock in case another goroutine already
	// refreshed sa.state between RUnlock and Lock.
	claimsRevision = sa.getAllocatedClaimsRevision()
	if sa.hasSliceTracker && sa.stateValid && sa.claimsRevision == claimsRevision {
		return sa.state.Clone(), nil
	}
	return sa.syncDeviceAffinityLocked()
}

// listSlices returns all ResourceSlices from the configured ResourceSliceLister
// with DeviceTaintRules applied.
func (sa *sharingAffinityTracker) listSlices() ([]*draapi.ResourceSlice, error) {
	if sa == nil || sa.resourceSliceLister == nil {
		return nil, nil
	}
	return sa.resourceSliceLister.ListWithDeviceTaintRules()
}

// syncDeviceAffinityLocked reconciles device sharing affinity locks against the
// current ResourceClaims and ResourceSlices. The caller must hold the owning
// claimTracker's inFlightMutex.
func (sa *sharingAffinityTracker) syncDeviceAffinityLocked() (structured.DeviceAffinityStateCollection, error) {
	return sa.syncDeviceAffinityWithSlicesLocked(nil, nil)
}

// syncDeviceAffinityWithSlicesLocked reconciles establishedDeviceLocks and
// incompleteDevices from active shared allocations (observedClaims plus
// inFlightAllocations) and ResourceSlice pool metadata, returning a freshly
// built DeviceAffinityStateCollection. The caller must hold the owning
// claimTracker's inFlightMutex.
func (sa *sharingAffinityTracker) syncDeviceAffinityWithSlicesLocked(observedClaims map[types.UID]*resourceapi.ResourceClaim, resourceSlices []*draapi.ResourceSlice) (structured.DeviceAffinityStateCollection, error) {
	if sa == nil {
		return nil, nil
	}
	var inFlightAllocations map[types.UID]inFlightAllocation
	if sa.claimTracker != nil {
		inFlightAllocations = sa.claimTracker.inFlightAllocations
	}
	out := make(structured.DeviceAffinityStateCollection)

	var claimsRevision int64
	hasClaimsRevision := false
	if observedClaims == nil {
		observedClaims, claimsRevision = sa.getAllocatedClaims()
		hasClaimsRevision = true
	}
	callerProvidedSlices := resourceSlices != nil
	activeClaims := make(map[types.UID]activeClaimEntry, len(observedClaims)+len(inFlightAllocations))
	for uid, claim := range observedClaims {
		if claim != nil && claim.Status.Allocation != nil {
			activeClaims[uid] = activeClaimEntry{claim: claim}
		}
	}
	for uid, inFlight := range inFlightAllocations {
		if inFlight.claim == nil || inFlight.claim.Status.Allocation == nil {
			continue
		}
		if existing, ok := activeClaims[uid]; ok {
			// A claim can be in both maps during PreBind after UpdateStatus is
			// observed by the informer and before MaybeRemoveClaimPendingAllocation
			// runs; attach inFlight.affinity to reuse pre-computed Filter values.
			existing.affinity = inFlight.affinity
			activeClaims[uid] = existing
		} else {
			activeClaims[uid] = activeClaimEntry{
				claim:    inFlight.claim,
				affinity: inFlight.affinity,
			}
		}
	}

	// Evict cached CEL extraction results for claims that have been deallocated,
	// deleted, or unreserved so the cache stays bounded to active shared claims.
	for key := range sa.claimExtractionCache {
		if _, ok := activeClaims[key.claimUID]; !ok {
			delete(sa.claimExtractionCache, key)
		}
	}

	ownersByDevice := make(map[structured.DeviceID][]activeDeviceOwner)
	for _, entry := range activeClaims {
		for _, res := range entry.claim.Status.Allocation.Devices.Results {
			if res.ShareID == nil || (res.AdminAccess != nil && *res.AdminAccess) {
				continue
			}
			deviceID := structured.MakeDeviceID(res.Driver, res.Pool, res.Device)
			ownersByDevice[deviceID] = append(ownersByDevice[deviceID], activeDeviceOwner{
				claim: entry.claim,
				// affinity is non-nil only for in-flight claims; observed claims
				// fall back to CEL extraction in resolveOwnerValuesLocked.
				affinity: entry.affinity,
				request:  res.Request,
			})
		}
	}

	for deviceID := range sa.establishedDeviceLocks {
		if len(ownersByDevice[deviceID]) == 0 {
			delete(sa.establishedDeviceLocks, deviceID)
		}
	}
	for deviceID := range sa.incompleteDevices {
		if len(ownersByDevice[deviceID]) == 0 {
			sa.incompleteDevices.Delete(deviceID)
		}
	}

	if len(ownersByDevice) == 0 {
		if hasClaimsRevision && !callerProvidedSlices {
			sa.state = out.Clone()
			sa.claimsRevision = claimsRevision
			sa.stateValid = sa.hasSliceTracker
		}
		sa.updateDeviceMetricsLocked()
		return out, nil
	}

	if resourceSlices == nil {
		var err error
		resourceSlices, err = sa.listSlices()
		if err != nil {
			sa.stateValid = false
			return out, err
		}
	}

	poolViews := make(map[schedulerapi.PoolID]structured.PoolMetadataView)
	getPoolView := func(driver, pool string) structured.PoolMetadataView {
		poolID := schedulerapi.PoolID{Driver: driver, Pool: pool}
		if view, ok := poolViews[poolID]; ok {
			return view
		}
		view := structured.ComputePoolMetadataView(resourceSlices, driver, pool)
		poolViews[poolID] = view
		return view
	}

	for deviceID, owners := range ownersByDevice {
		slices.SortFunc(owners, func(a, b activeDeviceOwner) int {
			if cmp := strings.Compare(string(a.claim.UID), string(b.claim.UID)); cmp != 0 {
				return cmp
			}
			return strings.Compare(a.request, b.request)
		})

		driver, pool := deviceID.Driver.String(), deviceID.Pool.String()
		view := getPoolView(driver, pool)

		if !view.Complete {
			sa.incompleteDevices.Insert(deviceID)
			out[deviceID] = structured.DeviceAffinityState{
				Status: structured.AffinityStatusUnreconstructable,
			}
			continue
		}
		sa.incompleteDevices.Delete(deviceID)

		if len(view.Extractors) == 0 {
			baseline := sa.establishedDeviceLocks[deviceID]
			if baseline != nil && (baseline.ambiguous || len(baseline.established) > 0) {
				baseline.ambiguous = true
				baseline.established = nil
				out[deviceID] = structured.DeviceAffinityState{
					Status: structured.AffinityStatusUnreconstructable,
				}
			}
			continue
		}

		baseline := sa.establishedDeviceLocks[deviceID]
		if baseline == nil {
			baseline = &deviceLockBaseline{}
			sa.establishedDeviceLocks[deviceID] = baseline
		}

		if !baseline.ambiguous {
			poolExtractors, poolErr := sa.compilePoolExtractors(view, driver, pool)
			if poolErr != nil {
				baseline.ambiguous = true
				baseline.established = nil
			} else {
				var commonValues map[string]string
				for _, owner := range owners {
					values, ok := sa.resolveOwnerValuesLocked(deviceID, owner, poolExtractors, view)
					if !ok || len(values) == 0 {
						baseline.ambiguous = true
						baseline.established = nil
						break
					}
					if commonValues == nil {
						commonValues = values
					} else if !maps.Equal(commonValues, values) {
						baseline.ambiguous = true
						baseline.established = nil
						break
					}
				}
				if !baseline.ambiguous {
					if len(baseline.established) > 0 && !maps.Equal(baseline.established, commonValues) {
						baseline.ambiguous = true
						baseline.established = nil
					} else {
						baseline.established = maps.Clone(commonValues)
					}
				}
			}
		}

		if baseline.ambiguous {
			out[deviceID] = structured.DeviceAffinityState{
				Status: structured.AffinityStatusUnreconstructable,
			}
		} else {
			out[deviceID] = structured.DeviceAffinityState{
				Status:         structured.AffinityStatusLocked,
				LockedAffinity: maps.Clone(baseline.established),
			}
		}
	}

	if hasClaimsRevision && !callerProvidedSlices {
		sa.state = out.Clone()
		sa.claimsRevision = claimsRevision
		sa.stateValid = sa.hasSliceTracker
	}
	sa.updateDeviceMetricsLocked()
	return out, nil
}

// compilePoolExtractors validates extractor key uniqueness and compiles the CEL
// extractors for the given driver and pool generation.
func (sa *sharingAffinityTracker) compilePoolExtractors(view structured.PoolMetadataView, driver, pool string) (sharingaffinity.PoolExtractors, error) {
	extractors := sharingaffinity.PoolExtractors{
		Key:        sharingaffinity.PoolKey{Driver: driver, Name: pool, Generation: view.Generation},
		Extractors: view.Extractors,
	}
	names := sets.New[string]()
	for _, extractor := range view.Extractors {
		if names.Has(extractor.Name) {
			return extractors, fmt.Errorf("duplicate sharing affinity key %q", extractor.Name)
		}
		names.Insert(extractor.Name)
	}
	return sa.extractor.CompilePoolExtractors(sa.ctx, extractors)
}

// resolveOwnerValuesLocked returns the extracted sharing affinity key-value map
// for an active device owner, using its pre-computed in-flight ClaimAffinity when
// it matches the pool generation and digest, or evaluating CEL against the claim's
// opaque parameters otherwise. The caller must hold the owning claimTracker's
// inFlightMutex.
func (sa *sharingAffinityTracker) resolveOwnerValuesLocked(deviceID structured.DeviceID, owner activeDeviceOwner, poolExtractors sharingaffinity.PoolExtractors, view structured.PoolMetadataView) (map[string]string, bool) {
	if values, ok := findMatchingDeviceAffinity(owner.affinity, deviceID, owner.request, view); ok {
		return values, true
	}
	return sa.extractOwnerValuesLocked(owner.claim, owner.request, poolExtractors, view.Digest)
}

// findMatchingDeviceAffinity checks whether a pre-computed ClaimAffinity contains
// a consistent, non-empty affinity map for the given device, request, and pool
// generation/digest.
func findMatchingDeviceAffinity(affinity *structured.ClaimAffinity, deviceID structured.DeviceID, request string, view structured.PoolMetadataView) (map[string]string, bool) {
	if affinity == nil {
		return nil, false
	}
	var matched map[string]string
	for _, d := range affinity.Devices {
		if d.Device != deviceID || d.Request != request {
			continue
		}
		if !d.HasSharingAffinity || d.Pool.Generation != view.Generation || d.Digest != view.Digest || len(d.Values) == 0 {
			return nil, false
		}
		if matched != nil && !maps.Equal(matched, d.Values) {
			return nil, false
		}
		matched = d.Values
	}
	if matched == nil {
		return nil, false
	}
	return matched, true
}

// extractOwnerValuesLocked evaluates the pool's CEL extractors against a claim's
// DeviceClaimConfiguration for the given request, caching the result in
// claimExtractionCache. The caller must hold the owning claimTracker's
// inFlightMutex.
func (sa *sharingAffinityTracker) extractOwnerValuesLocked(claim *resourceapi.ResourceClaim, request string, poolExtractors sharingaffinity.PoolExtractors, poolDigest string) (map[string]string, bool) {
	if claim == nil || sa == nil {
		return nil, false
	}
	key := claimExtractionKey{
		claimUID:          claim.UID,
		resourceVersion:   claim.ResourceVersion,
		request:           request,
		driver:            poolExtractors.Key.Driver,
		pool:              poolExtractors.Key.Name,
		poolGeneration:    poolExtractors.Key.Generation,
		poolDigest:        poolDigest,
		claimConfigDigest: hashClaimDeviceConfigs(claim.Spec.Devices.Config),
	}
	if cached, ok := sa.claimExtractionCache[key]; ok {
		if !cached.valid {
			return nil, false
		}
		return cached.values, true
	}

	extractCtx := klog.NewContext(sa.ctx, klog.LoggerWithValues(klog.FromContext(sa.ctx), "claim", klog.KObj(claim)))
	extracted, err := sa.extractor.ExtractRequestAffinity(extractCtx, poolExtractors, parseResultRequest(request), claim.Spec.Devices.Config)
	if err != nil || !extracted.AllExtracted || len(extracted.Values) == 0 {
		sa.claimExtractionCache[key] = claimExtractionEntry{valid: false}
		return nil, false
	}
	cloned := maps.Clone(extracted.Values)
	sa.claimExtractionCache[key] = claimExtractionEntry{
		values: cloned,
		valid:  true,
	}
	return cloned, true
}

// hashClaimDeviceConfigs computes a SHA-256 digest over a slice of
// DeviceClaimConfiguration entries for extraction cache keying.
func hashClaimDeviceConfigs(configs []resourceapi.DeviceClaimConfiguration) [32]byte {
	h := sha256.New()
	writeString := func(s string) {
		var lenBuf [8]byte
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(s)))
		_, _ = h.Write(lenBuf[:])
		_, _ = h.Write([]byte(s))
	}
	var countBuf [8]byte
	binary.LittleEndian.PutUint64(countBuf[:], uint64(len(configs)))
	_, _ = h.Write(countBuf[:])
	for _, cfg := range configs {
		binary.LittleEndian.PutUint64(countBuf[:], uint64(len(cfg.Requests)))
		_, _ = h.Write(countBuf[:])
		for _, req := range cfg.Requests {
			writeString(req)
		}
		if cfg.Opaque != nil {
			writeString(cfg.Opaque.Driver)
			binary.LittleEndian.PutUint64(countBuf[:], uint64(len(cfg.Opaque.Parameters.Raw)))
			_, _ = h.Write(countBuf[:])
			_, _ = h.Write(cfg.Opaque.Parameters.Raw)
		} else {
			writeString("")
		}
	}
	var sum [32]byte
	copy(sum[:], h.Sum(nil))
	return sum
}

// parseResultRequest splits a DeviceRequestAllocationResult request string into
// its parent request and subrequest components, if present.
func parseResultRequest(name string) sharingaffinity.Request {
	parent, subrequest, found := strings.Cut(name, "/")
	if found {
		return sharingaffinity.Request{ParentName: parent, Name: subrequest}
	}
	return sharingaffinity.Request{Name: name}
}

// cloneClaimAffinity returns a deep copy of a ClaimAffinity.
func cloneClaimAffinity(affinity *structured.ClaimAffinity) *structured.ClaimAffinity {
	if affinity == nil {
		return nil
	}
	cloned := &structured.ClaimAffinity{
		Devices: make([]structured.DeviceAffinity, len(affinity.Devices)),
	}
	for i, d := range affinity.Devices {
		d.Values = maps.Clone(d.Values)
		cloned.Devices[i] = d
	}
	return cloned
}

// hasStaleNoSharingAffinity reports whether a pre-computed ClaimAffinity for a
// device and request is stale relative to a pool view that currently has no
// sharing affinity extractors.
func hasStaleNoSharingAffinity(affinity *structured.ClaimAffinity, deviceID structured.DeviceID, request string, view structured.PoolMetadataView) bool {
	if affinity == nil {
		return false
	}
	for _, d := range affinity.Devices {
		if d.Device != deviceID || d.Request != request {
			continue
		}
		if d.HasSharingAffinity || (d.Digest != "" && d.Digest != structured.EmptyExtractorDigest) || (d.Pool.Generation != 0 && d.Pool.Generation != view.Generation) {
			return true
		}
	}
	return false
}

// signalClaimPendingAllocation records an in-flight claim allocation during
// Reserve, validating and locking device sharing affinity when DRASharingAffinity
// is enabled.
func (pl *DynamicResources) signalClaimPendingAllocation(claimUID types.UID, allocatedClaim *resourceapi.ResourceClaim, affinity *structured.ClaimAffinity) error {
	if pl.sharingAffinityEnabled() {
		if tracker, ok := pl.draManager.ResourceClaims().(*claimTracker); ok {
			return tracker.ReserveClaimWithAffinity(claimUID, allocatedClaim, affinity)
		}
	}
	return pl.draManager.ResourceClaims().SignalClaimPendingAllocation(claimUID, allocatedClaim)
}

// ReserveClaimWithAffinity validates that a claim's proposed device allocation and
// extracted ClaimAffinity are still consistent with current ResourceSlice metadata
// and established device locks, then records the in-flight reservation and locks
// the target shared devices.
func (c *claimTracker) ReserveClaimWithAffinity(claimUID types.UID, allocatedClaim *resourceapi.ResourceClaim, affinity *structured.ClaimAffinity) error {
	if allocatedClaim == nil || allocatedClaim.UID != claimUID || allocatedClaim.Status.Allocation == nil {
		return fmt.Errorf("resource claim %s has no allocation: %w", claimUID, errSharingAffinityStale)
	}
	if !isSpecialClaimName(allocatedClaim.Name) && c.cache != nil {
		assumedClaim, err := c.Get(allocatedClaim.Namespace, allocatedClaim.Name)
		if err != nil {
			return fmt.Errorf("look up assumed claim %s/%s, UID=%s: %w", allocatedClaim.Namespace, allocatedClaim.Name, claimUID, err)
		}
		if assumedClaim.UID != claimUID {
			return fmt.Errorf("claim %s/%s UID changed since Filter: %w", allocatedClaim.Namespace, allocatedClaim.Name, errSharingAffinityStale)
		}
		if c.sharingAffinity != nil && assumedClaim.ResourceVersion != "" && allocatedClaim.ResourceVersion != "" && assumedClaim.ResourceVersion != allocatedClaim.ResourceVersion {
			return fmt.Errorf("claim %s/%s changed since Filter: %w", allocatedClaim.Namespace, allocatedClaim.Name, errSharingAffinityStale)
		}
		if assumedClaim.Status.Allocation != nil {
			if c.sharingAffinity != nil && !apiequality.Semantic.DeepEqual(assumedClaim.Status.Allocation, allocatedClaim.Status.Allocation) {
				return fmt.Errorf("claim %s/%s has a different persisted allocation: %w", allocatedClaim.Namespace, allocatedClaim.Name, errSharingAffinityConflict)
			}
			c.logger.V(6).Info("Claim is already allocated, not creating in-flight", "claim", klog.KObj(assumedClaim), "uid", claimUID, "version", assumedClaim.ResourceVersion)
			return nil
		}
	}

	if c.sharingAffinity == nil {
		return c.SignalClaimPendingAllocation(claimUID, allocatedClaim)
	}

	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()

	sa := c.sharingAffinity
	existingInFlight, alreadyInFlight := c.inFlightAllocations[claimUID]
	if affinity == nil && alreadyInFlight {
		affinity = existingInFlight.affinity
	}

	resourceSlices, err := sa.listSlices()
	if err != nil {
		return err
	}

	observedClaims, claimsRevision := sa.getAllocatedClaims()
	sharingState, err := sa.syncDeviceAffinityWithSlicesLocked(observedClaims, resourceSlices)
	if err != nil {
		return err
	}

	pendingDeviceLocks := make(map[structured.DeviceID]map[string]string)
	var compatibleReuses int

	for _, res := range allocatedClaim.Status.Allocation.Devices.Results {
		contributes := res.ShareID != nil && (res.AdminAccess == nil || !*res.AdminAccess)
		deviceID := structured.MakeDeviceID(res.Driver, res.Pool, res.Device)
		view := structured.ComputePoolMetadataView(resourceSlices, res.Driver, res.Pool)
		if !view.Complete {
			return fmt.Errorf("device %s in pool %s/%s: %w", deviceID, res.Driver, res.Pool, errSharingAffinityStale)
		}
		if len(view.Extractors) == 0 {
			if existing, ok := sharingState[deviceID]; ok && existing.Status == structured.AffinityStatusUnreconstructable {
				return fmt.Errorf("device %s is unreconstructable: %w", deviceID, errSharingAffinityConflict)
			}
			if hasStaleNoSharingAffinity(affinity, deviceID, res.Request, view) {
				return fmt.Errorf("device %s request %s affinity mismatch for pool %s/%s generation %d: %w",
					deviceID, res.Request, res.Driver, res.Pool, view.Generation, errSharingAffinityStale)
			}
			continue
		}
		if _, err := sa.compilePoolExtractors(view, res.Driver, res.Pool); err != nil {
			return fmt.Errorf("device %s in invalid pool %s/%s: %w", deviceID, res.Driver, res.Pool, err)
		}

		values, ok := findMatchingDeviceAffinity(affinity, deviceID, res.Request, view)
		if !ok {
			return fmt.Errorf("device %s request %s affinity mismatch for pool %s/%s generation %d: %w",
				deviceID, res.Request, res.Driver, res.Pool, view.Generation, errSharingAffinityStale)
		}

		existingLocked := false
		if existing, ok := sharingState[deviceID]; ok {
			if existing.Status != structured.AffinityStatusLocked || !maps.Equal(existing.LockedAffinity, values) {
				return fmt.Errorf("device %s locked affinity mismatch: %w", deviceID, errSharingAffinityConflict)
			}
			existingLocked = true
		}
		_, pendingLocked := pendingDeviceLocks[deviceID]
		if contributes {
			if prevValues, ok := pendingDeviceLocks[deviceID]; ok && !maps.Equal(prevValues, values) {
				return fmt.Errorf("device %s self-conflicting affinity in claim %s: %w", deviceID, claimUID, errSharingAffinityConflict)
			}
			pendingDeviceLocks[deviceID] = values
		}
		if !alreadyInFlight && (existingLocked || pendingLocked) {
			compatibleReuses++
		}
	}

	if alreadyInFlight {
		existingInFlight.sharers++
		if existingInFlight.affinity == nil && affinity != nil {
			existingInFlight.affinity = cloneClaimAffinity(affinity)
		}
		c.inFlightAllocations[claimUID] = existingInFlight
	} else {
		c.inFlightAllocations[claimUID] = inFlightAllocation{
			claim:    allocatedClaim,
			affinity: cloneClaimAffinity(affinity),
			sharers:  1,
		}
	}
	for deviceID, values := range pendingDeviceLocks {
		sa.incompleteDevices.Delete(deviceID)
		sa.establishedDeviceLocks[deviceID] = &deviceLockBaseline{
			established: maps.Clone(values),
		}
		if sharingState == nil {
			sharingState = make(structured.DeviceAffinityStateCollection)
		}
		sharingState[deviceID] = structured.DeviceAffinityState{
			Status:         structured.AffinityStatusLocked,
			LockedAffinity: maps.Clone(values),
		}
	}
	sa.state = sharingState.Clone()
	sa.claimsRevision = claimsRevision
	sa.stateValid = sa.hasSliceTracker
	sa.updateDeviceMetricsLocked()
	if compatibleReuses > 0 && schedmetrics.SharingAffinityCompatibleReuseTotal != nil {
		schedmetrics.SharingAffinityCompatibleReuseTotal.Add(float64(compatibleReuses))
	}
	return nil
}
