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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
)

var (
	errSharingAffinityConflict = errors.New("sharing affinity conflict on device")
	errSharingAffinityStale    = errors.New("sharing affinity pool metadata changed or incomplete")
)

type deviceLockBaseline struct {
	established map[string]string
	ambiguous   bool
}

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

type claimExtractionEntry struct {
	values map[string]string
	valid  bool
}

type activeClaimEntry struct {
	claim    *resourceapi.ResourceClaim
	affinity *structured.ClaimAffinity
}

type activeDeviceOwner struct {
	claim    *resourceapi.ResourceClaim
	affinity *structured.ClaimAffinity
	request  string
}

func (c *claimTracker) configureSharingAffinity(enabled bool) {
	c.inFlightMutex.Lock()
	c.enabledSharingAffinity = enabled
	if enabled {
		sharingaffinity.SetMetricsRecorder(sharingaffinity.MetricsRecorder{
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
		})
		if c.extractor == nil {
			c.extractor = sharingaffinity.NewExtractor(1024)
		}
		if c.establishedDeviceLocks == nil {
			c.establishedDeviceLocks = make(map[structured.DeviceID]*deviceLockBaseline)
		}
		if c.incompleteDevices == nil {
			c.incompleteDevices = sets.New[structured.DeviceID]()
		}
		if c.claimExtractionCache == nil {
			c.claimExtractionCache = make(map[claimExtractionKey]claimExtractionEntry)
		}
	} else {
		clear(c.establishedDeviceLocks)
		c.incompleteDevices.Clear()
		c.updateSharingAffinityDeviceMetricsLocked()
	}
	registerHandler := enabled && !c.sliceHandlerRegistered && c.resourceSliceTracker != nil
	if registerHandler {
		c.sliceHandlerRegistered = true
	}
	c.inFlightMutex.Unlock()

	if enabled {
		c.allocatedDevices.onClaimChanged = c.onResourceClaimOrSliceChanged
		if registerHandler {
			c.resourceSliceTracker.AddEventHandler(cache.ResourceEventHandlerFuncs{
				AddFunc:    func(_ any) { c.onResourceClaimOrSliceChanged() },
				UpdateFunc: func(_, _ any) { c.onResourceClaimOrSliceChanged() },
				DeleteFunc: func(_ any) { c.onResourceClaimOrSliceChanged() },
			})
		}
		c.onResourceClaimOrSliceChanged()
	} else {
		c.allocatedDevices.onClaimChanged = nil
	}
}

func (c *claimTracker) updateSharingAffinityDeviceMetricsLocked() {
	var locked, unreconstructable int
	for deviceID, baseline := range c.establishedDeviceLocks {
		if c.incompleteDevices.Has(deviceID) || (baseline != nil && baseline.ambiguous) {
			unreconstructable++
		} else if baseline != nil && len(baseline.established) > 0 {
			locked++
		}
	}
	for deviceID := range c.incompleteDevices {
		if _, ok := c.establishedDeviceLocks[deviceID]; !ok {
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

func (c *claimTracker) onResourceClaimOrSliceChanged() {
	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()
	if !c.enabledSharingAffinity {
		return
	}
	_, _ = c.syncDeviceAffinityLocked(nil, false)
}

func (c *claimTracker) pruneDrainedDeviceLocksLocked() {
	if !c.enabledSharingAffinity || (len(c.establishedDeviceLocks) == 0 && len(c.incompleteDevices) == 0) {
		return
	}
	observedClaims, _ := c.allocatedDevices.Claims()
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
	for _, inFlight := range c.inFlightAllocations {
		recordClaimDevices(inFlight.claim)
	}
	for deviceID := range c.establishedDeviceLocks {
		if !activeDevices.Has(deviceID) {
			delete(c.establishedDeviceLocks, deviceID)
		}
	}
	for deviceID := range c.incompleteDevices {
		if !activeDevices.Has(deviceID) {
			c.incompleteDevices.Delete(deviceID)
		}
	}
	c.updateSharingAffinityDeviceMetricsLocked()
}

func (c *claimTracker) gatherDeviceSharingAffinity(observedClaims map[types.UID]*resourceapi.ResourceClaim) (structured.SharingAffinityState, error) {
	if !c.enabledSharingAffinity {
		return structured.SharingAffinityState{}, nil
	}
	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()
	return c.syncDeviceAffinityLocked(observedClaims, true)
}

func (c *claimTracker) listSlices() ([]*draapi.ResourceSlice, error) {
	if c.resourceSliceLister == nil {
		return nil, nil
	}
	return c.resourceSliceLister.ListWithDeviceTaintRules()
}

func (c *claimTracker) syncDeviceAffinityLocked(observedClaims map[types.UID]*resourceapi.ResourceClaim, buildState bool) (structured.SharingAffinityState, error) {
	return c.syncDeviceAffinityWithSlicesLocked(observedClaims, nil, buildState)
}

func (c *claimTracker) syncDeviceAffinityWithSlicesLocked(observedClaims map[types.UID]*resourceapi.ResourceClaim, resourceSlices []*draapi.ResourceSlice, buildState bool) (structured.SharingAffinityState, error) {
	if !c.enabledSharingAffinity {
		return structured.SharingAffinityState{}, nil
	}
	var out structured.SharingAffinityState
	if buildState {
		out = structured.SharingAffinityState{
			DeviceStates: make(structured.AffinityStateCollection),
			StalePools:   sets.New[schedulerapi.PoolID](),
		}
	}

	if observedClaims == nil {
		observedClaims, _ = c.allocatedDevices.Claims()
	}
	activeClaims := make(map[types.UID]activeClaimEntry, len(observedClaims)+len(c.inFlightAllocations))
	for uid, claim := range observedClaims {
		if claim != nil && claim.Status.Allocation != nil {
			activeClaims[uid] = activeClaimEntry{claim: claim}
		}
	}
	for uid, inFlight := range c.inFlightAllocations {
		if inFlight.claim == nil || inFlight.claim.Status.Allocation == nil {
			continue
		}
		if existing, ok := activeClaims[uid]; ok {
			existing.affinity = inFlight.affinity
			activeClaims[uid] = existing
		} else {
			activeClaims[uid] = activeClaimEntry{
				claim:    inFlight.claim,
				affinity: inFlight.affinity,
			}
		}
	}

	for key := range c.claimExtractionCache {
		if _, ok := activeClaims[key.claimUID]; !ok {
			delete(c.claimExtractionCache, key)
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
				claim:    entry.claim,
				affinity: entry.affinity,
				request:  res.Request,
			})
		}
	}

	for deviceID := range c.establishedDeviceLocks {
		if len(ownersByDevice[deviceID]) == 0 {
			delete(c.establishedDeviceLocks, deviceID)
		}
	}
	for deviceID := range c.incompleteDevices {
		if len(ownersByDevice[deviceID]) == 0 {
			c.incompleteDevices.Delete(deviceID)
		}
	}

	if len(ownersByDevice) == 0 && !buildState {
		c.updateSharingAffinityDeviceMetricsLocked()
		return out, nil
	}

	if resourceSlices == nil {
		var err error
		resourceSlices, err = c.listSlices()
		if err != nil {
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

	if buildState {
		for _, slice := range resourceSlices {
			if slice == nil {
				continue
			}
			driverName, poolName := slice.Spec.Driver.String(), slice.Spec.Pool.Name.String()
			view := getPoolView(driverName, poolName)
			if !view.Complete {
				out.StalePools.Insert(schedulerapi.PoolID{Driver: driverName, Pool: poolName})
			}
		}
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
			c.incompleteDevices.Insert(deviceID)
			if buildState {
				out.StalePools.Insert(schedulerapi.PoolID{Driver: driver, Pool: pool})
				out.DeviceStates[deviceID] = structured.DeviceAffinityState{
					Status: structured.AffinityStatusUnreconstructable,
				}
			}
			continue
		}
		c.incompleteDevices.Delete(deviceID)

		if len(view.Extractors) == 0 {
			baseline := c.establishedDeviceLocks[deviceID]
			if baseline != nil && (baseline.ambiguous || len(baseline.established) > 0) {
				baseline.ambiguous = true
				baseline.established = nil
				if buildState {
					out.DeviceStates[deviceID] = structured.DeviceAffinityState{
						Status: structured.AffinityStatusUnreconstructable,
					}
				}
			}
			continue
		}

		baseline := c.establishedDeviceLocks[deviceID]
		if baseline == nil {
			baseline = &deviceLockBaseline{}
			c.establishedDeviceLocks[deviceID] = baseline
		}

		if !baseline.ambiguous {
			poolExtractors, poolErr := c.compilePoolExtractors(view, driver, pool)
			if poolErr != nil {
				baseline.ambiguous = true
				baseline.established = nil
			} else {
				var commonValues map[string]string
				for _, owner := range owners {
					values, ok := c.resolveOwnerValuesLocked(deviceID, owner, poolExtractors, view)
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

		if buildState {
			if baseline.ambiguous {
				out.DeviceStates[deviceID] = structured.DeviceAffinityState{
					Status: structured.AffinityStatusUnreconstructable,
				}
			} else {
				out.DeviceStates[deviceID] = structured.DeviceAffinityState{
					Status:         structured.AffinityStatusLocked,
					LockedAffinity: maps.Clone(baseline.established),
				}
			}
		}
	}

	c.updateSharingAffinityDeviceMetricsLocked()
	return out, nil
}

func (c *claimTracker) compilePoolExtractors(view structured.PoolMetadataView, driver, pool string) (sharingaffinity.PoolExtractors, error) {
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
	if err := c.extractor.Compile(c.ctx, extractors); err != nil {
		return extractors, err
	}
	return extractors, nil
}

func (c *claimTracker) resolveOwnerValuesLocked(deviceID structured.DeviceID, owner activeDeviceOwner, poolExtractors sharingaffinity.PoolExtractors, view structured.PoolMetadataView) (map[string]string, bool) {
	if values, ok := findMatchingDeviceAffinity(owner.affinity, deviceID, owner.request, view); ok {
		return values, true
	}
	return c.extractOwnerValuesLocked(owner.claim, owner.request, poolExtractors, view.Digest)
}

func findMatchingDeviceAffinity(affinity *structured.ClaimAffinity, deviceID structured.DeviceID, request string, view structured.PoolMetadataView) (map[string]string, bool) {
	if affinity == nil {
		return nil, false
	}
	var matched map[string]string
	for _, d := range affinity.Devices {
		if d.Device != deviceID || d.Request != request {
			continue
		}
		if !d.Constrained || d.Pool.Generation != view.Generation || d.Digest != view.Digest || len(d.Values) == 0 {
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

func (c *claimTracker) extractOwnerValuesLocked(claim *resourceapi.ResourceClaim, request string, poolExtractors sharingaffinity.PoolExtractors, poolDigest string) (map[string]string, bool) {
	if claim == nil {
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
	if cached, ok := c.claimExtractionCache[key]; ok {
		if !cached.valid {
			return nil, false
		}
		return cached.values, true
	}

	extractCtx := klog.NewContext(c.ctx, klog.LoggerWithValues(klog.FromContext(c.ctx), "claim", klog.KObj(claim)))
	extracted, err := c.extractor.Extract(extractCtx, poolExtractors, parseResultRequest(request), claim.Spec.Devices.Config)
	if err != nil || !extracted.Satisfied || len(extracted.Values) == 0 {
		c.claimExtractionCache[key] = claimExtractionEntry{valid: false}
		return nil, false
	}
	cloned := maps.Clone(extracted.Values)
	c.claimExtractionCache[key] = claimExtractionEntry{
		values: cloned,
		valid:  true,
	}
	return cloned, true
}

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

func parseResultRequest(name string) sharingaffinity.Request {
	parent, subrequest, found := strings.Cut(name, "/")
	if found {
		return sharingaffinity.Request{ParentName: parent, Name: subrequest}
	}
	return sharingaffinity.Request{Name: name}
}

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

func hasStaleUnconstrainedProvenance(affinity *structured.ClaimAffinity, deviceID structured.DeviceID, request string, view structured.PoolMetadataView) bool {
	if affinity == nil {
		return false
	}
	for _, d := range affinity.Devices {
		if d.Device != deviceID || d.Request != request {
			continue
		}
		if d.Constrained || (d.Digest != "" && d.Digest != structured.EmptyExtractorDigest) || (d.Pool.Generation != 0 && d.Pool.Generation != view.Generation) {
			return true
		}
	}
	return false
}

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
		if c.enabledSharingAffinity && assumedClaim.ResourceVersion != "" && allocatedClaim.ResourceVersion != "" && assumedClaim.ResourceVersion != allocatedClaim.ResourceVersion {
			return fmt.Errorf("claim %s/%s changed since Filter: %w", allocatedClaim.Namespace, allocatedClaim.Name, errSharingAffinityStale)
		}
		if assumedClaim.Status.Allocation != nil {
			if c.enabledSharingAffinity && !apiequality.Semantic.DeepEqual(assumedClaim.Status.Allocation, allocatedClaim.Status.Allocation) {
				return fmt.Errorf("claim %s/%s has a different persisted allocation: %w", allocatedClaim.Namespace, allocatedClaim.Name, errSharingAffinityConflict)
			}
			c.logger.V(6).Info("Claim is already allocated, not creating in-flight", "claim", klog.KObj(assumedClaim), "uid", claimUID, "version", assumedClaim.ResourceVersion)
			return nil
		}
	}

	if !c.enabledSharingAffinity {
		return c.SignalClaimPendingAllocation(claimUID, allocatedClaim)
	}

	c.inFlightMutex.Lock()
	defer c.inFlightMutex.Unlock()

	existingInFlight, alreadyInFlight := c.inFlightAllocations[claimUID]
	if affinity == nil && alreadyInFlight {
		affinity = existingInFlight.affinity
	}

	resourceSlices, err := c.listSlices()
	if err != nil {
		return err
	}

	sharingState, err := c.syncDeviceAffinityWithSlicesLocked(nil, resourceSlices, true)
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
			if existing, ok := sharingState.DeviceStates[deviceID]; ok && existing.Status == structured.AffinityStatusUnreconstructable {
				return fmt.Errorf("device %s is unreconstructable: %w", deviceID, errSharingAffinityConflict)
			}
			if hasStaleUnconstrainedProvenance(affinity, deviceID, res.Request, view) {
				return fmt.Errorf("device %s request %s provenance mismatch for pool %s/%s generation %d: %w",
					deviceID, res.Request, res.Driver, res.Pool, view.Generation, errSharingAffinityStale)
			}
			continue
		}
		if _, err := c.compilePoolExtractors(view, res.Driver, res.Pool); err != nil {
			return fmt.Errorf("device %s in invalid pool %s/%s: %w", deviceID, res.Driver, res.Pool, err)
		}

		values, ok := findMatchingDeviceAffinity(affinity, deviceID, res.Request, view)
		if !ok {
			return fmt.Errorf("device %s request %s provenance mismatch for pool %s/%s generation %d: %w",
				deviceID, res.Request, res.Driver, res.Pool, view.Generation, errSharingAffinityStale)
		}

		existingLocked := false
		if existing, ok := sharingState.DeviceStates[deviceID]; ok {
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
		c.incompleteDevices.Delete(deviceID)
		c.establishedDeviceLocks[deviceID] = &deviceLockBaseline{
			established: maps.Clone(values),
		}
	}
	c.updateSharingAffinityDeviceMetricsLocked()
	if compatibleReuses > 0 && schedmetrics.SharingAffinityCompatibleReuseTotal != nil {
		schedmetrics.SharingAffinityCompatibleReuseTotal.Add(float64(compatibleReuses))
	}
	return nil
}

