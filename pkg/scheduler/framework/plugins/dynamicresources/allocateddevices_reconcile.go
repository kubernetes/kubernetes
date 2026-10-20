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
	"maps"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
)

type ownerID struct {
	uid      types.UID
	inFlight bool
}

type ownerPoolKey struct {
	uid      types.UID
	inFlight bool
	pool     poolName
}

type ownerPoolRecord struct {
	token         uint64
	claim         *resourceapi.ResourceClaim
	claimRV       string
	devices       sets.Set[structured.DeviceID]
	contributions claimAffinity
	view          poolMeta
	epoch         uint64
	pending       bool
	reason        string
	inactive      bool
	heldFor       types.NamespacedName
}

type poolOutcome int

const (
	outcomeSettled poolOutcome = iota
	outcomeWaitEvent
	outcomeRetry
	outcomeSuperseded
)

var claimRegistrationSynced = func(registration cache.ResourceEventHandlerRegistration) cache.InformerSynced {
	return registration.HasSynced
}

func allocationResults(claim *resourceapi.ResourceClaim) []resourceapi.DeviceRequestAllocationResult {
	if claim == nil || claim.Status.Allocation == nil {
		return nil
	}
	return claim.Status.Allocation.Devices.Results
}

func sameExtractionInputs(a, b *resourceapi.ResourceClaim) bool {
	return a != nil && b != nil && a.UID == b.UID &&
		apiequality.Semantic.DeepEqual(a.Spec.Devices.Config, b.Spec.Devices.Config) &&
		apiequality.Semantic.DeepEqual(allocationResults(a), allocationResults(b))
}

func claimPoolDevices(claim *resourceapi.ResourceClaim) map[poolName]sets.Set[structured.DeviceID] {
	pools := make(map[poolName]sets.Set[structured.DeviceID])
	for _, result := range allocationResults(claim) {
		// Match extractByPool: admin access does not own affinity state,
		// so it must not create a record that extraction can never settle.
		if result.AdminAccess != nil && *result.AdminAccess {
			continue
		}
		key := poolName{result.Driver, result.Pool}
		if pools[key] == nil {
			pools[key] = sets.New[structured.DeviceID]()
		}
		pools[key].Insert(structured.MakeDeviceID(result.Driver, result.Pool, result.Device))
	}
	return pools
}

func (a *allocatedDevices) beginComputeLocked(id ownerID, claim *resourceapi.ResourceClaim, pools sets.Set[poolName]) map[poolName]uint64 {
	tokens := make(map[poolName]uint64, len(pools))
	devices := claimPoolDevices(claim)
	if a.ownerPools[id] == nil {
		a.ownerPools[id] = sets.New[poolName]()
	}
	for pool := range pools {
		key := ownerPoolKey{id.uid, id.inFlight, pool}
		rec := a.owners[key]
		if rec == nil {
			rec = &ownerPoolRecord{}
			a.owners[key] = rec
		}
		a.nextToken++
		rec.token, rec.claim, rec.claimRV = a.nextToken, claim, claim.ResourceVersion
		// Keep old devices until commit too: an unfinished replacement must
		// neither drain nor expose its former device's baseline.
		rec.devices = rec.devices.Union(devices[pool])
		rec.pending, rec.reason, rec.inactive = true, "computing", false
		a.ownerPools[id].Insert(pool)
		tokens[pool] = rec.token
	}
	a.revision++
	return tokens
}

func (d *deviceAffinity) setBacker(id affinityContributionID, key ownerPoolKey, c affinityContribution) {
	if d.backers == nil {
		d.backers = make(map[affinityContributionID]map[ownerPoolKey]affinityContribution)
	}
	if d.backers[id] == nil {
		d.backers[id] = make(map[ownerPoolKey]affinityContribution)
	}
	c.values = maps.Clone(c.values)
	d.backers[id][key] = c
	d.refreshBacker(id)
}

// onlyBackedBy reports whether every contribution comes from uid; with an
// empty uid it reports whether there are no contributions at all.
func (d *deviceAffinity) onlyBackedBy(uid types.UID) bool {
	if uid == "" {
		return len(d.owners) == 0
	}
	for _, backers := range d.backers {
		for key := range backers {
			if key.uid != uid {
				return false
			}
		}
	}
	return true
}

func (d *deviceAffinity) dropBacker(id affinityContributionID, key ownerPoolKey) {
	delete(d.backers[id], key)
	if len(d.backers[id]) == 0 {
		delete(d.backers, id)
		delete(d.owners, id)
		return
	}
	d.refreshBacker(id)
}

func (d *deviceAffinity) refreshBacker(id affinityContributionID) {
	var effective affinityContribution
	first := true
	for _, c := range d.backers[id] {
		if first {
			effective = c
			first = false
		} else if effective.status != c.status || !maps.Equal(effective.values, c.values) {
			effective.status = structured.AffinityStatusUnreconstructable
			effective.values = nil
		}
	}
	if d.owners == nil {
		d.owners = make(map[affinityContributionID]affinityContribution)
	}
	d.owners[id] = effective
}

func (a *allocatedDevices) recordCoherentLocked(key ownerPoolKey, rec *ownerPoolRecord) bool {
	if rec.pending {
		return false
	}
	if a.metadata == nil {
		return true
	}
	pub := a.metadata.pools[key.pool]
	return rec.epoch == pub.epoch && sameMetadata(rec.view, pub)
}

func hasExtractors(view poolMeta) bool {
	return view.digest != "" && view.digest != structured.EmptyExtractorDigest
}

func (a *allocatedDevices) poolReadyLocked(key poolName) bool {
	if !a.startupReady || a.metadata != nil && !a.metadata.synced {
		return false
	}
	if !a.affinityRelevantLocked(key, poolMeta{}, false) {
		return true
	}
	for k, rec := range a.owners {
		if k.pool == key && !a.recordCoherentLocked(k, rec) {
			return false
		}
	}
	return true
}

func (a *allocatedDevices) commitPoolLocked(logger klog.Logger, key ownerPoolKey, token uint64, result poolExtraction) poolOutcome {
	rec := a.owners[key]
	if rec == nil || rec.token != token {
		return outcomeSuperseded
	}
	pending := func(reason string, outcome poolOutcome) poolOutcome {
		rec.pending, rec.reason = true, reason
		a.revision++
		logger.V(4).Info("Sharing affinity owner pool pending", "claimUID", key.uid, "driver", key.pool.driver, "pool", key.pool.pool, "reason", reason)
		return outcome
	}
	if !key.inFlight && a.tracker != nil {
		current, err := a.tracker.Get(rec.claim.Namespace, rec.claim.Name)
		if err != nil || current.UID != rec.claim.UID {
			return pending("cacheGone", outcomeWaitEvent)
		}
		if current != rec.claim && !(current.ResourceVersion == rec.claimRV && sameExtractionInputs(current, rec.claim)) {
			return pending("cacheAhead", outcomeWaitEvent)
		}
	}
	if result.reason == "listError" {
		return pending("listError", outcomeRetry)
	}
	if result.pending || !result.view.complete {
		return pending("incomplete", outcomeWaitEvent)
	}
	pub := result.view
	if a.metadata != nil {
		pub = a.metadata.pools[key.pool]
		if !sameMetadata(result.view, pub) {
			return pending("metadataAhead", outcomeWaitEvent)
		}
	}
	affected := rec.devices.Union(sets.New(result.devices...))
	for _, old := range rec.contributions {
		a.affinityDevices[old.device].dropBacker(old.id, key)
	}
	rec.contributions = result.contributions
	for _, c := range result.contributions {
		d := a.affinityDevices[c.device]
		if d == nil {
			d = &deviceAffinity{}
			a.affinityDevices[c.device] = d
		}
		d.setBacker(c.id, key, c)
	}
	rec.devices = sets.New(result.devices...)
	rec.view, rec.epoch, rec.pending, rec.reason, rec.inactive = result.view, pub.epoch, false, "", false
	a.finalizeDevicesLocked(logger, affected)
	a.clearDrainedLocked()
	a.revision++
	return outcomeSettled
}

func (a *allocatedDevices) finalizeDevicesLocked(logger klog.Logger, devices sets.Set[structured.DeviceID]) {
	if a.startupReady {
		a.finalizeDevicesUngatedLocked(logger, devices)
	}
}

func (a *allocatedDevices) finalizeDevicesUngatedLocked(logger klog.Logger, devices sets.Set[structured.DeviceID]) {
	for device := range devices {
		d := a.affinityDevices[device]
		if d == nil || d.ambiguous {
			continue
		}
		coherent, missing := true, false
		hasRecord := false
		for key, rec := range a.owners {
			if !rec.devices.Has(device) {
				continue
			}
			hasRecord = true
			if !a.recordCoherentLocked(key, rec) {
				coherent = false
				break
			}
			backed := false
			for _, c := range rec.contributions {
				if c.device == device {
					backed = true
					break
				}
			}
			// Admin access observes a device without owning its modal state.
			missing = missing || !backed && claimOwnsDevice(rec.claim, device)
		}
		if !coherent {
			continue
		}
		if !hasRecord && !a.hasOwnerLocked(device) {
			continue
		}
		state, present := foldDeviceAffinity(d)
		conflict := present && state.Status == structured.AffinityStatusUnreconstructable
		if d.established != nil {
			conflict = conflict || missing || !present || !maps.Equal(d.established, state.LockedAffinity)
		}
		if conflict {
			d.ambiguous = true
			a.fenced.Insert(device)
			outcome := "conflict"
			if d.established != nil {
				outcome = "metadataChanged"
			}
			logger.Error(errAffinityConflict, "Sharing affinity unreconstructable for allocated claim", "device", device, "outcome", outcome)
		} else if present && state.Status == structured.AffinityStatusLocked {
			d.established = maps.Clone(state.LockedAffinity)
		}
	}
}

func claimOwnsDevice(claim *resourceapi.ResourceClaim, device structured.DeviceID) bool {
	found := false
	for _, result := range allocationResults(claim) {
		if structured.MakeDeviceID(result.Driver, result.Pool, result.Device) != device {
			continue
		}
		found = true
		if result.AdminAccess == nil || !*result.AdminAccess {
			return true
		}
	}
	// A pending record may still retain a device from its previous allocation.
	return !found
}

func (a *allocatedDevices) drainDeviceLocked(device structured.DeviceID) {
	delete(a.affinityDevices, device)
	a.fenced.Delete(device)
}

func (a *allocatedDevices) clearDrainedLocked() {
	for device, d := range a.affinityDevices {
		if a.drainableLocked(device, d, "") {
			a.drainDeviceLocked(device)
		}
	}
}

// drainableLocked reports whether no owner other than exclude survives on device.
// Contributions alone are insufficient: after affinity metadata is removed,
// surviving owners keep records and capacity but contribute nothing.
func (a *allocatedDevices) drainableLocked(device structured.DeviceID, d *deviceAffinity, exclude types.UID) bool {
	if !d.onlyBackedBy(exclude) || a.hasOwnerLocked(device) {
		return false
	}
	for key, rec := range a.owners {
		if key.uid != exclude && rec.devices.Has(device) && claimOwnsDevice(rec.claim, device) {
			return false
		}
	}
	return true
}

func (a *allocatedDevices) removeRecordLocked(key ownerPoolKey, affected sets.Set[structured.DeviceID]) {
	rec := a.owners[key]
	if rec == nil {
		return
	}
	affected.Insert(rec.devices.UnsortedList()...)
	for _, c := range rec.contributions {
		a.affinityDevices[c.device].dropBacker(c.id, key)
	}
	delete(a.owners, key)
	id := ownerID{key.uid, key.inFlight}
	a.ownerPools[id].Delete(key.pool)
	if len(a.ownerPools[id]) == 0 {
		delete(a.ownerPools, id)
	}
	last := true
	for other := range a.owners {
		if other.pool == key.pool {
			last = false
			break
		}
	}
	if last {
		a.enqueuePoolLocked(key.pool)
	}
	a.revision++
}

func (a *allocatedDevices) enqueuePoolLocked(pool poolName) {
	if a.reconcile != nil {
		a.reconcile.Add(pool)
	}
}

type observedCapacity struct {
	dedicated  []structured.DeviceID
	shared     []structured.SharedDeviceID
	capacities []structured.DeviceConsumedCapacity
}

func (a *allocatedDevices) observedCapacity(claim *resourceapi.ResourceClaim) observedCapacity {
	var delta observedCapacity
	if claim == nil {
		return delta
	}
	foreachAllocatedDevice(claim, func(device structured.DeviceID) {
		delta.dedicated = append(delta.dedicated, device)
	}, a.enabledConsumableCapacity, func(id structured.SharedDeviceID) {
		delta.shared = append(delta.shared, id)
	}, func(capacity structured.DeviceConsumedCapacity) {
		delta.capacities = append(delta.capacities, capacity)
	})
	return delta
}

func (a *allocatedDevices) changeObservedCapacityLocked(delta observedCapacity, add bool) {
	for _, device := range delta.dedicated {
		if add {
			a.ids.Insert(device)
		} else {
			a.ids.Delete(device)
		}
	}
	for _, id := range delta.shared {
		if add {
			a.shareIDs.Insert(id)
		} else {
			a.shareIDs.Delete(id)
		}
	}
	for _, capacity := range delta.capacities {
		if add {
			a.capacities.Insert(capacity)
		} else {
			a.capacities.Remove(capacity)
		}
	}
}

func (a *allocatedDevices) lockOwners() func() {
	if a.tracker != nil {
		a.tracker.inFlightMutex.Lock()
	}
	a.mutex.Lock()
	return func() {
		a.mutex.Unlock()
		if a.tracker != nil {
			a.tracker.inFlightMutex.Unlock()
		}
	}
}

func (a *allocatedDevices) replaceObserved(old, claim *resourceapi.ResourceClaim) {
	oldCapacity, newCapacity := a.observedCapacity(old), a.observedCapacity(claim)
	unlock := a.lockOwners()
	if sameExtractionInputs(old, claim) {
		settled := true
		for pool := range a.ownerPools[ownerID{uid: claim.UID}] {
			if a.owners[ownerPoolKey{uid: claim.UID, pool: pool}].pending {
				settled = false
				break
			}
		}
		if settled {
			for pool := range a.ownerPools[ownerID{uid: claim.UID}] {
				rec := a.owners[ownerPoolKey{uid: claim.UID, pool: pool}]
				rec.claim, rec.claimRV = claim, claim.ResourceVersion
			}
			unlock()
			return
		}
	}
	changed := sets.New[poolName]()
	affected := sets.New[structured.DeviceID]()
	releaseHeld := func(name types.NamespacedName, finalize, resetDrained bool) {
		held, ok := a.holds[name]
		if !ok {
			return
		}
		for pool := range a.ownerPools[held].Clone() {
			changed.Insert(pool)
			a.removeRecordLocked(ownerPoolKey{held.uid, false, pool}, affected)
		}
		delete(a.holds, name)
		if resetDrained {
			// The replacement already owns the device; only other owners keep
			// the departed owner's fence.
			a.changeObservedCapacityLocked(newCapacity, false)
			for device := range affected {
				d := a.affinityDevices[device]
				if d == nil || !a.drainableLocked(device, d, claim.UID) {
					continue
				}
				if len(d.owners) == 0 {
					a.drainDeviceLocked(device)
					continue
				}
				// A worker already settled the replacement: keep its
				// contributions, but drop the departed owner's baseline.
				d.established, d.ambiguous = nil, false
				a.fenced.Delete(device)
			}
			a.changeObservedCapacityLocked(newCapacity, true)
			// Only devices the replacement owns through a still-pending
			// record get finalized by its commit. Everything else (devices
			// outside the allocation, admin-only references, worker-settled
			// records) must be finalized now, so changed maps are fenced and
			// a reset baseline is re-established before readiness.
			deferred := sets.New[structured.DeviceID]()
			for key, rec := range a.owners {
				if key.uid != claim.UID || !rec.pending {
					continue
				}
				for device := range rec.devices {
					if claimOwnsDevice(claim, device) {
						deferred.Insert(device)
					}
				}
			}
			a.finalizeDevicesLocked(a.logger, affected.Difference(deferred))
		}
		if finalize {
			a.finalizeDevicesLocked(a.logger, affected)
			a.clearDrainedLocked()
		}
	}
	if claim != nil {
		name := types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name}
		if claim.Status.Allocation == nil {
			releaseHeld(name, true, false)
		}
	}
	a.changeObservedCapacityLocked(oldCapacity, false)
	newPools := claimPoolDevices(claim)
	if old != nil {
		id := ownerID{uid: old.UID}
		hold := false
		if claim == nil && a.tracker != nil {
			visible, err := a.tracker.Get(old.Namespace, old.Name)
			hold = err == nil && visible.UID != old.UID
		}
		for pool := range a.ownerPools[id] {
			changed.Insert(pool)
			key := ownerPoolKey{uid: old.UID, pool: pool}
			if hold {
				rec := a.owners[key]
				a.nextToken++
				rec.token, rec.pending, rec.reason = a.nextToken, true, "replacementQueued"
				rec.heldFor = types.NamespacedName{Namespace: old.Namespace, Name: old.Name}
				a.holds[rec.heldFor] = id
			} else if claim == nil || old.UID != claim.UID || newPools[pool] == nil {
				a.removeRecordLocked(key, affected)
			}
		}
		delete(a.claimAffinity, old.UID)
	}
	if old != nil && claim != nil && old.UID != claim.UID {
		a.finalizeDevicesLocked(a.logger, affected)
		a.clearDrainedLocked()
	}
	a.changeObservedCapacityLocked(newCapacity, true)
	var tokens map[poolName]uint64
	if claim != nil && len(newPools) != 0 {
		pools := sets.New[poolName]()
		for pool := range newPools {
			pools.Insert(pool)
			changed.Insert(pool)
		}
		tokens = a.beginComputeLocked(ownerID{uid: claim.UID}, claim, pools)
	}
	if claim != nil && len(tokens) == 0 {
		// No owner records (e.g. admin-only): nothing will commit to end
		// the predecessor's hold.
		releaseHeld(types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name}, true, false)
	}
	a.finalizeDevicesLocked(a.logger, affected)
	a.clearDrainedLocked()
	a.revision++
	unlock()
	if claim != nil && len(tokens) != 0 {
		results := a.affinityAcct.extractByPool(klog.NewContext(context.Background(), a.logger), claim)
		if a.observedAfterComputeHook != nil {
			a.observedAfterComputeHook()
		}
		unlock = a.lockOwners()
		releaseHeld(types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name}, false, true)
		for _, result := range results {
			if a.commitPoolLocked(a.logger, ownerPoolKey{uid: claim.UID, pool: result.pool}, tokens[result.pool], result) == outcomeRetry {
				a.enqueuePoolLocked(result.pool)
			}
		}
		unlock()
	}
	if a.tracker != nil {
		a.tracker.notifyExclusionChanges(a.logger, changed, false)
	}
}

func (a *allocatedDevices) completeStartupLocked(logger klog.Logger) sets.Set[poolName] {
	changed := sets.New[poolName]()
	if a.startupReady || a.metadata == nil || !a.metadata.synced || !a.claimsSynced {
		return changed
	}
	devices := sets.New[structured.DeviceID]()
	for key, rec := range a.owners {
		pub := a.metadata.pools[key.pool]
		if rec.epoch != pub.epoch || !sameMetadata(rec.view, pub) {
			rec.pending, rec.reason = true, "metadataAhead"
		}
		devices.Insert(rec.devices.UnsortedList()...)
		changed.Insert(key.pool)
	}
	a.finalizeDevicesUngatedLocked(logger, devices)
	a.startupReady = true
	for pool := range a.metadata.pools {
		changed.Insert(pool)
	}
	for pool := range changed {
		a.enqueuePoolLocked(pool)
	}
	a.revision++
	return changed
}

func (a *allocatedDevices) createInFlightRecordsLocked(claim *resourceapi.ResourceClaim, results []poolExtraction) {
	pools := sets.New[poolName]()
	for _, result := range results {
		pools.Insert(result.pool)
	}
	tokens := a.beginComputeLocked(ownerID{uid: claim.UID, inFlight: true}, claim, pools)
	for _, result := range results {
		key := ownerPoolKey{uid: claim.UID, inFlight: true, pool: result.pool}
		if a.commitPoolLocked(a.logger, key, tokens[result.pool], result) != outcomeSettled {
			a.enqueuePoolLocked(result.pool)
		}
	}
}

func (a *allocatedDevices) deleteInFlightRecordsLocked(uid types.UID) {
	affected := sets.New[structured.DeviceID]()
	for pool := range a.ownerPools[ownerID{uid: uid, inFlight: true}] {
		a.removeRecordLocked(ownerPoolKey{uid: uid, inFlight: true, pool: pool}, affected)
	}
	a.finalizeDevicesLocked(a.logger, affected)
	a.clearDrainedLocked()
}

type reconcileItem struct {
	key   ownerPoolKey
	token uint64
	claim *resourceapi.ResourceClaim
}

func (c *claimTracker) initReconciler(ctx context.Context) {
	a := c.allocatedDevices
	a.tracker = c
	a.startupReady = false
	a.reconcile = workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.NewTypedItemExponentialFailureRateLimiter[poolName](5*time.Millisecond, time.Second),
		workqueue.TypedRateLimitingQueueConfig[poolName]{Name: "dra_sharing_affinity"})
	c.reconcilerDone = make(chan struct{})
	go c.runReconciler(ctx)
}

func (c *claimTracker) runReconciler(ctx context.Context) {
	defer close(c.reconcilerDone)
	go func() { <-ctx.Done(); c.allocatedDevices.reconcile.ShutDown() }()
	for c.processNext(ctx) {
	}
}

func (c *claimTracker) beginPool(pool poolName) []reconcileItem {
	a := c.allocatedDevices
	unlock := a.lockOwners()
	defer unlock()
	// Callers run only after metadata is installed; guard anyway.
	if a.metadata == nil {
		return nil
	}
	var items []reconcileItem
	removed := sets.New[structured.DeviceID]()
	for key, rec := range a.owners {
		if key.pool != pool {
			continue
		}
		if rec.heldFor.Name != "" {
			visible, err := c.Get(rec.heldFor.Namespace, rec.heldFor.Name)
			if err != nil || visible.UID == key.uid {
				// Release every pool of the held owner: the other pools would
				// otherwise stay pending with no event left to wake them.
				delete(a.holds, rec.heldFor)
				id := ownerID{uid: key.uid, inFlight: key.inFlight}
				for other := range a.ownerPools[id].Clone() {
					a.removeRecordLocked(ownerPoolKey{key.uid, key.inFlight, other}, removed)
					if other != pool {
						a.enqueuePoolLocked(other)
					}
				}
			}
			continue
		}
		recomputeInactive := false
		if rec.inactive && !hasExtractors(rec.view) {
			pub := a.metadata.pools[pool]
			if sameMetadata(rec.view, pub) {
				rec.epoch, rec.pending, rec.reason, rec.inactive = pub.epoch, false, "", false
			} else if !hasExtractors(pub) {
				rec.pending, rec.reason = false, ""
			} else {
				recomputeInactive = true
			}
			if !recomputeInactive {
				continue
			}
		}
		if a.recordCoherentLocked(key, rec) {
			continue
		}
		a.nextToken++
		rec.token, rec.pending, rec.reason = a.nextToken, true, "computing"
		items = append(items, reconcileItem{key, rec.token, rec.claim})
	}
	a.finalizeDevicesLocked(c.logger, removed)
	a.clearDrainedLocked()
	if len(items) != 0 || len(removed) != 0 {
		a.revision++
	}
	return items
}

func (c *claimTracker) commitPoolBatch(logger klog.Logger, pool poolName, items []reconcileItem, results []poolExtraction) poolOutcome {
	a := c.allocatedDevices
	unlock := a.lockOwners()
	defer unlock()
	outcome := outcomeSettled
	for i, item := range items {
		if a.commitPoolLocked(logger, item.key, item.token, results[i]) == outcomeRetry {
			outcome = outcomeRetry
		}
	}
	return outcome
}

func (c *claimTracker) processNext(ctx context.Context) bool {
	queue := c.allocatedDevices.reconcile
	pool, shutdown := queue.Get()
	if shutdown {
		return false
	}
	defer queue.Done(pool)
	if ctx.Err() != nil {
		queue.Forget(pool)
		return false
	}
	if c.reconcileProcessHook != nil {
		c.reconcileProcessHook(pool)
	}
	items := c.beginPool(pool)
	results := make([]poolExtraction, len(items))
	for i, item := range items {
		// Reconcile only this pool. Config remains claim-scoped, but results
		// from unrelated pools must not consume CEL budgets or list retries.
		claim := *item.claim
		allocation := *claim.Status.Allocation
		claim.Status.Allocation = &allocation
		allocation.Devices.Results = nil
		for _, result := range allocationResults(item.claim) {
			if result.Driver == pool.driver && result.Pool == pool.pool {
				allocation.Devices.Results = append(allocation.Devices.Results, result)
			}
		}
		found := false
		for _, result := range c.affinity.extractByPool(ctx, &claim) {
			if result.pool == pool {
				results[i] = result
				found = true
				break
			}
		}
		if !found {
			// A zero-value extraction commits as incomplete and waits for an event.
			c.logger.V(4).Info("No extraction result for reconciled pool", "pool", pool, "claim", klog.KObj(item.claim))
		}
		if c.reconcileAfterComputeHook != nil {
			c.reconcileAfterComputeHook(pool, item.key.uid)
		}
	}
	if ctx.Err() != nil {
		queue.Forget(pool)
		return false
	}
	if c.commitPoolBatch(c.logger, pool, items, results) == outcomeRetry {
		queue.AddRateLimited(pool)
	} else {
		queue.Forget(pool)
	}
	if c.reconcileAfterCommitHook != nil {
		c.reconcileAfterCommitHook(pool)
	}
	c.notifyExclusionChanges(c.logger, sets.New(pool), false)
	return true
}
