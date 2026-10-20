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
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
)

// ShareID is not necessarily unique within an allocation's results. Colliding
// identities must be normalized before recording ownership.
type affinityContributionID struct {
	claimUID types.UID
	device   structured.DeviceID
	shareID  types.UID
}

type affinityContribution struct {
	device  structured.DeviceID
	id      affinityContributionID
	pool    sharingaffinity.PoolKey
	request string
	status  structured.AffinityStatus
	values  map[string]string
}

type claimAffinity []affinityContribution

type deviceAffinity struct {
	owners map[affinityContributionID]affinityContribution
	// Observed and in-flight records may back the same effective owner.
	backers map[affinityContributionID]map[ownerPoolKey]affinityContribution
	// Reconciliation compares against the last coherent lock until final drain.
	established map[string]string
	// Once ambiguous, a device must drain before its remaining owners can be
	// trusted again, even if the owner which exposed the conflict is removed.
	ambiguous bool
}

type poolSliceLister interface {
	ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error)
}

type affinityAccounting struct {
	extractor *sharingaffinity.Extractor
	slices    poolSliceLister
	// evalContextHook lets tests observe lock state from inside CEL
	// evaluation, which polls the context between programs.
	evalContextHook func(context.Context) context.Context
}

var (
	errAffinityPoolIncomplete = errors.New("sharing affinity pool metadata incomplete")
	errAffinityUnsatisfied    = errors.New("sharing affinity extraction unsatisfied")
	errAffinityConflict       = errors.New("conflicting sharing affinity contributions")
)

func lookupPoolExtractors(lister poolSliceLister, driver, pool string) (sharingaffinity.PoolExtractors, error) {
	result := sharingaffinity.PoolExtractors{Key: sharingaffinity.PoolKey{Driver: driver, Name: pool}}
	all, err := lister.ListPatchedResourceSlices()
	if err != nil {
		return result, fmt.Errorf("list ResourceSlices for pool %s/%s: %w", driver, pool, err)
	}
	var selected []*resourceapi.ResourceSlice
	for _, slice := range all {
		if slice.Spec.Driver != driver || slice.Spec.Pool.Name != pool {
			continue
		}
		generation := slice.Spec.Pool.Generation
		if len(selected) == 0 || generation > result.Key.Generation {
			result.Key.Generation = generation
			selected = selected[:0]
		}
		if generation == result.Key.Generation {
			selected = append(selected, slice)
		}
	}
	if len(selected) == 0 {
		return result, fmt.Errorf("pool %s/%s has no ResourceSlices: %w", driver, pool, errAffinityPoolIncomplete)
	}
	expected := selected[0].Spec.Pool.ResourceSliceCount
	seen := make(map[string]struct{}, len(selected))
	for _, slice := range selected {
		if expected <= 0 || slice.Spec.Pool.ResourceSliceCount != expected || int64(len(selected)) != expected {
			return result, fmt.Errorf("pool %s/%s generation %d has %d slices, expected %d (slice %s advertises %d): %w",
				driver, pool, result.Key.Generation, len(selected), expected, slice.Name, slice.Spec.Pool.ResourceSliceCount, errAffinityPoolIncomplete)
		}
		if _, found := seen[slice.Name]; found {
			return result, fmt.Errorf("pool %s/%s generation %d repeats ResourceSlice %s: %w",
				driver, pool, result.Key.Generation, slice.Name, errAffinityPoolIncomplete)
		}
		seen[slice.Name] = struct{}{}
	}
	slices.SortFunc(selected, func(a, b *resourceapi.ResourceSlice) int {
		return strings.Compare(a.Name, b.Name)
	})
	for _, slice := range selected {
		result.Extractors = append(result.Extractors, slice.Spec.SharingAffinity...)
	}
	return result, nil
}

func parseResultRequest(name string) sharingaffinity.Request {
	parent, subrequest, found := strings.Cut(name, "/")
	if found {
		return sharingaffinity.Request{ParentName: parent, Name: subrequest}
	}
	return sharingaffinity.Request{Name: name}
}

// computeClaimAffinity must run outside accounting locks: CEL and slice listing
// do not need mutable ownership state. Removal uses these stored results.
func (a *affinityAccounting) computeClaimAffinity(ctx context.Context, claim *resourceapi.ResourceClaim) claimAffinity {
	if a == nil || claim == nil || claim.Status.Allocation == nil {
		return nil
	}
	logger := klog.FromContext(ctx).WithValues("claim", klog.KObj(claim), "claimUID", claim.UID)
	ctx = klog.NewContext(ctx, logger)
	type poolLookup struct {
		pool sharingaffinity.PoolExtractors
		err  error
	}
	// Cache only metadata, and only for this computation. Each request gets an
	// independent Extract call and therefore an independent runtime budget.
	pools := make(map[sharingaffinity.PoolKey]poolLookup)
	var contributions []affinityContribution
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.ShareID == nil || result.AdminAccess != nil && *result.AdminAccess {
			continue
		}
		key := sharingaffinity.PoolKey{Driver: result.Driver, Name: result.Pool}
		lookup, found := pools[key]
		if !found {
			lookup.pool, lookup.err = lookupPoolExtractors(a.slices, result.Driver, result.Pool)
			pools[key] = lookup
		}
		if lookup.err == nil && len(lookup.pool.Extractors) == 0 {
			continue
		}
		device := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
		c := affinityContribution{
			device:  device,
			id:      affinityContributionID{claimUID: claim.UID, device: device, shareID: *result.ShareID},
			pool:    lookup.pool.Key,
			request: result.Request,
			status:  structured.AffinityStatusUnreconstructable,
		}
		log := logger.WithValues("device", device, "request", result.Request, "pool", c.pool.Name, "generation", c.pool.Generation)
		err := lookup.err
		var extracted sharingaffinity.Result
		if err == nil {
			extracted, err = a.extractor.Extract(ctx, lookup.pool, parseResultRequest(result.Request), claim.Spec.Devices.Config)
			if err == nil && !extracted.Satisfied {
				err = fmt.Errorf("%w: missing extractors %v", errAffinityUnsatisfied, extracted.Unsatisfied)
			}
		}
		if err != nil {
			err = fmt.Errorf("claim %s result %q device %s: %w", claim.UID, result.Request, device, err)
			switch {
			case errors.Is(err, errAffinityPoolIncomplete):
				log.V(2).Info("Sharing affinity unreconstructable for allocated claim", "reason", "poolIncomplete", "err", err)
			case errors.Is(err, errAffinityUnsatisfied):
				log.V(2).Info("Sharing affinity unreconstructable for allocated claim", "reason", "unsatisfied", "unsatisfied", extracted.Unsatisfied, "err", err)
			default:
				outcome := "runtime"
				switch sharingaffinity.Classify(err) {
				case sharingaffinity.OutcomeInvalidDefinition:
					outcome = "invalidDefinition"
				case sharingaffinity.OutcomeClaimInconsistent:
					outcome = "claimInconsistent"
				}
				log.Error(err, "Sharing affinity unreconstructable for allocated claim", "outcome", outcome)
			}
		} else {
			c.status = structured.AffinityStatusLocked
			c.values = maps.Clone(extracted.Values)
		}
		log.V(5).Info("Sharing affinity contribution", "status", c.status, "keys", slices.Sorted(maps.Keys(c.values)))
		contributions = append(contributions, c)
	}
	return normalizeClaimAffinity(logger, claim.UID, contributions)
}

func normalizeClaimAffinity(logger klog.Logger, claimUID types.UID, in []affinityContribution) claimAffinity {
	var result claimAffinity
	indices := make(map[affinityContributionID]int, len(in))
	conflicts := make(map[affinityContributionID]bool)
	for _, c := range in {
		index, found := indices[c.id]
		if !found {
			c.values = maps.Clone(c.values)
			indices[c.id] = len(result)
			result = append(result, c)
			continue
		}
		previous := &result[index]
		if conflicts[c.id] || previous.status == c.status && maps.Equal(previous.values, c.values) {
			continue
		}
		logger.Error(fmt.Errorf("claim %s device %s: %w", claimUID, c.device, errAffinityConflict),
			"Sharing affinity unreconstructable for allocated claim",
			"claimUID", claimUID, "device", c.device, "request", c.request, "previousRequest", previous.request,
			"pool", c.pool.Name, "generation", c.pool.Generation, "outcome", "conflict",
			"keys", slices.Sorted(maps.Keys(c.values)), "previousKeys", slices.Sorted(maps.Keys(previous.values)))
		previous.status = structured.AffinityStatusUnreconstructable
		previous.values = nil
		conflicts[c.id] = true
	}
	return result
}

func foldDeviceAffinity(d *deviceAffinity) (structured.DeviceAffinityState, bool) {
	if d == nil {
		return structured.DeviceAffinityState{}, false
	}
	unreconstructable := structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}
	if d.ambiguous {
		return unreconstructable, true
	}
	if len(d.owners) == 0 {
		return structured.DeviceAffinityState{}, false
	}
	var values map[string]string
	for _, c := range d.owners {
		if c.status != structured.AffinityStatusLocked || len(c.values) == 0 {
			return unreconstructable, true
		}
		if values != nil && !maps.Equal(values, c.values) {
			return unreconstructable, true
		}
		values = c.values
	}
	return structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: maps.Clone(values)}, true
}

func (d *deviceAffinity) insert(logger klog.Logger, c affinityContribution) bool {
	if d.owners == nil {
		d.owners = make(map[affinityContributionID]affinityContribution)
	}
	conflict := false
	for id, previous := range d.owners {
		if id == c.id {
			conflict = previous.status != c.status || !maps.Equal(previous.values, c.values)
		} else if previous.status == structured.AffinityStatusLocked && c.status == structured.AffinityStatusLocked {
			conflict = !maps.Equal(previous.values, c.values)
		}
		if conflict {
			if !d.ambiguous {
				logger.Error(fmt.Errorf("device %s: %w", c.device, errAffinityConflict),
					"Sharing affinity unreconstructable for allocated claim",
					"claimUID", c.id.claimUID, "previousClaimUID", previous.id.claimUID,
					"device", c.device, "request", c.request, "previousRequest", previous.request,
					"pool", c.pool.Name, "generation", c.pool.Generation, "outcome", "conflict",
					"keys", slices.Sorted(maps.Keys(c.values)), "previousKeys", slices.Sorted(maps.Keys(previous.values)))
			}
			break
		}
	}
	d.ambiguous = d.ambiguous || conflict || c.status != structured.AffinityStatusLocked || len(c.values) == 0
	// A repeated identity is one owner, even if its two observations disagree.
	// Neither observation can clear the sticky ambiguity established above.
	if _, found := d.owners[c.id]; !found {
		c.values = maps.Clone(c.values)
		d.owners[c.id] = c
	}
	return conflict
}

func (d *deviceAffinity) remove(id affinityContributionID) bool {
	delete(d.owners, id)
	if len(d.owners) == 0 {
		d.ambiguous = false
		return true
	}
	return false
}

func (d *deviceAffinity) clone() *deviceAffinity {
	if d == nil {
		return nil
	}
	result := &deviceAffinity{
		owners:      maps.Clone(d.owners),
		backers:     maps.Clone(d.backers),
		established: maps.Clone(d.established),
		ambiguous:   d.ambiguous,
	}
	for id, c := range result.owners {
		c.values = maps.Clone(c.values)
		result.owners[id] = c
	}
	for id, backers := range result.backers {
		cloned := maps.Clone(backers)
		for key, c := range cloned {
			c.values = maps.Clone(c.values)
			cloned[key] = c
		}
		result.backers[id] = cloned
	}
	return result
}
