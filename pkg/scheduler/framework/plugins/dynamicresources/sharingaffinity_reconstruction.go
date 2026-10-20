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

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
)

// poolExtraction distinguishes incomplete metadata from a coherent snapshot whose
// claim configuration cannot reconstruct an affinity lock.
type poolExtraction struct {
	pool          poolName
	view          poolMeta
	devices       []structured.DeviceID
	contributions claimAffinity
	pending       bool
	reason        string
}

// extractByPool must run outside accounting locks. Provenance and CEL use the same
// cached slice listing, so publication can reject a superseded computation.
func (a *affinityAccounting) extractByPool(ctx context.Context, claim *resourceapi.ResourceClaim) []poolExtraction {
	if a == nil || claim == nil || claim.Status.Allocation == nil {
		return nil
	}
	byPool := make(map[poolName]int)
	var extracted []poolExtraction
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.AdminAccess != nil && *result.AdminAccess {
			continue
		}
		key := poolName{result.Driver, result.Pool}
		index, found := byPool[key]
		if !found {
			index = len(extracted)
			byPool[key] = index
			extracted = append(extracted, poolExtraction{pool: key})
		}
		// Dedicated and affinity-inactive allocations still participate in
		// device lifetime and must survive later metadata changes.
		extracted[index].devices = append(extracted[index].devices,
			structured.MakeDeviceID(result.Driver, result.Pool, result.Device))
	}
	if len(extracted) == 0 {
		return nil
	}

	all, err := a.slices.ListPatchedResourceSlices()
	logger := klog.FromContext(ctx).WithValues("claim", klog.KObj(claim), "claimUID", claim.UID)
	if err != nil {
		logger.Error(err, "Listing ResourceSlices for sharing affinity reconstruction")
		for i := range extracted {
			extracted[i].pending = true
			extracted[i].reason = "listError"
		}
		return extracted
	}
	slicesByPool := groupSlicesByPool(all)
	for i := range extracted {
		entry := &extracted[i]
		entry.view = groupedMetadataView(slicesByPool[entry.pool], entry.pool)
		if !entry.view.complete {
			entry.pending = true
			entry.reason = "incomplete"
			logger.V(4).Info("Sharing affinity reconstruction waiting for complete pool metadata",
				"driver", entry.pool.driver, "pool", entry.pool.pool, "generation", entry.view.generation, "reason", entry.reason)
		}
	}

	// Exclude transiently incomplete pools before using the existing extraction
	// path: it intentionally classifies missing metadata as unknown affinity.
	// Copy only the containers we change, leaving informer objects untouched.
	readyClaim := *claim
	readyAllocation := *claim.Status.Allocation
	readyAllocation.Devices.Results = nil
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.AdminAccess != nil && *result.AdminAccess {
			continue
		}
		if !extracted[byPool[poolName{result.Driver, result.Pool}]].pending {
			readyAllocation.Devices.Results = append(readyAllocation.Devices.Results, result)
		}
	}
	readyClaim.Status.Allocation = &readyAllocation
	if a.evalContextHook != nil {
		ctx = a.evalContextHook(ctx)
	}
	snapshot := &affinityAccounting{extractor: a.extractor, slices: metadataSliceList{slices: all}}
	for _, contribution := range snapshot.computeClaimAffinity(ctx, &readyClaim) {
		index := byPool[poolName{contribution.pool.Driver, contribution.pool.Name}]
		extracted[index].contributions = append(extracted[index].contributions, contribution)
	}
	return extracted
}
