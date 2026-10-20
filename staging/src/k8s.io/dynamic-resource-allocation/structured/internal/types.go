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

package internal

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
)

// ErrFailedAllocationOnNode is an empty sentinel for errors.Is.
// See allocator.go for details.
var ErrFailedAllocationOnNode = errors.New("")

// ClaimAffinity contains provenance in allocation-result order.
type ClaimAffinity struct {
	Devices []DeviceAffinity
}

// DeviceAffinity is scheduling provenance, not persisted API state.
type DeviceAffinity struct {
	ResultIndex int
	Request     string
	Device      DeviceID
	Pool        sharingaffinity.PoolKey
	Digest      string
	Constrained bool
	Values      map[string]string
	Contributes bool
}

// PoolMetadataView describes the complete highest generation without running CEL.
type PoolMetadataView struct {
	Complete   bool
	Generation int64
	Digest     string
	Extractors []resourceapi.SharingAffinityExtractor
}

var EmptyExtractorDigest = ExtractorDigest(nil)

// ExtractorDigest is independent of ResourceSlice and extractor ordering.
func ExtractorDigest(extractors []resourceapi.SharingAffinityExtractor) string {
	entries := make([]string, 0, len(extractors))
	for _, extractor := range extractors {
		data, _ := json.Marshal(extractor)
		entries = append(entries, string(data))
	}
	slices.Sort(entries)
	data, _ := json.Marshal(entries)
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

// ComputePoolMetadataView includes metadata slices which do not target this node.
func ComputePoolMetadataView(allSlices []*resourceapi.ResourceSlice, driver, pool string) PoolMetadataView {
	var selected []*resourceapi.ResourceSlice
	var view PoolMetadataView
	for _, slice := range allSlices {
		if slice.Spec.Driver != driver || slice.Spec.Pool.Name != pool {
			continue
		}
		if len(selected) == 0 || slice.Spec.Pool.Generation > view.Generation {
			selected = nil
			view.Generation = slice.Spec.Pool.Generation
		}
		if slice.Spec.Pool.Generation == view.Generation {
			selected = append(selected, slice)
		}
	}
	view.Digest = EmptyExtractorDigest
	if len(selected) == 0 {
		return view
	}
	slices.SortFunc(selected, func(a, b *resourceapi.ResourceSlice) int {
		return strings.Compare(a.Name, b.Name)
	})
	count := selected[0].Spec.Pool.ResourceSliceCount
	view.Complete = count > 0 && int64(len(selected)) == count
	names := sets.New[string]()
	for _, slice := range selected {
		if slice.Spec.Pool.ResourceSliceCount != count || names.Has(slice.Name) {
			view.Complete = false
		}
		names.Insert(slice.Name)
		view.Extractors = append(view.Extractors, slice.Spec.SharingAffinity...)
	}
	view.Digest = ExtractorDigest(view.Extractors)
	return view
}

type DeviceClassLister interface {
	// List returns a list of all DeviceClasses.
	List() ([]*resourceapi.DeviceClass, error)
	// Get returns the DeviceClass with the given className.
	Get(className string) (*resourceapi.DeviceClass, error)
}

// Allocator is intentionally not documented here. See the main package for docs.
//
// This interface is also broader than the public one.
type Allocator interface {
	Allocate(ctx context.Context, node *v1.Node, claims []*resourceapi.ResourceClaim) (finalResult []resourceapi.AllocationResult, finalErr error)
	Channel() AllocatorChannel
}

// AllocatorExtended is an optional interface. Not all variants implement it.
type AllocatorExtended interface {
	// Stats shows statistics from the allocation process.
	// May return nil if not implemented.
	GetStats() Stats
}

// Stats shows statistics from the allocation process.
type Stats struct {
	// NumAllocateOneInvocations counts the number of times the allocateOne function
	// got called.
	NumAllocateOneInvocations int64
}

type AllocatorChannel string

const (
	Experimental = "experimental"
	Stable       = "stable"
	Incubating   = "incubating"
)

// Features control optional functionality during ResourceClaim allocation.
// Each entry must correspond to at least one control flow change. Entries can
// be removed when the control flow change is no longer necessary (= feature is
// considered stable and always enabled).
//
// This often corresponds to feature gates, but not always: if a KEP implementation
// depends on a set of feature gates, then a single entry here should control whether
// that implementation is active.
type Features struct {
	// Sorted alphabetically. When adding a new entry, also extend Set and FeaturesAll.

	AdminAccess             bool
	CompatibilityGroups     bool
	ConsumableCapacity      bool
	DerivedAttributes       bool
	DeviceBindingAndStatus  bool
	DeviceTaints            bool
	FractionalCapacityRange bool
	ListTypeAttributes      bool
	OptionalNodeOperations  bool
	PartitionableDevices    bool
	PrioritizedList         bool
	SharingAffinity         bool
}

// Set returns all features which are set to true.
// The names of the features happen to match the Kubernetes
// feature gates where applicable. Plain strings are used
// because not all allocator features necessarily have to
// be Kubernetes feature gates and this package must not
// depend on those definitions.
func (f Features) Set() sets.Set[string] {
	enabled := sets.New[string]()
	if f.AdminAccess {
		enabled.Insert("DRAAdminAccess")
	}
	if f.CompatibilityGroups {
		enabled.Insert("DRADeviceCompatibilityGroups")
	}
	if f.ConsumableCapacity {
		enabled.Insert("DRAConsumableCapacity")
	}
	if f.DerivedAttributes {
		enabled.Insert("DRADerivedAttributes")
	}
	if f.DeviceTaints {
		enabled.Insert("DRADeviceTaints")
	}
	if f.FractionalCapacityRange {
		enabled.Insert("DRAFractionalCapacityRange")
	}
	if f.ListTypeAttributes {
		enabled.Insert("DRAListTypeAttributes")
	}
	if f.OptionalNodeOperations {
		enabled.Insert("DRAOptionalNodeOperations")
	}
	if f.PartitionableDevices {
		enabled.Insert("DRAPartitionableDevices")
	}
	if f.PrioritizedList {
		enabled.Insert("DRAPrioritizedList")
	}
	if f.SharingAffinity {
		enabled.Insert("DRASharingAffinity")
	}
	if f.DeviceBindingAndStatus {
		enabled.Insert("DRADeviceBindingConditions+DRAResourceClaimDeviceStatus")
	}
	return enabled
}

var FeaturesAll = Features{
	AdminAccess:             true,
	CompatibilityGroups:     true,
	ConsumableCapacity:      true,
	DerivedAttributes:       true,
	DeviceBindingAndStatus:  true,
	DeviceTaints:            true,
	FractionalCapacityRange: true,
	ListTypeAttributes:      true,
	OptionalNodeOperations:  true,
	PartitionableDevices:    true,
	PrioritizedList:         true,
	SharingAffinity:         true,
}
