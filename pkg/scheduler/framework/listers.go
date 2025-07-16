/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
)

// NodeInfoLister interface represents anything that can list/get NodeInfo objects from node name.
type NodeInfoLister interface {
	// List returns the list of NodeInfos.
	List() ([]*NodeInfo, error)
	// HavePodsWithAffinityList returns the list of NodeInfos of nodes with pods with affinity terms.
	HavePodsWithAffinityList() ([]*NodeInfo, error)
	// HavePodsWithRequiredAntiAffinityList returns the list of NodeInfos of nodes with pods with required anti-affinity terms.
	HavePodsWithRequiredAntiAffinityList() ([]*NodeInfo, error)
	// Get returns the NodeInfo of the given node name.
	Get(nodeName string) (*NodeInfo, error)
}

// StorageInfoLister interface represents anything that handles storage-related operations and resources.
type StorageInfoLister interface {
	// IsPVCUsedByPods returns true/false on whether the PVC is used by one or more scheduled pods,
	// keyed in the format "namespace/name".
	IsPVCUsedByPods(key string) bool
}

// SharedLister groups scheduler-specific listers.
type SharedLister interface {
	NodeInfos() NodeInfoLister
	StorageInfos() StorageInfoLister
}

// ResourceSliceLister can be used to obtain ResourceSlices.
type ResourceSliceLister interface {
	// ListWithDeviceTaintRules returns a list of all ResourceSlices with DeviceTaintRules applied
	// if the DRADeviceTaints feature is enabled, otherwise without them.
	//
	// k8s.io/dynamic-resource-allocation/resourceslice/tracker provides an implementation
	// of the necessary logic. That tracker can be instantiated as a replacement for
	// a normal ResourceSlice informer and provides a ListPatchedResourceSlices method.
	ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error)
}

// DeviceClassLister can be used to obtain DeviceClasses.
type DeviceClassLister interface {
	// List returns a list of all DeviceClasses.
	List() ([]*resourceapi.DeviceClass, error)
	// Get returns the DeviceClass with the given className.
	Get(className string) (*resourceapi.DeviceClass, error)
}

// ResourceClaimTracker can be used to obtain ResourceClaims, and track changes to ResourceClaims in-memory.
//
// If the claims are meant to be allocated in the API during the binding phase (when used by scheduler), the tracker helps avoid
// race conditions between scheduling and binding phases (as well as between the binding phase and the informer cache update).
//
// If the binding phase is not run (e.g. when used by Cluster Autoscaler which only runs the scheduling phase, and simulates binding in-memory),
// the tracker allows the framework user to obtain the claim allocations produced by the DRA plugin, and persist them outside of the API (e.g. in-memory).
type ResourceClaimTracker interface {
	// List lists ResourceClaims. The result is guaranteed to immediately include any changes made via AssumeClaimAfterAPICall(),
	// and SignalClaimPendingAllocation().
	List() ([]*resourceapi.ResourceClaim, error)
	// Get works like List(), but for a single claim.
	Get(namespace, claimName string) (*resourceapi.ResourceClaim, error)
	// ListAllAllocatedDevices lists all allocated Devices from allocated ResourceClaims. The result is guaranteed to immediately include
	// any changes made via AssumeClaimAfterAPICall(), and SignalClaimPendingAllocation().
	ListAllAllocatedDevices() (sets.Set[structured.DeviceID], error)

	// SignalClaimPendingAllocation signals to the tracker that the given ResourceClaim will be allocated via an API call in the
	// binding phase. This change is immediately reflected in the result of List() and the other accessors.
	SignalClaimPendingAllocation(claimUID types.UID, allocatedClaim *resourceapi.ResourceClaim) error
	// ClaimHasPendingAllocation answers whether a given claim has a pending allocation during the binding phase. It can be used to avoid
	// race conditions in subsequent scheduling phases.
	ClaimHasPendingAllocation(claimUID types.UID) bool
	// RemoveClaimPendingAllocation removes the pending allocation for the given ResourceClaim from the tracker if any was signaled via
	// SignalClaimPendingAllocation(). Returns whether there was a pending allocation to remove. List() and the other accessors immediately
	// stop reflecting the pending allocation in the results.
	RemoveClaimPendingAllocation(claimUID types.UID) (deleted bool)

	// AssumeClaimAfterAPICall signals to the tracker that an API call modifying the given ResourceClaim was made in the binding phase, and the
	// changes should be reflected in informers very soon. This change is immediately reflected in the result of List() and the other accessors.
	// This mechanism can be used to avoid race conditions between the informer update and subsequent scheduling phases.
	AssumeClaimAfterAPICall(claim *resourceapi.ResourceClaim) error
	// AssumedClaimRestore signals to the tracker that something went wrong with the API call modifying the given ResourceClaim, and
	// the changes won't be reflected in informers after all. List() and the other accessors immediately stop reflecting the assumed change,
	// and go back to the informer version.
	AssumedClaimRestore(namespace, claimName string)
}

// SharedDRAManager can be used to obtain DRA objects, and track modifications to them in-memory - mainly by the DRA plugin.
// The plugin's default implementation obtains the objects from the API. A different implementation can be
// plugged into the framework in order to simulate the state of DRA objects. For example, Cluster Autoscaler
// can use this to provide the correct DRA object state to the DRA plugin when simulating scheduling changes in-memory.
type SharedDRAManager interface {
	ResourceClaims() ResourceClaimTracker
	ResourceSlices() ResourceSliceLister
	DeviceClasses() DeviceClassLister
}
