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

package dynamicresources

import (
	"fmt"
	"slices"
	"sort"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/extended"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/utils/ptr"
)

// Extended Resources Backed by DRA - Scheduler Plugin Workflow by each extension points
//
// PreFilter - preFilterExtendedResources()
// - for pods using extended resources, find existing claim or create in-memory claim with temporary name "<extended-resources>"
// - the in-memory claim is used to track and allocate resources, claim object is created in PreBind extension point.
// - store the claim in stateData for Filter extension point
//
// Filter - filterExtendedResources()
// - if stale claim with Spec is identified, return Unschedulable for PostFilter extension point to cleanup
// - check which resources satisfied by device plugin vs need DRA
// - if extended resources need to be allocated through DRA, create node-specific claim
//
// PostFilter
// - if extended resource claim has real name (not "<extended-resources>"):
//   - it's stale from prior cycle -> delete it -> trigger retry
//
// Reserve
// - Store allocation results from Filter in stateData
// - Mark the claim as "allocation in-flight" via SignalClaimPendingAllocation()
//
// Unreserve
// - Remove claim from in-flight allocations and restore assume cache
// - Delete claim from API server if it has real name
//
// PreBind - bindClaim()
// - For "<extended-resources>" claims: create in API server and update stateData
// - Update claim status: add finalizer, allocation, and pod reservation
// - Store in assume cache (poll for extended resource claims)
// - Update pod.Status.ExtendedResourceClaimStatus with request mappings

const (
	// specialClaimInMemName is the name of the special resource claim that
	// exists only in memory. The claim will get a generated name when it is
	// written to API server.
	//
	// It's intentionally not a valid ResourceClaim name to avoid conflicts with
	// some actual ResourceClaim in the apiserver.
	specialClaimInMemName = "<extended-resources>"
)

// hasDeviceClassMappedExtendedResource returns true when the given resource list has an extended resource, that has
// a mapping to a device class.
func hasDeviceClassMappedExtendedResource(reqs v1.ResourceList, deviceClassMapping map[v1.ResourceName]string) bool {
	for rName, rValue := range reqs {
		if rValue.IsZero() {
			// We only care about the resources requested by the pod we are trying to schedule.
			continue
		}
		if schedutil.IsDRAExtendedResourceName(rName) {
			_, ok := deviceClassMapping[rName]
			if ok {
				return true
			}
		}
	}
	return false
}

// findExtendedResourceClaim looks for the extended resource claim, i.e., the claim with special annotation
// set to "true", and with the pod as owner. It must be called with all ResourceClaims in the cluster.
// The returned ResourceClaim is read-only.
func findExtendedResourceClaim(pod *v1.Pod, resourceClaims []*resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	for _, c := range resourceClaims {
		if c.Annotations[resourceapi.ExtendedResourceClaimAnnotation] == "true" {
			for _, or := range c.OwnerReferences {
				if or.Name == pod.Name && *or.Controller && or.UID == pod.UID {
					return c
				}
			}
		}
	}
	return nil
}

// preFilterExtendedResources checks if there is any extended resource in the
// pod requests that has a device class mapping, i.e., there is a device class
// that has spec.ExtendedResourceName or its implicit extended resource name
// matching the given extended resource in that pod requests.
//
// It looks for the special resource claim for the pod created from prior scheduling
// cycle. If not found, it creates the special claim with no Requests in the Spec,
// with a temporary UID, and the specialClaimInMemName name.
// Either way, the special claim is stored in state.claims.
//
// In addition, draExtendedResource is also stored in the cycle state.
//
// It returns the special ResourceClaim and an error status. It returns nil for both
// if the feature is disabled or not required for the Pod.
func (pl *DynamicResources) preFilterExtendedResources(pod *v1.Pod, logger klog.Logger, s *stateData) (*resourceapi.ResourceClaim, *fwk.Status) {
	if !pl.fts.EnableDRAExtendedResource {
		return nil, nil
	}

	deviceClassMapping, err := extended.DeviceClassMapping(pl.draManager)
	if err != nil {
		return nil, statusError(logger, err, "retrieving extended resource to DeviceClass mapping")
	}

	reqs := resourcehelper.PodRequests(pod, resourcehelper.PodResourcesOptions{})
	hasExtendedResource := hasDeviceClassMappedExtendedResource(reqs, deviceClassMapping)
	if !hasExtendedResource {
		return nil, nil
	}

	s.draExtendedResource.resourceToDeviceClass = deviceClassMapping
	r := framework.NewResource(reqs)
	s.draExtendedResource.podScalarResources = r.ScalarResources

	resourceClaims, err := pl.draManager.ResourceClaims().List()
	if err != nil {
		return nil, statusError(logger, err, "listing ResourceClaims")
	}

	// Check if the special resource claim has been created from prior scheduling cycle.
	//
	// If it was already allocated earlier, that allocation might not be valid anymore.
	// We could try to check that, but it depends on various factors that are difficult to
	// cover (basically needs to replicate allocator logic) and if it turns out that the
	// allocation is stale, we would have to schedule with those allocated devices not
	// available for a new allocation. This situation should be rare (= binding failure),
	// so we solve it via brute-force
	// - Kick off deallocation in the background.
	// - Mark the pod as unschedulable. Successful deallocation will make it schedulable again.
	extendedResourceClaim := findExtendedResourceClaim(pod, resourceClaims)
	if extendedResourceClaim == nil {
		// Create one special claim for all extended resources backed by DRA in the Pod.
		// Create the ResourceClaim with pod as owner, with a generated name that uses
		// <pod name>-extended-resources- as base. The final name will get truncated if it
		// would be too long.
		extendedResourceClaim = &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: pod.Namespace,
				Name:      specialClaimInMemName,
				// fake temporary UID for use in SignalClaimPendingAllocation
				UID:          types.UID(uuid.NewUUID()),
				GenerateName: pod.Name + "-extended-resources-",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion:         "v1",
						Kind:               "Pod",
						Name:               pod.Name,
						UID:                pod.UID,
						Controller:         ptr.To(true),
						BlockOwnerDeletion: ptr.To(true),
					},
				},
				Annotations: map[string]string{
					resourceapi.ExtendedResourceClaimAnnotation: "true",
				},
			},
			Spec: resourceapi.ResourceClaimSpec{},
		}
	}
	return extendedResourceClaim, nil
}

// filterExtendedResources computes the special claim's Requests based on the
// node's Allocatable. It returns the special claim updated to match what needs
// to be allocated through DRA for the node or nil if nothing needs to be allocated.
//
// It returns an error when the pod's extended resource requests cannot be allocated
// from node's Allocatable, nor matching any device class's explicit or implicit
// ExtendedResourceName.
func (pl *DynamicResources) filterExtendedResources(state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, logger klog.Logger) (*resourceapi.ResourceClaim, *fwk.Status) {
	extendedResourceClaim := state.claims.extendedResourceClaim()
	if extendedResourceClaim == nil {
		// Nothing to do.
		return nil, nil
	}

	// The claim is from the prior scheduling cycle, return unschedulable such that it can be
	// deleted at the PostFilter phase, and retry anew.
	if extendedResourceClaim.Spec.Devices.Requests != nil {
		return nil, statusUnschedulable(logger, "cannot schedule extended resource claim", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "claim", klog.KObj(extendedResourceClaim))
	}

	extendedResources := make(map[v1.ResourceName]int64)
	hasExtendedResource := false
	for rName, rQuant := range state.draExtendedResource.podScalarResources {
		if !schedutil.IsDRAExtendedResourceName(rName) {
			continue
		}
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}

		allocatable, okScalar := nodeInfo.GetAllocatable().GetScalarResources()[rName]
		_, okDynamic := state.draExtendedResource.resourceToDeviceClass[rName]
		if okDynamic {
			if allocatable > 0 {
				// node provides the resource via device plugin
				extendedResources[rName] = 0
			} else {
				// node needs to provide the resource via DRA
				extendedResources[rName] = rQuant
				hasExtendedResource = true
			}
		} else if !okScalar {
			// has request neither provided by device plugin, nor backed by DRA,
			// hence the pod does not fit the node.
			return nil, statusUnschedulable(logger, "cannot fit resource", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "resource", rName)
		}
	}

	// No extended resources backed by DRA on this node.
	// The pod may have extended resources, but they are all backed by device
	// plugin, hence the noderesources plugin should have checked if the node
	// can fit the pod.
	// This dynamic resources plugin Filter phase has nothing left to do.
	if state.claims.noUserClaim() && !hasExtendedResource {
		// It cannot be allocated when reaching here, as the claim from prior scheduling cycle
		// would return unschedulable earlier in this function.
		return nil, nil
	}

	// Each node needs its own, potentially different variant of the claim.
	nodeExtendedResourceClaim := extendedResourceClaim.DeepCopy()
	nodeExtendedResourceClaim.Spec.Devices.Requests = createDeviceRequests(pod, extendedResources, state.draExtendedResource.resourceToDeviceClass)

	if extendedResourceClaim.Status.Allocation != nil {
		// If it is already allocated, then we cannot simply allocate it again.
		//
		// It cannot be allocated when reaching here, as the claim found from prior scheduling cycle
		// would return unschedulable earlier in this function.
		return nil, nil
	}

	return nodeExtendedResourceClaim, nil
}

// createDeviceRequests computes the special claim's Requests based on the pod's extended resources
// that are not satisfied by the node's Allocatable.
//
// the device request name has the format: container-%d-request-%d,
// the first %d is the container's index in the pod's initContainer and containers
// the second %d is the extended resource's index in that container's sorted resource requests.
func createDeviceRequests(pod *v1.Pod, extendedResources map[v1.ResourceName]int64, deviceClassMapping map[v1.ResourceName]string) []resourceapi.DeviceRequest {
	var deviceRequests []resourceapi.DeviceRequest
	// Creating the extended resource claim's Requests by
	// iterating over the containers, and the resources in the containers,
	// and create one request per <container, extended resource>.

	// pod level resources currently have only cpu and memory, they are not considered here for now.
	// if extended resources are added to pod level resources in the future, they need to be
	// supported separately.
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	for r := range extendedResources {
		for i, c := range containers {
			creqs := c.Resources.Requests
			if creqs == nil {
				continue
			}
			var rQuant resource.Quantity
			var ok bool
			if rQuant, ok = creqs[r]; !ok {
				continue
			}
			crq, ok := (&rQuant).AsInt64()
			if !ok || crq == 0 {
				continue
			}
			className, ok := deviceClassMapping[r]
			// skip if the request does not map to a device class
			if !ok || className == "" {
				continue
			}
			keys := make([]string, 0, len(creqs))
			for k := range creqs {
				keys = append(keys, k.String())
			}
			// resource requests in a container is a map, their names must
			// be sorted to determine the resource's index order.
			slice.SortStrings(keys)
			ridx := 0
			for j := range keys {
				if keys[j] == r.String() {
					ridx = j
					break
				}
			}
			// i is the index of the container if the list of initContainers + containers.
			// ridx is the index of the extended resource request in the sorted all requests in the container.
			// crq is the quantity of the extended resource request.
			deviceRequests = append(deviceRequests,
				resourceapi.DeviceRequest{
					Name: fmt.Sprintf("container-%d-request-%d", i, ridx), // need to be container name index - extended resource name index
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: className, // map external resource name -> device class name
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           crq,
					},
				})
		}
	}
	sort.Slice(deviceRequests, func(i, j int) bool {
		return deviceRequests[i].Name < deviceRequests[j].Name
	})
	return deviceRequests
}

// isSpecialClaimName return true when the name is the specialClaimInMemName.
func isSpecialClaimName(name string) bool {
	return name == specialClaimInMemName
}

// createRequestMappings creates the requestMappings for the special extended resource claim.
// For each device request in the claim, it finds the container name, and
// the extended resource name in that container matching the device request.
// the device request name has the format: container-%d-request-%d,
// the first %d is the container's index in the pod's initContainer and containers
// the second %d is the extended resource's index in that container's sorted resource requests.
func createRequestMappings(claim *resourceapi.ResourceClaim, pod *v1.Pod) []v1.ContainerExtendedResourceRequest {
	var cer []v1.ContainerExtendedResourceRequest
	deviceReqNames := make([]string, 0, len(claim.Spec.Devices.Requests))
	for _, r := range claim.Spec.Devices.Requests {
		deviceReqNames = append(deviceReqNames, r.Name)
	}
	// pod level resources currently have only cpu and memory, they are not considered here for now.
	// if extended resources are added to pod level resources in the future, they need to be
	// supported separately.
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	for i, c := range containers {
		creqs := c.Resources.Requests
		keys := make([]string, 0, len(creqs))
		for k := range creqs {
			keys = append(keys, k.String())
		}
		// resource requests in a container is a map, their names must
		// be sorted to determine the resource's index order.
		slice.SortStrings(keys)
		for rName := range creqs {
			ridx := 0
			for j := range keys {
				if keys[j] == rName.String() {
					ridx = j
					break
				}
			}
			for _, devReqName := range deviceReqNames {
				// During filter phase, device request name is set to be
				// container name index "-" extended resource name index
				if fmt.Sprintf("container-%d-request-%d", i, ridx) == devReqName {
					cer = append(cer,
						v1.ContainerExtendedResourceRequest{
							ContainerName: c.Name,
							ResourceName:  rName.String(),
							RequestName:   devReqName,
						})
				}
			}
		}
	}
	return cer
}
