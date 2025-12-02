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
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/scheduler/metrics"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/util/retry"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
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

	// AssumeExtendedResourceTimeoutDefaultSeconds is the default timeout for waiting
	// for the extended resource claim to be updated in assumed cache.
	AssumeExtendedResourceTimeoutDefaultSeconds = 120
)

// draExtendedResource stores data for extended resources backed by DRA.
// It will remain empty when the DRAExtendedResource feature is disabled.
type draExtendedResource struct {
	// May have extended resource backed by DRA.
	podScalarResources map[v1.ResourceName]int64
}

// hasDeviceClassMappedExtendedResource returns true when the given resource list has an extended resource, that has
// a mapping to a device class.
func hasDeviceClassMappedExtendedResource(reqs v1.ResourceList, cache fwk.DeviceClassResolver) bool {
	for rName, rValue := range reqs {
		if rValue.IsZero() {
			// We only care about the resources requested by the pod we are trying to schedule.
			continue
		}
		if schedutil.IsDRAExtendedResourceName(rName) {
			if cache.GetDeviceClass(rName) != nil {
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

	// Try to build device class mapping from cache
	cache := pl.draManager.DeviceClassResolver()
	reqs := resourcehelper.PodRequests(pod, resourcehelper.PodResourcesOptions{})

	hasExtendedResource := hasDeviceClassMappedExtendedResource(reqs, cache)
	if !hasExtendedResource {
		return nil, nil
	}
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
	if extendedResourceClaim != nil {
		return extendedResourceClaim, nil
	}
	// Create one special claim for all extended resources backed by DRA in the Pod.
	// Create the ResourceClaim with pod as owner, with a generated name that uses
	// <pod name>-extended-resources- as base. The final name will get truncated if it
	// would be too long.
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: pod.Namespace,
			Name:      specialClaimInMemName,
			// fake temporary UID for use in SignalClaimPendingAllocation
			UID:          types.UID(uuid.NewUUID()),
			GenerateName: pod.Name + "-extended-resources-",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Pod",
					Name:       pod.Name,
					UID:        pod.UID,
					Controller: ptr.To(true),
				},
			},
			Annotations: map[string]string{
				resourceapi.ExtendedResourceClaimAnnotation: "true",
			},
		},
		Spec: resourceapi.ResourceClaimSpec{},
	}, nil
}

// filterExtendedResources computes the special claim's Requests based on the
// node's Allocatable. It returns:
// - nil if nothing needs to be allocated, all the extended resources are satisfied by device plugin, or
// - the special claim updated to match what needs to be allocated through DRA for the node
//
// It returns an error when the pod's extended resource requests cannot be allocated
// from node's Allocatable, nor matching any device class's explicit or implicit
// ExtendedResourceName.
func (pl *DynamicResources) filterExtendedResources(state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, logger klog.Logger) (*resourceapi.ResourceClaim, []v1.ContainerExtendedResourceRequest, *fwk.Status) {
	extendedResourceClaim := state.claims.extendedResourceClaim()
	if extendedResourceClaim == nil {
		// Nothing to do.
		return nil, nil, nil
	}

	// The claim is from the prior scheduling cycle, return unschedulable such that it can be
	// deleted at the PostFilter phase, and retry anew.
	if extendedResourceClaim.Spec.Devices.Requests != nil {
		return nil, nil, statusUnschedulable(logger, "cannot schedule extended resource claim", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "claim", klog.KObj(extendedResourceClaim))
	}

	extendedResources := make(map[v1.ResourceName]int64)
	hasExtendedResource := false
	cache := pl.draManager.DeviceClassResolver()
	for rName, rQuant := range state.draExtendedResource.podScalarResources {
		if !schedutil.IsDRAExtendedResourceName(rName) {
			continue
		}
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}
		allocatable, okScalar := nodeInfo.GetAllocatable().GetScalarResources()[rName]
		isBackedByDRA := cache.GetDeviceClass(rName) != nil
		if isBackedByDRA && allocatable == 0 {
			// node needs to provide the resource via DRA
			extendedResources[rName] = rQuant
			hasExtendedResource = true
		} else if !okScalar {
			// has request neither provided by device plugin, nor backed by DRA,
			// hence the pod does not fit the node.
			return nil, nil, statusUnschedulable(logger, "cannot fit resource", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "resource", rName)
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
		return nil, nil, nil
	}

	if extendedResourceClaim.Status.Allocation != nil {
		// If it is already allocated, then we cannot simply allocate it again.
		//
		// It cannot be allocated when reaching here, as the claim found from prior scheduling cycle
		// would return unschedulable earlier in this function.
		return nil, nil, nil
	}

	// Each node needs its own, potentially different variant of the claim.
	nodeExtendedResourceClaim := extendedResourceClaim.DeepCopy()
	reqs, mappings := createRequestsAndMappings(pod, extendedResources, logger, cache)
	nodeExtendedResourceClaim.Spec.Devices.Requests = reqs

	return nodeExtendedResourceClaim, mappings, nil
}

// isSpecialClaimName return true when the name is the specialClaimInMemName.
func isSpecialClaimName(name string) bool {
	return name == specialClaimInMemName
}

// deleteClaim deletes the claim after removing the finalizer from the claim, if there is any.
func (pl *DynamicResources) deleteClaim(ctx context.Context, claim *resourceapi.ResourceClaim, logger klog.Logger) error {
	refreshClaim := false
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if refreshClaim {
			updatedClaim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("get resourceclaim %s/%s: %w", claim.Namespace, claim.Name, err)
			}
			claim = updatedClaim
		} else {
			refreshClaim = true
		}
		// Remove the finalizer to unblock removal first.
		builtinControllerFinalizer := slices.Index(claim.Finalizers, resourceapi.Finalizer)
		if builtinControllerFinalizer >= 0 {
			claim.Finalizers = slices.Delete(claim.Finalizers, builtinControllerFinalizer, builtinControllerFinalizer+1)
		}

		_, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Update(ctx, claim, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("update resourceclaim %s/%s: %w", claim.Namespace, claim.Name, err)
		}
		return nil
	})
	if retryErr != nil {
		return retryErr
	}

	logger.V(5).Info("Delete", "resourceclaim", klog.KObj(claim))
	err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{})
	if err != nil {
		return err
	}
	return nil
}

func partitionContainerIndices(containers []v1.Container, numInitContainers int) ([]int, []int) {
	longLivedContainerIndices := make([]int, 0, len(containers))
	shortLivedInitContainerIndices := make([]int, 0, numInitContainers)
	for i, c := range containers {
		isInit := i < numInitContainers
		isSidecar := c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyAlways
		if isInit && !isSidecar {
			shortLivedInitContainerIndices = append(shortLivedInitContainerIndices, i)
			continue
		}
		longLivedContainerIndices = append(longLivedContainerIndices, i)
	}
	return longLivedContainerIndices, shortLivedInitContainerIndices
}

// createResourceRequestAndMappings returns the request and mappings for the given container and resource.
// reusableRequests is a list of other DeviceRequests this container can use before requesting its own.
// items in reusableRequests may be nil.
// The returned request may be nil if no additional request was required.
// The returned mappings may be empty if this container does not use this resource.
func createResourceRequestAndMappings(containerIndex int, container *v1.Container, rName v1.ResourceName, className string, reusableRequests []*resourceapi.DeviceRequest) (*resourceapi.DeviceRequest, []v1.ContainerExtendedResourceRequest) {
	var mappings []v1.ContainerExtendedResourceRequest
	creqs := container.Resources.Requests
	if creqs == nil {
		return nil, nil
	}
	var rQuant resource.Quantity
	var ok bool
	if rQuant, ok = creqs[rName]; !ok {
		return nil, nil
	}
	crq, ok := (&rQuant).AsInt64()
	if !ok || crq == 0 {
		return nil, nil
	}
	sum := int64(0)
	for _, r := range reusableRequests {
		if r != nil {
			sum += r.Exactly.Count
			mappings = append(mappings, v1.ContainerExtendedResourceRequest{
				ContainerName: container.Name,
				ResourceName:  rName.String(),
				RequestName:   r.Name,
			})
			if sum >= crq {
				return nil, mappings
			}
		}
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
		if keys[j] == rName.String() {
			ridx = j
			break
		}
	}
	// containerIndex is the index of the container in the list of initContainers + containers.
	// ridx is the index of the extended resource request in the sorted all requests in the container.
	// crq is the quantity of the extended resource request.
	reqName := fmt.Sprintf("container-%d-request-%d", containerIndex, ridx)
	deviceReq := resourceapi.DeviceRequest{
		Name: reqName, // need to be container name index - extended resource name index
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: className,
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			Count:           crq - sum, // the extra devices to request
		},
	}
	mappings = append(mappings, v1.ContainerExtendedResourceRequest{
		ContainerName: container.Name,
		ResourceName:  rName.String(),
		RequestName:   reqName,
	})

	return &deviceReq, mappings
}

func createRequestsAndMappings(pod *v1.Pod, extendedResources map[v1.ResourceName]int64, logger klog.Logger, deviceClassMapping fwk.DeviceClassResolver) ([]resourceapi.DeviceRequest, []v1.ContainerExtendedResourceRequest) {
	containers := slices.Clone(pod.Spec.InitContainers)
	containers = append(containers, pod.Spec.Containers...)
	longLivedContainerIndices, shortLivedInitContainerIndices := partitionContainerIndices(containers, len(pod.Spec.InitContainers))

	// all requests across all containers and resource types
	var deviceRequests []resourceapi.DeviceRequest
	// all mappings across all containers and resource types
	var mappings []v1.ContainerExtendedResourceRequest

	// Sort resource names to ensure deterministic ordering of device requests and mappings.
	// Maps have non-deterministic iteration order in Go, so we extract and sort the keys.
	resourceNames := make([]v1.ResourceName, 0, len(extendedResources))
	for resource := range extendedResources {
		resourceNames = append(resourceNames, resource)
	}
	sort.Slice(resourceNames, func(i, j int) bool {
		return resourceNames[i] < resourceNames[j]
	})

	for _, resource := range resourceNames {
		class := deviceClassMapping.GetDeviceClass(resource)
		// skip if the resource does not map to a device class
		if class == nil {
			continue
		}

		// shortLivedResourceMappings is the mapping of container+resource→request for short lived containers (init non-sidecar container)
		var shortLivedResourceMappings []v1.ContainerExtendedResourceRequest
		// longLivedResourceMappings is the mapping of container+resource→request for long lived containers (init sidecar or regular container)
		var longLivedResourceMappings []v1.ContainerExtendedResourceRequest

		// longLivedResourceRequests is the list of requests for a given resource by long-lived containers.
		// The length of this list is the same as the length of containers.
		// Entries may be nil if the container at that index did not produce a request for that resource.
		// Requests at later indices are reusable by non-sidecar initContainers at earlier indices.
		longLivedResourceRequests := make([]*resourceapi.DeviceRequest, len(containers))
		for _, i := range longLivedContainerIndices {
			containerRequest, containerMappings := createResourceRequestAndMappings(i, &containers[i], resource, class.Name, nil)
			longLivedResourceRequests[i] = containerRequest                                     // might be nil
			longLivedResourceMappings = append(longLivedResourceMappings, containerMappings...) // might be zero-length
		}

		// maxShortLivedResourceRequest is the maximum request for a given resource by short-lived containers
		var maxShortLivedResourceRequest *resourceapi.DeviceRequest
		// shortLivedRequestNames is all request names for a given resource by short-lived containers. All mappings to any name in
		// this set will be replaced by maxShortLivedResourceRequest.Name.
		shortLivedRequestNames := sets.New[string]()
		for _, i := range shortLivedInitContainerIndices {
			containerRequest, containerMappings := createResourceRequestAndMappings(i, &containers[i], resource, class.Name, longLivedResourceRequests[i:])
			if containerRequest != nil {
				shortLivedRequestNames.Insert(containerRequest.Name)
				if maxShortLivedResourceRequest == nil || maxShortLivedResourceRequest.Exactly.Count < containerRequest.Exactly.Count {
					maxShortLivedResourceRequest = containerRequest
				}
			}
			shortLivedResourceMappings = append(shortLivedResourceMappings, containerMappings...) // might be zero-length
		}

		// rewrite mappings to short-lived requests to use the maximum short-lived request name
		if maxShortLivedResourceRequest != nil && len(shortLivedRequestNames) > 1 {
			shortLivedRequestNames.Delete(maxShortLivedResourceRequest.Name)
			for i := range shortLivedResourceMappings {
				if shortLivedRequestNames.Has(shortLivedResourceMappings[i].RequestName) {
					shortLivedResourceMappings[i].RequestName = maxShortLivedResourceRequest.Name
				}
			}
		}

		// append non-nil requests
		if maxShortLivedResourceRequest != nil {
			deviceRequests = append(deviceRequests, *maxShortLivedResourceRequest)
		}
		for _, request := range longLivedResourceRequests {
			if request != nil {
				deviceRequests = append(deviceRequests, *request)
			}
		}
		// append mappings
		mappings = append(mappings, longLivedResourceMappings...)
		mappings = append(mappings, shortLivedResourceMappings...)
	}

	sort.Slice(deviceRequests, func(i, j int) bool {
		return deviceRequests[i].Name < deviceRequests[j].Name
	})
	return deviceRequests, mappings
}

// waitForExtendedClaimInAssumeCache polls the assume cache until the extended resource claim
// becomes visible. This is necessary because extended resource claims are created in the API
// server, and the informer update may not have reached the assume cache yet.
//
// AssumeClaimAfterAPICall returns ErrNotFound when the informer update hasn't arrived,
// so we poll with a timeout.
func (pl *DynamicResources) waitForExtendedClaimInAssumeCache(
	ctx context.Context,
	logger klog.Logger,
	claim *resourceapi.ResourceClaim,
) {
	pollErr := wait.PollUntilContextTimeout(
		ctx,
		1*time.Second,
		time.Duration(AssumeExtendedResourceTimeoutDefaultSeconds)*time.Second,
		true,
		func(ctx context.Context) (bool, error) {
			if err := pl.draManager.ResourceClaims().AssumeClaimAfterAPICall(claim); err != nil {
				if errors.Is(err, assumecache.ErrNotFound) {
					return false, nil
				}
				logger.V(5).Info("Claim not stored in assume cache", "claim", klog.KObj(claim), "err", err)
				return false, err
			}
			return true, nil
		},
	)

	if pollErr != nil {
		logger.V(5).Info("Claim not stored in assume cache after retries", "claim", klog.KObj(claim), "err", pollErr)
		// Note: We log but don't fail - the claim was created successfully
	}
}

// createExtendedResourceClaimInAPI creates an extended resource claim in the API server.
func (pl *DynamicResources) createExtendedResourceClaimInAPI(
	ctx context.Context,
	logger klog.Logger,
	pod *v1.Pod,
	nodeName string,
	state *stateData,
) (*resourceapi.ResourceClaim, error) {
	logger.V(5).Info("preparing to create claim for extended resources", "pod", klog.KObj(pod), "node", nodeName)
	// Get the node-specific claim that was prepared during Filter phase
	nodeAllocation, ok := state.nodeAllocations[nodeName]
	if !ok || nodeAllocation.extendedResourceClaim == nil {
		return nil, fmt.Errorf("extended resource claim not found for node %s", nodeName)
	}
	claim := nodeAllocation.extendedResourceClaim.DeepCopy()

	logger.V(5).Info("create claim for extended resources", "pod", klog.KObj(pod), "node", nodeName, "resourceclaim", klog.Format(claim))
	// Clear fields which must or can not be set during creation.
	claim.Status.Allocation = nil
	claim.Name = ""
	claim.UID = ""

	createdClaim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Create(ctx, claim, metav1.CreateOptions{})
	if err != nil {
		metrics.ResourceClaimCreatesTotal.WithLabelValues("failure").Inc()
		return nil, fmt.Errorf("create claim for extended resources %v: %w", klog.KObj(claim), err)
	}
	metrics.ResourceClaimCreatesTotal.WithLabelValues("success").Inc()
	logger.V(5).Info("created claim for extended resources", "pod", klog.KObj(pod), "node", nodeName, "resourceclaim", klog.Format(createdClaim))

	return createdClaim, nil
}

// patchPodExtendedResourceClaimStatus updates the pod's status with information about
// the extended resource claim.
func (pl *DynamicResources) patchPodExtendedResourceClaimStatus(
	ctx context.Context,
	pod *v1.Pod,
	claim *resourceapi.ResourceClaim,
	nodeName string,
	state *stateData,
) error {
	var cer []v1.ContainerExtendedResourceRequest
	if nodeAllocation, ok := state.nodeAllocations[nodeName]; ok {
		cer = nodeAllocation.containerResourceRequestMappings
	}
	if len(cer) == 0 {
		return fmt.Errorf("nil or empty request mappings, no update of pod %s/%s ExtendedResourceClaimStatus", pod.Namespace, pod.Name)
	}

	podStatusCopy := pod.Status.DeepCopy()
	podStatusCopy.ExtendedResourceClaimStatus = &v1.PodExtendedResourceClaimStatus{
		RequestMappings:   cer,
		ResourceClaimName: claim.Name,
	}
	err := schedutil.PatchPodStatus(ctx, pl.clientset, pod.Name, pod.Namespace, &pod.Status, podStatusCopy)
	if err != nil {
		return fmt.Errorf("update pod %s/%s ExtendedResourceClaimStatus: %w", pod.Namespace, pod.Name, err)
	}
	return nil
}

// unreserveExtendedResourceClaim cleans up the scheduler-owned extended resource claim
// when scheduling fails. It reverts the assume cache, and deletes the claim from the API
// server if it was already created.
func (pl *DynamicResources) unreserveExtendedResourceClaim(ctx context.Context, logger klog.Logger, pod *v1.Pod, state *stateData) {
	extendedResourceClaim := state.claims.extendedResourceClaim()
	if extendedResourceClaim == nil {
		// there is no extended resource claim
		return
	}

	// If the claim was marked as pending allocation (in-flight), remove that marker and restore
	// the assumed claim state to what it was before this scheduling attempt.
	if deleted := pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(state.claims.getInitialExtendedResourceClaimUID()); deleted {
		pl.draManager.ResourceClaims().AssumedClaimRestore(extendedResourceClaim.Namespace, extendedResourceClaim.Name)
	}
	if isSpecialClaimName(extendedResourceClaim.Name) {
		// In memory temporary extended resource claim does not need to be deleted
		return
	}
	// Claim was written to API server, need to delete it to prevent orphaned resources.
	logger.V(5).Info("delete extended resource backed by DRA", "resourceclaim", klog.KObj(extendedResourceClaim), "pod", klog.KObj(pod), "claim.UID", extendedResourceClaim.UID)
	extendedResourceClaim = extendedResourceClaim.DeepCopy()
	if err := pl.deleteClaim(ctx, extendedResourceClaim, logger); err != nil {
		logger.Error(err, "delete", "resourceclaim", klog.KObj(extendedResourceClaim))
	}
}
