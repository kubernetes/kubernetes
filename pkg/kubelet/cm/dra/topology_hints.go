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

package dra

import (
	"context"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	topologyNUMAAttributeSuffix = "/numaNode"
	topologyResourcePrefix      = "claim:"
)

var _ topologymanager.HintProvider = (*Manager)(nil)

type topologyLookupResult int

const (
	topologyLookupUnavailable topologyLookupResult = iota
	topologyLookupAvailable
	topologyLookupError
)

// GetTopologyHints implements topologymanager.HintProvider for container scope.
func (m *Manager) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	logger := klog.TODO()
	claimRequests := m.collectContainerClaimRequests(pod, container)
	return m.getTopologyHintsForClaimRequests(logger, pod, claimRequests)
}

// GetPodTopologyHints implements topologymanager.HintProvider for pod scope.
func (m *Manager) GetPodTopologyHints(pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	logger := klog.TODO()
	claimRequests := map[string]sets.Set[string]{}

	allContainers := make([]v1.Container, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	allContainers = append(allContainers, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	for i := range allContainers {
		mergeClaimRequests(claimRequests, m.collectContainerClaimRequests(pod, &allContainers[i]))
	}

	return m.getTopologyHintsForClaimRequests(logger, pod, claimRequests)
}

// Allocate is a no-op because DRA devices are already allocated by the scheduler.
func (m *Manager) Allocate(_ *v1.Pod, _ *v1.Container) error {
	return nil
}

// AllocatePod is a no-op because DRA devices are already allocated by the scheduler.
func (m *Manager) AllocatePod(_ *v1.Pod) error {
	return nil
}

func (m *Manager) getTopologyHintsForClaimRequests(logger klog.Logger, pod *v1.Pod, claimRequests map[string]sets.Set[string]) map[string][]topologymanager.TopologyHint {
	if len(claimRequests) == 0 {
		return nil
	}

	if m.kubeClient == nil || pod.Spec.NodeName == "" {
		return emptyTopologyHints(claimRequests)
	}

	sliceList, err := m.kubeClient.ResourceV1().ResourceSlices().List(context.TODO(), metav1.ListOptions{
		FieldSelector: fields.Set{resourceapi.ResourceSliceSelectorNodeName: pod.Spec.NodeName}.AsSelector().String(),
	})
	if err != nil {
		logger.Error(err, "Failed to list ResourceSlices for DRA topology hints", "pod", klog.KObj(pod))
		return emptyTopologyHints(claimRequests)
	}

	hints := make(map[string][]topologymanager.TopologyHint)
	for claimName, requests := range claimRequests {
		claim, err := m.kubeClient.ResourceV1().ResourceClaims(pod.Namespace).Get(context.TODO(), claimName, metav1.GetOptions{})
		if err != nil {
			logger.Error(err, "Failed to get ResourceClaim for DRA topology hints", "pod", klog.KObj(pod), "claim", klog.KRef(pod.Namespace, claimName))
			addEmptyRequestHints(hints, claimName, requests)
			continue
		}
		if claim.Status.Allocation == nil {
			addEmptyRequestHints(hints, claimName, requests)
			continue
		}

		requestNames := sets.List(requests)
		if len(requestNames) == 0 {
			requestNames = []string{""}
		}
		for _, requestName := range requestNames {
			resourceName := topologyResourceName(claimName, requestName)
			hints[resourceName] = buildTopologyHintsForRequest(logger, pod.Spec.NodeName, sliceList.Items, claim.Status.Allocation.Devices.Results, requestName)
		}
	}

	return hints
}

func (m *Manager) collectContainerClaimRequests(pod *v1.Pod, container *v1.Container) map[string]sets.Set[string] {
	claimRequests := make(map[string]sets.Set[string])

	containerClaimsMap := make(map[string][]string, len(container.Resources.Claims))
	for _, claim := range container.Resources.Claims {
		containerClaimsMap[claim.Name] = append(containerClaimsMap[claim.Name], claim.Request)
	}

	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		requests, ok := containerClaimsMap[podClaim.Name]
		if !ok {
			continue
		}

		claimName, _, err := resourceclaim.Name(pod, podClaim)
		if err != nil || claimName == nil {
			continue
		}
		insertClaimRequests(claimRequests, *claimName, requests...)
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DRAExtendedResource) && pod.Status.ExtendedResourceClaimStatus != nil {
		extendedResourceRequests := make(map[string]bool)
		for rName, rValue := range container.Resources.Requests {
			if !rValue.IsZero() && schedutil.IsDRAExtendedResourceName(rName) {
				extendedResourceRequests[rName.String()] = true
			}
		}

		for _, rm := range pod.Status.ExtendedResourceClaimStatus.RequestMappings {
			if rm.ContainerName == container.Name && extendedResourceRequests[rm.ResourceName] {
				insertClaimRequests(claimRequests, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName, rm.RequestName)
			}
		}
	}

	return claimRequests
}

func mergeClaimRequests(dst, src map[string]sets.Set[string]) {
	for claimName, requests := range src {
		if _, ok := dst[claimName]; !ok {
			dst[claimName] = sets.New[string]()
		}
		dst[claimName].Insert(sets.List(requests)...)
	}
}

func insertClaimRequests(claimRequests map[string]sets.Set[string], claimName string, requests ...string) {
	if _, ok := claimRequests[claimName]; !ok {
		claimRequests[claimName] = sets.New[string]()
	}
	claimRequests[claimName].Insert(requests...)
}

func emptyTopologyHints(claimRequests map[string]sets.Set[string]) map[string][]topologymanager.TopologyHint {
	hints := make(map[string][]topologymanager.TopologyHint)
	for claimName, requests := range claimRequests {
		addEmptyRequestHints(hints, claimName, requests)
	}
	return hints
}

func addEmptyRequestHints(hints map[string][]topologymanager.TopologyHint, claimName string, requests sets.Set[string]) {
	requestNames := sets.List(requests)
	if len(requestNames) == 0 {
		requestNames = []string{""}
	}
	for _, requestName := range requestNames {
		hints[topologyResourceName(claimName, requestName)] = []topologymanager.TopologyHint{}
	}
}

func topologyResourceName(claimName, requestName string) string {
	if requestName == "" {
		return topologyResourcePrefix + claimName
	}
	return fmt.Sprintf("%s%s/%s", topologyResourcePrefix, claimName, requestName)
}

func buildTopologyHintsForRequest(logger klog.Logger, nodeName string, slices []resourceapi.ResourceSlice, results []resourceapi.DeviceRequestAllocationResult, requestName string) []topologymanager.TopologyHint {
	matchingResults := matchingAllocationResults(results, requestName)
	if len(matchingResults) == 0 {
		return []topologymanager.TopologyHint{}
	}

	numaNodes := sets.New[int]()
	for _, result := range matchingResults {
		nodes, lookupResult := lookupDeviceNUMANodes(logger, nodeName, slices, result)
		switch lookupResult {
		case topologyLookupError:
			return []topologymanager.TopologyHint{}
		case topologyLookupUnavailable:
			return nil
		case topologyLookupAvailable:
			numaNodes.Insert(nodes...)
		}
	}

	mask, err := bitmask.NewBitMask(sets.List(numaNodes)...)
	if err != nil {
		logger.Error(err, "Failed to build DRA NUMA affinity bitmask", "nodes", sets.List(numaNodes))
		return []topologymanager.TopologyHint{}
	}

	return []topologymanager.TopologyHint{{
		NUMANodeAffinity: mask,
		Preferred:        true,
	}}
}

func matchingAllocationResults(results []resourceapi.DeviceRequestAllocationResult, requestName string) []resourceapi.DeviceRequestAllocationResult {
	if requestName == "" {
		return results
	}

	matchingResults := make([]resourceapi.DeviceRequestAllocationResult, 0, len(results))
	for _, result := range results {
		if requestName == result.Request || requestName == resourceclaim.BaseRequestRef(result.Request) {
			matchingResults = append(matchingResults, result)
		}
	}
	return matchingResults
}

func lookupDeviceNUMANodes(logger klog.Logger, nodeName string, slices []resourceapi.ResourceSlice, result resourceapi.DeviceRequestAllocationResult) ([]int, topologyLookupResult) {
	for i := range slices {
		slice := &slices[i]
		if slice.Spec.Driver != result.Driver || slice.Spec.Pool.Name != result.Pool {
			continue
		}

		for j := range slice.Spec.Devices {
			device := &slice.Spec.Devices[j]
			if device.Name != result.Device || !deviceAccessibleFromNode(slice, device, nodeName) {
				continue
			}

			nodes, ok := extractNUMANodes(device.Attributes)
			if !ok {
				return nil, topologyLookupUnavailable
			}
			return nodes, topologyLookupAvailable
		}
	}

	logger.Info("Unable to resolve allocated DRA device in ResourceSlices for topology hinting",
		"driver", result.Driver, "pool", result.Pool, "device", result.Device, "node", nodeName)
	return nil, topologyLookupError
}

func deviceAccessibleFromNode(slice *resourceapi.ResourceSlice, device *resourceapi.Device, nodeName string) bool {
	if slice.Spec.PerDeviceNodeSelection != nil && *slice.Spec.PerDeviceNodeSelection {
		return nodeSelectionAllowsNode(device.NodeName, device.AllNodes, nodeName)
	}
	return nodeSelectionAllowsNode(slice.Spec.NodeName, slice.Spec.AllNodes, nodeName)
}

func nodeSelectionAllowsNode(selectedNode *string, allNodes *bool, nodeName string) bool {
	if selectedNode != nil {
		return *selectedNode == nodeName
	}
	if allNodes != nil && *allNodes {
		return true
	}
	return true
}

func extractNUMANodes(attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) ([]int, bool) {
	for name, attribute := range attributes {
		if !strings.HasSuffix(string(name), topologyNUMAAttributeSuffix) {
			continue
		}
		return parseNUMANodes(attribute)
	}
	return nil, false
}

func parseNUMANodes(attribute resourceapi.DeviceAttribute) ([]int, bool) {
	switch {
	case attribute.IntValue != nil:
		if *attribute.IntValue < 0 {
			return nil, false
		}
		return []int{int(*attribute.IntValue)}, true
	case len(attribute.IntValues) > 0:
		nodes := sets.New[int]()
		for _, value := range attribute.IntValues {
			if value < 0 {
				return nil, false
			}
			nodes.Insert(int(value))
		}
		return sets.List(nodes), true
	default:
		return nil, false
	}
}
