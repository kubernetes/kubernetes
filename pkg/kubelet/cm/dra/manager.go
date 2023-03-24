/*
Copyright 2022 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	dra "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// draManagerStateFileName is the file name where dra manager stores its state
const draManagerStateFileName = "dra_manager_state"

// ManagerImpl is the structure in charge of managing DRA resource Plugins.
type ManagerImpl struct {
	// cache contains cached claim info
	cache *claimInfoCache

	// KubeClient reference
	kubeClient clientset.Interface
}

// NewManagerImpl creates a new manager.
func NewManagerImpl(kubeClient clientset.Interface, stateFileDirectory string) (*ManagerImpl, error) {
	klog.V(2).InfoS("Creating DRA manager")

	claimInfoCache, err := newClaimInfoCache(stateFileDirectory, draManagerStateFileName)
	if err != nil {
		return nil, fmt.Errorf("failed to create claimInfo cache: %+v", err)
	}

	manager := &ManagerImpl{
		cache:      claimInfoCache,
		kubeClient: kubeClient,
	}

	return manager, nil
}

// PrepareResources attempts to prepare all of the required resource
// plugin resources for the input container, issue an NodePrepareResource rpc request
// for each new resource requirement, process their responses and update the cached
// containerResources on success.
func (m *ManagerImpl) PrepareResources(pod *v1.Pod) error {
	// Process resources for each resource claim referenced by container
	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		for range container.Resources.Claims {
			for i := range pod.Spec.ResourceClaims {
				claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
				klog.V(3).InfoS("Processing resource", "claim", claimName, "pod", pod.Name)

				// Resource is already prepared, add pod UID to it
				if claimInfo := m.cache.get(claimName, pod.Namespace); claimInfo != nil {
					// We delay checkpointing of this change until this call
					// returns successfully. It is OK to do this because we
					// will only return successfully from this call if the
					// checkpoint has succeeded. That means if the kubelet is
					// ever restarted before this checkpoint succeeds, the pod
					// whose resources are being prepared would never have
					// started, so it's OK (actually correct) to not include it
					// in the cache.
					claimInfo.addPodReference(pod.UID)
					continue
				}

				// Query claim object from the API server
				resourceClaim, err := m.kubeClient.ResourceV1alpha2().ResourceClaims(pod.Namespace).Get(
					context.TODO(),
					claimName,
					metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %+v", claimName, pod.Name, err)
				}

				// Check if pod is in the ReservedFor for the claim
				if !resourceclaim.IsReservedForPod(pod, resourceClaim) {
					return fmt.Errorf("pod %s(%s) is not allowed to use resource claim %s(%s)",
						pod.Name, pod.UID, claimName, resourceClaim.UID)
				}

				// Grab the allocation.resourceHandles. If there are no
				// allocation.resourceHandles, create a single resourceHandle with no
				// content. This will trigger processing of this claim by a single
				// kubelet plugin whose name matches resourceClaim.Status.DriverName.
				resourceHandles := resourceClaim.Status.Allocation.ResourceHandles
				if len(resourceHandles) == 0 {
					resourceHandles = make([]resourcev1alpha2.ResourceHandle, 1)
				}

				// Create a claimInfo object to store the relevant claim info.
				claimInfo := newClaimInfo(
					resourceClaim.Status.DriverName,
					resourceClaim.Spec.ResourceClassName,
					resourceClaim.UID,
					resourceClaim.Name,
					resourceClaim.Namespace,
					sets.New(string(pod.UID)),
				)

				// Walk through each resourceHandle
				for _, resourceHandle := range resourceHandles {
					// If no DriverName is provided in the resourceHandle, we
					// use the DriverName from the status
					pluginName := resourceHandle.DriverName
					if pluginName == "" {
						pluginName = resourceClaim.Status.DriverName
					}

					// Call NodePrepareResource RPC for each resourceHandle
					client, err := dra.NewDRAPluginClient(pluginName)
					if err != nil {
						return fmt.Errorf("failed to get DRA Plugin client for plugin name %s, err=%+v", pluginName, err)
					}
					response, err := client.NodePrepareResource(
						context.Background(),
						resourceClaim.Namespace,
						resourceClaim.UID,
						resourceClaim.Name,
						resourceHandle.Data)
					if err != nil {
						return fmt.Errorf("NodePrepareResource failed, claim UID: %s, claim name: %s, resource handle: %s, err: %+v",
							resourceClaim.UID, resourceClaim.Name, resourceHandle.Data, err)
					}
					klog.V(3).InfoS("NodePrepareResource succeeded", "pluginName", pluginName, "response", response)

					// Add the CDI Devices returned by NodePrepareResource to
					// the claimInfo object.
					err = claimInfo.addCDIDevices(pluginName, response.CdiDevices)
					if err != nil {
						return fmt.Errorf("failed to add CDIDevices to claimInfo %+v: %+v", claimInfo, err)
					}

					// TODO: We (re)add the claimInfo object to the cache and
					// sync it to the checkpoint *after* the
					// NodePrepareResource call has completed. This will cause
					// issues if the kubelet gets restarted between
					// NodePrepareResource and syncToCheckpoint. It will result
					// in not calling NodeUnprepareResource for this claim
					// because no claimInfo will be synced back to the cache
					// for it after the restart. We need to resolve this issue
					// before moving to beta.
					m.cache.add(claimInfo)

					// Checkpoint to reduce redundant calls to
					// NodePrepareResource() after a kubelet restart.
					err = m.cache.syncToCheckpoint()
					if err != nil {
						return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
					}
				}
			}
		}
	}
	// Checkpoint to capture all of the previous addPodReference() calls.
	err := m.cache.syncToCheckpoint()
	if err != nil {
		return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
	}
	return nil
}

// GetResources gets a ContainerInfo object from the claimInfo cache.
// This information is used by the caller to update a container config.
func (m *ManagerImpl) GetResources(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	annotations := []kubecontainer.Annotation{}
	cdiDevices := []kubecontainer.CDIDevice{}

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])

		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}

			claimInfo := m.cache.get(claimName, pod.Namespace)
			if claimInfo == nil {
				return nil, fmt.Errorf("unable to get resource for namespace: %s, claim: %s", pod.Namespace, claimName)
			}

			klog.V(3).InfoS("Add resource annotations", "claim", claimName, "annotations", claimInfo.annotations)
			annotations = append(annotations, claimInfo.annotations...)
			for _, devices := range claimInfo.CDIDevices {
				for _, device := range devices {
					cdiDevices = append(cdiDevices, kubecontainer.CDIDevice{Name: device})
				}
			}
		}
	}

	return &ContainerInfo{Annotations: annotations, CDIDevices: cdiDevices}, nil
}

// UnprepareResources calls a plugin's NodeUnprepareResource API for each resource claim owned by a pod.
// This function is idempotent and may be called multiple times against the same pod.
// As such, calls to the underlying NodeUnprepareResource API are skipped for claims that have
// already been successfully unprepared.
func (m *ManagerImpl) UnprepareResources(pod *v1.Pod) error {
	// Call NodeUnprepareResource RPC for every resource claim referenced by the pod
	for i := range pod.Spec.ResourceClaims {
		claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		claimInfo := m.cache.get(claimName, pod.Namespace)

		// Skip calling NodeUnprepareResource if claim info is not cached
		if claimInfo == nil {
			continue
		}

		// Skip calling NodeUnprepareResource if other pods are still referencing it
		if len(claimInfo.PodUIDs) > 1 {
			// We delay checkpointing of this change until this call returns successfully.
			// It is OK to do this because we will only return successfully from this call if
			// the checkpoint has succeeded. That means if the kubelet is ever restarted
			// before this checkpoint succeeds, we will simply call into this (idempotent)
			// function again.
			claimInfo.deletePodReference(pod.UID)
			continue
		}

		// Query claim object from the API server
		resourceClaim, err := m.kubeClient.ResourceV1alpha2().ResourceClaims(pod.Namespace).Get(
			context.TODO(),
			claimName,
			metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %+v", claimName, pod.Name, err)
		}

		// Grab the allocation.resourceHandles. If there are no
		// allocation.resourceHandles, create a single resourceHandle with no
		// content. This will trigger processing of this claim by a single
		// kubelet plugin whose name matches resourceClaim.Status.DriverName.
		resourceHandles := resourceClaim.Status.Allocation.ResourceHandles
		if len(resourceHandles) == 0 {
			resourceHandles = make([]resourcev1alpha2.ResourceHandle, 1)
		}

		// Loop through all plugins and call NodeUnprepareResource only for the
		// last pod that references the claim
		for _, resourceHandle := range resourceHandles {
			// If no DriverName is provided in the resourceHandle, we
			// use the DriverName from the status
			pluginName := resourceHandle.DriverName
			if pluginName == "" {
				pluginName = claimInfo.DriverName
			}

			// Call NodeUnprepareResource RPC for each resourceHandle
			client, err := dra.NewDRAPluginClient(pluginName)
			if err != nil {
				return fmt.Errorf("failed to get DRA Plugin client for plugin name %s, err=%+v", pluginName, err)
			}
			response, err := client.NodeUnprepareResource(
				context.Background(),
				claimInfo.Namespace,
				claimInfo.ClaimUID,
				claimInfo.ClaimName,
				resourceHandle.Data)
			if err != nil {
				return fmt.Errorf(
					"NodeUnprepareResource failed, pod: %s, claim UID: %s, claim name: %s, CDI devices: %s, err: %+v",
					pod.Name, claimInfo.ClaimUID, claimInfo.ClaimName, claimInfo.CDIDevices, err)
			}
			klog.V(3).InfoS("NodeUnprepareResource succeeded", "response", response)
		}

		// Delete last pod UID only if all NodeUnprepareResource calls succeed.
		// This ensures that the status manager doesn't enter termination status
		// for the pod. This logic is implemented in
		// m.PodMightNeedToUnprepareResources and claimInfo.hasPodReference.
		claimInfo.deletePodReference(pod.UID)
		m.cache.delete(claimInfo.ClaimName, pod.Namespace)

		// Checkpoint to reduce redundant calls to NodeUnPrepareResource() after a kubelet restart.
		err = m.cache.syncToCheckpoint()
		if err != nil {
			return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
		}
	}
	// Checkpoint to capture all of the previous deletePodReference() calls.
	err := m.cache.syncToCheckpoint()
	if err != nil {
		return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
	}
	return nil
}

// PodMightNeedToUnprepareResources returns true if the pod might need to
// unprepare resources
func (m *ManagerImpl) PodMightNeedToUnprepareResources(UID types.UID) bool {
	return m.cache.hasPodReference(UID)
}

// GetCongtainerClaimInfos gets Container's ClaimInfo
func (m *ManagerImpl) GetContainerClaimInfos(pod *v1.Pod, container *v1.Container) ([]*ClaimInfo, error) {
	claimInfos := make([]*ClaimInfo, 0, len(pod.Spec.ResourceClaims))

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])

		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}
			claimInfo := m.cache.get(claimName, pod.Namespace)
			if claimInfo == nil {
				return nil, fmt.Errorf("unable to get resource for namespace: %s, claim: %s", pod.Namespace, claimName)
			}
			claimInfos = append(claimInfos, claimInfo)
		}
	}
	return claimInfos, nil
}
