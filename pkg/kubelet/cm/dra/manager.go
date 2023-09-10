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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
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
// plugin resources for the input container, issue NodePrepareResources rpc requests
// for each new resource requirement, process their responses and update the cached
// containerResources on success.
func (m *ManagerImpl) PrepareResources(pod *v1.Pod) error {
	batches := make(map[string][]*drapb.Claim)
	claimInfos := make(map[types.UID]*ClaimInfo)
	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		klog.V(3).InfoS("Processing resource", "podClaim", podClaim.Name, "pod", pod.Name)
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			return fmt.Errorf("prepare resource claim: %v", err)
		}

		if claimName == nil {
			// Nothing to do.
			continue
		}
		// Query claim object from the API server
		resourceClaim, err := m.kubeClient.ResourceV1alpha2().ResourceClaims(pod.Namespace).Get(
			context.TODO(),
			*claimName,
			metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %+v", *claimName, pod.Name, err)
		}

		if mustCheckOwner {
			if err = resourceclaim.IsForPod(pod, resourceClaim); err != nil {
				return err
			}
		}

		// Check if pod is in the ReservedFor for the claim
		if !resourceclaim.IsReservedForPod(pod, resourceClaim) {
			return fmt.Errorf("pod %s(%s) is not allowed to use resource claim %s(%s)",
				pod.Name, pod.UID, *claimName, resourceClaim.UID)
		}

		// If no container actually uses the claim, then we don't need
		// to prepare it.
		if !claimIsUsedByPod(podClaim, pod) {
			klog.V(5).InfoS("Skipping unused resource", "claim", claimName, "pod", pod.Name)
			continue
		}

		claimInfo := m.cache.get(*claimName, pod.Namespace)
		if claimInfo == nil {
			// claim does not exist in cache, create new claimInfo object
			// to be processed later.
			claimInfo = newClaimInfoFromResourceClaim(resourceClaim)
		}

		// We delay checkpointing of this change until this call
		// returns successfully. It is OK to do this because we
		// will only return successfully from this call if the
		// checkpoint has succeeded. That means if the kubelet is
		// ever restarted before this checkpoint succeeds, the pod
		// whose resources are being prepared would never have
		// started, so it's OK (actually correct) to not include it
		// in the cache.
		claimInfo.addPodReference(pod.UID)

		if claimInfo.prepared {
			// Already prepared this claim, no need to prepare it again
			continue
		}

		// Loop through all plugins and prepare for calling NodePrepareResources.
		for _, resourceHandle := range claimInfo.ResourceHandles {
			// If no DriverName is provided in the resourceHandle, we
			// use the DriverName from the status
			pluginName := resourceHandle.DriverName
			if pluginName == "" {
				pluginName = resourceClaim.Status.DriverName
			}
			claim := &drapb.Claim{
				Namespace:      resourceClaim.Namespace,
				Uid:            string(resourceClaim.UID),
				Name:           resourceClaim.Name,
				ResourceHandle: resourceHandle.Data,
			}
			batches[pluginName] = append(batches[pluginName], claim)
		}
		claimInfos[resourceClaim.UID] = claimInfo
	}

	// Call NodePrepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for pluginName, claims := range batches {
		// Call NodePrepareResources RPC for all resource handles.
		client, err := dra.NewDRAPluginClient(pluginName)
		if err != nil {
			return fmt.Errorf("failed to get DRA Plugin client for plugin name %s: %v", pluginName, err)
		}
		response, err := client.NodePrepareResources(context.Background(), &drapb.NodePrepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodePrepareResources failed: %v", err)
		}
		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodePrepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.Error != "" {
				return fmt.Errorf("NodePrepareResources failed for claim %s/%s: %s", reqClaim.Namespace, reqClaim.Name, result.Error)
			}

			claimInfo := claimInfos[types.UID(claimUID)]

			// Add the CDI Devices returned by NodePrepareResources to
			// the claimInfo object.
			err = claimInfo.addCDIDevices(pluginName, result.CDIDevices)
			if err != nil {
				return fmt.Errorf("failed to add CDIDevices to claimInfo %+v: %+v", claimInfo, err)
			}
			// mark claim as (successfully) prepared by manager, so next time we dont prepare it.
			claimInfo.prepared = true

			// TODO: We (re)add the claimInfo object to the cache and
			// sync it to the checkpoint *after* the
			// NodePrepareResources call has completed. This will cause
			// issues if the kubelet gets restarted between
			// NodePrepareResources and syncToCheckpoint. It will result
			// in not calling NodeUnprepareResources for this claim
			// because no claimInfo will be synced back to the cache
			// for it after the restart. We need to resolve this issue
			// before moving to beta.
			m.cache.add(claimInfo)
		}

		// Checkpoint to reduce redundant calls to
		// NodePrepareResources after a kubelet restart.
		err = m.cache.syncToCheckpoint()
		if err != nil {
			return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodePrepareResources left out %d claims", unfinished)
		}
	}
	// Checkpoint to capture all of the previous addPodReference() calls.
	err := m.cache.syncToCheckpoint()
	if err != nil {
		return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
	}
	return nil
}

func lookupClaimRequest(claims []*drapb.Claim, claimUID string) *drapb.Claim {
	for _, claim := range claims {
		if claim.Uid == claimUID {
			return claim
		}
	}
	return nil
}

func claimIsUsedByPod(podClaim *v1.PodResourceClaim, pod *v1.Pod) bool {
	if claimIsUsedByContainers(podClaim, pod.Spec.InitContainers) {
		return true
	}
	if claimIsUsedByContainers(podClaim, pod.Spec.Containers) {
		return true
	}
	return false
}

func claimIsUsedByContainers(podClaim *v1.PodResourceClaim, containers []v1.Container) bool {
	for i := range containers {
		if claimIsUsedByContainer(podClaim, &containers[i]) {
			return true
		}
	}
	return false
}

func claimIsUsedByContainer(podClaim *v1.PodResourceClaim, container *v1.Container) bool {
	for _, c := range container.Resources.Claims {
		if c.Name == podClaim.Name {
			return true
		}
	}
	return false
}

// GetResources gets a ContainerInfo object from the claimInfo cache.
// This information is used by the caller to update a container config.
func (m *ManagerImpl) GetResources(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	annotations := []kubecontainer.Annotation{}
	cdiDevices := []kubecontainer.CDIDevice{}

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return nil, fmt.Errorf("list resource claims: %v", err)
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}

			claimInfo := m.cache.get(*claimName, pod.Namespace)
			if claimInfo == nil {
				return nil, fmt.Errorf("unable to get resource for namespace: %s, claim: %s", pod.Namespace, *claimName)
			}

			claimInfo.RLock()
			claimAnnotations := claimInfo.annotationsAsList()
			klog.V(3).InfoS("Add resource annotations", "claim", *claimName, "annotations", claimAnnotations)
			annotations = append(annotations, claimAnnotations...)
			for _, devices := range claimInfo.CDIDevices {
				for _, device := range devices {
					cdiDevices = append(cdiDevices, kubecontainer.CDIDevice{Name: device})
				}
			}
			claimInfo.RUnlock()
		}
	}

	return &ContainerInfo{Annotations: annotations, CDIDevices: cdiDevices}, nil
}

// UnprepareResources calls a plugin's NodeUnprepareResource API for each resource claim owned by a pod.
// This function is idempotent and may be called multiple times against the same pod.
// As such, calls to the underlying NodeUnprepareResource API are skipped for claims that have
// already been successfully unprepared.
func (m *ManagerImpl) UnprepareResources(pod *v1.Pod) error {
	batches := make(map[string][]*drapb.Claim)
	claimInfos := make(map[types.UID]*ClaimInfo)
	for i := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return fmt.Errorf("unprepare resource claim: %v", err)
		}

		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}

		claimInfo := m.cache.get(*claimName, pod.Namespace)

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

		// Loop through all plugins and prepare for calling NodeUnprepareResources.
		for _, resourceHandle := range claimInfo.ResourceHandles {
			// If no DriverName is provided in the resourceHandle, we
			// use the DriverName from the status
			pluginName := resourceHandle.DriverName
			if pluginName == "" {
				pluginName = claimInfo.DriverName
			}

			claim := &drapb.Claim{
				Namespace:      claimInfo.Namespace,
				Uid:            string(claimInfo.ClaimUID),
				Name:           claimInfo.ClaimName,
				ResourceHandle: resourceHandle.Data,
			}
			batches[pluginName] = append(batches[pluginName], claim)
		}
		claimInfos[claimInfo.ClaimUID] = claimInfo
	}

	// Call NodeUnprepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for pluginName, claims := range batches {
		// Call NodeUnprepareResources RPC for all resource handles.
		client, err := dra.NewDRAPluginClient(pluginName)
		if err != nil {
			return fmt.Errorf("failed to get DRA Plugin client for plugin name %s: %v", pluginName, err)
		}
		response, err := client.NodeUnprepareResources(context.Background(), &drapb.NodeUnprepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodeUnprepareResources failed: %v", err)
		}

		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodeUnprepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.Error != "" {
				return fmt.Errorf("NodeUnprepareResources failed for claim %s/%s: %s", reqClaim.Namespace, reqClaim.Name, err)
			}

			// Delete last pod UID only if unprepare succeeds.
			// This ensures that the status manager doesn't enter termination status
			// for the pod. This logic is implemented in
			// m.PodMightNeedToUnprepareResources and claimInfo.hasPodReference.
			claimInfo := claimInfos[types.UID(claimUID)]
			claimInfo.deletePodReference(pod.UID)
			m.cache.delete(claimInfo.ClaimName, pod.Namespace)
		}

		// Checkpoint to reduce redundant calls to NodeUnprepareResources after a kubelet restart.
		err = m.cache.syncToCheckpoint()
		if err != nil {
			return fmt.Errorf("failed to checkpoint claimInfo state, err: %+v", err)
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodeUnprepareResources left out %d claims", unfinished)
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
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return nil, fmt.Errorf("determine resource claim information: %v", err)
		}

		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}
			claimInfo := m.cache.get(*claimName, pod.Namespace)
			if claimInfo == nil {
				return nil, fmt.Errorf("unable to get resource for namespace: %s, claim: %s", pod.Namespace, *claimName)
			}
			claimInfos = append(claimInfos, claimInfo)
		}
	}
	return claimInfos, nil
}
