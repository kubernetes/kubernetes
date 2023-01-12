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
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	dra "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ManagerImpl is the structure in charge of managing DRA resource Plugins.
type ManagerImpl struct {
	// cache contains cached claim info
	cache *claimInfoCache

	// KubeClient reference
	kubeClient clientset.Interface
}

// NewManagerImpl creates a new manager.
func NewManagerImpl(kubeClient clientset.Interface) (*ManagerImpl, error) {
	klog.V(2).InfoS("Creating DRA manager")

	manager := &ManagerImpl{
		cache:      newClaimInfoCache(),
		kubeClient: kubeClient,
	}

	return manager, nil
}

// Generate container annotations using CDI UpdateAnnotations API.
func generateCDIAnnotations(
	claimUID types.UID,
	driverName string,
	cdiDevices []string,
) ([]kubecontainer.Annotation, error) {
	annotations, err := updateAnnotations(map[string]string{}, driverName, string(claimUID), cdiDevices)
	if err != nil {
		return nil, fmt.Errorf("can't generate CDI annotations: %+v", err)
	}

	kubeAnnotations := []kubecontainer.Annotation{}
	for key, value := range annotations {
		kubeAnnotations = append(kubeAnnotations, kubecontainer.Annotation{Name: key, Value: value})
	}

	return kubeAnnotations, nil
}

// prepareContainerResources attempts to prepare all of required resource
// plugin resources for the input container, issue an NodePrepareResource rpc request
// for each new resource requirement, process their responses and update the cached
// containerResources on success.
func (m *ManagerImpl) prepareContainerResources(pod *v1.Pod, container *v1.Container) error {
	// Process resources for each resource claim referenced by container
	for range container.Resources.Claims {
		for i, podResourceClaim := range pod.Spec.ResourceClaims {
			claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
			klog.V(3).InfoS("Processing resource", "claim", claimName, "pod", pod.Name)

			if claimInfo := m.cache.get(claimName, pod.Namespace); claimInfo != nil {
				// resource is already prepared, add pod UID to it
				claimInfo.addPodReference(pod.UID)

				continue
			}

			// Query claim object from the API server
			resourceClaim, err := m.kubeClient.ResourceV1alpha1().ResourceClaims(pod.Namespace).Get(
				context.TODO(),
				claimName,
				metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %+v", claimName, pod.Name, err)
			}

			// Check if pod is in the ReservedFor for the claim
			if !resourceclaim.IsReservedForPod(pod, resourceClaim) {
				return fmt.Errorf("pod %s(%s) is not allowed to use resource claim %s(%s)",
					pod.Name, pod.UID, podResourceClaim.Name, resourceClaim.UID)
			}

			// Call NodePrepareResource RPC
			driverName := resourceClaim.Status.DriverName

			client, err := dra.NewDRAPluginClient(driverName)
			if err != nil {
				return fmt.Errorf("failed to get DRA Plugin client for plugin name %s, err=%+v", driverName, err)
			}

			response, err := client.NodePrepareResource(
				context.Background(),
				resourceClaim.Namespace,
				resourceClaim.UID,
				resourceClaim.Name,
				resourceClaim.Status.Allocation.ResourceHandle)
			if err != nil {
				return fmt.Errorf("NodePrepareResource failed, claim UID: %s, claim name: %s, resource handle: %s, err: %+v",
					resourceClaim.UID, resourceClaim.Name, resourceClaim.Status.Allocation.ResourceHandle, err)
			}

			klog.V(3).InfoS("NodePrepareResource succeeded", "response", response)

			annotations, err := generateCDIAnnotations(resourceClaim.UID, driverName, response.CdiDevices)
			if err != nil {
				return fmt.Errorf("failed to generate container annotations, err: %+v", err)
			}

			// Cache prepared resource
			err = m.cache.add(
				resourceClaim.Name,
				resourceClaim.Namespace,
				&claimInfo{
					driverName:  driverName,
					claimUID:    resourceClaim.UID,
					claimName:   resourceClaim.Name,
					namespace:   resourceClaim.Namespace,
					podUIDs:     sets.New(string(pod.UID)),
					cdiDevices:  response.CdiDevices,
					annotations: annotations,
				})
			if err != nil {
				return fmt.Errorf(
					"failed to cache prepared resource, claim: %s(%s), err: %+v",
					resourceClaim.Name,
					resourceClaim.UID,
					err,
				)
			}
		}
	}

	return nil
}

// getContainerInfo gets a container info from the claimInfo cache.
// This information is used by the caller to update a container config.
func (m *ManagerImpl) getContainerInfo(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	annotations := []kubecontainer.Annotation{}

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

			klog.V(3).InfoS("add resource annotations", "claim", claimName, "annotations", claimInfo.annotations)
			annotations = append(annotations, claimInfo.annotations...)
		}
	}

	return &ContainerInfo{Annotations: annotations}, nil
}

// PrepareResources calls plugin NodePrepareResource from the registered DRA resource plugins.
func (m *ManagerImpl) PrepareResources(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	if err := m.prepareContainerResources(pod, container); err != nil {
		return nil, err
	}

	return m.getContainerInfo(pod, container)
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
		if len(claimInfo.podUIDs) > 1 {
			claimInfo.deletePodReference(pod.UID)
			continue
		}

		// Call NodeUnprepareResource only for the last pod that references the claim
		client, err := dra.NewDRAPluginClient(claimInfo.driverName)
		if err != nil {
			return fmt.Errorf("failed to get DRA Plugin client for plugin name %s, err=%+v", claimInfo.driverName, err)
		}

		response, err := client.NodeUnprepareResource(
			context.Background(),
			claimInfo.namespace,
			claimInfo.claimUID,
			claimInfo.claimName,
			claimInfo.cdiDevices)
		if err != nil {
			return fmt.Errorf(
				"NodeUnprepareResource failed, pod: %s, claim UID: %s, claim name: %s, CDI devices: %s, err: %+v",
				pod.Name,
				claimInfo.claimUID,
				claimInfo.claimName,
				claimInfo.cdiDevices, err)
		}

		// Delete last pod UID only if NodeUnprepareResource call succeeds.
		// This ensures that status manager doesn't enter termination status
		// for the pod. This logic is implemented in the m.PodMightNeedToUnprepareResources
		// and in the claimInfo.hasPodReference.
		claimInfo.deletePodReference(pod.UID)

		klog.V(3).InfoS("NodeUnprepareResource succeeded", "response", response)
		// delete resource from the cache
		m.cache.delete(claimInfo.claimName, pod.Namespace)
	}

	return nil
}

// PodMightNeedToUnprepareResources returns true if the pod might need to
// unprepare resources
func (m *ManagerImpl) PodMightNeedToUnprepareResources(UID types.UID) bool {
	return m.cache.hasPodReference(UID)
}
