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

package state

import (
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
)

// UpdatePodFromCheckpoint overlays mutable fields that are gated on node allocation (e.g.
// container resources, pod-level resources, and memory-backed emptyDir volume size limits)
// from the checkpointed pod onto the incoming pod.
//
// Fields in PodSpec fall into three categories:
//  1. Immutable fields (e.g. NodeName, Volumes list, Container names): Identical between pod and storedPod.
//  2. Allocatable fields (e.g. Container resources, pod-level resources, emptyDir limits): Mutable, but
//     gated on node allocation admission. These must reflect the node's committed allocation (storedPod)
//     until newly requested specs are admitted and checkpointed.
//  3. Allocation-bypass fields (e.g. Container images, tolerations, ActiveDeadlineSeconds): Mutable
//     without node allocation admission. These reflect the latest incoming spec (pod).
//
// Starting from the incoming pod and selectively overlaying Category 2 allocatable fields from
// storedPod ensures that Category 3 mutations are preserved, forward compatibility is maintained
// for new PodSpec fields, and legacy V1 checkpoints (which only recorded Category 2 fields) can
// cleanly hydrate into full PodSpecs without dropping non-resource metadata.
//
// This function returns a deep copy of the pod only if updates were applied.
func UpdatePodFromCheckpoint(pod *v1.Pod, storedPod *v1.Pod) (*v1.Pod, bool) {
	if pod == nil || storedPod == nil {
		return pod, false
	}

	updated := false
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodLevelResourcesVerticalScaling) {
		pod, updated = updatePodLevelResourcesFromCheckpoint(pod, storedPod)
	}
	pod, updated = updateContainerResourcesFromCheckpoint(pod, storedPod, updated)
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		pod, updated = updateEmptyDirVolumeLimitsFromCheckpoint(pod, storedPod, updated)
	}

	return pod, updated
}

func updateContainerResourcesFromCheckpoint(pod *v1.Pod, storedPod *v1.Pod, alreadyUpdated bool) (*v1.Pod, bool) {
	updated := alreadyUpdated
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		if storedPod == nil {
			return v1.ResourceRequirements{}, false
		}
		for sc := range podutil.ContainerIter(&storedPod.Spec, podutil.InitContainers|podutil.Containers) {
			if sc.Name == c.Name {
				if !apiequality.Semantic.DeepEqual(c.Resources, sc.Resources) {
					// Stored state differs from pod spec, retrieve the stored resources
					if !updated {
						// If this is the first update to be performed, copy the pod
						pod = pod.DeepCopy()
						updated = true
					}
					return sc.Resources, true
				}
			}
		}
		return v1.ResourceRequirements{}, false
	}

	for i, c := range pod.Spec.Containers {
		if cAlloc, updated := containerAlloc(c); updated {
			// Stored state differs from pod spec, update
			pod.Spec.Containers[i].Resources = cAlloc
		}
	}
	for i, c := range pod.Spec.InitContainers {
		if cAlloc, updated := containerAlloc(c); updated {
			// Stored state differs from pod spec, update
			pod.Spec.InitContainers[i].Resources = cAlloc
		}
	}
	return pod, updated
}

func updatePodLevelResourcesFromCheckpoint(pod *v1.Pod, storedPod *v1.Pod) (*v1.Pod, bool) {
	pAlloc := storedPod.Spec.Resources
	if pAlloc == nil {
		return pod, false
	}
	if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, pAlloc) {
		// Stored state differs from pod spec, retrieve the stored resources
		pod = pod.DeepCopy()
		pod.Spec.Resources = pAlloc.DeepCopy()
		return pod, true
	}
	return pod, false
}

func updateEmptyDirVolumeLimitsFromCheckpoint(pod *v1.Pod, storedPod *v1.Pod, alreadyUpdated bool) (*v1.Pod, bool) {
	updated := alreadyUpdated
	for i, vol := range pod.Spec.Volumes {
		if !volHasMemoryBackedEmptyDirSizeLimit(&vol) {
			continue
		}

		var alloc *resource.Quantity
		for _, sv := range storedPod.Spec.Volumes {
			if sv.Name == vol.Name && sv.EmptyDir != nil {
				alloc = sv.EmptyDir.SizeLimit
				break
			}
		}

		if alloc != nil && alloc.Cmp(*vol.EmptyDir.SizeLimit) != 0 {
			if !updated {
				pod = pod.DeepCopy()
				updated = true
			}
			allocCopy := alloc.DeepCopy()
			pod.Spec.Volumes[i].EmptyDir.SizeLimit = &allocCopy
		}
	}
	return pod, updated
}

func volHasMemoryBackedEmptyDirSizeLimit(vol *v1.Volume) bool {
	return vol != nil && vol.EmptyDir != nil && vol.EmptyDir.Medium == v1.StorageMediumMemory && vol.EmptyDir.SizeLimit != nil
}
