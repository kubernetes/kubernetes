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
	"encoding/json"
	"slices"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// PodResourceInfo stores resource requirements for containers within a pod.
type PodResourceInfo struct {
	// ContainerResources maps container names to their respective ResourceRequirements.
	ContainerResources map[string]v1.ResourceRequirements

	// PodLevelResources represents resource requirements that apply to the entire pod, if any.
	PodLevelResources *v1.ResourceRequirements

	// EmptyDirVolumeLimits maps emptyDir volume names to their respective resource limits, if any.
	EmptyDirVolumeLimits map[string]*resource.Quantity
}

// PodResourceInfoMap maps pod UIDs to their corresponding PodResourceInfo,
// tracking resource requirements for all containers within each pod.
type PodResourceInfoMap map[types.UID]PodResourceInfo

// PodResourceCheckpointInfo is the V1 checkpoint payload, stored as JSON in Checkpoint.Data.
type PodResourceCheckpointInfo struct {
	Entries PodResourceInfoMap `json:"entries,omitempty"`
}

// Clone returns a copy of PodResourceInfoMap
func (pr PodResourceInfoMap) Clone() PodResourceInfoMap {
	prCopy := make(PodResourceInfoMap)
	for podUID, podInfo := range pr {
		newPodInfo := PodResourceInfo{
			ContainerResources: make(map[string]v1.ResourceRequirements),
			PodLevelResources:  podInfo.PodLevelResources.DeepCopy(),
		}
		for containerName, containerInfo := range podInfo.ContainerResources {
			newPodInfo.ContainerResources[containerName] = *containerInfo.DeepCopy()
		}
		if podInfo.EmptyDirVolumeLimits != nil {
			newPodInfo.EmptyDirVolumeLimits = make(map[string]*resource.Quantity)
			for volumeName, volumeLimit := range podInfo.EmptyDirVolumeLimits {
				if volumeLimit == nil {
					newPodInfo.EmptyDirVolumeLimits[volumeName] = nil
				} else {
					vl := volumeLimit.DeepCopy()
					newPodInfo.EmptyDirVolumeLimits[volumeName] = &vl
				}
			}
		}
		prCopy[podUID] = newPodInfo
	}
	return prCopy
}

// migrateV1ToV2 converts the JSON payload of a V1 checkpoint into a PodList.
//
// V1 did not record container types, so every container ends up in Spec.Containers, and nothing but
// resources is known about the pods (no images, no volume medium, ...). That is all the actuated
// state ever keeps, and the allocated state is completed once the pod is admitted again and its
// full spec is stored.
//
// Pods, containers and volumes are emitted sorted so that the result, and any checkpoint written
// from it, does not depend on map iteration order.
func migrateV1ToV2(data string) (*v1.PodList, error) {
	var checkpointData PodResourceCheckpointInfo
	if err := json.Unmarshal([]byte(data), &checkpointData); err != nil {
		return nil, err
	}

	uids := make([]types.UID, 0, len(checkpointData.Entries))
	for uid := range checkpointData.Entries {
		uids = append(uids, uid)
	}
	slices.Sort(uids)

	podList := &v1.PodList{Items: make([]v1.Pod, 0, len(uids))}
	for _, uid := range uids {
		entry := checkpointData.Entries[uid]
		pod := v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: uid}}

		containerNames := make([]string, 0, len(entry.ContainerResources))
		for name := range entry.ContainerResources {
			containerNames = append(containerNames, name)
		}
		slices.Sort(containerNames)
		for _, name := range containerNames {
			resources := entry.ContainerResources[name]
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name:      name,
				Resources: *resources.DeepCopy(),
			})
		}

		if entry.PodLevelResources != nil {
			pod.Spec.Resources = entry.PodLevelResources.DeepCopy()
		}

		volumeNames := make([]string, 0, len(entry.EmptyDirVolumeLimits))
		for name := range entry.EmptyDirVolumeLimits {
			volumeNames = append(volumeNames, name)
		}
		slices.Sort(volumeNames)
		for _, name := range volumeNames {
			var limit *resource.Quantity
			if l := entry.EmptyDirVolumeLimits[name]; l != nil {
				lc := l.DeepCopy()
				limit = &lc
			}
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: name,
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: limit},
				},
			})
		}

		podList.Items = append(podList.Items, pod)
	}
	return podList, nil
}

// podListToV1Entries is the inverse of migrateV1ToV2: it derives the V1 representation of the pods
// so that a kubelet that predates V2 can read the checkpoint after a rollback.
//
// The result is a superset of what V1 recorded (V1 skipped, for example, non-resizable containers),
// which V1 readers tolerate because they only look entries up by name. Volumes without a size limit
// are left out, since V1 readers dereference the stored limit.
func podListToV1Entries(podList *v1.PodList) PodResourceInfoMap {
	entries := make(PodResourceInfoMap, len(podList.Items))
	for _, pod := range podList.Items {
		info := PodResourceInfo{
			ContainerResources: make(map[string]v1.ResourceRequirements),
		}
		for _, c := range pod.Spec.InitContainers {
			info.ContainerResources[c.Name] = *c.Resources.DeepCopy()
		}
		for _, c := range pod.Spec.Containers {
			info.ContainerResources[c.Name] = *c.Resources.DeepCopy()
		}
		for _, c := range pod.Spec.EphemeralContainers {
			info.ContainerResources[c.Name] = *c.Resources.DeepCopy()
		}
		if pod.Spec.Resources != nil {
			info.PodLevelResources = pod.Spec.Resources.DeepCopy()
		}
		for _, vol := range pod.Spec.Volumes {
			if vol.EmptyDir == nil || vol.EmptyDir.SizeLimit == nil {
				continue
			}
			if info.EmptyDirVolumeLimits == nil {
				info.EmptyDirVolumeLimits = make(map[string]*resource.Quantity)
			}
			limit := vol.EmptyDir.SizeLimit.DeepCopy()
			info.EmptyDirVolumeLimits[vol.Name] = &limit
		}
		entries[pod.UID] = info
	}
	return entries
}
