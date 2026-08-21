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

type PodResourceCheckpointInfo struct {
	Entries PodResourceInfoMap `json:"entries,omitempty"`
}

// migrateV1ToV2 migrates the legacy JSON V1 checkpoint data to the V2 PodList format.
// This results in an incomplete podList as the full podSpec is not available to us
// in the V1 checkpoint, but will self-heal as the pod is synced:
//   - When AddPod is called for a particular pod, the checkpoint has sufficient information to
//     correctly perform UpdatePodFromAllocation to allow the pod to be re-admitted based on its
//     previously allocated resources. AddPod then calls out to SetAllocatedResources which would
//     trigger the entire PodSpec to be written to the allocation checkpoint. The test
//     TestAllocationManager_Upgrade_CheckpointMigration in allocation_manager_test.go
//     verifies this behavior.
//   - During the first syncPod for a particular pod, InitializeActuatedPod fills out the remaining
//     for the actuated checkpoint. The test TestInitializeActuatedPod_HydrateMigratedState in
//     kuberuntime_manager_test.go verifies this behavior.
func migrateV1ToV2(data string) (*v1.PodList, error) {
	var checkpointData PodResourceCheckpointInfo
	if err := json.Unmarshal([]byte(data), &checkpointData); err != nil {
		return nil, err
	}

	podList := &v1.PodList{
		Items: make([]v1.Pod, 0, len(checkpointData.Entries)),
	}
	for podUID, entry := range checkpointData.Entries {
		pod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: podUID,
			},
		}
		// The legacy V1 checkpoint format stored all container resource allocations in a single map without
		// distinguishing between init containers and regular application containers. During migration, we append
		// all entries to pod.Spec.Containers.
		// On startup, UpdatePodFromCheckpoint searches across both InitContainers and Containers matching
		// by container name, ensuring init container allocations are correctly restored. Once the pod is
		// admitted, AddPod calls SetAllocatedResources which writes the complete, properly partitioned PodSpec
		// to the checkpoint, self-healing the state.
		for containerName, resources := range entry.ContainerResources {
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name:      containerName,
				Resources: *resources.DeepCopy(),
			})
		}
		if entry.PodLevelResources != nil {
			pod.Spec.Resources = entry.PodLevelResources.DeepCopy()
		}
		for volName, limit := range entry.EmptyDirVolumeLimits {
			var limitCopy *resource.Quantity
			if limit != nil {
				lc := limit.DeepCopy()
				limitCopy = &lc
			}
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: volName,
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{
						// Only memory-backed emptyDir volumes were supported
						// in the V1 checkpoint format.
						Medium:    v1.StorageMediumMemory,
						SizeLimit: limitCopy,
					},
				},
			})
		}
		podList.Items = append(podList.Items, pod)
	}
	return podList, nil
}
