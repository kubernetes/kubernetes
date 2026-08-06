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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

func TestStateMemory_EmptyDirVolumeLimits(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodResourceInfoMap{})

	podUID := types.UID("pod-1")
	volName := "volume-1"

	// Get on nonexistent pod should return (nil, false)
	qty, exists := state.GetEmptyDirVolumeLimit(podUID, volName)
	assert.Nil(t, qty)
	assert.False(t, exists)

	// Set volume limit on nonexistent pod should implicitly initialize the pod and insert it
	targetLimit := resource.MustParse("256Mi")
	err := state.SetEmptyDirVolumeLimit(podUID, volName, &targetLimit)
	require.NoError(t, err)

	// Get volume limit should return the parsed value and true
	qty, exists = state.GetEmptyDirVolumeLimit(podUID, volName)
	require.True(t, exists)
	require.NotNil(t, qty)
	assert.True(t, targetLimit.Equal(*qty))

	// Get on existing pod with nonexistent volume should return (nil, false)
	qtyNonexistent, existsNonexistent := state.GetEmptyDirVolumeLimit(podUID, "nonexistent-volume")
	assert.Nil(t, qtyNonexistent)
	assert.False(t, existsNonexistent)

	// Returned quantity should be a deep copy (mutability check)
	qty.Set(1024 * 1024 * 512) // Modify the returned Quantity value to 512Mi
	refreshedQty, exists := state.GetEmptyDirVolumeLimit(podUID, volName)
	require.True(t, exists)
	assert.True(t, targetLimit.Equal(*refreshedQty), "Modifying the returned Quantity pointer should not alter Kubelet's internal memory state")

	// Set another volume on the same pod should keep existing limits intact
	anotherVolName := "volume-2"
	anotherLimit := resource.MustParse("128Mi")
	err = state.SetEmptyDirVolumeLimit(podUID, anotherVolName, &anotherLimit)
	require.NoError(t, err)

	// Verify both volumes are present and correct
	qty1, exists1 := state.GetEmptyDirVolumeLimit(podUID, volName)
	assert.True(t, exists1)
	assert.True(t, targetLimit.Equal(*qty1))

	qty2, exists2 := state.GetEmptyDirVolumeLimit(podUID, anotherVolName)
	assert.True(t, exists2)
	assert.True(t, anotherLimit.Equal(*qty2))
}

func TestStateMemory_ResourceIsolation(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodResourceInfoMap{})

	podUID := types.UID("pod-1")
	containerName := "container-1"
	volumeName := "volume-1"

	// Set container resources
	containerResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("250m"),
		},
	}
	err := state.SetContainerResources(logger, podUID, containerName, containerResources)
	require.NoError(t, err)

	// Verify container resources are set, and others are nil/empty
	res, found := state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))

	podRes, found := state.GetPodLevelResources(podUID)
	assert.False(t, found)
	assert.Nil(t, podRes)

	volLimit, found := state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.False(t, found)
	assert.Nil(t, volLimit)

	// Set pod-level resources
	podResources := &v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("512Mi"),
		},
	}
	err = state.SetPodLevelResources(logger, podUID, podResources)
	require.NoError(t, err)

	// Verify pod-level resources are set, AND container resources are still intact
	podRes, found = state.GetPodLevelResources(podUID)
	assert.True(t, found)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))

	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))

	volLimit, found = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.False(t, found)
	assert.Nil(t, volLimit)

	// Set emptyDir volume limit
	targetLimit := resource.MustParse("256Mi")
	err = state.SetEmptyDirVolumeLimit(podUID, volumeName, &targetLimit)
	require.NoError(t, err)

	// Verify volume limit is set, AND both container and pod-level resources are still intact
	volLimit, found = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.True(t, found)
	assert.True(t, targetLimit.Equal(*volLimit))

	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))

	podRes, found = state.GetPodLevelResources(podUID)
	assert.True(t, found)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))

	// Update container resources again
	updatedContainerResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU: resource.MustParse("500m"),
		},
	}
	err = state.SetContainerResources(logger, podUID, containerName, updatedContainerResources)
	require.NoError(t, err)

	// Verify container resources are updated, AND both pod-level resources and volume limits are still intact
	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, updatedContainerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))

	podRes, found = state.GetPodLevelResources(podUID)
	assert.True(t, found)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))

	volLimit, found = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.True(t, found)
	assert.True(t, targetLimit.Equal(*volLimit))
}
