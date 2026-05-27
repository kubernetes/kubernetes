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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

func TestStateMemory_EmptyDirVolumeLimits(t *testing.T) {
	state := NewStateMemory(nil)

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
