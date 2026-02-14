//go:build !windows

package userns

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestUserNsManagerFlexibleRecording(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	// Configure manager with a default length of 65536
	userNsLength := uint32(65536)
	testUserNsPodsManager := &testUserNsPodsManager{
		userNsLength:   userNsLength,
		mappingFirstID: 65536,
		mappingLen:     65536 * 10, // 10 blocks
		maxPods:        10,
	}
	idsPerPod := int64(userNsLength)
	m, err := MakeUserNsManager(logger, testUserNsPodsManager, &idsPerPod)
	require.NoError(t, err)

	// 1. Record a pod with a different length (e.g. half block)
	err = m.record(logger, "small-pod", 65536, 32768)
	require.NoError(t, err)
	assert.True(t, m.isSet(65536), "block should be marked as used")

	// 2. Record another pod in the same block (this should now work as it's from "disk" simulation)
	err = m.record(logger, "another-small-pod", 65536+32768, 32768)
	require.NoError(t, err)
	assert.True(t, m.isSet(65536), "block should still be marked as used")

	// 3. Release one pod, block should remain used
	m.Release(logger, "small-pod")
	assert.True(t, m.isSet(65536), "block should still be used by another-small-pod")

	// 4. Release second pod, block should be freed
	m.Release(logger, "another-small-pod")
	assert.False(t, m.isSet(65536), "block should be freed")

	// 5. Record a pod that spans across two blocks
	// Block 0: 65536 to 131071
	// Block 1: 131072 to 196607
	err = m.record(logger, "spanning-pod", 65536+32768, 65536)
	require.NoError(t, err)
	assert.True(t, m.isSet(65536), "block 0 should be used")
	assert.True(t, m.isSet(131072), "block 1 should be used")

	// 6. Release spanning pod
	m.Release(logger, "spanning-pod")
	assert.False(t, m.isSet(65536), "block 0 should be freed")
	assert.False(t, m.isSet(131072), "block 1 should be freed")

	// 7. Record a pod completely out of range (simulating config change)
	// We expect this to record in usedBy but NOT in bitmap, and NOT fail.
	err = m.record(logger, "out-of-range-pod", 1000000, 65536)
	require.NoError(t, err)
	assert.True(t, m.podAllocated("out-of-range-pod"))

	// Ensure it didn't mess up the bitmap
	for i := 0; i < 10; i++ {
		assert.False(t, m.used.Has(i), "bitmap should be empty")
	}

	m.Release(logger, "out-of-range-pod")
	assert.False(t, m.podAllocated("out-of-range-pod"))
}
