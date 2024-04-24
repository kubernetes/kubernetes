/*
Copyright 2024 The Kubernetes Authors.

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

package userns

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
)

func TestMakeUserNsManagerSwitch(t *testing.T) {
	// Create the manager with the feature gate enabled, to record some pods on disk.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	pods := []types.UID{"pod-1", "pod-2"}

	testUserNsPodsManager := &testUserNsPodsManager{
		podDir: t.TempDir(),
		// List the same pods we will record, so the second time we create the userns
		// manager, it will find these pods on disk with userns data.
		podList: pods,
	}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// Record the pods on disk.
	for _, podUID := range pods {
		pod := v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}}
		_, err := m.GetOrCreateUserNamespaceMappings(&pod, "")
		require.NoError(t, err, "failed to record userns range for pod %v", podUID)
	}

	// Test re-init works when the feature gate is disabled and there were some
	// pods written on disk.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)
	m2, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// The feature gate is off, no pods should be allocated.
	for _, pod := range pods {
		ok := m2.podAllocated(pod)
		assert.False(t, ok, "pod %q should not be allocated", pod)
	}
}

func TestGetOrCreateUserNamespaceMappingsSwitch(t *testing.T) {
	// Enable the feature gate to create some pods on disk.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	pods := []types.UID{"pod-1", "pod-2"}

	testUserNsPodsManager := &testUserNsPodsManager{
		podDir: t.TempDir(),
		// List the same pods we will record, so the second time we create the userns
		// manager, it will find these pods on disk with userns data.
		podList: pods,
	}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// Record the pods on disk.
	for _, podUID := range pods {
		pod := v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}}
		_, err := m.GetOrCreateUserNamespaceMappings(&pod, "")
		require.NoError(t, err, "failed to record userns range for pod %v", podUID)
	}

	// Test no-op when the feature gate is disabled and there were some
	// pods registered on disk.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)
	// Create a new manager with the feature gate off and verify the userns range is nil.
	m2, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	for _, podUID := range pods {
		pod := v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}}
		userns, err := m2.GetOrCreateUserNamespaceMappings(&pod, "")

		assert.NoError(t, err, "failed to record userns range for pod %v", podUID)
		assert.Nil(t, userns, "userns range should be nil for pod %v", podUID)
	}
}

func TestCleanupOrphanedPodUsernsAllocationsSwitch(t *testing.T) {
	// Enable the feature gate to create some pods on disk.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	listPods := []types.UID{"pod-1", "pod-2"}
	pods := []types.UID{"pod-3", "pod-4"}
	testUserNsPodsManager := &testUserNsPodsManager{
		podDir:  t.TempDir(),
		podList: listPods,
	}

	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// Record the pods on disk.
	for _, podUID := range pods {
		pod := v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}}
		_, err := m.GetOrCreateUserNamespaceMappings(&pod, "")
		require.NoError(t, err, "failed to record userns range for pod %v", podUID)
	}

	// Test cleanup works when the feature gate is disabled and there were some
	// pods registered.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)
	err = m.CleanupOrphanedPodUsernsAllocations(nil, nil)
	require.NoError(t, err)

	// The feature gate is off, no pods should be allocated.
	for _, pod := range append(listPods, pods...) {
		ok := m.podAllocated(pod)
		assert.False(t, ok, "pod %q should not be allocated", pod)
	}
}
