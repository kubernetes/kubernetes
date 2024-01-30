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

package userns

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type testUserNsPodsManager struct {
	podDir  string
	podList []types.UID
}

func (m *testUserNsPodsManager) GetPodDir(podUID types.UID) string {
	if m.podDir == "" {
		return "/tmp/non-existant-dir.This-is-not-used-in-tests"
	}
	return m.podDir
}

func (m *testUserNsPodsManager) ListPodsFromDisk() ([]types.UID, error) {
	if len(m.podList) == 0 {
		return nil, nil
	}
	return m.podList, nil
}

func TestUserNsManagerAllocate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	assert.Equal(t, true, m.isSet(0*65536), "m.isSet(0) should be true")

	allocated, length, err := m.allocateOne("one")
	assert.NoError(t, err)
	assert.Equal(t, userNsLength, int(length), "m.isSet(%d).length=%v", allocated, length)
	assert.Equal(t, true, m.isSet(allocated), "m.isSet(%d)", allocated)

	allocated2, length2, err := m.allocateOne("two")
	assert.NoError(t, err)
	assert.NotEqual(t, allocated, allocated2, "allocated != allocated2")
	assert.Equal(t, length, length2, "length == length2")

	// verify that re-adding the same pod with the same settings won't fail
	err = m.record("two", allocated2, length2)
	assert.NoError(t, err)
	// but it fails if anyting is different
	err = m.record("two", allocated2+1, length2)
	assert.Error(t, err)

	m.Release("one")
	m.Release("two")
	assert.Equal(t, false, m.isSet(allocated), "m.isSet(%d)", allocated)
	assert.Equal(t, false, m.isSet(allocated2), "m.nsSet(%d)", allocated2)

	var allocs []uint32
	for i := 0; i < 1000; i++ {
		allocated, length, err = m.allocateOne(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, userNsLength, int(length), "length is not the expected. iter: %v", i)
		assert.NoError(t, err)
		allocs = append(allocs, allocated)
	}
	for i, v := range allocs {
		assert.Equal(t, true, m.isSet(v), "m.isSet(%d) should be true", v)
		m.Release(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, false, m.isSet(v), "m.isSet(%d) should be false", v)

		err = m.record(types.UID(fmt.Sprintf("%d", i)), v, userNsLength)
		assert.NoError(t, err)
		m.Release(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, false, m.isSet(v), "m.isSet(%d) should be false", v)
	}
}

func TestUserNsManagerParseUserNsFile(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	cases := []struct {
		name    string
		file    string
		success bool
	}{
		{
			name: "basic",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: true,
		},
		{
			name: "invalid length",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":0 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":0 } ]
                               }`,
			success: false,
		},
		{
			name: "wrong offset",
			file: `{
	                        "uidMappings":[ {"hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ {"hostId":1, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "two GID mappings",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":userNsLength } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":userNsLength }, { "hostId":196608, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "two UID mappings",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 }, { "hostId":196608, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "no root UID",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":1, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "no root GID",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":1, "length":65536 } ]
                               }`,
			success: false,
		},
	}

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	assert.NoError(t, err)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// We don't validate the result. It was parsed with the json parser, we trust that.
			_, err = m.parseUserNsFileAndRecord(types.UID(tc.name), []byte(tc.file))
			if (tc.success && err == nil) || (!tc.success && err != nil) {
				return
			}

			t.Errorf("expected success: %v but got error: %v", tc.success, err)
		})
	}
}

func TestGetOrCreateUserNamespaceMappings(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	trueVal := true
	falseVal := false

	cases := []struct {
		name    string
		pod     *v1.Pod
		expMode runtimeapi.NamespaceMode
		success bool
	}{
		{
			name:    "no user namespace",
			pod:     &v1.Pod{},
			expMode: runtimeapi.NamespaceMode_NODE,
			success: true,
		},
		{
			name: "opt-in to host user namespace",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &trueVal,
				},
			},
			expMode: runtimeapi.NamespaceMode_NODE,
			success: true,
		},
		{
			name: "user namespace",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &falseVal,
				},
			},
			expMode: runtimeapi.NamespaceMode_POD,
			success: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// These tests will create the userns file, so use an existing podDir.
			testUserNsPodsManager := &testUserNsPodsManager{podDir: t.TempDir()}
			m, err := MakeUserNsManager(testUserNsPodsManager)
			assert.NoError(t, err)

			userns, err := m.GetOrCreateUserNamespaceMappings(tc.pod)
			if (tc.success && err != nil) || (!tc.success && err == nil) {
				t.Errorf("expected success: %v but got error: %v", tc.success, err)
			}

			if userns.GetMode() != tc.expMode {
				t.Errorf("expected mode: %v but got: %v", tc.expMode, userns.GetMode())
			}
		})
	}
}

func TestCleanupOrphanedPodUsernsAllocations(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	cases := []struct {
		name                 string
		runningPods          []*kubecontainer.Pod
		pods                 []*v1.Pod
		listPods             []types.UID /* pods to list */
		podSetBeforeCleanup  []types.UID /* pods to record before cleanup */
		podSetAfterCleanup   []types.UID /* pods set expected after cleanup */
		podUnsetAfterCleanup []types.UID /* pods set expected after cleanup */
	}{
		{
			name:     "no stale pods",
			listPods: []types.UID{"pod-1", "pod-2"},
		},
		{
			name:                 "no stale pods set",
			podSetBeforeCleanup:  []types.UID{"pod-1", "pod-2"},
			listPods:             []types.UID{"pod-1", "pod-2"},
			podUnsetAfterCleanup: []types.UID{"pod-1", "pod-2"},
		},
		{
			name:                 "one running pod",
			listPods:             []types.UID{"pod-1", "pod-2"},
			podSetBeforeCleanup:  []types.UID{"pod-1", "pod-2"},
			runningPods:          []*kubecontainer.Pod{{ID: "pod-1"}},
			podSetAfterCleanup:   []types.UID{"pod-1"},
			podUnsetAfterCleanup: []types.UID{"pod-2"},
		},
		{
			name:                 "pod set before cleanup but not listed ==> unset",
			podSetBeforeCleanup:  []types.UID{"pod-1", "pod-2"},
			runningPods:          []*kubecontainer.Pod{{ID: "pod-1"}},
			podUnsetAfterCleanup: []types.UID{"pod-1", "pod-2"},
		},
		{
			name:                 "one pod",
			listPods:             []types.UID{"pod-1", "pod-2"},
			podSetBeforeCleanup:  []types.UID{"pod-1", "pod-2"},
			pods:                 []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{UID: "pod-1"}}},
			podSetAfterCleanup:   []types.UID{"pod-1"},
			podUnsetAfterCleanup: []types.UID{"pod-2"},
		},
		{
			name:                 "no listed pods ==> all unset",
			podSetBeforeCleanup:  []types.UID{"pod-1", "pod-2"},
			podUnsetAfterCleanup: []types.UID{"pod-1", "pod-2"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testUserNsPodsManager := &testUserNsPodsManager{
				podList: tc.listPods,
			}
			m, err := MakeUserNsManager(testUserNsPodsManager)
			require.NoError(t, err)

			// Record the userns range as used
			for i, pod := range tc.podSetBeforeCleanup {
				err := m.record(pod, uint32((i+1)*65536), 65536)
				require.NoError(t, err)
			}

			err = m.CleanupOrphanedPodUsernsAllocations(tc.pods, tc.runningPods)
			require.NoError(t, err)

			for _, pod := range tc.podSetAfterCleanup {
				ok := m.podAllocated(pod)
				assert.True(t, ok, "pod %q should be allocated", pod)
			}

			for _, pod := range tc.podUnsetAfterCleanup {
				ok := m.podAllocated(pod)
				assert.False(t, ok, "pod %q should not be allocated", pod)
			}
		})
	}
}

func TestAllocateMaxPods(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// The first maxPods allocations should succeed.
	for i := 0; i < maxPods; i++ {
		_, _, err = m.allocateOne(types.UID(fmt.Sprintf("%d", i)))
		require.NoError(t, err)
	}

	// The next allocation should fail, hitting maxPods.
	_, _, err = m.allocateOne(types.UID(fmt.Sprintf("%d", maxPods+1)))
	assert.Error(t, err)
}

func TestRecordMaxPods(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	// The first maxPods allocations should succeed.
	for i := 0; i < maxPods; i++ {
		err = m.record(types.UID(fmt.Sprintf("%d", i)), uint32((i+1)*65536), 65536)
		require.NoError(t, err)
	}

	// The next allocation should fail, hitting maxPods.
	err = m.record(types.UID(fmt.Sprintf("%d", maxPods+1)), uint32((maxPods+1)*65536), 65536)
	assert.Error(t, err)
}
