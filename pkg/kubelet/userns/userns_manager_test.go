//go:build !windows

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
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	testUserNsLength = uint32(65536)
	// skip the first block
	minimumMappingUID = testUserNsLength
	// allocate enough space for 2000 user namespaces
	mappingLen  = testUserNsLength * 2000
	testMaxPods = 110
)

type testUserNsPodsManager struct {
	podDir         string
	podList        []types.UID
	userns         bool
	maxPods        int
	mappingFirstID uint32
	mappingLen     uint32
	userNsLength   uint32
}

func (m *testUserNsPodsManager) GetPodDir(podUID types.UID) string {
	if m.podDir == "" {
		return "/tmp/non-existent-dir.This-is-not-used-in-tests"
	}
	return m.podDir
}

func (m *testUserNsPodsManager) ListPodsFromDisk() ([]types.UID, error) {
	if len(m.podList) == 0 {
		return nil, nil
	}
	return m.podList, nil
}

func (m *testUserNsPodsManager) HandlerSupportsUserNamespaces(runtimeHandler string) (bool, error) {
	if runtimeHandler == "error" {
		return false, errors.New("unknown runtime")
	}
	return m.userns, nil
}

func (m *testUserNsPodsManager) GetKubeletMappings(logger klog.Logger, idsPerPod uint32) (uint32, uint32, error) {
	if m.mappingFirstID != 0 {
		return m.mappingFirstID, m.mappingLen, nil
	}
	return minimumMappingUID, mappingLen, nil
}

func (m *testUserNsPodsManager) GetMaxPods() int {
	if m.maxPods != 0 {
		return m.maxPods
	}

	return testMaxPods
}

func TestUserNsManagerAllocate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	customUserNsLength := uint32(1048576)

	cases := []struct {
		name           string
		userNsLength   uint32
		mappingFirstID uint32
		mappingLen     uint32
	}{
		{
			name:           "default",
			userNsLength:   testUserNsLength,
			mappingFirstID: minimumMappingUID,
			mappingLen:     mappingLen,
		},
		{
			name:           "custom",
			userNsLength:   customUserNsLength,
			mappingFirstID: customUserNsLength,
			mappingLen:     customUserNsLength * 2000,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testUserNsPodsManager := &testUserNsPodsManager{
				userNsLength:   tc.userNsLength,
				mappingFirstID: tc.mappingFirstID,
				mappingLen:     tc.mappingLen,
			}
			idsPerPod := int64(tc.userNsLength)
			m, err := MakeUserNsManager(logger, testUserNsPodsManager, &idsPerPod)
			require.NoError(t, err)

			allocated, length, err := m.allocateOne(logger, "one")
			require.NoError(t, err)
			assert.Equal(t, tc.userNsLength, length, "m.isSet(%d).length=%v", allocated, length)
			assert.True(t, m.isSet(allocated), "m.isSet(%d)", allocated)

			allocated2, length2, err := m.allocateOne(logger, "two")
			require.NoError(t, err)
			assert.NotEqual(t, allocated, allocated2, "allocated != allocated2")
			assert.Equal(t, length, length2, "length == length2")

			// verify that re-adding the same pod with the same settings won't fail
			err = m.record(logger, "two", allocated2, length2)
			require.NoError(t, err)
			// but it fails if anyting is different
			err = m.record(logger, "two", allocated2+1, length2)
			require.Error(t, err)

			m.Release(logger, "one")
			m.Release(logger, "two")
			assert.False(t, m.isSet(allocated), "m.isSet(%d)", allocated)
			assert.False(t, m.isSet(allocated2), "m.nsSet(%d)", allocated2)

			var allocs []uint32
			for i := 0; i < 1000; i++ {
				allocated, length, err = m.allocateOne(logger, types.UID(fmt.Sprintf("%d", i)))
				assert.Equal(t, tc.userNsLength, length, "length is not the expected. iter: %v", i)
				require.NoError(t, err)
				assert.GreaterOrEqual(t, allocated, tc.mappingFirstID)
				// The last ID of the userns range (allocated+userNsLength) should be within bounds.
				assert.LessOrEqual(t, allocated, tc.mappingFirstID+tc.mappingLen-tc.userNsLength)
				allocs = append(allocs, allocated)
			}
			for i, v := range allocs {
				assert.True(t, m.isSet(v), "m.isSet(%d) should be true", v)
				m.Release(logger, types.UID(fmt.Sprintf("%d", i)))
				assert.False(t, m.isSet(v), "m.isSet(%d) should be false", v)

				err = m.record(logger, types.UID(fmt.Sprintf("%d", i)), v, tc.userNsLength)
				require.NoError(t, err)
				m.Release(logger, types.UID(fmt.Sprintf("%d", i)))
				assert.False(t, m.isSet(v), "m.isSet(%d) should be false", v)
			}
		})
	}
}

func TestMakeUserNsManager(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	cases := []struct {
		name           string
		mappingFirstID uint32
		mappingLen     uint32
		maxPods        int
		success        bool
	}{
		{
			name:    "default",
			success: true,
		},
		{
			name:           "firstID not multiple",
			mappingFirstID: 65536 + 1,
		},
		{
			name:           "firstID is less than 65535",
			mappingFirstID: 1,
		},
		{
			name:           "mappingLen not multiple",
			mappingFirstID: 65536,
			mappingLen:     65536 + 1,
		},
		{
			name:           "range can't fit maxPods",
			mappingFirstID: 65536,
			mappingLen:     65536,
			maxPods:        2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testUserNsPodsManager := &testUserNsPodsManager{
				podDir:         t.TempDir(),
				mappingFirstID: tc.mappingFirstID,
				mappingLen:     tc.mappingLen,
				maxPods:        tc.maxPods,
			}
			_, err := MakeUserNsManager(logger, testUserNsPodsManager, nil)

			if tc.success {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
			}
		})
	}
}

func TestUserNsManagerParseUserNsFile(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	// Built rather than hand written, so a fixture cannot quietly stop being JSON.
	mapping := func(hostID, containerID, length uint32) string {
		return fmt.Sprintf(`{"hostId":%d,"containerId":%d,"length":%d}`, hostID, containerID, length)
	}
	mappingsFile := func(uid, gid string) string {
		return fmt.Sprintf(`{"uidMappings":[%s],"gidMappings":[%s]}`, uid, gid)
	}

	// The second block of the pool, so the one after it is still in range.
	const firstID = minimumMappingUID + testUserNsLength
	produced := mapping(firstID, 0, testUserNsLength)
	nextRange := mapping(firstID+testUserNsLength, 0, testUserNsLength)

	cases := []struct {
		name    string
		file    string
		wantErr string        // empty means the file is accepted
		want    userNamespace // what an accepted file has to parse to
	}{{
		name: "one UID mapping and an identical GID mapping",
		file: mappingsFile(produced, produced),
		want: userNamespace{
			UIDMappings: []idMapping{{HostId: firstID, ContainerId: 0, Length: testUserNsLength}},
			GIDMappings: []idMapping{{HostId: firstID, ContainerId: 0, Length: testUserNsLength}},
		},
	}, {
		name:    "truncated file",
		file:    `{"uidMappings":`,
		wantErr: "invalid user namespace mappings file",
	}, {
		name:    "no UID mapping",
		file:    mappingsFile("", produced),
		wantErr: "no more than one mapping allowed",
	}, {
		name:    "two UID mappings",
		file:    mappingsFile(produced+","+nextRange, produced),
		wantErr: "no more than one mapping allowed",
	}, {
		name:    "no GID mapping",
		file:    mappingsFile(produced, ""),
		wantErr: "GID and UID mappings should be identical",
	}, {
		name:    "two GID mappings",
		file:    mappingsFile(produced, produced+","+nextRange),
		wantErr: "GID and UID mappings should be identical",
	}, {
		name:    "GID mapping at another host ID",
		file:    mappingsFile(produced, nextRange),
		wantErr: "GID and UID mapping should be identical",
	}, {
		name:    "GID mapping of another length",
		file:    mappingsFile(produced, mapping(firstID, 0, testUserNsLength*2)),
		wantErr: "GID and UID mapping should be identical",
	}, {
		name:    "container ID 0 not mapped",
		file:    mappingsFile(mapping(firstID, 1, testUserNsLength), mapping(firstID, 1, testUserNsLength)),
		wantErr: "UID 0 must be mapped",
	}, {
		name:    "zero mapping length",
		file:    mappingsFile(mapping(firstID, 0, 0), mapping(firstID, 0, 0)),
		wantErr: "wrong user namespace length",
	}, {
		// Same wrong length in both, so it reaches record() and not the identity check.
		name:    "nonzero length other than the configured IDs per pod",
		file:    mappingsFile(mapping(firstID, 0, testUserNsLength*2), mapping(firstID, 0, testUserNsLength*2)),
		wantErr: "wrong user namespace length",
	}, {
		name:    "host ID is not a multiple of the unit",
		file:    mappingsFile(mapping(firstID+1, 0, testUserNsLength), mapping(firstID+1, 0, testUserNsLength)),
		wantErr: "wrong user namespace offset",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			// One manager per case: a shared one lets an earlier case hold this range.
			m, err := MakeUserNsManager(logger, &testUserNsPodsManager{}, nil)
			require.NoError(t, err)

			podUID := types.UID(tc.name)
			freeBefore := m.used.Free()

			got, err := m.parseUserNsFileAndRecord(logger, podUID, []byte(tc.file))
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				// A rejected file must leave the allocator as it was.
				require.Empty(t, m.usedBy)
				require.Equal(t, freeBefore, m.used.Free())
				return
			}

			require.NoError(t, err)
			// GetOrCreateUserNamespaceMappings passes this value on to the runtime.
			require.Equal(t, tc.want, got)

			// The other half of the function: the range is reserved, and to this pod.
			require.Len(t, m.usedBy, 1)
			recorded, found := m.usedBy[podUID]
			require.True(t, found)
			require.Equal(t, tc.want.UIDMappings[0].HostId, recorded)
			require.True(t, m.isSet(recorded))
			require.Equal(t, freeBefore-1, m.used.Free())
		})
	}
}
func TestGetOrCreateUserNamespaceMappings(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	trueVal := true
	falseVal := false

	cases := []struct {
		name           string
		pod            *v1.Pod
		expMode        runtimeapi.NamespaceMode
		runtimeUserns  bool
		runtimeHandler string
		success        bool
	}{
		{
			name:    "no user namespace",
			pod:     &v1.Pod{},
			expMode: runtimeapi.NamespaceMode_NODE,
			success: true,
		},
		{
			name:    "nil pod",
			pod:     nil,
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
			expMode:       runtimeapi.NamespaceMode_POD,
			runtimeUserns: true,
			success:       true,
		},
		{
			name: "user namespace, but no runtime support",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &falseVal,
				},
			},
			runtimeUserns: false,
		},
		{
			name: "user namespace, but runtime returns error",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &falseVal,
				},
			},
			// This handler name makes the fake runtime return an error.
			runtimeHandler: "error",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			// These tests will create the userns file, so use an existing podDir.
			testUserNsPodsManager := &testUserNsPodsManager{
				podDir: t.TempDir(),
				userns: tc.runtimeUserns,
			}
			m, err := MakeUserNsManager(logger, testUserNsPodsManager, nil)
			assert.NoError(t, err)

			userns, err := m.GetOrCreateUserNamespaceMappings(logger, tc.pod, tc.runtimeHandler)
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
	logger, ctx := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

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
				podDir:  t.TempDir(),
				podList: tc.listPods,
			}
			m, err := MakeUserNsManager(logger, testUserNsPodsManager, nil)
			require.NoError(t, err)

			// Record the userns range as used
			for i, pod := range tc.podSetBeforeCleanup {
				err := m.record(logger, pod, uint32((i+1)*65536), 65536)
				require.NoError(t, err)
			}

			err = m.CleanupOrphanedPodUsernsAllocations(ctx, tc.pods, tc.runningPods)
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

type failingUserNsPodsManager struct {
	testUserNsPodsManager
}

func (m *failingUserNsPodsManager) ListPodsFromDisk() ([]types.UID, error) {
	return nil, os.ErrPermission
}

func TestMakeUserNsManagerFailsListPod(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	testUserNsPodsManager := &failingUserNsPodsManager{}
	_, err := MakeUserNsManager(logger, testUserNsPodsManager, nil)
	assert.Error(t, err)
	assert.ErrorContains(t, err, "read pods from disk")
}

func TestRecordBounds(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	// Allow exactly for 1 pod
	testUserNsPodsManager := &testUserNsPodsManager{
		mappingFirstID: 65536,
		mappingLen:     65536,
		maxPods:        1,
	}
	m, err := MakeUserNsManager(logger, testUserNsPodsManager, nil)
	require.NoError(t, err)

	// The first pod allocation should succeed.
	err = m.record(logger, types.UID(fmt.Sprintf("%d", 0)), 65536, 65536)
	require.NoError(t, err)

	// The next allocation should fail, as there is no space left.
	err = m.record(logger, types.UID(fmt.Sprintf("%d", 2)), uint32(2*65536), 65536)
	assert.Error(t, err)
	assert.ErrorContains(t, err, "out of range")
}
