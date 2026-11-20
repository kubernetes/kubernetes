/*
Copyright 2026 The Kubernetes Authors.

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

package kubelet

import (
	"testing"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	kubeletapitypes "k8s.io/kubelet/pkg/types"
)

const (
	knownPodUID   = types.UID("11111111-1111-1111-1111-111111111111")
	stalePodUID   = types.UID("22222222-2222-2222-2222-222222222222")
	knownPodSlice = "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod11111111_1111_1111_1111_111111111111.slice"
	stalePodSlice = "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod22222222_2222_2222_2222_222222222222.slice"
)

// fakeIsPodCgroup mirrors cm.PodContainerManager.IsPodCgroup for the fixtures
// used in these tests: it matches only the pod cgroups themselves, not the
// per-container cgroups nested under them.
func fakeIsPodCgroup(cgroupfs string) (bool, types.UID) {
	switch cgroupfs {
	case knownPodSlice:
		return true, knownPodUID
	case stalePodSlice:
		return true, stalePodUID
	}
	return false, ""
}

func TestPodUIDForContainer(t *testing.T) {
	testCases := []struct {
		name        string
		cgroupfs    string
		info        *cadvisorapi.ContainerInfo
		expectedUID types.UID
		expectFound bool
	}{
		{
			name:     "container scope with pod uid label takes precedence over path",
			cgroupfs: knownPodSlice + "/crio-aaaa.scope",
			info: &cadvisorapi.ContainerInfo{
				Spec: cadvisorapi.ContainerSpec{
					Labels: map[string]string{kubeletapitypes.KubernetesPodUIDLabel: string(stalePodUID)},
				},
			},
			expectedUID: stalePodUID,
			expectFound: true,
		},
		{
			name:        "container scope without label resolves via parent pod cgroup",
			cgroupfs:    stalePodSlice + "/crio-bbbb.scope",
			info:        &cadvisorapi.ContainerInfo{},
			expectedUID: stalePodUID,
			expectFound: true,
		},
		{
			name:        "container scope without label and nil info resolves via parent pod cgroup",
			cgroupfs:    knownPodSlice + "/crio-cccc.scope",
			info:        nil,
			expectedUID: knownPodUID,
			expectFound: true,
		},
		{
			name:        "pod cgroup itself resolves directly",
			cgroupfs:    stalePodSlice,
			info:        &cadvisorapi.ContainerInfo{},
			expectedUID: stalePodUID,
			expectFound: true,
		},
		{
			name:        "system container is not associated with a pod",
			cgroupfs:    "/system.slice/sssd.service",
			info:        &cadvisorapi.ContainerInfo{},
			expectedUID: "",
			expectFound: false,
		},
		{
			name:        "root cgroup is not associated with a pod",
			cgroupfs:    "/",
			info:        &cadvisorapi.ContainerInfo{},
			expectedUID: "",
			expectFound: false,
		},
		{
			name:        "kubepods aggregate cgroup is not associated with a pod",
			cgroupfs:    "/kubepods.slice/kubepods-burstable.slice",
			info:        &cadvisorapi.ContainerInfo{},
			expectedUID: "",
			expectFound: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			uid, found := podUIDForContainer(tc.cgroupfs, tc.info, fakeIsPodCgroup)
			assert.Equal(t, tc.expectFound, found)
			assert.Equal(t, tc.expectedUID, uid)
		})
	}
}

func TestFilterStalePodContainers(t *testing.T) {
	podExists := func(uid types.UID) bool {
		return uid == knownPodUID
	}
	containers := map[string]*cadvisorapi.ContainerInfo{
		// container of a pod known to the kubelet: must be kept
		knownPodSlice + "/crio-known.scope": {
			Spec: cadvisorapi.ContainerSpec{
				Labels: map[string]string{kubeletapitypes.KubernetesPodUIDLabel: string(knownPodUID)},
			},
		},
		// container of a rescheduled/deleted pod: must be removed
		stalePodSlice + "/crio-stale.scope": {
			Spec: cadvisorapi.ContainerSpec{
				Labels: map[string]string{kubeletapitypes.KubernetesPodUIDLabel: string(stalePodUID)},
			},
		},
		// leftover pod cgroup of a rescheduled/deleted pod: must be removed
		stalePodSlice: {},
		// non-pod container: must be kept
		"/system.slice/kubelet.service": {},
	}

	filterStalePodContainers(containers, podExists, fakeIsPodCgroup)

	assert.Contains(t, containers, knownPodSlice+"/crio-known.scope", "container of a known pod must be kept")
	assert.Contains(t, containers, "/system.slice/kubelet.service", "non-pod containers must be kept")
	assert.NotContains(t, containers, stalePodSlice+"/crio-stale.scope", "container of an unknown pod must be removed")
	assert.NotContains(t, containers, stalePodSlice, "leftover pod cgroup of an unknown pod must be removed")
	assert.Len(t, containers, 2)
}
