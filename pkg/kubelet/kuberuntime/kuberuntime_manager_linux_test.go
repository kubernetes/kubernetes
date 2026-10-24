//go:build linux

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

package kuberuntime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// A request-only memory resize must rewrite the pod cgroup's memory.low when
// TieredReservation derives it from the request, even though memory.max is
// unchanged. Runtimes are notified via UpdatePodSandboxResources first, then
// the pod cgroup is actuated before the containers grow.
func TestDoPodResizeActionRequestOnlyMemoryResizeUpdatesPodMemoryLow(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MemoryQoS, true)
	origCgroupMode := isCgroup2UnifiedMode
	isCgroup2UnifiedMode = func() bool { return true }
	t.Cleanup(func() { isCgroup2UnifiedMode = origCgroupMode })
	metrics.Register()

	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.memoryReservationPolicy = kubeletconfiginternal.TieredReservationMemoryReservationPolicy
	m.memoryThrottlingFactor = new(0.9)

	mockCM := cmtesting.NewMockContainerManager(t)
	mockCM.EXPECT().PodHasExclusiveCPUs(logger, mock.Anything).Return(false).Maybe()
	mockCM.EXPECT().ContainerHasExclusiveCPUs(logger, mock.Anything, mock.Anything).Return(false).Maybe()
	m.containerManager = mockCM
	mockPCM := cmtesting.NewMockPodContainerManager(t)
	mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

	// Pod cgroup currently protects the old request; the limit is unchanged.
	mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceMemory).Return(&cm.ResourceConfig{
		Memory:  ptr.To[int64](512 * 1024 * 1024),
		Unified: map[string]string{cm.Cgroup2MemoryLow: "268435456"},
	}, nil)
	mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceCPU).Return(&cm.ResourceConfig{
		CPUShares: ptr.To[uint64](102),
		CPUQuota:  ptr.To[int64](10000),
		CPUPeriod: ptr.To[uint64](100000),
	}, nil)

	var actuated []map[string]string
	mockPCM.EXPECT().SetPodCgroupConfig(logger, mock.Anything, mock.Anything).
		Run(func(logger klog.Logger, pod *v1.Pod, rc *cm.ResourceConfig) {
			actuated = append(actuated, rc.Unified)
		}).Return(nil).Times(1)

	pod, kps := makeBasePodAndStatus()
	_, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	for idx, fc := range fakeContainers {
		kps.ContainerStatuses[idx].ID = kubecontainer.ContainerID{Type: "testRuntime", ID: fc.Id}
	}
	pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("400Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("512Mi"),
		},
	}

	m.runtimeHelper = &containertest.FakeRuntimeHelper{}

	actions := podActions{
		ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
			v1.ResourceMemory: {{
				container:       &pod.Spec.Containers[0],
				kubeContainerID: kps.ContainerStatuses[0].ID,
				desiredContainerResources: resourceRequirements{
					memoryRequest: 400 * 1024 * 1024,
					memoryLimit:   512 * 1024 * 1024,
				},
				currentContainerResources: &resourceRequirements{
					memoryRequest: 256 * 1024 * 1024,
					memoryLimit:   512 * 1024 * 1024,
				},
			}},
		},
		SandboxID: "sandbox-id",
	}
	resizeResult := m.doPodResizeAction(tCtx, pod, kps, actions)

	require.NoError(t, resizeResult.Error, resizeResult.Message)
	require.Len(t, actuated, 1, "request-only resize must rewrite the pod cgroup once")
	assert.Equal(t, map[string]string{cm.Cgroup2MemoryLow: "419430400"}, actuated[0])
	mockCM.AssertExpectations(t)
	mockPCM.AssertExpectations(t)
}
