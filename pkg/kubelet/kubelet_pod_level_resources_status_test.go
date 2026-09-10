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

package kubelet

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	cm "k8s.io/kubernetes/pkg/kubelet/cm"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
)

// newBufferedTestLogger returns a logger whose entries are captured in
// memory so tests can assert on the emitted log levels.
func newBufferedTestLogger(t *testing.T) klog.Logger {
	return ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true)))
}

// capturedErrorLogs returns the messages of the error-level entries captured
// by the given ktesting logger.
func capturedErrorLogs(t *testing.T, logger klog.Logger) []string {
	t.Helper()
	underlier, ok := logger.GetSink().(ktesting.Underlier)
	require.True(t, ok, "expected a ktesting LogSink, got %T", logger.GetSink())
	var messages []string
	for _, entry := range underlier.GetBuffer().Data() {
		if entry.Type == ktesting.LogError {
			messages = append(messages, entry.Message)
		}
	}
	return messages
}

// capturedInfoLogs returns the messages of the info-level entries captured
// by the given ktesting logger.
func capturedInfoLogs(t *testing.T, logger klog.Logger) []string {
	t.Helper()
	underlier, ok := logger.GetSink().(ktesting.Underlier)
	require.True(t, ok, "expected a ktesting LogSink, got %T", logger.GetSink())
	var messages []string
	for _, entry := range underlier.GetBuffer().Data() {
		if entry.Type == ktesting.LogInfo {
			messages = append(messages, entry.Message)
		}
	}
	return messages
}

func TestConvertToAPIPodLevelResourcesStatus(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodLevelResources:                       true,
		features.InPlacePodLevelResourcesVerticalScaling: true,
	})

	testCases := []struct {
		name             string
		allocatedCPU     string
		readbackShares   uint64
		expectedMilliCPU int64
	}{
		{
			// 50m -> shares=51 -> weight=2 -> shares(readback)=28 -> 28m.
			// Round-trip match: preserve allocated 50m.
			name:             "preserve allocated request for lossy v2 roundtrip readback",
			allocatedCPU:     "50m",
			readbackShares:   28,
			expectedMilliCPU: 50,
		},
		{
			// Readback shares do not match the round-tripped allocated shares:
			// treat as a real actuation and report the readback milliCPU.
			name:             "use actuated request when readback is not a v2 roundtrip",
			allocatedCPU:     "50m",
			readbackShares:   102, // SharesToMilliCPU(102) == 100
			expectedMilliCPU: 100,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger := newBufferedTestLogger(t)

			pod := &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse(tc.allocatedCPU),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
					Containers: []v1.Container{{Name: "pause"}},
				},
				Status: v1.PodStatus{Phase: v1.PodRunning},
			}

			cpuPeriod := uint64(100000)
			cpuQuota := int64(0)
			readbackShares := tc.readbackShares
			cpuCfg := &cm.ResourceConfig{
				CPUShares: &readbackShares,
				CPUPeriod: &cpuPeriod,
				CPUQuota:  &cpuQuota,
			}

			mockPCM := cmtesting.NewMockPodContainerManager(t)
			mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceMemory).Return(nil, nil)
			mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceCPU).Return(cpuCfg, nil)

			mockCM := cmtesting.NewMockContainerManager(t)
			mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

			kl := &Kubelet{containerManager: mockCM}
			got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{}, v1.PodRunning)
			require.NotNil(t, got)
			require.Equal(t, tc.expectedMilliCPU, got.Requests.Cpu().MilliValue())
		})
	}
}

func TestConvertToAPIPodLevelResourcesStatusMissingCgroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodLevelResources:                       true,
		features.InPlacePodLevelResourcesVerticalScaling: true,
	})

	// runningPod returns a pod whose (stale) API status is still Running,
	// like a pod going through teardown after its cgroup was removed.
	runningPod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: "bar",
			},
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("50m"),
						v1.ResourceMemory: resource.MustParse("50Mi"),
					},
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("100Mi"),
					},
				},
				Containers: []v1.Container{{Name: "pause"}},
			},
			Status: v1.PodStatus{Phase: v1.PodRunning},
		}
	}

	// wrappedENOENT mimics the error returned by GetPodCgroupConfig when the
	// pod cgroup files are gone: the underlying os.ErrNotExist is wrapped.
	wrappedENOENT := func() error {
		return fmt.Errorf("failed to read memory.max for cgroup /kubepods/burstable/poduid: %w", os.ErrNotExist)
	}

	t.Run("non-running phase skips cgroup reads and reports effective allocated resources", func(t *testing.T) {
		logger := newBufferedTestLogger(t)
		pod := runningPod()

		// The old status was reported by a running pod and carries cgroup
		// readback values that must not be preserved once the pod is no
		// longer running.
		oldPodStatus := v1.PodStatus{
			Phase: v1.PodRunning,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("99m")},
				Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("999Mi")},
			},
		}

		// No cgroup reads are expected: the strict mocks fail the test if
		// NewPodContainerManager or GetPodCgroupConfig is called.
		mockCM := cmtesting.NewMockContainerManager(t)

		kl := &Kubelet{containerManager: mockCM}
		got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, oldPodStatus, v1.PodSucceeded)
		require.NotNil(t, got)
		require.True(t, got.Requests.Cpu().Equal(resource.MustParse("50m")))
		require.True(t, got.Requests.Memory().Equal(resource.MustParse("50Mi")))
		require.True(t, got.Limits.Cpu().Equal(resource.MustParse("100m")))
		require.True(t, got.Limits.Memory().Equal(resource.MustParse("100Mi")))
		require.Empty(t, capturedErrorLogs(t, logger))
	})

	t.Run("ENOENT from cgroup read falls back to allocated resources without error log", func(t *testing.T) {
		logger := newBufferedTestLogger(t)
		pod := runningPod()

		mockPCM := cmtesting.NewMockPodContainerManager(t)
		mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceMemory).Return(nil, wrappedENOENT())

		mockCM := cmtesting.NewMockContainerManager(t)
		mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

		kl := &Kubelet{containerManager: mockCM}
		got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{}, v1.PodRunning)
		require.NotNil(t, got)
		require.True(t, got.Requests.Cpu().Equal(resource.MustParse("50m")))
		require.True(t, got.Requests.Memory().Equal(resource.MustParse("50Mi")))
		require.True(t, got.Limits.Cpu().Equal(resource.MustParse("100m")))
		require.True(t, got.Limits.Memory().Equal(resource.MustParse("100Mi")))

		// The missing cgroup is expected during teardown: it must not be
		// logged at error level, only at a lower verbosity.
		require.Empty(t, capturedErrorLogs(t, logger))
		require.Contains(t, capturedInfoLogs(t, logger), "failed to read memory cgroup config for the pod")
	})

	t.Run("non-ENOENT cgroup read error is logged at error level", func(t *testing.T) {
		logger := newBufferedTestLogger(t)
		pod := runningPod()

		readErr := fmt.Errorf("failed to read memory.max for cgroup /kubepods/burstable/poduid: permission denied")

		mockPCM := cmtesting.NewMockPodContainerManager(t)
		mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceMemory).Return(nil, readErr)
		mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceCPU).Return(nil, readErr)

		mockCM := cmtesting.NewMockContainerManager(t)
		mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

		kl := &Kubelet{containerManager: mockCM}
		got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{}, v1.PodRunning)
		require.NotNil(t, got)
		// The read failed, so the allocated values are reported.
		require.True(t, got.Requests.Cpu().Equal(resource.MustParse("50m")))
		require.True(t, got.Limits.Memory().Equal(resource.MustParse("100Mi")))

		errorLogs := capturedErrorLogs(t, logger)
		require.Contains(t, errorLogs, "failed to read memory cgroup config for the pod")
		require.Contains(t, errorLogs, "failed to read cpu cgroup config for the pod")
	})

	t.Run("ENOENT fallback reports complete resources for container-only pod", func(t *testing.T) {
		logger := newBufferedTestLogger(t)
		// A pod without pod-level resources (Spec.Resources == nil) whose
		// effective resources are aggregated from its containers.
		pod := runningPod()
		pod.Spec.Resources = nil
		pod.Spec.Containers = []v1.Container{{
			Name: "pause",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("25m"),
					v1.ResourceMemory: resource.MustParse("25Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("200m"),
					v1.ResourceMemory: resource.MustParse("200Mi"),
				},
			},
		}}

		mockPCM := cmtesting.NewMockPodContainerManager(t)
		mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceMemory).Return(nil, wrappedENOENT())

		mockCM := cmtesting.NewMockContainerManager(t)
		mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

		kl := &Kubelet{containerManager: mockCM}
		got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{}, v1.PodRunning)
		require.NotNil(t, got)
		// The fallback reports the full effective allocated resources
		// instead of a partial list with only requests.memory.
		require.True(t, got.Requests.Cpu().Equal(resource.MustParse("25m")))
		require.True(t, got.Requests.Memory().Equal(resource.MustParse("25Mi")))
		require.True(t, got.Limits.Cpu().Equal(resource.MustParse("200m")))
		require.True(t, got.Limits.Memory().Equal(resource.MustParse("200Mi")))
		require.Empty(t, capturedErrorLogs(t, logger))
	})
}
