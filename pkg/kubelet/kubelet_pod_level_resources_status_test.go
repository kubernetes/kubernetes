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
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	cm "k8s.io/kubernetes/pkg/kubelet/cm"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
)

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
			logger, _ := ktesting.NewTestContext(t)

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
			got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{})
			require.NotNil(t, got)
			require.Equal(t, tc.expectedMilliCPU, got.Requests.Cpu().MilliValue())
		})
	}
}

func TestConvertToAPIPodLevelResourcesStatusContainerLevelResourcesStable(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodLevelResources:                       true,
		features.InPlacePodLevelResourcesVerticalScaling: true,
	})

	testCases := []struct {
		name string
		pod  *v1.Pod
		// readbackShares is the lossy cpu.shares readback of the pod cgroup.
		readbackShares uint64
		// cpuQuota is the cpu.cfs_quota_us readback, over a 100000us period.
		cpuQuota int64
	}{
		{
			// 156m -> shares=159 -> weight=6 -> shares(readback)=133 -> 130m.
			name: "container requests only",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name: "c",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:              resource.MustParse("156m"),
								v1.ResourceMemory:           resource.MustParse("256Mi"),
								v1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("350m"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					}},
				},
			},
			readbackShares: 133,
			cpuQuota:       35000,
		},
		{
			// (156m + 10m overhead) -> shares=169 -> weight=7 -> shares(readback)=159 -> 156m.
			// The pod cgroup quota covers the overhead too: (350m + 10m) -> 36000us.
			name: "container requests plus overhead",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("10m"),
					},
					Containers: []v1.Container{{
						Name: "c",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("156m"),
								v1.ResourceMemory: resource.MustParse("256Mi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("350m"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					}},
				},
			},
			readbackShares: 159,
			cpuQuota:       36000,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			cpuPeriod := uint64(100000)
			cpuQuota := tc.cpuQuota
			readbackShares := tc.readbackShares
			cpuCfg := &cm.ResourceConfig{
				CPUShares: &readbackShares,
				CPUPeriod: &cpuPeriod,
				CPUQuota:  &cpuQuota,
			}

			pod := tc.pod.DeepCopy()

			mockPCM := cmtesting.NewMockPodContainerManager(t)
			mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceMemory).Return(nil, nil)
			mockPCM.EXPECT().GetPodCgroupConfig(pod, v1.ResourceCPU).Return(cpuCfg, nil)

			mockCM := cmtesting.NewMockContainerManager(t)
			mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)

			kl := &Kubelet{containerManager: mockCM}

			// Status reported while the pod is not running yet: the allocated
			// resources, aggregated from the containers and the overhead.
			pod.Status.Phase = v1.PodPending
			beforeRunning := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{})

			// Same pod once it reports Running, with the previous status in hand.
			pod.Status.Phase = v1.PodRunning
			afterRunning := kl.convertToAPIPodLevelResourcesStatus(logger, pod, v1.PodStatus{
				Phase:     v1.PodRunning,
				Resources: beforeRunning,
			})

			require.True(t, apiequality.Semantic.DeepEqual(beforeRunning.Requests, afterRunning.Requests),
				"status.resources.requests changed once the pod started running:\nbefore: %v\nafter:  %v",
				beforeRunning.Requests, afterRunning.Requests)
			require.True(t, apiequality.Semantic.DeepEqual(beforeRunning.Limits, afterRunning.Limits),
				"status.resources.limits changed once the pod started running:\nbefore: %v\nafter:  %v",
				beforeRunning.Limits, afterRunning.Limits)
		})
	}
}
