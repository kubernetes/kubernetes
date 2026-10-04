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
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	cm "k8s.io/kubernetes/pkg/kubelet/cm"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
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
