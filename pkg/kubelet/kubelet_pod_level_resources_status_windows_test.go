//go:build windows

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
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
)

// TestConvertToAPIPodLevelResourcesStatus verifies that on Windows, where
// in-place pod-level resize is not supported and pods are not backed by
// cgroups, the kubelet reports the allocated pod-level resources without
// consulting the pod container manager at all.
//
// See https://github.com/kubernetes/kubernetes/issues/141760
func TestConvertToAPIPodLevelResourcesStatus(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodLevelResources:                       true,
		features.InPlacePodLevelResourcesVerticalScaling: true,
	})

	logger, _ := ktesting.NewTestContext(t)

	pod := &v1.Pod{
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

	mockCM := cmtesting.NewMockContainerManager(t)

	// Synthetic oldPodStatus status whose values deliberately do not match the pod
	// spec.
	//
	// Before the platform guard, the failed cgroup read left every readback nil
	// and preserveOldResourcesValue copied the previous status into the result.
	// With the guard, it now returns early from the allocated spec and
	// the previous status is never consulted, so these values must not appear.
	oldPodStatus := v1.PodStatus{
		Phase: v1.PodRunning,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("999m")},
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("999m"), v1.ResourceMemory: resource.MustParse("999Mi")},
		},
	}

	kl := &Kubelet{containerManager: mockCM}
	got := kl.convertToAPIPodLevelResourcesStatus(logger, pod, oldPodStatus)
	require.NotNil(t, got)

	// Result must be the allocated (spec) resources, **not the stale/synthetic status**.
	require.Equal(t, getEffectiveAllocatedResources(pod), got)
	require.Equal(t, int64(50), got.Requests.Cpu().MilliValue(), "stale cpu request leaked from previous status")
	require.Equal(t, int64(100), got.Limits.Cpu().MilliValue(), "stale cpu limit leaked from previous status")
	require.Equal(t, int64(100*1024*1024), got.Limits.Memory().Value(), "stale memory limit leaked from previous status")
}
