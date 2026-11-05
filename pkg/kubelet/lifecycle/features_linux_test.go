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

package lifecycle

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestIsPodLevelResourcesSupported(t *testing.T) {
	podWithoutPLR := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "c1"}},
		},
	}
	podWithCPUPLR := &v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}
	podWithEphemeralStoragePLR := &v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceEphemeralStorage: resource.MustParse("1Gi")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}

	tests := []struct {
		name                            string
		pod                             *v1.Pod
		podLevelResourcesEnabled        bool
		podLevelEphemeralStorageEnabled bool
		want                            PodAdmitResult
	}{
		{
			name: "pod without pod-level resources admitted when gates disabled",
			pod:  podWithoutPLR,
			want: PodAdmitResult{Admit: true},
		},
		{
			name:                     "pod with pod-level CPU rejected when PodLevelResources disabled",
			pod:                      podWithCPUPLR,
			podLevelResourcesEnabled: false,
			want: PodAdmitResult{
				Admit:   false,
				Reason:  PodLevelResourcesNotAdmittedReason,
				Message: "PodLevelResources feature gate is disabled",
			},
		},
		{
			name:                            "pod with pod-level CPU admitted when PodLevelResourcesEphemeralStorage disabled",
			pod:                             podWithCPUPLR,
			podLevelResourcesEnabled:        true,
			podLevelEphemeralStorageEnabled: false,
			want:                            PodAdmitResult{Admit: true},
		},
		{
			name:                            "pod with pod-level ephemeral-storage rejected when PodLevelResourcesEphemeralStorage disabled",
			pod:                             podWithEphemeralStoragePLR,
			podLevelResourcesEnabled:        true,
			podLevelEphemeralStorageEnabled: false,
			want: PodAdmitResult{
				Admit:   false,
				Reason:  PodLevelResourcesNotAdmittedReason,
				Message: "PodLevelResourcesEphemeralStorage feature gate is disabled",
			},
		},
		{
			name:                            "pod with pod-level ephemeral-storage admitted when both gates enabled",
			pod:                             podWithEphemeralStoragePLR,
			podLevelResourcesEnabled:        true,
			podLevelEphemeralStorageEnabled: true,
			want:                            PodAdmitResult{Admit: true},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.PodLevelResources:                 tc.podLevelResourcesEnabled,
				features.PodLevelResourcesEphemeralStorage: tc.podLevelResourcesEnabled && tc.podLevelEphemeralStorageEnabled,
			})
			got := isPodLevelResourcesSupported(tc.pod)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("isPodLevelResourcesSupported() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
