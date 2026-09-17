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
	"context"
	"testing"

	libcontainercgroups "github.com/opencontainers/cgroups"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestIsPerPodPIDLimitSupported(t *testing.T) {
	podWithPIDLimit := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pid-pod"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("2048")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}
	podWithoutPIDLimit := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "plain-pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "c1"}},
		},
	}

	tests := []struct {
		name        string
		gateEnabled bool
		pod         *v1.Pod
		expectAdmit bool
	}{
		{
			name:        "pod without pid limit admitted with gate on",
			gateEnabled: true,
			pod:         podWithoutPIDLimit,
			expectAdmit: true,
		},
		{
			name:        "pod without pid limit admitted with gate off",
			gateEnabled: false,
			pod:         podWithoutPIDLimit,
			expectAdmit: true,
		},
		{
			name:        "pod with pid limit rejected when gate is off",
			gateEnabled: false,
			pod:         podWithPIDLimit,
			expectAdmit: false,
		},
		{
			name:        "pod with pid limit with gate on admitted iff cgroupsv2",
			gateEnabled: true,
			pod:         podWithPIDLimit,
			// The cgroup mode of the host running the test decides the outcome.
			expectAdmit: libcontainercgroups.IsCgroup2UnifiedMode(),
		},
		{
			// Static pods bypass apiserver validation; the kubelet must
			// reject out-of-range values itself.
			name:        "pod with pid limit below the allowed range rejected",
			gateEnabled: true,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pid-pod-low"},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("64")},
					},
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			expectAdmit: false,
		},
		{
			name:        "pod with pid limit above the allowed range rejected",
			gateEnabled: true,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pid-pod-high"},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("32768")},
					},
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			expectAdmit: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerPodPIDLimit, tc.gateEnabled)
			result := isPerPodPIDLimitSupported(tc.pod)
			if result.Admit != tc.expectAdmit {
				t.Errorf("expected Admit=%v, got Admit=%v (reason=%q message=%q)", tc.expectAdmit, result.Admit, result.Reason, result.Message)
			}
			if !result.Admit && result.Reason != PerPodPIDLimitNotAdmittedReason {
				t.Errorf("expected reason %q, got %q", PerPodPIDLimitNotAdmittedReason, result.Reason)
			}
		})
	}

	// The pid-limit check must also be reachable through the combined
	// podFeaturesAdmitHandler, after the pod-level-resources check.
	t.Run("via podFeaturesAdmitHandler", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerPodPIDLimit, false)
		handler := NewPodFeaturesAdmitHandler()
		result := handler.Admit(context.Background(), &PodAdmitAttributes{Pod: podWithPIDLimit})
		if result.Admit {
			t.Fatal("expected handler to reject pod with pid limit when the gate is disabled")
		}
		if result.Reason != PerPodPIDLimitNotAdmittedReason {
			t.Fatalf("expected reason %q, got %q", PerPodPIDLimitNotAdmittedReason, result.Reason)
		}
	})
}
