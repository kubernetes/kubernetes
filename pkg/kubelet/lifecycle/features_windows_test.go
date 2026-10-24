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

package lifecycle

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestIsPerPodPIDLimitSupportedWindows(t *testing.T) {
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

	// Windows can never enforce pod-level PID limits: pods specifying one are
	// rejected regardless of feature gate state.
	for _, gateEnabled := range []bool{true, false} {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerPodPIDLimit, gateEnabled)

		if result := isPerPodPIDLimitSupported(podWithPIDLimit); result.Admit {
			t.Errorf("gate=%t: expected pod with pid limit to be rejected on Windows", gateEnabled)
		} else if result.Reason != PerPodPIDLimitNotAdmittedReason {
			t.Errorf("gate=%t: expected reason %q, got %q", gateEnabled, PerPodPIDLimitNotAdmittedReason, result.Reason)
		}

		if result := isPerPodPIDLimitSupported(podWithoutPIDLimit); !result.Admit {
			t.Errorf("gate=%t: expected pod without pid limit to be admitted, got reason %q", gateEnabled, result.Reason)
		}
	}
}
