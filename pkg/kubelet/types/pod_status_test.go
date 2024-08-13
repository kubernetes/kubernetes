/*
Copyright 2018 The Kubernetes Authors.

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

package types

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodConditionByKubelet(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodReadyToStartContainersCondition, true)

	trueCases := []v1.PodConditionType{
		v1.PodScheduled,
		v1.PodReady,
		v1.PodInitialized,
		v1.ContainersReady,
		v1.PodReadyToStartContainers,
	}

	for _, tc := range trueCases {
		if !PodConditionByKubelet(tc) {
			t.Errorf("Expect %q to be condition owned by kubelet.", tc)
		}
	}

	falseCases := []v1.PodConditionType{
		v1.PodConditionType("abcd"),
		v1.PodConditionType(v1.PodReasonUnschedulable),
	}

	for _, tc := range falseCases {
		if PodConditionByKubelet(tc) {
			t.Errorf("Expect %q NOT to be condition owned by kubelet.", tc)
		}
	}
}

func TestPodConditionSharedByKubelet(t *testing.T) {
	trueCases := []v1.PodConditionType{
		v1.DisruptionTarget,
	}

	for _, tc := range trueCases {
		if !PodConditionSharedByKubelet(tc) {
			t.Errorf("Expect %q to be condition shared by kubelet.", tc)
		}
	}

	falseCases := []v1.PodConditionType{
		v1.PodConditionType("abcd"),
		v1.PodConditionType(v1.PodReasonUnschedulable),
	}

	for _, tc := range falseCases {
		if PodConditionSharedByKubelet(tc) {
			t.Errorf("Expect %q NOT to be condition shared by kubelet.", tc)
		}
	}
}
