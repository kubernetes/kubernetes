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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodConditionByKubelet(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodHasNetworkCondition, true)()

	for _, enablePodDisruptionConditions := range []bool{false, true} {
		t.Run(fmt.Sprintf("enablePodDisruptionConditions=%v", enablePodDisruptionConditions), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDisruptionConditions, enablePodDisruptionConditions)()
			trueCases := []v1.PodConditionType{
				v1.PodScheduled,
				v1.PodReady,
				v1.PodInitialized,
				v1.ContainersReady,
				PodHasNetwork,
			}

			if enablePodDisruptionConditions {
				trueCases = append(trueCases, v1.AlphaNoCompatGuaranteeDisruptionTarget)
				trueCases = append(trueCases, ResourceExhausted)
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
		})
	}
}

func TestPodFailureConditions(t *testing.T) {
	testCases := map[string]struct {
		enablePodDisruptionConditions bool
		wantConditions                []v1.PodConditionType
	}{
		"get PodFailureConditions; PodDisruptionConditions disabled": {
			enablePodDisruptionConditions: false,
			wantConditions:                []v1.PodConditionType{},
		},
		"get PodFailureConditions; PodDisruptionConditions enabled": {
			enablePodDisruptionConditions: true,
			wantConditions: []v1.PodConditionType{
				v1.AlphaNoCompatGuaranteeDisruptionTarget,
				ResourceExhausted,
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDisruptionConditions, tc.enablePodDisruptionConditions)()
			if diff := cmp.Diff(tc.wantConditions, PodFailureConditions()); diff != "" {
				t.Fatalf("unexpected output: %s", diff)
			}
		})
	}
}
