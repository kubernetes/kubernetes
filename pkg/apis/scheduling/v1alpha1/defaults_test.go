/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1_test

import (
	"reflect"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"
	ptr "k8s.io/utils/ptr"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultPreempting(t *testing.T) {
	priorityClass := &v1alpha1.PriorityClass{}

	output := roundTrip(t, runtime.Object(priorityClass)).(*v1alpha1.PriorityClass)
	if output.PreemptionPolicy == nil || *output.PreemptionPolicy != apiv1.PreemptLowerPriority {
		t.Errorf("Expected PriorityClass.Preempting value: %+v\ngot: %+v\n", apiv1.PreemptLowerPriority, output.PreemptionPolicy)
	}
}

func TestSetDefaultWorkload(t *testing.T) {
	workload := &v1alpha1.Workload{
		Spec: v1alpha1.WorkloadSpec{
			PodGroups: []v1alpha1.PodGroup{
				{
					Name: "test-podgroup",
					Policy: v1alpha1.PodGroupPolicy{
						Gang: &v1alpha1.GangSchedulingPolicy{},
					},
				},
			},
		},
	}

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		expectedDisruptionMode        *v1alpha1.DisruptionMode
	}{
		{
			name:                          "workload-aware preemption disabled",
			enableWorkloadAwarePreemption: false,
			expectedDisruptionMode:        nil,
		},
		{
			name:                          "workload-aware preemption enabled",
			enableWorkloadAwarePreemption: true,
			expectedDisruptionMode:        ptr.To(v1alpha1.DisruptionModePod),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			if tc.enableWorkloadAwarePreemption {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadAwarePreemption, true)
			}
			workloadCopy := workload.DeepCopy()
			output := roundTrip(t, runtime.Object(workloadCopy)).(*v1alpha1.Workload)
			gang := output.Spec.PodGroups[0].Policy.Gang
			if tc.expectedDisruptionMode == nil && gang.DisruptionMode != nil {
				t.Fatalf("Expected Workload.Spec.PodGroups[0].Policy.Gang.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, gang.DisruptionMode)
			}
			if tc.expectedDisruptionMode != nil {
				if gang.DisruptionMode == nil || *gang.DisruptionMode != *tc.expectedDisruptionMode {
					t.Fatalf("Expected Workload.Spec.PodGroups[0].Policy.Gang.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, gang.DisruptionMode)
				}
			}
		})
	}
}
