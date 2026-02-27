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

package v1alpha2_test

import (
	"reflect"
	"testing"

	"k8s.io/api/scheduling/v1alpha2"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha2.SchemeGroupVersion)
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

func TestSetDefaultWorkload(t *testing.T) {
	workload := &v1alpha2.Workload{
		Spec: v1alpha2.WorkloadSpec{
			PodGroupTemplates: []v1alpha2.PodGroupTemplate{
				{
					Name: "test-podgroup",
					SchedulingPolicy: v1alpha2.PodGroupSchedulingPolicy{
						Gang: &v1alpha2.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		expectedDisruptionMode        *v1alpha2.DisruptionMode
	}{
		{
			name:                          "workload-aware preemption disabled",
			enableWorkloadAwarePreemption: false,
			expectedDisruptionMode:        nil,
		},
		{
			name:                          "workload-aware preemption enabled",
			enableWorkloadAwarePreemption: true,
			expectedDisruptionMode:        new(v1alpha2.DisruptionModePod),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption: tc.enableWorkloadAwarePreemption,
			})
			workloadCopy := workload.DeepCopy()
			output := roundTrip(t, runtime.Object(workloadCopy)).(*v1alpha2.Workload)
			disruptionMode := output.Spec.PodGroupTemplates[0].DisruptionMode
			if tc.expectedDisruptionMode == nil && disruptionMode != nil {
				t.Fatalf("Expected Workload.Spec.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, disruptionMode)
			}
			if tc.expectedDisruptionMode != nil {
				if disruptionMode == nil || *disruptionMode != *tc.expectedDisruptionMode {
					t.Fatalf("Expected Workload.Spec.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, disruptionMode)
				}
			}
		})
	}
}

func TestSetDefaultPodGroup(t *testing.T) {
	pg := &v1alpha2.PodGroup{
		Spec: v1alpha2.PodGroupSpec{
			SchedulingPolicy: v1alpha2.PodGroupSchedulingPolicy{
				Gang: &v1alpha2.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
		},
	}

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		expectedDisruptionMode        *v1alpha2.DisruptionMode
	}{
		{
			name:                          "workload-aware preemption disabled",
			enableWorkloadAwarePreemption: false,
			expectedDisruptionMode:        nil,
		},
		{
			name:                          "workload-aware preemption enabled",
			enableWorkloadAwarePreemption: true,
			expectedDisruptionMode:        new(v1alpha2.DisruptionModePod),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption: tc.enableWorkloadAwarePreemption,
			})
			pgCopy := pg.DeepCopy()
			output := roundTrip(t, runtime.Object(pgCopy)).(*v1alpha2.PodGroup)
			disruptionMode := output.Spec.DisruptionMode
			if tc.expectedDisruptionMode == nil && disruptionMode != nil {
				t.Fatalf("Expected PodGroup.Spec.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, disruptionMode)
			}
			if tc.expectedDisruptionMode != nil {
				if disruptionMode == nil || *disruptionMode != *tc.expectedDisruptionMode {
					t.Fatalf("Expected PodGroup.Spec.DisruptionMode value: %+v\ngot: %+v\n", tc.expectedDisruptionMode, disruptionMode)
				}
			}
		})
	}
}
