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

package v1beta1_test

import (
	"reflect"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1beta1.SchemeGroupVersion)
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
	priorityClass := &v1beta1.PriorityClass{}

	output := roundTrip(t, runtime.Object(priorityClass)).(*v1beta1.PriorityClass)
	if output.PreemptionPolicy == nil || *output.PreemptionPolicy != apiv1.PreemptLowerPriority {
		t.Errorf("Expected PriorityClass.Preempting value: %+v\ngot: %+v\n", apiv1.PreemptLowerPriority, output.PreemptionPolicy)
	}
}

func TestSetDefaultsPodGroup(t *testing.T) {
	var (
		preemptNever         = v1beta1.PreemptNever
		preemptLowerPriority = v1beta1.PreemptLowerPriority
	)

	tests := []struct {
		name                           string
		enablePodGroupPreemptionPolicy bool
		podGroup                       *v1beta1.PodGroup
		expectedPolicy                 *v1beta1.PreemptionPolicy
	}{
		{
			name:                           "feature gate disabled, policy is nil",
			podGroup:                       &v1beta1.PodGroup{},
			enablePodGroupPreemptionPolicy: false,
			expectedPolicy:                 nil,
		},
		{
			name: "feature gate disabled, policy is set",
			podGroup: &v1beta1.PodGroup{
				Spec: v1beta1.PodGroupSpec{
					PreemptionPolicy: &preemptNever,
				},
			},
			enablePodGroupPreemptionPolicy: false,
			expectedPolicy:                 &preemptNever,
		},
		{
			name:                           "feature gate enabled, policy is nil",
			podGroup:                       &v1beta1.PodGroup{},
			enablePodGroupPreemptionPolicy: true,
			expectedPolicy:                 &preemptLowerPriority,
		},
		{
			name: "feature gate enabled, policy is set",
			podGroup: &v1beta1.PodGroup{
				Spec: v1beta1.PodGroupSpec{
					PreemptionPolicy: &preemptNever,
				},
			},
			enablePodGroupPreemptionPolicy: true,
			expectedPolicy:                 &preemptNever,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          tc.enablePodGroupPreemptionPolicy,
				features.PodGroupPreemptionPolicy: tc.enablePodGroupPreemptionPolicy,
			})

			output := roundTrip(t, runtime.Object(tc.podGroup)).(*v1beta1.PodGroup)
			if !reflect.DeepEqual(output.Spec.PreemptionPolicy, tc.expectedPolicy) {
				t.Errorf("Expected PreemptionPolicy: %v, got: %v", tc.expectedPolicy, output.Spec.PreemptionPolicy)
			}
		})
	}
}
