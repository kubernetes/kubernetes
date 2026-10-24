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

package v1alpha3_test

import (
	"reflect"
	"testing"

	"k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha3.SchemeGroupVersion)
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

func TestSetDefaultsPodGroup(t *testing.T) {
	var (
		preemptNever         = v1alpha3.PreemptNever
		preemptLowerPriority = v1alpha3.PreemptLowerPriority
	)

	tests := []struct {
		name           string
		podGroup       *v1alpha3.PodGroup
		expectedPolicy *v1alpha3.PreemptionPolicy
	}{
		{
			name:           "policy is nil",
			podGroup:       &v1alpha3.PodGroup{},
			expectedPolicy: &preemptLowerPriority,
		},
		{
			name: "policy is set to PreemptNever",
			podGroup: &v1alpha3.PodGroup{
				Spec: v1alpha3.PodGroupSpec{
					PreemptionPolicy: &preemptNever,
				},
			},
			expectedPolicy: &preemptNever,
		},
		{
			name: "policy is set to PreemptLowerPriority",
			podGroup: &v1alpha3.PodGroup{
				Spec: v1alpha3.PodGroupSpec{
					PreemptionPolicy: &preemptLowerPriority,
				},
			},
			expectedPolicy: &preemptLowerPriority,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			output := roundTrip(t, runtime.Object(tc.podGroup)).(*v1alpha3.PodGroup)
			if !reflect.DeepEqual(output.Spec.PreemptionPolicy, tc.expectedPolicy) {
				t.Errorf("Expected PreemptionPolicy: %v, got: %v", tc.expectedPolicy, output.Spec.PreemptionPolicy)
			}
		})
	}
}

func TestSetDefaultsCompositePodGroup(t *testing.T) {
	var (
		preemptNever         = v1alpha3.PreemptNever
		preemptLowerPriority = v1alpha3.PreemptLowerPriority
	)

	tests := []struct {
		name              string
		compositePodGroup *v1alpha3.CompositePodGroup
		expectedPolicy    *v1alpha3.PreemptionPolicy
	}{
		{
			name:              "policy is nil",
			compositePodGroup: &v1alpha3.CompositePodGroup{},
			expectedPolicy:    &preemptLowerPriority,
		},
		{
			name: "policy is set to PreemptNever",
			compositePodGroup: &v1alpha3.CompositePodGroup{
				Spec: v1alpha3.CompositePodGroupSpec{
					PreemptionPolicy: &preemptNever,
				},
			},
			expectedPolicy: &preemptNever,
		},
		{
			name: "policy is set to PreemptLowerPriority",
			compositePodGroup: &v1alpha3.CompositePodGroup{
				Spec: v1alpha3.CompositePodGroupSpec{
					PreemptionPolicy: &preemptLowerPriority,
				},
			},
			expectedPolicy: &preemptLowerPriority,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			output := roundTrip(t, runtime.Object(tc.compositePodGroup)).(*v1alpha3.CompositePodGroup)
			if !reflect.DeepEqual(output.Spec.PreemptionPolicy, tc.expectedPolicy) {
				t.Errorf("Expected PreemptionPolicy: %v, got: %v", tc.expectedPolicy, output.Spec.PreemptionPolicy)
			}
		})
	}
}
