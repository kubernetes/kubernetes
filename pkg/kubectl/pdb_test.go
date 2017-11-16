/*
Copyright 2017 The Kubernetes Authors.

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

package kubectl

import (
	policy "k8s.io/api/policy/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"testing"
)

func TestPodDisruptionBudgetV2Generator(t *testing.T) {
	minAvailableNumber := intstr.Parse("2")
	minAvailablePercent := intstr.Parse("50%")

	tests := map[string]struct {
		params    map[string]interface{}
		expected  *policy.PodDisruptionBudget
		expectErr bool
	}{
		"test valid case with number min-available": {
			params: map[string]interface{}{
				"name":          "foo",
				"selector":      "app=nginx",
				"min-available": "2",
				"max-available": "",
			},
			expected: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailableNumber,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": "nginx",
						},
					},
				},
			},
			expectErr: false,
		},
		"test valid case with percent min-available": {
			params: map[string]interface{}{
				"name":          "foo",
				"selector":      "app=nginx",
				"min-available": "50%",
				"max-available": "",
			},
			expected: &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailablePercent,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": "nginx",
						},
					},
				},
			},
			expectErr: false,
		},
		"test missing required param": {
			params: map[string]interface{}{
				"name":          "foo",
				"min-available": "2",
				"max-available": "",
			},
			expectErr: true,
		},
		"test with invalid format params": {
			params: map[string]interface{}{
				"name":          "foo",
				"selector":      "app=nginx",
				"min-available": 2,
				"max-available": "",
			},
			expectErr: true,
		},
		"test min-available/max-available all not be specified": {
			params: map[string]interface{}{
				"name":          "foo",
				"selector":      "app=nginx",
				"min-available": "",
				"max-available": "",
			},
			expectErr: true,
		},
		"test min-available and max-unavailable cannot be both specified": {
			params: map[string]interface{}{
				"name":          "foo",
				"selector":      "app=nginx",
				"min-available": "2",
				"max-available": "5",
			},
			expectErr: true,
		},
	}

	generator := PodDisruptionBudgetV2Generator{}
	for name, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !apiequality.Semantic.DeepEqual(obj.(*policy.PodDisruptionBudget), test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, obj.(*policy.PodDisruptionBudget))
		}
	}
}
