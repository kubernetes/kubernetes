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

package policy

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCheckAppArmor(t *testing.T) {

	testCases := []struct {
		name           string
		metaData       *metav1.ObjectMeta
		podSpec        *corev1.PodSpec
		expectedResult *CheckResult
	}{
		{
			name: "container with default AppArmor + extra annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				corev1.AppArmorBetaProfileNamePrefix + "test": "runtime/default",
				"env": "prod",
			},
			},
			podSpec:        &corev1.PodSpec{},
			expectedResult: &CheckResult{Allowed: true},
		},
		{
			name: "container with local AppArmor + extra annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				corev1.AppArmorBetaProfileNamePrefix + "test": "localhost/sec-profile01",
				"env": "dev",
			},
			},
			podSpec:        &corev1.PodSpec{},
			expectedResult: &CheckResult{Allowed: true},
		},
		{
			name: "container with no AppArmor annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				"env": "dev",
			},
			},
			podSpec:        &corev1.PodSpec{},
			expectedResult: &CheckResult{Allowed: true},
		},
		{
			name:           "container with no annotations",
			metaData:       &metav1.ObjectMeta{},
			podSpec:        &corev1.PodSpec{},
			expectedResult: &CheckResult{Allowed: true},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			result := appArmorProfile_1_0(testCase.metaData, nil)
			if result.Allowed != testCase.expectedResult.Allowed {
				t.Errorf("Expected result was Allowed=%v for annotations %v",
					testCase.expectedResult.Allowed, testCase.metaData.Annotations)
			}
		})
	}
}
