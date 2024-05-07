/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"
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

func TestAppArmorProfile(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "multiple containers",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						`container.apparmor.security.beta.kubernetes.io/`:  `bogus`,
						`container.apparmor.security.beta.kubernetes.io/a`: ``,
						`container.apparmor.security.beta.kubernetes.io/b`: `runtime/default`,
						`container.apparmor.security.beta.kubernetes.io/c`: `localhost/`,
						`container.apparmor.security.beta.kubernetes.io/d`: `localhost/foo`,
						`container.apparmor.security.beta.kubernetes.io/e`: `unconfined`,
						`container.apparmor.security.beta.kubernetes.io/f`: `unknown`,
					},
				},
			},
			expectReason: `forbidden AppArmor profiles`,
			expectDetail: strings.Join([]string{
				`container.apparmor.security.beta.kubernetes.io/="bogus"`,
				`container.apparmor.security.beta.kubernetes.io/e="unconfined"`,
				`container.apparmor.security.beta.kubernetes.io/f="unknown"`,
			}, ", "),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := appArmorProfile_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed {
				t.Fatal("expected disallowed")
			}
			if e, a := tc.expectReason, result.ForbiddenReason; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
		})
	}
}
