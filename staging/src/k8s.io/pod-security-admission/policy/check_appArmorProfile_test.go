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

func TestCheckAppArmor_Allowed(t *testing.T) {
	testCases := []struct {
		name     string
		metaData *metav1.ObjectMeta
		podSpec  *corev1.PodSpec
	}{
		{
			name: "container with default AppArmor + extra annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "test": "runtime/default",
				"env": "prod",
			}},
			podSpec: &corev1.PodSpec{},
		},
		{
			name: "container with local AppArmor + extra annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "test": "localhost/sec-profile01",
				"env": "dev",
			}},
			podSpec: &corev1.PodSpec{},
		},
		{
			name: "container with no AppArmor annotations",
			metaData: &metav1.ObjectMeta{Annotations: map[string]string{
				"env": "dev",
			}},
			podSpec: &corev1.PodSpec{},
		},
		{
			name:     "container with no annotations",
			metaData: &metav1.ObjectMeta{},
			podSpec:  &corev1.PodSpec{},
		},
		{
			name:     "pod with runtime default",
			metaData: &metav1.ObjectMeta{},
			podSpec: &corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					AppArmorProfile: &corev1.AppArmorProfile{
						Type: corev1.AppArmorProfileTypeRuntimeDefault,
					},
				},
			},
		},
		{
			name:     "container with localhost profile",
			metaData: &metav1.ObjectMeta{},
			podSpec: &corev1.PodSpec{
				Containers: []corev1.Container{{
					Name: "foo",
					SecurityContext: &corev1.SecurityContext{
						AppArmorProfile: &corev1.AppArmorProfile{
							Type: corev1.AppArmorProfileTypeRuntimeDefault,
						},
					},
				}},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			result := appArmorProfile_1_0(testCase.metaData, testCase.podSpec)
			if !result.Allowed {
				t.Errorf("Should be allowed")
			}
		})
	}
}

func TestCheckAppArmor_Forbidden(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "unconfined pod",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					SecurityContext: &corev1.PodSecurityContext{
						AppArmorProfile: &corev1.AppArmorProfile{
							Type: corev1.AppArmorProfileTypeUnconfined,
						},
					},
				},
			},
			expectReason: "forbidden AppArmor profile",
			expectDetail: `pod must not set AppArmor profile type to "Unconfined"`,
		},
		{
			name: "unconfined container",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					SecurityContext: &corev1.PodSecurityContext{
						AppArmorProfile: &corev1.AppArmorProfile{
							Type: corev1.AppArmorProfileTypeRuntimeDefault,
						},
					},
					Containers: []corev1.Container{{
						Name: "foo",
						SecurityContext: &corev1.SecurityContext{
							AppArmorProfile: &corev1.AppArmorProfile{
								Type: corev1.AppArmorProfileTypeUnconfined,
							},
						},
					}},
				},
			},
			expectReason: "forbidden AppArmor profile",
			expectDetail: `container "foo" must not set AppArmor profile type to "Unconfined"`,
		},
		{
			name: "unconfined init container",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					SecurityContext: &corev1.PodSecurityContext{
						AppArmorProfile: &corev1.AppArmorProfile{
							Type: corev1.AppArmorProfileTypeRuntimeDefault,
						},
					},
					Containers: []corev1.Container{{
						Name: "foo",
					}},
					InitContainers: []corev1.Container{{
						Name: "bar",
						SecurityContext: &corev1.SecurityContext{
							AppArmorProfile: &corev1.AppArmorProfile{
								Type: corev1.AppArmorProfileTypeUnconfined,
							},
						},
					}},
				},
			},
			expectReason: "forbidden AppArmor profile",
			expectDetail: `container "bar" must not set AppArmor profile type to "Unconfined"`,
		},
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
			expectReason: "forbidden AppArmor profiles",
			expectDetail: "annotations must not set AppArmor profile type to " + strings.Join([]string{
				`"container.apparmor.security.beta.kubernetes.io/="bogus""`,
				`"container.apparmor.security.beta.kubernetes.io/e="unconfined""`,
				`"container.apparmor.security.beta.kubernetes.io/f="unknown""`,
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
