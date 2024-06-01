/*
Copyright 2024 The Kubernetes Authors.

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
)

func TestHostProbesAndHostLifecycle(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "valid pod with unset hosts",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "a",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{}, // Host field not set (defaults to podIP)
							},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{}, // Host field not set (defaults to podIP)
							},
						},
						Lifecycle: &corev1.Lifecycle{
							PostStart: &corev1.LifecycleHandler{
								HTTPGet: &corev1.HTTPGetAction{}, // Host field not set (defaults to podIP)
							},
						},
					},
				},
			}},
		},
		{
			name: "invalid pod with local host IP as probe host",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "a",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "127.0.0.1"},
							},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{Host: "::1"},
							},
						},
						StartupProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "localhost"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "a" uses probe or lifecycle hosts "127.0.0.1", "::1", "localhost"`,
		},
		{
			name: "invalid httpget host in liveness probe",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "a",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "invalid.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "a" uses probe or lifecycle host "invalid.host"`,
		},
		{
			name: "invalid tcpsocket host in readiness probe",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "b",
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{Host: "invalid.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "b" uses probe or lifecycle host "invalid.host"`,
		},
		{
			name: "invalid httpget host in startup probe",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "c",
						StartupProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "invalid.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "c" uses probe or lifecycle host "invalid.host"`,
		},
		{
			name: "invalid poststart tcpsocket host",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "d",
						Lifecycle: &corev1.Lifecycle{
							PostStart: &corev1.LifecycleHandler{
								TCPSocket: &corev1.TCPSocketAction{Host: "invalid.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "d" uses probe or lifecycle host "invalid.host"`,
		},
		{
			name: "invalid prestop httpget host",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "e",
						Lifecycle: &corev1.Lifecycle{
							PreStop: &corev1.LifecycleHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "another.invalid.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "e" uses probe or lifecycle host "another.invalid.host"`,
		},
		{
			name: "multiple containers with multiple invalid hosts",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "valid"},
					{
						Name: "invalid1",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Host: "a.com"}},
						},
					},
					{
						Name: "invalid2",
						Lifecycle: &corev1.Lifecycle{
							PreStop: &corev1.LifecycleHandler{TCPSocket: &corev1.TCPSocketAction{Host: "b.com"}},
						},
						StartupProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Host: "a.com"}},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `containers "invalid1", "invalid2" use probe or lifecycle hosts "a.com", "b.com"`,
		},
		{
			name: "invalid ipv4 host in probe",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "a",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "8.8.8.8"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "a" uses probe or lifecycle host "8.8.8.8"`,
		},
		{
			name: "invalid ipv6 host in probe",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "a",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{Host: "2001:4860:4860::8888"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "a" uses probe or lifecycle host "2001:4860:4860::8888"`,
		},
		{
			name: "invalid host in initcontainer",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{
					{
						Name: "init",
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{Host: "invalid.init.host"},
							},
						},
					},
				},
			}},
			expectReason: "probe or lifecycle host",
			expectDetail: `container "init" uses probe or lifecycle host "invalid.init.host"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := hostProbesAndHostLifecycleV1Dot34(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if tc.expectReason == "" {
				if !result.Allowed {
					t.Fatalf("expected allowed, but got disallowed: %s", result.ForbiddenDetail)
				}
			} else {
				if result.Allowed {
					t.Fatal("expected disallowed, but got allowed")
				}
				if e, a := tc.expectReason, result.ForbiddenReason; e != a {
					t.Errorf("expected reason\n%s\ngot\n%s", e, a)
				}
				if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
					t.Errorf("expected detail\n%s\ngot\n%s", e, a)
				}
			}
		})
	}
}
