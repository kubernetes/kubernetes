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

package util

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestLookupContainerPortNumberByServicePort(t *testing.T) {
	tests := []struct {
		name          string
		svc           v1.Service
		pod           v1.Pod
		port          int32
		containerPort int32
		err           bool
	}{
		{
			name: "test success 1 (int port)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			port:          80,
			containerPort: 8080,
			err:           false,
		},
		{
			name: "test success 2 (clusterIP: None)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			port:          80,
			containerPort: 80,
			err:           false,
		},
		{
			name: "test success 3 (named port)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			port:          80,
			containerPort: 8080,
			err:           false,
		},
		{
			name: "test success (targetPort omitted)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{
							Port: 80,
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			port:          80,
			containerPort: 80,
			err:           false,
		},
		{
			name: "test failure 1 (cannot find a matching named port)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			port:          80,
			containerPort: -1,
			err:           true,
		},
		{
			name: "test failure 2 (cannot find a matching service port)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			port:          443,
			containerPort: 443,
			err:           true,
		},
		{
			name: "test failure 2 (cannot find a matching service port, but ClusterIP: None)",
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
					Ports: []v1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			port:          443,
			containerPort: 443,
			err:           true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			containerPort, err := LookupContainerPortNumberByServicePort(tt.svc, tt.pod, tt.port)
			if err != nil {
				if tt.err {
					if containerPort != tt.containerPort {
						t.Errorf("%v: expected port %v; got %v", tt.name, tt.containerPort, containerPort)
					}
					return
				}

				t.Errorf("%v: unexpected error: %v", tt.name, err)
				return
			}

			if tt.err {
				t.Errorf("%v: unexpected success", tt.name)
				return
			}

			if containerPort != tt.containerPort {
				t.Errorf("%v: expected port %v; got %v", tt.name, tt.containerPort, containerPort)
			}
		})
	}
}
