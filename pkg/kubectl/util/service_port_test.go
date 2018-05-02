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

	"k8s.io/apimachinery/pkg/util/intstr"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestLookupContainerPortNumberByName(t *testing.T) {
	cases := []struct {
		name     string
		pod      api.Pod
		portname string
		portnum  int32
		err      bool
	}{
		{
			name: "test success 1",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			portname: "http",
			portnum:  int32(80),
			err:      false,
		},
		{
			name: "test faulure 1",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			portname: "www",
			portnum:  int32(0),
			err:      true,
		},
	}

	for _, tc := range cases {
		portnum, err := LookupContainerPortNumberByName(tc.pod, tc.portname)
		if err != nil {
			if tc.err {
				continue
			}

			t.Errorf("%v: unexpected error: %v", tc.name, err)
			continue
		}

		if tc.err {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		if portnum != tc.portnum {
			t.Errorf("%v: expected port number %v; got %v", tc.name, tc.portnum, portnum)
		}
	}
}

func TestLookupContainerPortNumberByServicePort(t *testing.T) {
	cases := []struct {
		name          string
		svc           api.Service
		pod           api.Pod
		port          int32
		containerPort int32
		err           bool
	}{
		{
			name: "test success 1 (int port)",
			svc: api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt(8080),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					ClusterIP: api.ClusterIPNone,
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt(8080),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port: 80,
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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
			svc: api.Service{
				Spec: api.ServiceSpec{
					ClusterIP: api.ClusterIPNone,
					Ports: []api.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
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

	for _, tc := range cases {
		containerPort, err := LookupContainerPortNumberByServicePort(tc.svc, tc.pod, tc.port)
		if err != nil {
			if tc.err {
				if containerPort != tc.containerPort {
					t.Errorf("%v: expected port %v; got %v", tc.name, tc.containerPort, containerPort)
				}
				continue
			}

			t.Errorf("%v: unexpected error: %v", tc.name, err)
			continue
		}

		if tc.err {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		if containerPort != tc.containerPort {
			t.Errorf("%v: expected port %v; got %v", tc.name, tc.containerPort, containerPort)
		}
	}
}
