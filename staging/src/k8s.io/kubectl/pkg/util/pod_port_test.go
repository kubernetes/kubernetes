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
)

func TestLookupContainerPortNumberByName(t *testing.T) {
	tests := []struct {
		name     string
		pod      v1.Pod
		portname string
		portnum  int32
		err      bool
	}{
		{
			name: "test success 1",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
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
			portname: "www",
			portnum:  int32(0),
			err:      true,
		},
		{
			name: "test success 2",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "sql",
									ContainerPort: int32(3306)},
							},
						},
					},
				},
			},
			portname: "sql",
			portnum:  int32(3306),
			err:      false,
		}, {
			name: "test failure 2",
			pod: v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
					InitContainers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									Name:          "sql",
									ContainerPort: int32(3306)},
							},
						},
					},
				},
			},
			portname: "metrics",
			portnum:  int32(0),
			err:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			portnum, err := LookupContainerPortNumberByName(tt.pod, tt.portname)
			if err != nil {
				if tt.err {
					return
				}

				t.Errorf("%v: unexpected error: %v", tt.name, err)
				return
			}

			if tt.err {
				t.Errorf("%v: unexpected success", tt.name)
				return
			}

			if portnum != tt.portnum {
				t.Errorf("%v: expected port number %v; got %v", tt.name, tt.portnum, portnum)
			}
		})
	}
}
