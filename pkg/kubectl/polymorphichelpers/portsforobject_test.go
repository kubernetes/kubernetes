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

package polymorphichelpers

import (
	"testing"

	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestPortsForObject(t *testing.T) {
	tests := []struct {
		object    runtime.Object
		expectErr bool
	}{
		{
			object: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Ports: []api.ContainerPort{
								{
									ContainerPort: 101,
								},
							},
						},
					},
				},
			},
		},
		{
			object: &api.Service{
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Port: 101,
						},
					},
				},
			},
		},
		{
			object: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Ports: []api.ContainerPort{
										{
											ContainerPort: 101,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			object: &extensions.Deployment{
				Spec: extensions.DeploymentSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Ports: []api.ContainerPort{
										{
											ContainerPort: 101,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			object: &extensions.ReplicaSet{
				Spec: extensions.ReplicaSetSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Ports: []api.ContainerPort{
										{
											ContainerPort: 101,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			object:    &api.Node{},
			expectErr: true,
		},
	}
	expectedPorts := []string{"101"}

	for _, test := range tests {
		actual, err := portsForObject(test.object)
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(actual, expectedPorts) {
			t.Errorf("expected ports %v, but got %v", expectedPorts, actual)
		}
	}
}
