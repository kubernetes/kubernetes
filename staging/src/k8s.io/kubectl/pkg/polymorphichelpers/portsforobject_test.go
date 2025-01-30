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

	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestPortsForObject(t *testing.T) {
	tests := []struct {
		object    runtime.Object
		expectErr bool
		expected  []string
	}{
		{
			object: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 101,
								},
							},
						},
					},
				},
			},
			expected: []string{"101"},
		},
		{
			object: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 101,
								},
								{
									ContainerPort: 102,
								},
							},
						},
					},
				},
			},
			expected: []string{"101", "102"},
		},
		{
			object: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 101,
									Protocol:      corev1.ProtocolTCP,
								},
								{
									ContainerPort: 101,
									Protocol:      corev1.ProtocolUDP,
								},
								{
									ContainerPort: 102,
								},
							},
						},
					},
				},
			},
			expected: []string{"101", "102"},
		},
		{
			object: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 101,
									Protocol:      corev1.ProtocolTCP,
								},
								{
									ContainerPort: 102,
								},
								{
									ContainerPort: 101,
									Protocol:      corev1.ProtocolUDP,
								},
							},
						},
					},
				},
			},
			expected: []string{"101", "102"},
		},
		{
			object: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port: 101,
						},
					},
				},
			},
			expected: []string{"101"},
		},
		{
			object: &corev1.ReplicationController{
				Spec: corev1.ReplicationControllerSpec{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Ports: []corev1.ContainerPort{
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
			expected: []string{"101"},
		},
		{
			object: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Ports: []corev1.ContainerPort{
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
			expected: []string{"101"},
		},
		{
			object: &extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Ports: []corev1.ContainerPort{
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
			expected: []string{"101"},
		},
		{
			object:    &corev1.Node{},
			expectErr: true,
		},
	}

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
		if !reflect.DeepEqual(actual, test.expected) {
			t.Errorf("expected ports %v, but got %v", test.expected, actual)
		}
	}
}
