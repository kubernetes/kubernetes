/*
Copyright 2014 The Kubernetes Authors.

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

package securitycontext

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestInternalDetermineEffectiveSecurityContext(t *testing.T) {
	privileged := true
	readOnlyRootFilesystem := false
	runAsUser := int64(1)
	runAsRoot := true
	runAsNonRoot := true

	tests := map[string]struct {
		pod       *v1.Pod
		expect    *v1.SecurityContext
	}{
		// pod has PodSecurityContext, container has SecurityContext and need privileged
		"test1": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "ctr",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							SecurityContext: &v1.SecurityContext{
								Privileged:             &privileged,
								ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
								Capabilities: &v1.Capabilities{
									Add:  []v1.Capability{"foo"},
									Drop: []v1.Capability{"bar"},
								},
								SELinuxOptions: &v1.SELinuxOptions{
									User:  "container_user",
									Role:  "container_role",
									Type:  "container_type",
									Level: "container_level",
								},
								RunAsUser:    &runAsUser,
								RunAsNonRoot: &runAsRoot,
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{
						SELinuxOptions: &v1.SELinuxOptions{
							User:  "pod_user",
							Role:  "pod_role",
							Type:  "pod_type",
							Level: "pod_level",
						},
						RunAsUser:    &runAsUser,
						RunAsNonRoot: &runAsNonRoot,
					},
				},
			},
			expect: &v1.SecurityContext{
				Privileged:             &privileged,
				ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
				Capabilities: &v1.Capabilities{
					Add:  []v1.Capability{"foo"},
					Drop: []v1.Capability{"bar"},
				},
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "container_user",
					Role:  "container_role",
					Type:  "container_type",
					Level: "container_level",
				},
				RunAsUser:    &runAsUser,
				RunAsNonRoot: &runAsRoot,
			},
		},
		// pod has PodSecurityContext, container has SecurityContext and need no privileged
		"test2": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "ctr",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							SecurityContext: &v1.SecurityContext{
								ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
								Capabilities: &v1.Capabilities{
									Add:  []v1.Capability{"foo"},
									Drop: []v1.Capability{"bar"},
								},
								SELinuxOptions: &v1.SELinuxOptions{
									User:  "container_user",
									Role:  "container_role",
									Type:  "container_type",
									Level: "container_level",
								},
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{
						SELinuxOptions: &v1.SELinuxOptions{
							User:  "pod_user",
							Role:  "pod_role",
							Type:  "pod_type",
							Level: "pod_level",
						},
						RunAsUser:    &runAsUser,
						RunAsNonRoot: &runAsNonRoot,
					},
				},
			},
			expect: &v1.SecurityContext{
				Privileged: nil,
				ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
				Capabilities: &v1.Capabilities{
					Add:  []v1.Capability{"foo"},
					Drop: []v1.Capability{"bar"},
				},
				SELinuxOptions: &v1.SELinuxOptions{
					User:  "container_user",
					Role:  "container_role",
					Type:  "container_type",
					Level: "container_level",
				},
				RunAsUser:    &runAsUser,
				RunAsNonRoot: &runAsNonRoot,
			},
		},
	}

	for k, v := range tests {
		containers := v.pod.Spec.Containers
		actual := DetermineEffectiveSecurityContext(v.pod, &containers[0])
		if !reflect.DeepEqual(actual, v.expect) {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}
}
