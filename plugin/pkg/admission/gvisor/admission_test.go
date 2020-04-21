/*
Copyright 2020 The Kubernetes Authors.

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

package gvisor

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
)

func int64Ptr(p int64) *int64 {
	return &p
}

func boolPtr(p bool) *bool {
	return &p
}

func procMountTypePtr(p core.ProcMountType) *core.ProcMountType {
	return &p
}

func mountPropagationModePtr(p core.MountPropagationMode) *core.MountPropagationMode {
	return &p
}

func makePodCreateAttrs(pod *core.Pod) admission.Attributes {
	return admission.NewAttributesRecord(pod, nil, core.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, core.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})
}

func makePodUpdateAttrs(pod, oldPod *core.Pod) admission.Attributes {
	return admission.NewAttributesRecord(pod, oldPod, core.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, core.Resource("pods").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, &user.DefaultInfo{})
}

func makeRuntimeClassCreateAttrs(rc *node.RuntimeClass) admission.Attributes {
	return admission.NewAttributesRecord(rc, nil, core.Kind("RuntimeClass").WithVersion("version"), rc.Namespace, rc.Name, core.Resource("runtimeclasses").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})
}

func TestDeprecatedAnnotations(t *testing.T) {
	for name, test := range map[string]struct {
		annotations map[string]string
		expectErr   bool
	}{
		"runtime-handler.cri.kubernetes.io annotation": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "gvisor",
			},
			expectErr: true,
		},
		"io.kubernetes.cri.untrusted-workload annotation": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
			},
			expectErr: true,
		},
		"both gvisor annotations": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
				"runtime-handler.cri.kubernetes.io":    "gvisor",
			},
			expectErr: true,
		},
		"other annotation": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "false",
				"runtime-handler.cri.kubernetes.io":    "other",
			},
			expectErr: false,
		},
		"no annotation": {
			expectErr: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			pod := &core.Pod{}
			pod.Annotations = test.annotations

			err := checkDeprecatedAnnotation(pod)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateGVisorPod(t *testing.T) {
	for name, test := range map[string]struct {
		pod       core.Pod
		expectErr bool
	}{
		"regular pod": {
			pod:       core.Pod{},
			expectErr: false,
		},
		"pod with existing node selector": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					NodeSelector:     map[string]string{"other": "selector"},
				},
			},
			expectErr: false,
		},
		"pod with host path": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Volumes: []core.Volume{
						{
							Name: "test-host-path",
							VolumeSource: core.VolumeSource{
								HostPath: &core.HostPathVolumeSource{
									Path: "/test/host/path",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with host network": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						HostNetwork: true,
					},
				},
			},
			expectErr: true,
		},
		"pod with host pid": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						HostPID: true,
					},
				},
			},
			expectErr: true,
		},
		"pod with host ipc": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						HostIPC: true,
					},
				},
			},
			expectErr: true,
		},
		"pod with selinux options": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						SELinuxOptions: &core.SELinuxOptions{
							User:  "user",
							Role:  "role",
							Type:  "type",
							Level: "level",
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with FSGroup": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						FSGroup: int64Ptr(1234),
					},
				},
			},
			expectErr: true,
		},
		"pod with Sysctls": {
			pod: core.Pod{
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						Sysctls: []core.Sysctl{
							{
								Name:  "kernel.shm_rmid_forced",
								Value: "0",
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with empty SecurityContext container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:            "container",
							SecurityContext: &core.SecurityContext{},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with non-Privileged container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(false),
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with Privileged container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with SELinux container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								SELinuxOptions: &core.SELinuxOptions{
									User:  "user",
									Role:  "role",
									Type:  "type",
									Level: "level",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with non-AllowPrivilegeEscalation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								AllowPrivilegeEscalation: boolPtr(false),
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with AllowPrivilegeEscalation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								AllowPrivilegeEscalation: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with default ProcMount container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								ProcMount: procMountTypePtr(core.DefaultProcMount),
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with Unmasked ProcMount container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								ProcMount: procMountTypePtr(core.UnmaskedProcMount),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with VolumeDevices container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							VolumeDevices: []core.VolumeDevice{
								{
									Name:       "dev1",
									DevicePath: "/dev/dev1",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with VolumeMounts container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							VolumeMounts: []core.VolumeMount{
								{
									Name:      "volume1",
									MountPath: "/",
								},
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with None MountPropagation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							VolumeMounts: []core.VolumeMount{
								{
									Name:             "volume1",
									MountPath:        "/",
									MountPropagation: mountPropagationModePtr(core.MountPropagationNone),
								},
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with HostToContainer MountPropagation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							VolumeMounts: []core.VolumeMount{
								{
									Name:             "volume1",
									MountPath:        "/",
									MountPropagation: mountPropagationModePtr(core.MountPropagationHostToContainer),
								},
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with Bidirectional MountPropagation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name: "container",
							VolumeMounts: []core.VolumeMount{
								{
									Name:             "volume1",
									MountPath:        "/",
									MountPropagation: mountPropagationModePtr(core.MountPropagationBidirectional),
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"pod with Seccomp": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod": "test",
					},
				},
			},
			expectErr: true,
		},
		"pod with Seccomp container": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"container.seccomp.security.alpha.kubernetes.io/test": "test",
					},
				},
			},
			expectErr: true,
		},
		"pod with AppArmor container": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"container.apparmor.security.beta.kubernetes.io/test": "test",
					},
				},
			},
			expectErr: true,
		},
		"pod with non-Privileged init container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(false),
							},
						},
					},
				},
			},
			expectErr: false,
		},
		"pod with Privileged init container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateGVisorPod(&test.pod)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGvisor_Admit(t *testing.T) {
	gvisor := NewGvisor()

	createPodTests := map[string]struct {
		pod, expected core.Pod
		expectErr     bool
	}{
		"create regular pod": {
			pod:       core.Pod{},
			expectErr: false,
			expected:  core.Pod{},
		},
		"create pod with gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
		},
		"create pod with non-gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
		},
		"create gvisor pod with some disallowed options": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					SecurityContext: &core.PodSecurityContext{
						HostNetwork: true,
						HostPID:     true,
						HostIPC:     true,
						SELinuxOptions: &core.SELinuxOptions{
							User:  "user",
							Role:  "role",
							Type:  "type",
							Level: "level",
						},
						FSGroup: int64Ptr(1234),
						Sysctls: []core.Sysctl{
							{
								Name:  "kernel.shm_rmid_forced",
								Value: "0",
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container",
							VolumeDevices: []core.VolumeDevice{
								{
									Name:       "dev1",
									DevicePath: "/dev/dev1",
								},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":            "test",
						"container.seccomp.security.alpha.kubernetes.io/test": "test",
						"container.apparmor.security.beta.kubernetes.io/test": "test",
					},
				},
			},
			expectErr: true,
		},
	}

	for name, test := range createPodTests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodCreateAttrs(&test.pod)
			err := gvisor.Admit(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, test.pod)
			}
		})
	}

	gvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
		},
		Spec: core.PodSpec{
			Containers:       []core.Container{{Image: "my-image:v1"}},
			RuntimeClassName: stringPtr("gvisor"),
		},
	}
	deprecatedGvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
			Annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io":    "gvisor",
				"io.kubernetes.cri.untrusted-workload": "true",
			},
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	nativePod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "native-pod",
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	gvisorPodNewImage := gvisorPod.DeepCopy()
	gvisorPodNewImage.Spec.Containers[0].Image = "my-image:v2"

	updatePodTests := map[string]struct {
		oldPod, newPod, expected *core.Pod
		expectErr                bool
	}{
		"non-gvisor->non-gvisor": {
			oldPod:    nativePod.DeepCopy(),
			newPod:    nativePod.DeepCopy(),
			expectErr: false,
			expected:  nativePod.DeepCopy(),
		},
		"gvisor->deprecated-gvisor": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    deprecatedGvisorPod.DeepCopy(),
			expectErr: false, //  It's Validate's job to fail this case, not Admit's
			expected:  deprecatedGvisorPod.DeepCopy(),
		},
		"gvisor->modified-image": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    gvisorPodNewImage.DeepCopy(),
			expectErr: false,
			expected:  gvisorPodNewImage.DeepCopy(),
		},
	}
	for name, test := range updatePodTests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodUpdateAttrs(test.newPod, test.oldPod)
			err := gvisor.Admit(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, test.newPod)
			}
		})
	}

	gvisorPod = &core.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testname", Namespace: "testnamespace"},
		Spec: core.PodSpec{
			RuntimeClassName: stringPtr(gvisorRuntimeClass),
		},
	}
	expectedGvisorPod := gvisorPod.DeepCopy()

	otherTests := map[string]struct {
		obj         runtime.Object
		oldObj      runtime.Object
		kind        string
		namespace   string
		name        string
		resource    string
		subresource string
		operation   admission.Operation
		options     runtime.Object
		expectErr   bool
		expected    runtime.Object
	}{
		"other resource": {
			obj:       gvisorPod,
			kind:      "Foo",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "foos",
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: false,
			expected:  expectedGvisorPod,
		},
		"non-empty subresource": {
			obj:         gvisorPod,
			kind:        "Pod",
			namespace:   gvisorPod.Namespace,
			name:        gvisorPod.Name,
			resource:    "pods",
			subresource: "foo",
			operation:   admission.Create,
			options:     &metav1.CreateOptions{},
			expectErr:   false,
			expected:    expectedGvisorPod,
		},
		"non-create pod operation": {
			obj:       gvisorPod,
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Delete,
			options:   &metav1.DeleteOptions{},
			expectErr: false,
			expected:  expectedGvisorPod,
		},
		"create non-pod marked as kind pod": {
			obj:       &core.Service{},
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: true,
		},
	}

	for name, test := range otherTests {
		t.Run(name, func(t *testing.T) {
			attrs := admission.NewAttributesRecord(test.obj, test.oldObj, core.Kind(test.kind).WithVersion("version"), test.namespace, test.name, core.Resource(test.resource).WithVersion("version"), test.subresource, test.operation, test.options, false, nil)
			err := gvisor.Admit(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, test.obj)
			}
		})
	}
}

func TestAdmitPodCreate(t *testing.T) {
	tests := map[string]struct {
		pod       core.Pod
		expected  core.Pod
		expectErr bool
	}{
		"create regular pod": {
			pod:       core.Pod{},
			expectErr: false,
			expected:  core.Pod{},
		},
		"pod with deprecated annotation": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gvisor-pod",
					Annotations: map[string]string{
						"runtime-handler.cri.kubernetes.io":    "gvisor",
						"io.kubernetes.cri.untrusted-workload": "true",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Image: "my-image:v1"}},
				},
			},
			expectErr: true,
		},
		"create pod with gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
		},
		"create pod with non-gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
		},
		"gvisor pod with host path": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					Volumes: []core.Volume{
						{
							Name: "test-host-path",
							VolumeSource: core.VolumeSource{
								HostPath: &core.HostPathVolumeSource{
									Path: "/test/host/path",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"empty pod": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
		},
		"pod NET_RAW capability": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					InitContainers: []core.Container{
						{
							Name: "init-container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{},
								},
							},
						},
						{
							Name:            "init-container-without-capabilities",
							SecurityContext: &core.SecurityContext{},
						},
						{
							Name: "init-container-without-security-context",
						},
						{
							Name: "init-container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{},
								},
							},
						},
						{
							Name:            "container-without-capabilities",
							SecurityContext: &core.SecurityContext{},
						},
						{
							Name: "container-without-security-context",
						},
						{
							Name: "container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
				},
			},
			expectErr: false,
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					InitContainers: []core.Container{
						{
							Name: "init-container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-capabilities",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-security-context",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-capabilities",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-security-context",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodCreateAttrs(&test.pod)
			err := admitPodCreate(attrs)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, test.pod)
			}
		})
	}
}

func TestMutateGVisorPod(t *testing.T) {
	for name, test := range map[string]struct {
		pod      core.Pod
		expected core.Pod
	}{
		"empty pod": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
		},
		"pod NET_RAW capability": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					InitContainers: []core.Container{
						{
							Name: "init-container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{},
								},
							},
						},
						{
							Name:            "init-container-without-capabilities",
							SecurityContext: &core.SecurityContext{},
						},
						{
							Name: "init-container-without-security-context",
						},
						{
							Name: "init-container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{},
								},
							},
						},
						{
							Name:            "container-without-capabilities",
							SecurityContext: &core.SecurityContext{},
						},
						{
							Name: "container-without-security-context",
						},
						{
							Name: "container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
				},
			},
			expected: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					InitContainers: []core.Container{
						{
							Name: "init-container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-capabilities",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-without-security-context",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "init-container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container-with-net-raw-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-net-raw-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-net-raw",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-capabilities",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-without-security-context",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"NET_RAW"},
								},
							},
						},
						{
							Name: "container-with-all-caps-added",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Add: []core.Capability{"ALL"},
								},
							},
						},
						{
							Name: "container-with-all-caps-dropped",
							SecurityContext: &core.SecurityContext{
								Capabilities: &core.Capabilities{
									Drop: []core.Capability{"ALL"},
								},
							},
						},
					},
				},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			pod := test.pod
			mutateGVisorPod(&pod)
			assert.Equal(t, test.expected, pod)
		})
	}
}

func TestNodeSelectorConflict(t *testing.T) {
	pod := core.Pod{
		Spec: core.PodSpec{
			RuntimeClassName: stringPtr(gvisorRuntimeClass),
			NodeSelector:     map[string]string{gvisorNodeKey: "other"},
		},
	}
	err := validateGVisorPod(&pod)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "conflict:")
}

func createEmptyDir(name string, medium core.StorageMedium) core.Volume {
	return core.Volume{
		Name: name,
		VolumeSource: core.VolumeSource{
			EmptyDir: &core.EmptyDirVolumeSource{
				Medium: medium,
			},
		},
	}
}

func createContainer(name string, readonly bool, prop core.MountPropagationMode, volumes ...string) core.Container {
	c := core.Container{
		Name:         name,
		VolumeMounts: make([]core.VolumeMount, 0, len(volumes)),
	}
	for _, v := range volumes {
		c.VolumeMounts = append(c.VolumeMounts, core.VolumeMount{
			Name:             v,
			ReadOnly:         readonly,
			MountPropagation: mountPropagationModePtr(prop),
		})
	}
	return c
}

type annotation struct {
	name    string
	typ     string
	share   string
	options string
}

func createAnnotations(annons ...annotation) map[string]string {
	rv := map[string]string{}
	for _, a := range annons {
		rv["dev.gvisor.spec.mount."+a.name+".type"] = a.typ
		rv["dev.gvisor.spec.mount."+a.name+".share"] = a.share
		rv["dev.gvisor.spec.mount."+a.name+".options"] = a.options
	}
	return rv
}

func TestVolumeHints(t *testing.T) {
	type test struct {
		name       string
		volumes    []core.Volume
		containers []core.Container
		want       map[string]string
	}
	var tests []test

	// Add tests that are the same for all mediums.
	for _, medium := range []core.StorageMedium{core.StorageMediumDefault, core.StorageMediumMemory, core.StorageMediumHugePages} {
		typ, err := getMountType(medium)
		assert.NoError(t, err)

		tests = append(tests,
			test{
				name: typ + ": volume not used",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					createContainer("container", false, core.MountPropagationNone),
				},
				want: nil,
			},
			test{
				name: typ + ": access type mismatch",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					createContainer("container", false, core.MountPropagationNone),
					createContainer("container", true, core.MountPropagationNone),
				},
				want: nil,
			},
			test{
				name: typ + ": propagation mismatch",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					createContainer("container", false, core.MountPropagationNone),
					createContainer("container", false, core.MountPropagationHostToContainer),
				},
				want: nil,
			},
			test{
				name: typ + ": subpath",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					{
						Name: "container",
						VolumeMounts: []core.VolumeMount{
							{
								Name:    "empty",
								SubPath: "/subpath",
							},
						},
					},
				},
				want: nil,
			},
			test{
				name: typ + ": subpathexpr",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					{
						Name: "container",
						VolumeMounts: []core.VolumeMount{
							{
								Name:        "empty",
								SubPathExpr: "/subpath",
							},
						},
					},
				},
				want: nil,
			},
			test{
				name:    typ + ": not empty",
				volumes: []core.Volume{{Name: "non-empty"}},
				containers: []core.Container{
					createContainer("container", false, core.MountPropagationNone, "non-empty"),
				},
				want: nil,
			},
			test{
				name: typ + ": default propagation",
				volumes: []core.Volume{
					createEmptyDir("empty", medium),
				},
				containers: []core.Container{
					{
						Name:         "container",
						VolumeMounts: []core.VolumeMount{{Name: "empty"}},
					},
				},
				want: createAnnotations(annotation{
					name:    "empty",
					typ:     typ,
					share:   "container",
					options: "rw,rprivate",
				}),
			},
		)

		for _, readonly := range []bool{true, false} {
			for _, prop := range []core.MountPropagationMode{core.MountPropagationNone, core.MountPropagationHostToContainer} {
				options := "rw"
				if readonly {
					options = "ro"
				}
				if prop == core.MountPropagationNone {
					options += ",rprivate"
				} else {
					options += ",rslave"
				}

				tests = append(tests,
					test{
						name: fmt.Sprintf("%s+readonly(%t)+%s: container", typ, readonly, prop),
						volumes: []core.Volume{
							createEmptyDir("empty", medium),
						},
						containers: []core.Container{
							createContainer("container", readonly, prop, "empty"),
						},
						want: createAnnotations(annotation{
							name:    "empty",
							typ:     typ,
							share:   "container",
							options: options,
						}),
					},
					test{
						name: fmt.Sprintf("%s+readonly(%t)+%s: container + empty container", typ, readonly, prop),
						volumes: []core.Volume{
							createEmptyDir("empty", medium),
						},
						containers: []core.Container{
							createContainer("container1", readonly, prop, "empty"),
							createContainer("container2", readonly, prop),
						},
						want: createAnnotations(annotation{
							name:    "empty",
							typ:     typ,
							share:   "container",
							options: options,
						}),
					},
					test{
						name: fmt.Sprintf("%s+readonly(%t)+%s: pod", typ, readonly, prop),
						volumes: []core.Volume{
							createEmptyDir("empty", medium),
						},
						containers: []core.Container{
							createContainer("container1", readonly, prop, "empty"),
							createContainer("container2", readonly, prop, "empty"),
						},
						want: createAnnotations(annotation{
							name:    "empty",
							typ:     typ,
							share:   "pod",
							options: options,
						}),
					},
					test{
						name: fmt.Sprintf("%s+readonly(%t)+%s: two mounts", typ, readonly, prop),
						volumes: []core.Volume{
							createEmptyDir("empty1", medium),
							createEmptyDir("empty2", medium),
						},
						containers: []core.Container{
							createContainer("container1", readonly, prop, "empty1", "empty2"),
						},
						want: createAnnotations(
							annotation{
								name:    "empty1",
								typ:     typ,
								share:   "container",
								options: options,
							},
							annotation{
								name:    "empty2",
								typ:     typ,
								share:   "container",
								options: options,
							}),
					},
				)
			}
		}
	}

	tests = append(tests,
		test{
			name: "combo",
			volumes: []core.Volume{
				createEmptyDir("empty-default", core.StorageMediumDefault),
				createEmptyDir("empty-memory", core.StorageMediumMemory),
				createEmptyDir("empty-huge", core.StorageMediumHugePages),
				createEmptyDir("empty-memory-shared", core.StorageMediumMemory),
				{Name: "non-empty"},
			},
			containers: []core.Container{
				createContainer("container", false, core.MountPropagationNone, "empty-default", "empty-memory", "empty-huge", "non-empty"),
				createContainer("container", false, core.MountPropagationNone, "empty-memory-shared", "non-empty"),
				createContainer("container", false, core.MountPropagationNone, "empty-memory-shared"),
				createContainer("container", false, core.MountPropagationNone),
			},
			want: createAnnotations(
				annotation{
					name:    "empty-default",
					typ:     "bind",
					share:   "container",
					options: "rw,rprivate",
				},
				annotation{
					name:    "empty-memory",
					typ:     "tmpfs",
					share:   "container",
					options: "rw,rprivate",
				},
				annotation{
					name:    "empty-huge",
					typ:     "tmpfs",
					share:   "container",
					options: "rw,rprivate",
				},
				annotation{
					name:    "empty-memory-shared",
					typ:     "tmpfs",
					share:   "pod",
					options: "rw,rprivate",
				},
			),
		},
	)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for i := range tc.containers {
				// Shift containers to run with different orders.
				containers := append(tc.containers[i:], tc.containers[0:i]...)
				pod := core.Pod{
					Spec: core.PodSpec{
						Volumes:    tc.volumes,
						Containers: containers,
					},
				}
				mutateGVisorPod(&pod)
				assert.Equal(t, tc.want, pod.Annotations)

				// Now make one of the containers an init container. End result should
				// be the same.
				pod.Spec.InitContainers = []core.Container{containers[0]}
				pod.Spec.Containers = containers[1:]
				mutateGVisorPod(&pod)
				assert.Equal(t, tc.want, pod.Annotations)
			}
		})
	}
}

func TestGvisor_Validate(t *testing.T) {
	gvisor := NewGvisor()

	createPodTests := map[string]struct {
		pod       core.Pod
		expectErr bool
	}{
		"create regular pod": {
			pod:       core.Pod{},
			expectErr: false,
		},
		"pod with deprecated annotation": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gvisor-pod",
					Annotations: map[string]string{
						"runtime-handler.cri.kubernetes.io":    "gvisor",
						"io.kubernetes.cri.untrusted-workload": "true",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Image: "my-image:v1"}},
				},
			},
			expectErr: true,
		},
		"create pod with gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expectErr: false,
		},
		"create pod with non-gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
			expectErr: false,
		},
		"create gvisor pod with some disallowed options": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					SecurityContext: &core.PodSecurityContext{
						HostNetwork: true,
						HostPID:     true,
						HostIPC:     true,
						SELinuxOptions: &core.SELinuxOptions{
							User:  "user",
							Role:  "role",
							Type:  "type",
							Level: "level",
						},
						FSGroup: int64Ptr(1234),
						Sysctls: []core.Sysctl{
							{
								Name:  "kernel.shm_rmid_forced",
								Value: "0",
							},
						},
					},
					Containers: []core.Container{
						{
							Name: "container",
							VolumeDevices: []core.VolumeDevice{
								{
									Name:       "dev1",
									DevicePath: "/dev/dev1",
								},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":            "test",
						"container.seccomp.security.alpha.kubernetes.io/test": "test",
						"container.apparmor.security.beta.kubernetes.io/test": "test",
					},
				},
			},
			expectErr: true,
		},
	}

	for name, test := range createPodTests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodCreateAttrs(&test.pod)
			err := gvisor.Validate(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	gvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
		},
		Spec: core.PodSpec{
			Containers:       []core.Container{{Image: "my-image:v1"}},
			RuntimeClassName: stringPtr("gvisor"),
		},
	}
	deprecatedGvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
			Annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io":    "gvisor",
				"io.kubernetes.cri.untrusted-workload": "true",
			},
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	nativePod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "native-pod",
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	gvisorPodNewImage := gvisorPod.DeepCopy()
	gvisorPodNewImage.Spec.Containers[0].Image = "my-image:v2"

	updatePodTests := map[string]struct {
		oldPod, newPod *core.Pod
		expectErr      bool
	}{
		"non-gvisor->non-gvisor": {
			oldPod:    nativePod.DeepCopy(),
			newPod:    nativePod.DeepCopy(),
			expectErr: false,
		},
		"gvisor->deprecated-gvisor": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    deprecatedGvisorPod.DeepCopy(),
			expectErr: true,
		},
		"gvisor->modified-image": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    gvisorPodNewImage.DeepCopy(),
			expectErr: false,
		},
	}
	for name, test := range updatePodTests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodUpdateAttrs(test.newPod, test.oldPod)
			err := gvisor.Validate(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	runtimeclassTests := map[string]struct {
		obj       *node.RuntimeClass
		oldObj    runtime.Object
		operation admission.Operation
		options   runtime.Object
		expectErr bool
	}{
		"create gvisor rtc": {
			obj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
				Handler:    "gvisor",
			},
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: false,
		},
		"update gvisor rtc": {
			obj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
			},
			oldObj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
				Handler:    "gvisor",
			},
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			expectErr: false,
		},
		"delete gvisor rtc": {
			obj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
			},
			operation: admission.Delete,
			options:   &metav1.DeleteOptions{},
			expectErr: false,
		},
		"create with non-gvisor handler": {
			obj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "gvisor"},
				Handler:    "foo",
			},
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: true,
		},
		"create non-gvisor runtimeclass": {
			obj: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
			},
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: false,
		},
	}

	for testName, test := range runtimeclassTests {
		t.Run(testName, func(t *testing.T) {
			attrs := admission.NewAttributesRecord(test.obj, test.oldObj, node.Kind("RuntimeClass").WithVersion("version"), "testnamespace", test.obj.Name, node.Resource("runtimeclasses").WithVersion("version"), "", test.operation, test.options, false, &user.DefaultInfo{})
			err := gvisor.Validate(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	gvisorPod = &core.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testname", Namespace: "testnamespace"},
		Spec: core.PodSpec{
			RuntimeClassName: stringPtr(gvisorRuntimeClass),
		},
	}

	otherTests := map[string]struct {
		obj         runtime.Object
		oldObj      runtime.Object
		kind        string
		namespace   string
		name        string
		resource    string
		subresource string
		operation   admission.Operation
		options     runtime.Object
		expectErr   bool
	}{
		"other resource": {
			obj:       gvisorPod,
			kind:      "Foo",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "foos",
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: false,
		},
		"non-empty subresource": {
			obj:         gvisorPod,
			kind:        "Pod",
			namespace:   gvisorPod.Namespace,
			name:        gvisorPod.Name,
			resource:    "pods",
			subresource: "foo",
			operation:   admission.Create,
			options:     &metav1.CreateOptions{},
			expectErr:   false,
		},
		"non-create/update pod operation": {
			obj:       gvisorPod,
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Delete,
			options:   &metav1.DeleteOptions{},
			expectErr: false,
		},
		"create non-pod marked as kind pod": {
			obj:       &core.Service{},
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			expectErr: true,
		},
		"update non-pod marked as kind pod": {
			obj:       gvisorPod,
			oldObj:    &core.Service{},
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			expectErr: true,
		},
		"update pod to non-pod marked as kind pod": {
			obj:       &core.Service{},
			oldObj:    gvisorPod,
			kind:      "Pod",
			namespace: gvisorPod.Namespace,
			name:      gvisorPod.Name,
			resource:  "pods",
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			expectErr: true,
		},
	}

	for name, test := range otherTests {
		t.Run(name, func(t *testing.T) {
			attrs := admission.NewAttributesRecord(test.obj, test.oldObj, core.Kind(test.kind).WithVersion("version"), test.namespace, test.name, core.Resource(test.resource).WithVersion("version"), test.subresource, test.operation, test.options, false, nil)
			err := gvisor.Validate(context.TODO(), attrs, nil)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	t.Run("non-rtc object marked as kind rtc", func(t *testing.T) {
		attrs := admission.NewAttributesRecord(gvisorPod, nil, node.Kind("RuntimeClass").WithVersion("version"), "testnamespace", gvisorRuntimeClass, node.Resource("runtimeclasses").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
		err := gvisor.Validate(context.TODO(), attrs, nil)
		assert.Error(t, err)
	})
}

// non-pod
// no runtimeclass or difference runtimeclass
// some subset of validation tests
func TestValidatePodCreate(t *testing.T) {
	tests := map[string]struct {
		pod       core.Pod
		expectErr bool
	}{
		"create regular pod": {
			pod:       core.Pod{},
			expectErr: false,
		},
		"pod with deprecated annotation": {
			pod: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gvisor-pod",
					Annotations: map[string]string{
						"runtime-handler.cri.kubernetes.io":    "gvisor",
						"io.kubernetes.cri.untrusted-workload": "true",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Image: "my-image:v1"}},
				},
			},
			expectErr: true,
		},
		"create pod with gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			expectErr: false,
		},
		"create pod with non-gvisor runtimeclass": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr("other"),
				},
			},
			expectErr: false,
		},
		"create gvisor pod with existing node selector": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					NodeSelector:     map[string]string{"other": "selector"},
				},
			},
			expectErr: false,
		},
		"create gvisor pod with non-gvisor runtime node selector": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					NodeSelector:     map[string]string{gvisorNodeKey: "other"},
				},
			},
			expectErr: true,
		},
		"gvisor pod with host path": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					Volumes: []core.Volume{
						{
							Name: "test-host-path",
							VolumeSource: core.VolumeSource{
								HostPath: &core.HostPathVolumeSource{
									Path: "/test/host/path",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with disallowed security context options": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					SecurityContext: &core.PodSecurityContext{
						HostNetwork: true,
						HostPID:     true,
						HostIPC:     true,
						SELinuxOptions: &core.SELinuxOptions{
							User:  "user",
							Role:  "role",
							Type:  "type",
							Level: "level",
						},
						FSGroup: int64Ptr(1234),
						Sysctls: []core.Sysctl{
							{
								Name:  "kernel.shm_rmid_forced",
								Value: "0",
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with Privileged container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with Privileged init container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					InitContainers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								Privileged: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with AllowPrivilegeEscalation container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					Containers: []core.Container{
						{
							Name: "container",
							SecurityContext: &core.SecurityContext{
								AllowPrivilegeEscalation: boolPtr(true),
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with VolumeDevices container": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
					Containers: []core.Container{
						{
							Name: "container",
							VolumeDevices: []core.VolumeDevice{
								{
									Name:       "dev1",
									DevicePath: "/dev/dev1",
								},
							},
						},
					},
				},
			},
			expectErr: true,
		},
		"gvisor pod with disallowed annotations": {
			pod: core.Pod{
				Spec: core.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":            "test",
						"container.seccomp.security.alpha.kubernetes.io/test": "test",
						"container.apparmor.security.beta.kubernetes.io/test": "test",
					},
				},
			},
			expectErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			err := validatePodCreate(makePodCreateAttrs(&test.pod))
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidatePodUpdate(t *testing.T) {
	gvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
		},
		Spec: core.PodSpec{
			Containers:       []core.Container{{Image: "my-image:v1"}},
			RuntimeClassName: stringPtr("gvisor"),
		},
	}
	deprecatedGvisorPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gvisor-pod",
			Annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io":    "gvisor",
				"io.kubernetes.cri.untrusted-workload": "true",
			},
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	nativePod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "native-pod",
		},
		Spec: core.PodSpec{
			Containers: []core.Container{{Image: "my-image:v1"}},
		},
	}
	gvisorPodNewImage := gvisorPod.DeepCopy()
	gvisorPodNewImage.Spec.Containers[0].Image = "my-image:v2"

	tests := map[string]struct {
		oldPod, newPod *core.Pod
		expectErr      bool
	}{
		"non-gvisor->non-gvisor": {
			oldPod:    nativePod.DeepCopy(),
			newPod:    nativePod.DeepCopy(),
			expectErr: false,
		},
		"gvisor->deprecated-gvisor": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    deprecatedGvisorPod.DeepCopy(),
			expectErr: true,
		},
		"gvisor->modified-image": {
			oldPod:    gvisorPod.DeepCopy(),
			newPod:    gvisorPodNewImage.DeepCopy(),
			expectErr: false,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			attrs := makePodUpdateAttrs(test.newPod, test.oldPod)

			err := validatePodUpdate(attrs)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateRuntimeClass(t *testing.T) {
	tests := []struct {
		name, handler string
		expectErr     bool
	}{
		{"", "gvisor", false},
		{"foo", "gvisor", false},
		{"foo", "bar", false},
		{"foo", "", false},
		{"gvisor", "gvisor", false},
		{"gvisor", "bar", true},
		{"gvisor", "", true},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s-%s-beta", test.name, test.handler), func(t *testing.T) {

			rc := &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: test.name},
				Handler:    test.handler,
			}
			attrs := makeRuntimeClassCreateAttrs(rc)
			err := validateRuntimeClass(attrs)
			if test.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
