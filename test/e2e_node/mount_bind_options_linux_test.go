/*
Copyright 2025 The Kubernetes Authors.

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

package e2enode

import (
	"context"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Mount bind options [LinuxOnly]", framework.WithFeatureGate(features.VolumeBindMountOptions), func() {
	f := framework.NewDefaultFramework("mount-bind-options")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("bind mount options enforcement", func() {
		f.It("should enforce noexec, nosuid, nodev on disk-backed emptyDir", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bind-opts-disk-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: busyboxImage,
							Name:  "test",
							Command: []string{"sh", "-c",
								"grep ' /mnt ' /proc/self/mountinfo | grep noexec | grep nosuid | grep nodev && " +
									"cp /bin/sh /mnt/test.sh && chmod +x /mnt/test.sh && " +
									"! /mnt/test.sh -c 'echo should-not-run'",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:             "vol",
									MountPath:        "/mnt",
									BindMountOptions: []string{"noexec", "nosuid", "nodev"},
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name:         "vol",
							VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
						},
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
		})

		f.It("should enforce noexec on tmpfs emptyDir", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bind-opts-tmpfs-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: busyboxImage,
							Name:  "test",
							Command: []string{"sh", "-c",
								"grep ' /mnt ' /proc/self/mountinfo | grep noexec && " +
									"cp /bin/sh /mnt/test.sh && chmod +x /mnt/test.sh && " +
									"! /mnt/test.sh -c 'echo should-not-run'",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:             "vol",
									MountPath:        "/mnt",
									BindMountOptions: []string{"noexec"},
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
							},
						},
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
		})
	})

	ginkgo.Context("per-container granularity", func() {
		f.It("should allow different bind options per container for the same volume", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bind-opts-granularity-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Image:   busyboxImage,
							Name:    "setup",
							Command: []string{"sh", "-c", "cp /bin/sh /mnt/test.sh && chmod +x /mnt/test.sh"},
							VolumeMounts: []v1.VolumeMount{
								{Name: "shared", MountPath: "/mnt"},
							},
						},
						{
							Image: busyboxImage,
							Name:  "noexec-container",
							Command: []string{"sh", "-c",
								"! /mnt/test.sh -c 'echo should-not-run'",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:             "shared",
									MountPath:        "/mnt",
									BindMountOptions: []string{"noexec"},
								},
							},
						},
						{
							Image: busyboxImage,
							Name:  "exec-container",
							Command: []string{"sh", "-c",
								"/mnt/test.sh -c 'echo exec-allowed'",
							},
							VolumeMounts: []v1.VolumeMount{
								{Name: "shared", MountPath: "/mnt"},
							},
						},
					},
					Containers: []v1.Container{
						{
							Image:   busyboxImage,
							Name:    "done",
							Command: []string{"echo", "pass"},
						},
					},
					Volumes: []v1.Volume{
						{
							Name:         "shared",
							VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
						},
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
		})
	})

	ginkgo.Context("control test", func() {
		f.It("should allow execution when bindMountOptions is not set", func(ctx context.Context) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bind-opts-control-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: busyboxImage,
							Name:  "test",
							Command: []string{"sh", "-c",
								"cp /bin/sh /mnt/test.sh && chmod +x /mnt/test.sh && /mnt/test.sh -c 'echo exec-allowed'",
							},
							VolumeMounts: []v1.VolumeMount{
								{Name: "vol", MountPath: "/mnt"},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name:         "vol",
							VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
						},
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, pod)
			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
		})
	})
})
