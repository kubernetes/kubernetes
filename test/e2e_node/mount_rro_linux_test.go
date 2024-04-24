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

package e2enode

import (
	"context"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// Usage:
// make test-e2e-node TEST_ARGS='--service-feature-gates=RecursiveReadOnlyMounts=true --kubelet-flags="--feature-gates=RecursiveReadOnlyMounts=true"' FOCUS="Mount recursive read-only" SKIP=""
var _ = SIGDescribe("Mount recursive read-only [LinuxOnly]", framework.WithSerial(), nodefeature.RecursiveReadOnlyMounts, func() {
	f := framework.NewDefaultFramework("mount-rro")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Describe("Mount recursive read-only", func() {
		ginkgo.Context("when the runtime supports recursive read-only mounts", func() {
			f.It("should accept recursive read-only mounts", func(ctx context.Context) {
				ginkgo.By("waiting for the node to be ready", func() {
					waitForNodeReady(ctx)
					if !supportsRRO(ctx, f) {
						e2eskipper.Skipf("runtime does not support recursive read-only mounts")
					}
				}) // By
				var pod *v1.Pod
				ginkgo.By("creating a pod", func() {
					pod = e2epod.NewPodClient(f).Create(ctx,
						podForRROSupported("mount-rro-"+string(uuid.NewUUID()), f.Namespace.Name))
					framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
					var err error
					pod, err = f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
				}) // By
				ginkgo.By("checking containerStatuses.volumeMounts", func() {
					gomega.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(3)) // "mount", "test", "unmount"
					volMountStatuses := pod.Status.InitContainerStatuses[1].VolumeMounts
					var verifiedVolMountStatuses int
					for _, f := range volMountStatuses {
						switch f.Name {
						case "mnt":
							switch f.MountPath {
							case "/mnt-rro", "/mnt-rro-if-possible":
								gomega.Expect(*f.RecursiveReadOnly).To(gomega.Equal(v1.RecursiveReadOnlyEnabled))
								verifiedVolMountStatuses++
							case "/mnt-rro-disabled", "/mnt-ro":
								gomega.Expect(*f.RecursiveReadOnly).To(gomega.Equal(v1.RecursiveReadOnlyDisabled))
								verifiedVolMountStatuses++
							case "/mnt-rw":
								gomega.Expect(f.RecursiveReadOnly).To(gomega.BeNil())
								verifiedVolMountStatuses++
							default:
								framework.Failf("unexpected mount path: %q", f.MountPath)
							}
						default: // implicit secret volumes, etc.
							// NOP
						}
					}
					gomega.Expect(verifiedVolMountStatuses).To(gomega.Equal(5))
				}) // By
			}) // It
			f.It("should reject invalid recursive read-only mounts", func(ctx context.Context) {
				ginkgo.By("waiting for the node to be ready", func() {
					waitForNodeReady(ctx)
					if !supportsRRO(ctx, f) {
						e2eskipper.Skipf("runtime does not support recursive read-only mounts")
					}
				}) // By
				ginkgo.By("specifying RRO without RO", func() {
					pod := &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "mount-rro-invalid-" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image:   busyboxImage,
									Name:    "busybox",
									Command: []string{"echo", "this container should fail"},
									VolumeMounts: []v1.VolumeMount{
										{
											Name:              "mnt",
											MountPath:         "/mnt",
											RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyEnabled),
										},
									},
								},
							},
							Volumes: []v1.Volume{
								{
									Name: "mnt",
									VolumeSource: v1.VolumeSource{
										EmptyDir: &v1.EmptyDirVolumeSource{},
									},
								},
							},
						},
					}
					_, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
					gomega.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("spec.containers[0].volumeMounts.recursiveReadOnly: Forbidden: may only be specified when readOnly is true")))
				}) // By
				// See also the unit test [pkg/kubelet.TestResolveRecursiveReadOnly] for more invalid conditions (e.g., incompatible mount propagation)
			}) // It
		}) // Context
		ginkgo.Context("when the runtime does not support recursive read-only mounts", func() {
			f.It("should accept non-recursive read-only mounts", func(ctx context.Context) {
				e2eskipper.SkipUnlessFeatureGateEnabled(features.RecursiveReadOnlyMounts)
				ginkgo.By("waiting for the node to be ready", func() {
					waitForNodeReady(ctx)
					if supportsRRO(ctx, f) {
						e2eskipper.Skipf("runtime supports recursive read-only mounts")
					}
				}) // By
				var pod *v1.Pod
				ginkgo.By("creating a pod", func() {
					pod = e2epod.NewPodClient(f).Create(ctx,
						podForRROUnsupported("mount-ro-"+string(uuid.NewUUID()), f.Namespace.Name))
					framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
					var err error
					pod, err = f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
				}) // By
				ginkgo.By("checking containerStatuses.volumeMounts", func() {
					gomega.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(3)) // "mount", "test", "unmount"
					volMountStatuses := pod.Status.InitContainerStatuses[1].VolumeMounts
					var verifiedVolMountStatuses int
					for _, f := range volMountStatuses {
						switch f.Name {
						case "mnt":
							switch f.MountPath {
							case "/mnt-rro-if-possible", "/mnt-rro-disabled", "/mnt-ro":
								gomega.Expect(*f.RecursiveReadOnly).To(gomega.Equal(v1.RecursiveReadOnlyDisabled))
								verifiedVolMountStatuses++
							case "/mnt-rw":
								gomega.Expect(f.RecursiveReadOnly).To(gomega.BeNil())
								verifiedVolMountStatuses++
							default:
								framework.Failf("unexpected mount path: %q", f.MountPath)
							}
						default: // implicit secret volumes, etc.
							// NOP
						}
					}
					gomega.Expect(verifiedVolMountStatuses).To(gomega.Equal(4))
				}) // By
			}) // It
			f.It("should reject recursive read-only mounts", func(ctx context.Context) {
				e2eskipper.SkipUnlessFeatureGateEnabled(features.RecursiveReadOnlyMounts)
				ginkgo.By("waiting for the node to be ready", func() {
					waitForNodeReady(ctx)
					if supportsRRO(ctx, f) {
						e2eskipper.Skipf("runtime supports recursive read-only mounts")
					}
				}) // By
				ginkgo.By("specifying RRO explicitly", func() {
					pod := &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "mount-rro-unsupported-" + string(uuid.NewUUID()),
							Namespace: f.Namespace.Name,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image:   busyboxImage,
									Name:    "busybox",
									Command: []string{"echo", "this container should fail"},
									VolumeMounts: []v1.VolumeMount{
										{
											Name:              "mnt",
											MountPath:         "/mnt",
											ReadOnly:          true,
											RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyEnabled),
										},
									},
								},
							},
							Volumes: []v1.Volume{
								{
									Name: "mnt",
									VolumeSource: v1.VolumeSource{
										EmptyDir: &v1.EmptyDirVolumeSource{},
									},
								},
							},
						},
					}
					pod = e2epod.NewPodClient(f).Create(ctx, pod)
					framework.ExpectNoError(e2epod.WaitForPodContainerToFail(ctx, f.ClientSet, pod.Namespace, pod.Name, 0, "CreateContainerConfigError", framework.PodStartShortTimeout))
					var err error
					pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)
					gomega.Expect(pod.Status.ContainerStatuses[0].State.Waiting.Message).To(
						gomega.ContainSubstring("failed to resolve recursive read-only mode: volume \"mnt\" requested recursive read-only mode, but it is not supported by the runtime"))
				}) // By
			}) // It
		}) // Context
	}) // Describe
}) // SIGDescribe

func supportsRRO(ctx context.Context, f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
	node := nodeList.Items[0]
	for _, f := range node.Status.RuntimeHandlers {
		if f.Name == "" && f.Features != nil && *f.Features.RecursiveReadOnlyMounts {
			return true
		}
	}
	return false
}

func podForRROSupported(name, ns string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			InitContainers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "mount",
					Command: []string{"sh", "-euxc", "mkdir -p /mnt/tmpfs && mount -t tmpfs none /mnt/tmpfs"},
					SecurityContext: &v1.SecurityContext{
						Privileged: ptr.To(true),
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:             "mnt",
							MountPath:        "/mnt",
							MountPropagation: ptr.To(v1.MountPropagationBidirectional),
						},
					},
				},
				{
					Image: busyboxImage,
					Name:  "test",
					Command: []string{"sh", "-euxc", `
for f in rro rro-if-possible; do touch /mnt-$f/tmpfs/foo 2>&1 | grep "Read-only"; done
for f in rro-disabled ro rw; do touch /mnt-$f/tmpfs/foo; done
`},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:              "mnt",
							MountPath:         "/mnt-rro",
							ReadOnly:          true,
							MountPropagation:  ptr.To(v1.MountPropagationNone), // explicit
							RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyEnabled),
						},
						{
							Name:              "mnt",
							MountPath:         "/mnt-rro-if-possible",
							ReadOnly:          true,
							RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyIfPossible),
						},
						{
							Name:              "mnt",
							MountPath:         "/mnt-rro-disabled",
							ReadOnly:          true,
							RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyDisabled), // explicit
						},
						{
							Name:      "mnt",
							MountPath: "/mnt-ro",
							ReadOnly:  true,
						},
						{
							Name:      "mnt",
							MountPath: "/mnt-rw",
						},
					},
				},
				{
					Image:   busyboxImage,
					Name:    "unmount",
					Command: []string{"umount", "/mnt/tmpfs"},
					SecurityContext: &v1.SecurityContext{
						Privileged: ptr.To(true),
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:             "mnt",
							MountPath:        "/mnt",
							MountPropagation: ptr.To(v1.MountPropagationBidirectional),
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "completion",
					Command: []string{"echo", "OK"},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "mnt",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
}

func podForRROUnsupported(name, ns string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			InitContainers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "mount",
					Command: []string{"sh", "-euxc", "mkdir -p /mnt/tmpfs && mount -t tmpfs none /mnt/tmpfs"},
					SecurityContext: &v1.SecurityContext{
						Privileged: ptr.To(true),
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:             "mnt",
							MountPath:        "/mnt",
							MountPropagation: ptr.To(v1.MountPropagationBidirectional),
						},
					},
				},
				{
					Image: busyboxImage,
					Name:  "test",
					Command: []string{"sh", "-euxc", `
for f in rro-if-possible rro-disabled ro rw; do touch /mnt-$f/tmpfs/foo; done
`},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:              "mnt",
							MountPath:         "/mnt-rro-if-possible",
							ReadOnly:          true,
							RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyIfPossible),
						},
						{
							Name:              "mnt",
							MountPath:         "/mnt-rro-disabled",
							ReadOnly:          true,
							RecursiveReadOnly: ptr.To(v1.RecursiveReadOnlyDisabled), // explicit
						},
						{
							Name:      "mnt",
							MountPath: "/mnt-ro",
							ReadOnly:  true,
						},
						{
							Name:      "mnt",
							MountPath: "/mnt-rw",
						},
					},
				},
				{
					Image:   busyboxImage,
					Name:    "unmount",
					Command: []string{"umount", "/mnt/tmpfs"},
					SecurityContext: &v1.SecurityContext{
						Privileged: ptr.To(true),
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:             "mnt",
							MountPath:        "/mnt",
							MountPropagation: ptr.To(v1.MountPropagationBidirectional),
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Image:   busyboxImage,
					Name:    "completion",
					Command: []string{"echo", "OK"},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "mnt",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
}
