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
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/opencontainers/selinux/go-selinux"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Run this single test locally using a running CRI-O instance by:
// make test-e2e-node CONTAINER_RUNTIME_ENDPOINT="unix:///var/run/crio/crio.sock" TEST_ARGS='--ginkgo.focus="ImageVolume" --feature-gates=ImageVolume=true --service-feature-gates=ImageVolume=true --kubelet-flags="--cgroup-root=/ --runtime-cgroups=/system.slice/crio.service --kubelet-cgroups=/system.slice/kubelet.service --fail-swap-on=false"'
var _ = SIGDescribe("ImageVolume", nodefeature.ImageVolume, feature.ImageVolume, func() {
	f := framework.NewDefaultFramework("image-volume-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	const (
		podName             = "test-pod"
		containerName       = "test-container"
		validImageRef       = "quay.io/crio/artifact:v1"
		invalidImageRef     = "localhost/invalid"
		volumeName          = "volume"
		volumePathPrefix    = "/volume"
		defaultSELinuxUser  = "system_u"
		defaultSELinuxRole  = "system_r"
		defaultSELinuxType  = "svirt_lxc_net_t"
		defaultSELinuxLevel = "s0:c1,c5"
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessFeatureGateEnabled(features.ImageVolume)
	})

	createPod := func(ctx context.Context, podName, nodeName string, volumes []v1.Volume, volumeMounts []v1.VolumeMount, selinuxOptions *v1.SELinuxOptions) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				NodeName:      nodeName,
				RestartPolicy: v1.RestartPolicyAlways,
				SecurityContext: &v1.PodSecurityContext{
					SELinuxOptions: selinuxOptions,
				},
				Containers: []v1.Container{
					{
						Name:         containerName,
						Image:        busyboxImage,
						Command:      ExecCommand(podName, execCommand{LoopForever: true}),
						VolumeMounts: volumeMounts,
					},
				},
				Volumes: volumes,
			},
		}

		ginkgo.By(fmt.Sprintf("Creating a pod (%s/%s)", f.Namespace.Name, podName))
		e2epod.NewPodClient(f).Create(ctx, pod)

	}

	f.It("should succeed with pod and pull policy of Always", func(ctx context.Context) {
		var selinuxOptions *v1.SELinuxOptions
		if selinux.GetEnabled() {
			selinuxOptions = &v1.SELinuxOptions{
				User:  defaultSELinuxUser,
				Role:  defaultSELinuxRole,
				Type:  defaultSELinuxType,
				Level: defaultSELinuxLevel,
			}
			ginkgo.By(fmt.Sprintf("Using SELinux on pod: %v", selinuxOptions))
		}

		createPod(ctx,
			podName,
			"",
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef, PullPolicy: v1.PullAlways}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePathPrefix))

		firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "dir", "file"))
		gomega.Expect(firstFileContents).To(gomega.Equal("1"))

		secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "file"))
		gomega.Expect(secondFileContents).To(gomega.Equal("2"))
	})

	f.It("should succeed with pod and multiple volumes", func(ctx context.Context) {
		var selinuxOptions *v1.SELinuxOptions
		if selinux.GetEnabled() {
			selinuxOptions = &v1.SELinuxOptions{
				User:  defaultSELinuxUser,
				Role:  defaultSELinuxRole,
				Type:  defaultSELinuxType,
				Level: defaultSELinuxLevel,
			}
			ginkgo.By(fmt.Sprintf("Using SELinux on pod: %v", selinuxOptions))
		}

		createPod(ctx,
			podName,
			"",
			[]v1.Volume{
				{Name: volumeName + "-0", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef}}},
				{Name: volumeName + "-1", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef}}},
			},
			[]v1.VolumeMount{
				{Name: volumeName + "-0", MountPath: volumePathPrefix + "-0"},
				{Name: volumeName + "-1", MountPath: volumePathPrefix + "-1"},
			},
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		for i := range 2 {
			volumePath := fmt.Sprintf("%s-%d", volumePathPrefix, i)
			ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePath))

			firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePath, "dir", "file"))
			gomega.Expect(firstFileContents).To(gomega.Equal("1"))

			secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePath, "file"))
			gomega.Expect(secondFileContents).To(gomega.Equal("2"))
		}
	})

	f.It("should fail if image volume is not existing", func(ctx context.Context) {
		var selinuxOptions *v1.SELinuxOptions
		if selinux.GetEnabled() {
			selinuxOptions = &v1.SELinuxOptions{
				User:  defaultSELinuxUser,
				Role:  defaultSELinuxRole,
				Type:  defaultSELinuxType,
				Level: defaultSELinuxLevel,
			}
			ginkgo.By(fmt.Sprintf("Using SELinux on pod: %v", selinuxOptions))
		}

		createPod(ctx,
			podName,
			"",
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: invalidImageRef}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to fail", f.Namespace.Name, podName))
		err := e2epod.WaitForPodContainerToFail(ctx, f.ClientSet, f.Namespace.Name, podName, 0, images.ErrImagePullBackOff.Error(), time.Minute)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)
	})

	f.It("should succeed if image volume is not existing but unused", func(ctx context.Context) {
		var selinuxOptions *v1.SELinuxOptions
		if selinux.GetEnabled() {
			selinuxOptions = &v1.SELinuxOptions{
				User:  defaultSELinuxUser,
				Role:  defaultSELinuxRole,
				Type:  defaultSELinuxType,
				Level: defaultSELinuxLevel,
			}
			ginkgo.By(fmt.Sprintf("Using SELinux on pod: %v", selinuxOptions))
		}

		createPod(ctx,
			podName,
			"",
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: invalidImageRef}}}},
			nil,
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Verifying the volume mount is not used for path: %s", volumePathPrefix))

		output := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/ls", filepath.Dir(volumePathPrefix))
		gomega.Expect(output).NotTo(gomega.ContainSubstring(strings.TrimPrefix(volumePathPrefix, "/")))
	})

	f.It("should succeed with multiple pods and same image on the same node", func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err, "Failed to get a ready schedulable node")

		baseName := "test-pod"
		anotherSELinuxLevel := "s0:c100,c200"

		for i := range 2 {
			podName := fmt.Sprintf("%s-%d", baseName, i)

			var selinuxOptions *v1.SELinuxOptions
			if selinux.GetEnabled() {
				if i == 0 {
					selinuxOptions = &v1.SELinuxOptions{
						User:  defaultSELinuxUser,
						Role:  defaultSELinuxRole,
						Type:  defaultSELinuxType,
						Level: defaultSELinuxLevel,
					}
				} else {
					selinuxOptions = &v1.SELinuxOptions{
						User:  defaultSELinuxUser,
						Role:  defaultSELinuxRole,
						Type:  defaultSELinuxType,
						Level: anotherSELinuxLevel,
					}
				}

				ginkgo.By(fmt.Sprintf("Using SELinux on pod %q: %v", podName, selinuxOptions))
			}

			createPod(ctx,
				podName,
				node.Name,
				[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef}}}},
				[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
				selinuxOptions,
			)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

			ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePathPrefix))

			firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "dir", "file"))
			gomega.Expect(firstFileContents).To(gomega.Equal("1"))

			secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "file"))
			gomega.Expect(secondFileContents).To(gomega.Equal("2"))
		}

		podName := baseName + "-0"
		ginkgo.By(fmt.Sprintf("Rechecking the pod (%s/%s) after another pod is running as expected", f.Namespace.Name, podName))
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePathPrefix))

		firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "dir", "file"))
		gomega.Expect(firstFileContents).To(gomega.Equal("1"))

		secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "file"))
		gomega.Expect(secondFileContents).To(gomega.Equal("2"))
	})
})
