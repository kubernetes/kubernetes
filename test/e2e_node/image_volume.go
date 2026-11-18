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

	"github.com/blang/semver/v4"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/opencontainers/selinux/go-selinux"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Run this single test locally using a running CRI-O instance by:
// make test-e2e-node CONTAINER_RUNTIME_ENDPOINT="unix:///var/run/crio/crio.sock" TEST_ARGS='--ginkgo.focus="ImageVolume" --kubelet-flags="--cgroup-root=/ --runtime-cgroups=/system.slice/crio.service --kubelet-cgroups=/system.slice/kubelet.service --fail-swap-on=false"'
var _ = SIGDescribe("ImageVolume", feature.ImageVolume, func() {
	f := framework.NewDefaultFramework("image-volume-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	volumeImage := imageutils.GetE2EImage(imageutils.Kitten)

	const (
		podName             = "test-pod"
		containerName       = "test-container"
		invalidImageRef     = "localhost/invalid"
		volumeName          = "volume"
		volumePathPrefix    = "/volume"
		defaultSELinuxUser  = "system_u"
		defaultSELinuxRole  = "system_r"
		defaultSELinuxType  = "svirt_lxc_net_t"
		defaultSELinuxLevel = "s0:c1,c5"
	)

	// TODO: remove when containerd 2.2 is available on all node e2e test platforms.
	requireContainerdVersion := func(ctx context.Context, givenVersion string) {
		runtime, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err, "Failed to get CRI client")

		resp, err := runtime.Version(ctx, "")
		framework.ExpectNoError(err, "Failed to get runtime version")

		// Other runtimes like CRI-O do not need to enforce any version requirement.
		if resp.GetRuntimeName() != "containerd" {
			return
		}

		// Throw away any version suffix which just causes semver parsing issues.
		version, _, _ := strings.Cut(resp.GetRuntimeVersion(), "-")

		parsedVersion, err := semver.ParseTolerant(version)
		framework.ExpectNoError(err, "Failed to parse CRI runtime version: "+version)

		requiredVersion, err := semver.ParseTolerant(givenVersion)
		framework.ExpectNoError(err, "Failed to parse required CRI runtime version")

		if parsedVersion.LT(requiredVersion) {
			e2eskipper.Skipf("runtime containerd is version %q, but test requires at least %q", resp.GetRuntimeVersion(), requiredVersion)
		}
	}

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

	verifyFileContents := func(podName, volumePath string) {
		ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePath))

		firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePath, "data.json"))
		gomega.Expect(firstFileContents).To(gomega.ContainSubstring("kitten.jpg"))

		secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePath, "etc", "os-release"))
		gomega.Expect(secondFileContents).To(gomega.ContainSubstring("Alpine Linux"))
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		requireContainerdVersion(ctx, "2.1")
	})

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
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage, PullPolicy: v1.PullAlways}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		verifyFileContents(podName, volumePathPrefix)
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
				{Name: volumeName + "-0", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}},
				{Name: volumeName + "-1", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}},
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
			verifyFileContents(podName, volumePath)
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
		framework.ExpectNoError(err, "Failed to await for the pod to be failed: (%s/%s)", f.Namespace.Name, podName)
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
				[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}}},
				[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
				selinuxOptions,
			)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

			verifyFileContents(podName, volumePathPrefix)
		}

		podName := baseName + "-0"
		ginkgo.By(fmt.Sprintf("Rechecking the pod (%s/%s) after another pod is running as expected", f.Namespace.Name, podName))
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		verifyFileContents(podName, volumePathPrefix)
	})

	f.It("should report the image volume digest in the pod status", framework.WithFeatureGate(features.ImageVolumeWithDigest), func(ctx context.Context) {
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
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
			selinuxOptions,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Getting the image ID of %s from the container runtime", volumeImage))
		runtimeService, imageService, err := getCRIClient(ctx)
		framework.ExpectNoError(err, "Failed to get CRI client")
		imageStatus, err := imageService.ImageStatus(ctx, &runtimeapi.ImageSpec{Image: volumeImage}, false)
		framework.ExpectNoError(err, "Failed to get image status: %s", volumeImage)
		imageID := imageStatus.GetImage().GetId()
		gomega.Expect(imageID).NotTo(gomega.BeEmpty(), "Failed to get image ID: %s", volumeImage)

		ginkgo.By(fmt.Sprintf("Getting the image volume digest reported by the container runtime for the pod (%s/%s)", f.Namespace.Name, podName))
		containers, err := runtimeService.ListContainers(ctx, &runtimeapi.ContainerFilter{
			LabelSelector: map[string]string{
				types.KubernetesPodNameLabel:      podName,
				types.KubernetesPodNamespaceLabel: f.Namespace.Name,
			},
		})
		framework.ExpectNoError(err, "Failed to list the containers of the pod (%s/%s)", f.Namespace.Name, podName)
		gomega.Expect(containers).NotTo(gomega.BeEmpty(), "Failed to list the containers of the pod (%s/%s)", f.Namespace.Name, podName)
		containerStatus, err := runtimeService.ContainerStatus(ctx, containers[0].GetId(), false)
		framework.ExpectNoError(err, "Failed to get the container status of the pod (%s/%s)", f.Namespace.Name, podName)
		var runtimeImageRef string
		for _, mount := range containerStatus.GetStatus().GetMounts() {
			if mount.GetContainerPath() == volumePathPrefix {
				runtimeImageRef = mount.GetImage().GetImageRef()
				break
			}
		}
		if runtimeImageRef == "" {
			e2eskipper.Skipf("container runtime does not report ImageSpec.ImageRef for image volume mounts")
		}

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to report the image volume digest", f.Namespace.Name, podName))
		var imageRef string
		err = framework.Gomega().Eventually(ctx, func(ctx context.Context) (string, error) {
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			for _, volumeMount := range pod.Status.ContainerStatuses[0].VolumeMounts {
				if volumeMount.Name == volumeName && volumeMount.VolumeStatus != nil && volumeMount.VolumeStatus.Image != nil {
					imageRef = volumeMount.VolumeStatus.Image.ImageRef
				}
			}
			return imageRef, nil
		}).WithPolling(5 * time.Second).WithTimeout(time.Minute).Should(gomega.Equal(runtimeImageRef))
		framework.ExpectNoError(err, "Failed to await the image volume digest in the pod status (%s/%s)", f.Namespace.Name, podName)

		gomega.Expect(imageRef).To(gomega.Equal(imageID), "image volume digest should match the image ID reported by the container runtime")
	})

	f.Context("subPath", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			requireContainerdVersion(ctx, "2.2")
		})

		f.It("should succeed when using a valid subPath", func(ctx context.Context) {
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
				[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}}},
				[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix, SubPath: "etc"}},
				selinuxOptions,
			)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to be running", f.Namespace.Name, podName))
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: (%s/%s)", f.Namespace.Name, podName)

			// Using this file is intentional because it's generally available on Linux and will not expose any secret information.
			fileContent := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "passwd"))
			gomega.Expect(fileContent).To(gomega.ContainSubstring("root:"))
		})

		f.It("should fail if subPath in volume is not existing", func(ctx context.Context) {
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
				[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: volumeImage}}}},
				[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix, SubPath: "not-existing"}},
				selinuxOptions,
			)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%s/%s) to fail", f.Namespace.Name, podName))
			err := e2epod.WaitForPodContainerToFail(ctx, f.ClientSet, f.Namespace.Name, podName, 0, kuberuntime.ErrCreateContainer.Error(), time.Minute)
			framework.ExpectNoError(err, "Failed to await for the pod to be failed: (%s/%s)", f.Namespace.Name, podName)
		})
	})
})
