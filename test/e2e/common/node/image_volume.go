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

package node

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/opencontainers/selinux/go-selinux"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe(framework.WithConformance(), "ImageVolume", func() {
	f := framework.NewDefaultFramework("image-volume")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	volumeImage := imageutils.GetE2EImage(imageutils.Kitten)

	const (
		podName             = "test-pod"
		containerName       = "test-container"
		volumeName          = "volume"
		volumePathPrefix    = "/volume"
		defaultSELinuxUser  = "system_u"
		defaultSELinuxRole  = "system_r"
		defaultSELinuxType  = "svirt_lxc_net_t" // that's the SELinux type used by Debian/Ubuntu. Keep it for maximum test compatibility.
		defaultSELinuxLevel = "s0:c1,c5"
	)

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
						Image:        imageutils.GetE2EImage(imageutils.BusyBox),
						Command:      []string{"/bin/sh", "-c", "while true; do echo test; sleep 1; done"},
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

	/*
		Release: v1.35
		Testname: Image Volume
		Description: Create a Pod using an image volume and a pull policy of Always.
		This test verifies that the image volume functionality is available by default and works as intended.
	*/
	framework.ConformanceIt("should succeed with pod and pull policy of Always", func(ctx context.Context) {
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
})
