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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// Run this single test locally using a running CRI-O instance by:
// make test-e2e-node CONTAINER_RUNTIME_ENDPOINT="unix:///var/run/crio/crio.sock" TEST_ARGS='--ginkgo.focus="ImageVolume" --feature-gates=ImageVolume=true --service-feature-gates=ImageVolume=true --kubelet-flags="--cgroup-root=/ --runtime-cgroups=/system.slice/crio.service --kubelet-cgroups=/system.slice/kubelet.service --fail-swap-on=false"'
var _ = SIGDescribe("ImageVolume", nodefeature.ImageVolume, func() {
	f := framework.NewDefaultFramework("image-volume-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	const (
		podName          = "test-pod"
		containerName    = "test-container"
		validImageRef    = "quay.io/crio/artifact:v1"
		invalidImageRef  = "localhost/invalid"
		volumeName       = "volume"
		volumePathPrefix = "/volume"
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessFeatureGateEnabled(features.ImageVolume)
	})

	createPod := func(ctx context.Context, volumes []v1.Volume, volumeMounts []v1.VolumeMount) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:            containerName,
						Image:           busyboxImage,
						Command:         ExecCommand(podName, execCommand{LoopForever: true}),
						SecurityContext: &v1.SecurityContext{Privileged: ptr.To(true)},
						VolumeMounts:    volumeMounts,
					},
				},
				Volumes: volumes,
			},
		}

		ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v)", f.Namespace.Name, podName))
		e2epod.NewPodClient(f).Create(ctx, pod)

	}

	f.It("should succeed with pod and pull policy of Always", func(ctx context.Context) {
		createPod(ctx,
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef, PullPolicy: v1.PullAlways}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Verifying the volume mount contents for path: %s", volumePathPrefix))

		firstFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "dir", "file"))
		gomega.Expect(firstFileContents).To(gomega.Equal("1"))

		secondFileContents := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/cat", filepath.Join(volumePathPrefix, "file"))
		gomega.Expect(secondFileContents).To(gomega.Equal("2"))
	})

	f.It("should succeed with pod and multiple volumes", func(ctx context.Context) {
		createPod(ctx,
			[]v1.Volume{
				{Name: volumeName + "-0", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef}}},
				{Name: volumeName + "-1", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: validImageRef}}},
			},
			[]v1.VolumeMount{
				{Name: volumeName + "-0", MountPath: volumePathPrefix + "-0"},
				{Name: volumeName + "-1", MountPath: volumePathPrefix + "-1"},
			},
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, podName)

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
		createPod(ctx,
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: invalidImageRef}}}},
			[]v1.VolumeMount{{Name: volumeName, MountPath: volumePathPrefix}},
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to fail", f.Namespace.Name, podName))
		err := e2epod.WaitForPodContainerToFail(ctx, f.ClientSet, f.Namespace.Name, podName, 0, images.ErrImagePullBackOff.Error(), time.Minute)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, podName)
	})

	f.It("should succeed if image volume is not existing but unused", func(ctx context.Context) {
		createPod(ctx,
			[]v1.Volume{{Name: volumeName, VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: invalidImageRef}}}},
			nil,
		)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running", f.Namespace.Name, podName))
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: (%v/%v)", f.Namespace.Name, podName)

		ginkgo.By(fmt.Sprintf("Verifying the volume mount is not used for path: %s", volumePathPrefix))

		output := e2epod.ExecCommandInContainer(f, podName, containerName, "/bin/ls", filepath.Dir(volumePathPrefix))
		gomega.Expect(output).NotTo(gomega.ContainSubstring(strings.TrimPrefix(volumePathPrefix, "/")))
	})
})
