/*
Copyright The Kubernetes Authors.

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
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// This test guards against a regression where a pod could get stuck in
// Terminating forever. When terminationMessagePath points to a file inside a
// Bidirectional-mounted emptyDir volume, the kubelet's bind mount of the
// termination message file propagates back to the host copy of the volume
// directory (MS_SHARED). emptyDir teardown then failed with
// "unlinkat ... device or resource busy" because os.RemoveAll cannot unlink a
// mount point, and the pod never finished deleting. See issue #115054.
//
// The test exercises both halves of the contract:
//  1. the termination message is read successfully (it is captured while the
//     container terminates, independently of volume teardown), and
//  2. the pod tears down cleanly after deletion (the regression hangs here).
var _ = SIGDescribe("EmptyDir bidirectional teardown [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("emptydir-bidirectional")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	const (
		podName             = "bidirectional-emptydir"
		containerName       = "writer"
		terminationMessage  = "bidirectional-teardown-ok"
		volumeName          = "cache-volume"
		volumeMountPath     = "/cache"
		terminationFilePath = "/cache/log.txt"
	)

	f.It("should read terminationMessagePath inside a Bidirectional emptyDir and tear down cleanly after deletion", func(ctx context.Context) {
		ginkgo.By("waiting for the node to be ready")
		waitForNodeReady(ctx)

		podClient := e2epod.NewPodClient(f)

		ginkgo.By("creating a privileged pod whose terminationMessagePath is inside a Bidirectional emptyDir")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:                     containerName,
						Image:                    busyboxImage,
						Command:                  []string{"sh", "-c", "echo '" + terminationMessage + "' > " + terminationFilePath},
						TerminationMessagePath:   terminationFilePath,
						TerminationMessagePolicy: v1.TerminationMessageReadFile,
						VolumeMounts: []v1.VolumeMount{
							{
								Name:             volumeName,
								MountPath:        volumeMountPath,
								MountPropagation: ptr.To(v1.MountPropagationBidirectional),
							},
						},
						SecurityContext: &v1.SecurityContext{
							Privileged: new(true),
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}
		pod = podClient.Create(ctx, pod)

		ginkgo.By("waiting for the container to terminate successfully")
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))

		ginkgo.By("verifying the termination message was read from the file inside the volume")
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(pod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		terminated := pod.Status.ContainerStatuses[0].State.Terminated
		gomega.Expect(terminated).NotTo(gomega.BeNil(), "container should be terminated")
		gomega.Expect(terminated.Message).To(gomega.ContainSubstring(terminationMessage),
			"termination message should be read from the file inside the Bidirectional emptyDir")

		ginkgo.By("deleting the pod and verifying it tears down cleanly")
		// Pre-fix, emptyDir teardown fails with "device or resource busy" on the
		// propagated submount and retries forever, so the pod never disappears
		// and this wait times out.
		err = f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, framework.PodDeleteTimeout))
	})
})
