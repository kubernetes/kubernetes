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
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// This test guards against a regression where deleting a pod leaves the
// kubelet unable to tear down one of its volumes. When terminationMessagePath
// points to a file inside a Bidirectional-mounted emptyDir volume, the
// runtime's bind mount of the termination message file propagates back to the
// host copy of the volume directory (MS_SHARED) and outlives the container.
// Volume teardown then failed with "unlinkat ... device or resource busy"
// because os.RemoveAll cannot unlink a mount point. See issue #115054.
// The volume manager now unmounts anything leaked into the pod's volume
// directory before handing the volume to the plugin's TearDown; emptyDir is
// used here because it is the simplest volume type that reproduces the leak.
//
// The pod must still be running when it is deleted. The API server removes an
// already-terminal pod immediately, which would bypass the kubelet's
// termination flow and hide the bug. For a running pod the kubelet only
// removes the pod object once every volume is torn down, so pre-fix the pod
// stays in Terminating and the delete wait below times out.
//
// The test checks three things:
//  1. the pod object goes away after a graceful delete (the regression hangs here),
//  2. the termination message was read from the file inside the volume, taken
//     from the final pod state carried by the DELETED watch event, and
//  3. the pod directory is removed from the node, so nothing leaked.
var _ = SIGDescribe("EmptyDir bidirectional teardown [LinuxOnly]", framework.WithNodeConformance(), func() {
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

	f.It("should delete a running pod whose terminationMessagePath is inside a Bidirectional emptyDir and remove its pod directory", func(ctx context.Context) {
		ginkgo.By("waiting for the node to be ready")
		waitForNodeReady(ctx)

		podClient := e2epod.NewPodClient(f)

		ginkgo.By("creating a privileged pod whose terminationMessagePath is inside a Bidirectional emptyDir")
		// The container writes the termination message up front and then
		// waits for SIGTERM, so the pod is still running when it is deleted
		// and the message is in place when the container exits.
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				RestartPolicy:                 v1.RestartPolicyNever,
				TerminationGracePeriodSeconds: ptr.To[int64](15),
				Containers: []v1.Container{
					{
						Name:  containerName,
						Image: busyboxImage,
						Command: []string{"sh", "-c",
							"echo '" + terminationMessage + "' > " + terminationFilePath + "; trap 'exit 0' TERM; while true; do sleep 1; done"},
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
		pod = podClient.CreateSync(ctx, pod)
		podDir := filepath.Join(framework.TestContext.KubeletRootDir, "pods", string(pod.UID))

		ginkgo.By("deleting the pod gracefully and waiting for the kubelet to remove it")
		// Watch the pod so the DELETED event, which carries the pod's final
		// state, can be inspected after the object is gone.
		lw := &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = fields.OneTermEqualSelector("metadata.name", pod.Name).String()
				return podClient.List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = fields.OneTermEqualSelector("metadata.name", pod.Name).String()
				return podClient.Watch(ctx, options)
			},
		}
		err := podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		watchCtx, cancel := watchtools.ContextWithOptionalTimeout(ctx, framework.PodDeleteTimeout)
		defer cancel()
		var finalPod *v1.Pod
		_, err = watchtools.UntilWithSync(watchCtx, lw, &v1.Pod{}, nil, func(e watch.Event) (bool, error) {
			if e.Type != watch.Deleted {
				return false, nil
			}
			finalPod = e.Object.(*v1.Pod)
			return true, nil
		})
		framework.ExpectNoError(err, "pod %s was not deleted within %v; the kubelet is likely unable to tear down the Bidirectional emptyDir", pod.Name, framework.PodDeleteTimeout)

		ginkgo.By("verifying the termination message was read from the file inside the volume")
		gomega.Expect(finalPod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		terminated := finalPod.Status.ContainerStatuses[0].State.Terminated
		gomega.Expect(terminated).NotTo(gomega.BeNil(), "container should be terminated in the final pod state")
		gomega.Expect(terminated.Message).To(gomega.ContainSubstring(terminationMessage),
			"termination message should be read from the file inside the Bidirectional emptyDir")

		ginkgo.By("verifying the pod directory was removed from the node")
		gomega.Eventually(ctx, func() error {
			_, err := os.Stat(podDir)
			if os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			return os.ErrExist
		}).WithTimeout(2*time.Minute).WithPolling(2*time.Second).Should(gomega.Succeed(),
			"pod directory %s should be removed once all volumes are torn down", podDir)
	})
})
