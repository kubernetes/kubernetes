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
	"golang.org/x/sys/unix"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/services"
	admissionapi "k8s.io/pod-security-admission/api"
)

// The kubelet is restarted with a host mount that affects all volume
// operations, so this test must not run concurrently with other node tests.
var _ = SIGDescribe("Mounts with an empty source [LinuxOnly]", framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("mount-empty-source")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.It("should start the kubelet and manage subPath volumes with an empty-source mount present", func(ctx context.Context) {
		const (
			volumeName = "memory-volume"
			subPath    = "subpath"
		)

		ginkgo.By("preparing the mount point")
		mountPath, err := os.MkdirTemp("", "mount-empty-source-")
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(path string) {
			framework.ExpectNoError(os.Remove(path))
		}, mountPath)
		// Resolve symlinks so the path matches the kernel's mount table entries.
		mountPath, err = filepath.EvalSymlinks(mountPath)
		framework.ExpectNoError(err)

		podClient := e2epod.NewPodClient(f)
		var pod *v1.Pod
		ginkgo.DeferCleanup(func(ctx context.Context) {
			if pod != nil {
				podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}
		})

		ginkgo.By("stopping the kubelet")
		restartKubelet := mustStopKubelet(ctx, f)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			if !e2enode.HealthCheck(kubeletHealthCheckURL) {
				restartKubelet(ctx)
			}
		})

		ginkgo.By("creating a host tmpfs mount with an empty source")
		// mount(8) may replace an empty source with "none", so use the syscall.
		framework.ExpectNoError(unix.Mount("", mountPath, "tmpfs", 0, "size=1m"))
		ginkgo.DeferCleanup(func() {
			if err := unix.Unmount(mountPath, 0); err != nil {
				framework.Logf("unmount %s failed (%v), retrying with MNT_DETACH", mountPath, err)
				framework.ExpectNoError(unix.Unmount(mountPath, unix.MNT_DETACH))
			}
		})

		ginkgo.By("starting the kubelet with the empty-source mount present")
		restartKubelet(ctx)

		ginkgo.By("running a pod that writes and reads a memory backed emptyDir through a subPath")
		pod = podClient.Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "empty-mount-source-"},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{{
					Name:    "test",
					Image:   busyboxImage,
					Command: []string{"sh", "-ec", "echo mount-source-test > /data/test; test \"$(cat /data/test)\" = mount-source-test; exec sleep 3600"},
					VolumeMounts: []v1.VolumeMount{{
						Name:      volumeName,
						MountPath: "/data",
						SubPath:   subPath,
					}},
				}},
				Volumes: []v1.Volume{{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
					},
				}},
			},
		})
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStart))

		podDir := filepath.Join(services.KubeletRootDirectory, "pods", string(pod.UID))
		volumePath := filepath.Join(podDir, "volumes", "kubernetes.io~empty-dir", volumeName)
		gomega.Eventually(ctx, func() (string, error) {
			data, err := os.ReadFile(filepath.Join(volumePath, subPath, "test"))
			return string(data), err
		}, f.Timeouts.PodStart, time.Second).Should(gomega.Equal("mount-source-test\n"))

		// The pod directory is only removed after the volume and its subPath bind
		// mounts are torn down.
		ginkgo.By("verifying the kubelet removes the pod directory while the empty-source mount still exists")
		podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		pod = nil
		gomega.Eventually(ctx, func() error {
			_, err := os.Stat(podDir)
			if os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			return os.ErrExist
		}, f.Timeouts.PodDelete, time.Second).Should(gomega.Succeed(), "pod directory %s should be removed", podDir)
	})
})
