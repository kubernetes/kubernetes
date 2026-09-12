//go:build linux

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
	"fmt"
	"os"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("StaticPod", framework.WithDisruptive(), framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("static-pod-apiserver-down")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when apiserver is unreachable", func() {
		ginkgo.It("should restart static pod when spec changes", func(ctx context.Context) {
			staticPodName := "test-static-pod"
			staticPodUID := types.UID("static-test-uid-123")
			mirrorPodName := fmt.Sprintf("%s-%s", staticPodName, framework.TestContext.NodeName)

			kubeletCfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			ginkgo.By("creating initial static pod with static UID while apiserver is up")
			staticPod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      staticPodName,
					Namespace: f.Namespace.Name,
					UID:       staticPodUID,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "test-container",
							Image: imageutils.GetE2EImage(imageutils.Pause),
							Env: []v1.EnvVar{
								{Name: "VERSION", Value: "v1"},
							},
						},
					},
				},
			}

			staticPodPath, err := createStaticPodFromPod(kubeletCfg.StaticPodPath, staticPod)
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func() {
				ginkgo.By("cleaning up static pod file")
				_ = os.Remove(staticPodPath)
			})

			ginkgo.By("waiting for mirror pod to appear in API server")
			var originalMirrorPod *v1.Pod
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(err).Should(gomega.Succeed())
				g.Expect(pod.Status.Phase).Should(gomega.Equal(v1.PodRunning))
				originalMirrorPod = pod
			}, 2*time.Minute, 2*time.Second).Should(gomega.Succeed())

			framework.Logf("Mirror pod created: %s (UID: %s, env: %s)",
				originalMirrorPod.Name, originalMirrorPod.UID, originalMirrorPod.Spec.Containers[0].Env[0].Value)

			ginkgo.By("pausing apiserver to simulate apiserver restart")
			err = PauseAPIServer()
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func() {
				ginkgo.By("resuming apiserver")
				_ = ResumeAPIServer()
			})

			ginkgo.By("restarting kubelet so it hasn't seen API source yet")
			restartKubelet(ctx, true)

			// Can't use waitForKubeletToStart becase the apiserver is down
			gomega.Eventually(ctx, func() bool {
				return e2enode.HealthCheck(kubeletHealthCheckURL)
			}, 2*time.Minute, 5*time.Second).Should(gomega.BeTrueBecause("expected kubelet to be healthy"))

			ginkgo.By("waiting for pod to start after kubelet restart")
			originalContainerID := waitForContainer(ctx, f.Namespace.Name, mirrorPodName, "test-container", 30*time.Second,
				func(g gomega.Gomega, foundPodID, foundContainerID string) {
					g.Expect(foundPodID).NotTo(gomega.BeEmpty(), "pod should exist")
					g.Expect(foundContainerID).NotTo(gomega.BeEmpty(), "should find container")
				})
			framework.Logf("Pod restarted after kubelet restart, container ID: %s", originalContainerID)

			ginkgo.By("deleting static pod file while apiserver is paused")
			err = os.Remove(staticPodPath)
			framework.ExpectNoError(err)

			time.Sleep(2 * time.Second)

			ginkgo.By("recreating static pod with updated spec (keeping static UID)")
			staticPod.Spec.Containers[0].Env[0].Value = "v2"
			staticPodPath, err = createStaticPodFromPod(kubeletCfg.StaticPodPath, staticPod)
			framework.ExpectNoError(err)

			ginkgo.By("verifying pod restarts locally via CRI while API is still paused")
			// The fix (SourceForPodReady) should allow deletion/recreation within ~40 seconds
			// Without the fix (AllReady check), deletion would be blocked indefinitely while API is paused
			newContainerID := waitForContainer(ctx, f.Namespace.Name, mirrorPodName, "test-container", 45*time.Second,
				func(g gomega.Gomega, foundPodID, foundContainerID string) {
					g.Expect(foundPodID).NotTo(gomega.BeEmpty(), "pod should exist")
					g.Expect(foundContainerID).NotTo(gomega.BeEmpty(), "should find container")
					g.Expect(foundContainerID).NotTo(gomega.Equal(originalContainerID), "should find new container with different ID")
				})

			framework.Logf("Static pod restarted locally while API was paused (old container: %s, new container: %s)",
				originalContainerID, newContainerID)

			ginkgo.By("verifying API server is still paused after pod restarted")
			// This confirms the restart happened without waiting for API to come back
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			gomega.Expect(err).Should(gomega.HaveOccurred(), "API should still be unavailable")

			ginkgo.By("resuming apiserver to restore connectivity")
			err = ResumeAPIServer()
			framework.ExpectNoError(err)

			ginkgo.By("deleting static pod file")
			err = os.Remove(staticPodPath)
			framework.ExpectNoError(err)

			ginkgo.By("verifying pod is deleted")
			waitForContainer(ctx, f.Namespace.Name, mirrorPodName, "test-container", 30*time.Second,
				func(g gomega.Gomega, foundPodID, foundContainerID string) {
					g.Expect(foundPodID).To(gomega.BeEmpty(), "pod should be deleted")
				})

			ginkgo.By("verifying mirror pod is deleted from API")
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrueBecause("mirror pod should be deleted"))
			}, 30*time.Second, 2*time.Second).Should(gomega.Succeed())

			framework.Logf("Static pod and mirror pod successfully deleted")
		})
	})
})

func waitForContainer(ctx context.Context, namespace, podName, containerName string, timeout time.Duration,
	verifier func(g gomega.Gomega, foundPodID, foundContainerID string)) string {
	var containerID string
	gomega.EventuallyWithOffset(1, ctx, func(g gomega.Gomega) {
		cricli, _, err := getCRIClient(ctx)
		g.Expect(err).Should(gomega.Succeed())

		pods, err := cricli.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
		g.Expect(err).Should(gomega.Succeed())

		var targetPodID string
		for _, pod := range pods {
			if pod.Metadata.Name == podName && pod.Metadata.Namespace == namespace {
				targetPodID = pod.Id
				break
			}
		}

		if targetPodID != "" {
			containers, err := cricli.ListContainers(ctx, &runtimeapi.ContainerFilter{PodSandboxId: targetPodID})
			g.Expect(err).Should(gomega.Succeed())

			for _, container := range containers {
				if container.Metadata.Name == containerName {
					containerID = container.Id
					break
				}
			}
		}

		verifier(g, targetPodID, containerID)
	}, timeout, 2*time.Second).Should(gomega.Succeed())
	return containerID
}
