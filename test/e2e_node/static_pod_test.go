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
	"fmt"
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("StaticPod", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("static-pod")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	f.Context("when the static pod has init container", func() {
		f.It("should be ready after init container is removed and kubelet restarts", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("create static pod")
			staticPod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "static",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "main",
							Image: imageutils.GetE2EImage(imageutils.Pause),
						},
					},
					InitContainers: []v1.Container{
						{
							Name:    "init",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"ls"},
						},
					},
				},
			}
			staticPodPath, err := createStaticPodFromPod(kubeletCfg.StaticPodPath, staticPod)
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func() {
				ginkgo.By("delete static pod")
				err = os.Remove(staticPodPath)
				framework.ExpectNoError(err)
			})

			var initCtrID string
			var startTime *metav1.Time
			mirrorPodName := fmt.Sprintf("%s-%s", staticPod.Name, framework.TestContext.NodeName)
			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				pod, err := f.ClientSet.CoreV1().Pods(staticPod.Namespace).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(err).Should(gomega.Succeed())
				g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
				cstatus := pod.Status.InitContainerStatuses[0]
				// Wait until the init container is terminated.
				g.Expect(cstatus.State.Terminated).NotTo(gomega.BeNil())
				g.Expect(cstatus.State.Terminated.ContainerID).NotTo(gomega.BeEmpty())
				initCtrID = cstatus.ContainerID
				startTime = pod.Status.StartTime
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())

			ginkgo.By("remove init container")
			removeInitContainer(ctx, initCtrID)

			ginkgo.By("restart kubelet")
			startKubelet := mustStopKubelet(ctx, f)
			startKubelet(ctx)
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("kubelet should be started"))

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				pod, err := f.ClientSet.CoreV1().Pods(staticPod.Namespace).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(pod.Status.StartTime).NotTo(gomega.Equal(startTime))
				g.Expect(err).Should(gomega.Succeed())
				g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
				cstatus := pod.Status.InitContainerStatuses[0]
				// Init container should be completed.
				g.Expect(cstatus.State.Terminated).NotTo(gomega.BeNil())
				g.Expect(cstatus.State.Terminated.Reason).To(gomega.Equal("Completed"))
				g.Expect(cstatus.State.Terminated.ExitCode).To(gomega.BeZero())
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())
		})
	})
	f.Context("when the static pod has sidecar container", func() {
		f.It("should be ready after sidecar container is removed and kubelet restarts", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("create static pod")
			staticPod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "static",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "main",
							Image: imageutils.GetE2EImage(imageutils.Pause),
						},
					},
					InitContainers: []v1.Container{
						{
							Name:          "init",
							Image:         imageutils.GetE2EImage(imageutils.Pause),
							RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways),
						},
					},
				},
			}
			staticPodPath, err := createStaticPodFromPod(kubeletCfg.StaticPodPath, staticPod)
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func() {
				ginkgo.By("delete static pod")
				err = os.Remove(staticPodPath)
				framework.ExpectNoError(err)
			})

			var sidecarCtrID string
			var startTime *metav1.Time
			mirrorPodName := fmt.Sprintf("%s-%s", staticPod.Name, framework.TestContext.NodeName)
			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				pod, err := f.ClientSet.CoreV1().Pods(staticPod.Namespace).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(err).Should(gomega.Succeed())
				g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
				cstatus := pod.Status.InitContainerStatuses[0]
				// Wait until the sidecar container starts running.
				g.Expect(cstatus.State.Running).NotTo(gomega.BeNil())
				sidecarCtrID = cstatus.ContainerID
				startTime = pod.Status.StartTime
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())

			// Stop kubelet first not to restart the sidecar container.
			ginkgo.By("stop kubelet")
			startKubelet := mustStopKubelet(ctx, f)

			ginkgo.By("remove sidecar container")
			removeInitContainer(ctx, sidecarCtrID)

			ginkgo.By("start kubelet")
			startKubelet(ctx)
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("kubelet should be started"))

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(g gomega.Gomega) {
				pod, err := f.ClientSet.CoreV1().Pods(staticPod.Namespace).Get(ctx, mirrorPodName, metav1.GetOptions{})
				g.Expect(pod.Status.StartTime).NotTo(gomega.Equal(startTime))
				g.Expect(err).Should(gomega.Succeed())
				g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
				cstatus := pod.Status.InitContainerStatuses[0]
				// Sidecar container should be restarted and running.
				g.Expect(cstatus.State.Running).NotTo(gomega.BeNil())
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())
		})
	})
})

func createStaticPodFromPod(dir string, pod *v1.Pod) (string, error) {
	name := pod.Name
	namespace := pod.Namespace
	file := staticPodPath(dir, name, namespace)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = f.Close()
	}()

	y := printers.YAMLPrinter{}
	return file, y.PrintObj(pod, f)
}

func removeInitContainer(ctx context.Context, ctrID string) {
	cricli, _, err := getCRIClient()
	framework.ExpectNoError(err)
	splitID := strings.Split(ctrID, "://")
	gomega.Expect(splitID).To(gomega.HaveLen(2))
	ctrID = splitID[1]
	// Make sure the container is stopped before removing it. This may fail.
	_ = cricli.StopContainer(ctx, ctrID, 0)
	err = cricli.RemoveContainer(ctx, ctrID)
	framework.ExpectNoError(err)
}
