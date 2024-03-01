/*
Copyright 2023 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
* Unknown pods are pods which are unknown pods to the kubelet, but are still
* running in the container runtime. If kubelet detects a pod which is not in
* the config (i.e. not present in API-server or static pod), but running as
* detected in container runtime, kubelet should aggressively terminate the pod.
*
* This situation can be encountered if a pod is running, then kubelet is
* stopped, and while stopped, the manifest is deleted (by force deleting the
* API pod or deleting the static pod manifest), and then restarting the
* kubelet. Upon restart, kubelet will see the pod as running via the container
* runtime, but it will not be present in the config, thus making the pod a
* "unknown pod". Kubelet should then proceed to terminate these unknown pods.
 */
var _ = SIGDescribe("Unknown Pods", framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("unknown-pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("when creating a mirror pod", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "unknown-test-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = kubeletCfg.StaticPodPath

			framework.Logf("create the static pod %v", staticPodName)
			err := createStaticPodWithGracePeriod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			framework.Logf("wait for the mirror pod %v to be running", mirrorPodName)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeNil())
		})

		ginkgo.It("the static pod should be terminated and cleaned up due to becoming a unknown pod due to being force deleted while kubelet is not running", func(ctx context.Context) {
			framework.Logf("Stopping the kubelet")
			startKubelet := stopKubelet()

			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// wait until the kubelet health check will fail
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalseBecause("expected kubelet health check to be failed"))

			framework.Logf("Delete the static pod manifest while the kubelet is not running")
			file := staticPodPath(podPath, staticPodName, ns)
			framework.Logf("deleting static pod manifest %q", file)
			err = os.Remove(file)
			framework.ExpectNoError(err)

			framework.Logf("Starting the kubelet")
			startKubelet()

			// wait until the kubelet health check will succeed
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

			framework.Logf("wait for the mirror pod %v to disappear", mirrorPodName)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())

			waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			framework.Logf("deleting the static pod %v", staticPodName)
			err := deleteStaticPod(podPath, staticPodName, ns)
			if !os.IsNotExist(err) {
				framework.ExpectNoError(err)
			}

			framework.Logf("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())
		})
	})

	ginkgo.Context("when creating a API pod", func() {
		var ns, podName string

		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			podName = "unknown-test-pause-pod-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "pause",
							Image: imageutils.GetPauseImageName(),
						},
					},
				},
			}

			e2epod.NewPodClient(f).CreateSync(ctx, pod)
		})

		ginkgo.It("the api pod should be terminated and cleaned up due to becoming a unknown pod due to being force deleted while kubelet is not running", func(ctx context.Context) {
			framework.Logf("Stopping the kubelet")
			startKubelet := stopKubelet()

			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// wait until the kubelet health check will fail
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalseBecause("expected kubelet health check to be failed"))

			framework.Logf("Delete the pod while the kubelet is not running")
			// Delete pod sync by name will force delete the pod, removing it from kubelet's config
			deletePodSyncByName(ctx, f, podName)

			framework.Logf("Starting the kubelet")
			startKubelet()

			// wait until the kubelet health check will succeed
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

			framework.Logf("wait for the pod %v to disappear", podName)
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, podName, ns)
			}, f.Timeouts.PodDelete, f.Timeouts.Poll).Should(gomega.BeNil())

			waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
		})
	})
})
