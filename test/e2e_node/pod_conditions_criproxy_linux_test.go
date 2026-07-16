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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// PodReadyToStartContainers condition timing tests with CRI Proxy delays (Linux only)
var _ = SIGDescribe("Pod conditions managed by Kubelet", func() {
	f := framework.NewDefaultFramework("pod-conditions")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	f.Context("including PodReadyToStartContainers condition", f.WithSerial(), framework.WithFeatureGate(features.PodReadyToStartContainersCondition), func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
		})

		f.Context("timing with CRI Proxy delays", feature.CriProxy, func() {
			ginkgo.BeforeEach(func() {
				if e2eCriProxy == nil {
					ginkgo.Skip("Skip the test since the CRI Proxy is undefined. Please run with --cri-proxy-enabled=true")
				}
				if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
					ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
				}
				ginkgo.DeferCleanup(func() error {
					return resetCRIProxyInjector(e2eCriProxy)
				})
			})

			ginkgo.It("a pod without init containers should report PodReadyToStartContainers condition set before image pull completes", runPodReadyToStartContainersTimingTest(f, false))
			ginkgo.It("a pod with init containers should report PodReadyToStartContainers condition set before image pull completes", runPodReadyToStartContainersTimingTest(f, true))
		})

		addAfterEachForCleaningUpPods(f)
	})
})

// newPullImageAlwaysPodWithInitContainer creates a pod with init container and ImagePullPolicy: Always
func newPullImageAlwaysPodWithInitContainer() *v1.Pod {
	podName := "cri-proxy-test-" + string(uuid.NewUUID())
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Image:           imageutils.GetE2EImage(imageutils.BusyBox),
					Name:            "init",
					ImagePullPolicy: v1.PullAlways,
					Command:         []string{"sh", "-c", "sleep 2"},
				},
			},
			Containers: []v1.Container{
				{
					Image:           imageutils.GetPauseImageName(),
					Name:            "main",
					ImagePullPolicy: v1.PullAlways,
				},
			},
		},
	}
	return pod
}

func runPodReadyToStartContainersTimingTest(f *framework.Framework, hasInitContainers bool) func(ctx context.Context) {
	return func(ctx context.Context) {
		const delayTime = 15 * time.Second

		ginkgo.By("Injecting delay into PullImage calls")
		err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.PullImage {
				ginkgo.By(fmt.Sprintf("Delaying PullImage by %v", delayTime))
				time.Sleep(delayTime)
			}
			return nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating test pod with ImagePullPolicy: Always")
		var testPod *v1.Pod
		if hasInitContainers {
			testPod = newPullImageAlwaysPodWithInitContainer()
		} else {
			testPod = newPullImageAlwaysPod()
		}
		testPod = e2epod.NewPodClient(f).Create(ctx, testPod)

		ginkgo.By("Waiting for PodReadyToStartContainers condition to be set")
		gomega.Eventually(func() error {
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, testPod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			_, err = getTransitionTimeForPodConditionWithStatus(pod, v1.PodReadyToStartContainers, true)
			return err
		}).WithPolling(500*time.Millisecond).WithTimeout(10*time.Second).Should(gomega.Succeed(),
			"PodReadyToStartContainers condition should be set to True within %v", 10*time.Second)

		ginkgo.By("Verifying condition timing, it should be set quickly before image pull delay")
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, testPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		conditionTime, err := getTransitionTimeForPodConditionWithStatus(pod, v1.PodReadyToStartContainers, true)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for pod to eventually become Running after image pull")
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, testPod))

		ginkgo.By("Verifying condition was set before image pull completed")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, testPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		podReadyTime, err := getTransitionTimeForPodConditionWithStatus(pod, v1.PodReady, true)
		framework.ExpectNoError(err)

		gomega.Expect(conditionTime.Before(podReadyTime)).To(gomega.BeTrueBecause(
			"PodReadyToStartContainers was set at %v but PodReady was set at %v - condition should be set before image pull completes",
			conditionTime, podReadyTime))
	}
}
