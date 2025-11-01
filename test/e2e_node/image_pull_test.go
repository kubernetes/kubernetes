//go:build linux
// +build linux

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
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	"k8s.io/kubernetes/test/utils/format"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// CriProxy injector is used to simulate and verify the image pull behavior.
// These tests need to run in serial to prevent caching of the images by other tests
// and to prevent the wait time of image pulls to be increased by other images.
var _ = SIGDescribe("Pull Image", feature.CriProxy, framework.WithSerial(), func() {

	f := framework.NewDefaultFramework("parallel-pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var testpods []*v1.Pod

	ginkgo.Context("parallel image pull with MaxParallelImagePulls=5", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.BeforeEach(func(ctx context.Context) {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
			testpods = prepareAndCleanup(ctx, f)
			gomega.Expect(len(testpods)).To(gomega.BeNumerically("<=", 5))
		})

		ginkgo.It("should pull immediately if no more than 5 pods", func(ctx context.Context) {
			var mu sync.Mutex
			timeout := 20 * time.Second
			callCh := make(chan struct{})
			callStatus := make(map[int]chan struct{})
			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.PullImage {
					mu.Lock()
					callID := len(callStatus)
					callStatus[callID] = callCh
					mu.Unlock()
					if callID == 0 {
						// wait for next call
						select {
						case <-callCh:
							return nil
						case <-time.After(timeout):
							return fmt.Errorf("no parallel image pull after %s", timeout)
						}
					} else {
						// send a signal to the first call
						callCh <- struct{}{}
					}
				}
				return nil
			})
			framework.ExpectNoError(err)

			for _, testpod := range testpods {
				pod := e2epod.NewPodClient(f).Create(ctx, testpod)
				ginkgo.DeferCleanup(deletePodSyncByName, f, pod.Name)
			}

			imagePulled, podStartTime, podEndTime, err := getPodImagePullDurations(ctx, f, testpods)
			framework.ExpectNoError(err)

			checkPodPullingOverlap(podStartTime, podEndTime, testpods)

			for _, img := range imagePulled {
				framework.Logf("Pod pull duration including waiting is %v, and the pulled duration is %v", img.pulledIncludeWaitingDuration, img.pulledDuration)
				// if a pod image pull hanged for more than 50%, it is a delayed pull.
				if float32(img.pulledIncludeWaitingDuration.Milliseconds())/float32(img.pulledDuration.Milliseconds()) > 1.5 {
					// as this is parallel image pulling, the waiting duration should be similar with the pulled duration.
					framework.Failf("There is a delayed image pulling, which is not expected for parallel image pulling.")
				}
			}
		})

	})
})

var _ = SIGDescribe("Pull Image", feature.CriProxy, framework.WithSerial(), func() {

	f := framework.NewDefaultFramework("serialize-pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("serialize image pull", func() {
		// this is the default behavior now.
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = true
			initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		})

		var testpods []*v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
			testpods = prepareAndCleanup(ctx, f)
			gomega.Expect(len(testpods)).To(gomega.BeNumerically("<=", 5))
		})

		ginkgo.It("should be waiting more", func(ctx context.Context) {
			// all serialize image pulls should timeout
			timeout := 20 * time.Second
			var mu sync.Mutex
			callCh := make(chan struct{})
			callStatus := make(map[int]chan struct{})
			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.PullImage {
					mu.Lock()
					callID := len(callStatus)
					callStatus[callID] = callCh
					mu.Unlock()
					if callID == 0 {
						// wait for next call
						select {
						case <-callCh:
							return errors.New("parallel image pull detected")
						case <-time.After(timeout):
							return nil
						}
					} else {
						// send a signal to the first call
						select {
						case callCh <- struct{}{}:
							return errors.New("parallel image pull detected")
						case <-time.After(timeout):
							return nil
						}
					}
				}
				return nil
			})
			framework.ExpectNoError(err)

			var pods []*v1.Pod
			for _, testpod := range testpods {
				pod := e2epod.NewPodClient(f).Create(ctx, testpod)
				ginkgo.DeferCleanup(deletePodSyncByName, f, pod.Name)
				pods = append(pods, pod)
			}
			for _, pod := range pods {
				err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Running", 2*time.Minute, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase == v1.PodRunning {
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err)
			}

			imagePulled, podStartTime, podEndTime, err := getPodImagePullDurations(ctx, f, testpods)
			framework.ExpectNoError(err)
			gomega.Expect(len(testpods)).To(gomega.BeComparableTo(len(imagePulled)))

			checkPodPullingOverlap(podStartTime, podEndTime, testpods)

			// if a pod image pull hanged for more than 50%, it is a delayed pull.
			var anyDelayedPull bool
			for _, img := range imagePulled {
				framework.Logf("Pod pull duration including waiting is %v, and the pulled duration is %v", img.pulledIncludeWaitingDuration, img.pulledDuration)
				if float32(img.pulledIncludeWaitingDuration.Milliseconds())/float32(img.pulledDuration.Milliseconds()) > 1.5 {
					anyDelayedPull = true
				}
			}
			// as this is serialize image pulling, the waiting duration should be almost double the duration with the pulled duration.
			// use 1.5 as a common ratio to avoid some overlap during pod creation
			if !anyDelayedPull {
				framework.Failf("All image pullings are not delayed, which is not expected for serilized image pull")
			}
		})

	})

	ginkgo.It("Image pull retry backs off on error.", func(ctx context.Context) {
		if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
			ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
		}

		// inject PullImage failed to trigger backoff
		expectedErr := fmt.Errorf("PullImage failed")
		err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.PullImage {
				return expectedErr
			}
			return nil
		})
		framework.ExpectNoError(err)

		pod := e2epod.NewPodClient(f).Create(ctx, newPullImageAlwaysPod())
		podErr := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "ImagePullBackOff", 1*time.Minute, func(pod *v1.Pod) (bool, error) {
			if len(pod.Status.ContainerStatuses) > 0 && pod.Status.Reason == images.ErrImagePullBackOff.Error() {
				return true, nil
			}
			return false, nil
		})
		gomega.Expect(podErr).To(gomega.HaveOccurred())

		eventMsg, err := getFailedToPullImageMsg(ctx, f, pod.Name)
		framework.ExpectNoError(err)
		isExpectedErrMsg := strings.Contains(eventMsg, expectedErr.Error())
		gomega.Expect(isExpectedErrMsg).To(gomega.BeTrueBecause("we injected an exception into the PullImage interface of the cri proxy"))

		// Hard wait 30 seconds for image pulls to repeatedly back off.
		time.Sleep(30 * time.Second)

		e, err := getImagePullAttempts(ctx, f, pod.Name)
		framework.ExpectNoError(err)
		// 3 would take 10s best case.
		gomega.Expect(e.Count).Should(gomega.BeNumerically(">=", 3))
		// 7 would take 310s best case, if the infra went slow.
		gomega.Expect(e.Count).Should(gomega.BeNumerically("<=", 7))

	})

})

var _ = SIGDescribe("Allow cancel pull image", framework.WithSerial(), framework.WithFeatureGate(features.CancelPullImage), func() {
	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var is internalapi.ImageManagerService

	ginkgo.BeforeEach(func() {
		var err error
		_, is, err = getCRIClient()
		framework.ExpectNoError(err)
	})

	ginkgo.Context("with parallel image pull", func() {
		// Configure kubelet to use parallel image pulls
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.It("Should not pull image after pod deletion with parallel pull", func(ctx context.Context) {
			// Get a test image
			image := imageutils.GetE2EImage(imageutils.Agnhost)

			// 1. Remove the image before creating the pod
			ginkgo.By("Removing the image before creating the pod")
			err := removeImageWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// Verify image is removed
			exists, err := isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not exist before test"))

			// 2. Create a pod with the image
			ginkgo.By("Creating a pod with the image")
			testpod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "image-pull-deletion-parallel-test",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "test-container",
						Image:           image,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			pod := e2epod.NewPodClient(f).Create(context.Background(), testpod)

			// 3. Waiting for the pod to be scheduled
			ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
			err = waitForPodConditionWithShortPoll(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", time.Minute, func(pod *v1.Pod) (bool, error) {
				return pod.Spec.NodeName != "", nil
			})
			framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

			// 4. Wait for the "Pulling" event and then delete the pod
			ginkgo.By("Waiting for the 'Pulling' event")
			gomega.Eventually(ctx, func(ctx context.Context) string {
				pullingEvent, err := getImagePullAttempts(ctx, f, pod.Name)
				framework.ExpectNoError(err)
				return pullingEvent.Reason
			}).WithTimeout(time.Minute).WithPolling(time.Millisecond * 10).Should(gomega.Equal(kubeletevents.PullingImage))

			err = e2epod.NewPodClient(f).Delete(ctx, testpod.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			// 5. Wait for 10 seconds and check if the image is pulled
			ginkgo.By("Waiting for 10 seconds")
			time.Sleep(10 * time.Second)

			ginkgo.By("Checking if the image is pulled")
			exists, err = isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// With parallel pull, the image might still be pulled even after pod deletion
			// If image is pulled, test should fail
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not be pulled after pod deletion with parallel pull"))

			// 6. Ensure the image is available after the test
			ginkgo.DeferCleanup(func() error {
				ginkgo.By("Pulling the image back after test")
				return pullImageWithCRI(context.Background(), is, image)
			})
		})

		ginkgo.It("canceling one of the concurrent image pulls should not affect the others", func(ctx context.Context) {
			// Get a test image
			image := imageutils.GetE2EImage(imageutils.Agnhost)

			// 1. Remove the image before creating the pod
			ginkgo.By("Removing the image before creating the pod")
			err := removeImageWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// Verify image is removed
			exists, err := isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not exist before test"))

			// 2. Create three pods with the same image concurrently
			ginkgo.By("Creating three pods with the same image concurrently")
			var pods []*v1.Pod
			for i := 0; i < 3; i++ {
				pod := e2epod.NewPodClient(f).Create(ctx, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("image-pull-cancel-%d", i),
						Namespace: f.Namespace.Name,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:            "test-container",
							Image:           image,
							ImagePullPolicy: v1.PullAlways,
						}},
						RestartPolicy: v1.RestartPolicyNever,
					},
				})
				pods = append(pods, pod)
			}

			// 3. Wait for all pods to be scheduled
			ginkgo.By("Waiting for all pods to be scheduled")
			for _, pod := range pods {
				err = waitForPodConditionWithShortPoll(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", time.Minute, func(pod *v1.Pod) (bool, error) {
					return pod.Spec.NodeName != "", nil
				})
				framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)
			}

			// 4. Wait for the "Pulling" event and then delete the pod
			ginkgo.By("Waiting for the 'Pulling' event")
			gomega.Eventually(ctx, func(ctx context.Context) string {
				pullingEvent, err := getImagePullAttempts(ctx, f, pods[0].Name)
				framework.ExpectNoError(err)
				return pullingEvent.Reason
			}).WithTimeout(time.Minute).WithPolling(time.Millisecond * 10).Should(gomega.Equal(kubeletevents.PullingImage))
			ginkgo.By("Deleting the first pod")

			err = e2epod.NewPodClient(f).Delete(ctx, pods[0].Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			// 5. Wait for 10 seconds and check if the other pods are running
			ginkgo.By("Waiting for 10 seconds and checking other pods")
			time.Sleep(10 * time.Second)

			// Check the remaining pods
			for i := 1; i < len(pods); i++ {
				pod := pods[i]
				ginkgo.By(fmt.Sprintf("Waiting for pod %s to be running", pod.Name))
				err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
				framework.ExpectNoError(err, "Pod %s should be running", pod.Name)
			}

			// 6. Ensure the image is available after the test
			ginkgo.DeferCleanup(func() error {
				ginkgo.By("Pulling the image back after test")
				return pullImageWithCRI(context.Background(), is, image)
			})
		})
	})

	ginkgo.Context("with serialized image pull", func() {
		// Configure kubelet to use serialized image pulls
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = true
			initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		})
		ginkgo.It("Should not pull image after pod deletion with serialized pull", func(ctx context.Context) {
			// Get a test image
			image := imageutils.GetE2EImage(imageutils.Agnhost)

			// 1. Remove the image before creating the pod
			ginkgo.By("Removing the image before creating the pod")
			err := removeImageWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// Verify image is removed
			exists, err := isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not exist before test"))

			// 2. Create a pod with the image
			ginkgo.By("Creating a pod with the image")
			testpod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "image-pull-deletion-serial-test",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "test-container",
						Image:           image,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			pod := e2epod.NewPodClient(f).Create(context.Background(), testpod)

			// 3. Waiting for the pod to be scheduled
			ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
			err = waitForPodConditionWithShortPoll(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", time.Minute, func(pod *v1.Pod) (bool, error) {
				return pod.Spec.NodeName != "", nil
			})
			framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

			// 4. Wait for the "Pulling" event and then delete the pod
			ginkgo.By("Waiting for the 'Pulling' event")
			gomega.Eventually(ctx, func(ctx context.Context) string {
				pullingEvent, err := getImagePullAttempts(ctx, f, pod.Name)
				framework.ExpectNoError(err)
				return pullingEvent.Reason
			}).WithTimeout(time.Minute).WithPolling(time.Millisecond * 10).Should(gomega.Equal(kubeletevents.PullingImage))

			ginkgo.By("Deleting the pod")
			err = e2epod.NewPodClient(f).Delete(context.Background(), pod.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			// 5. Wait for 10 seconds and check if the image is pulled
			ginkgo.By("Waiting for 10 seconds")
			time.Sleep(10 * time.Second)

			ginkgo.By("Checking if the image is pulled")
			exists, err = isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// With serialized pull, the image should be pulled after pod deletion
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not be pulled after pod deletion with serialized pull"))

			// 6. Ensure the image is available after the test
			ginkgo.DeferCleanup(func() error {
				ginkgo.By("Pulling the image back after test")
				return pullImageWithCRI(context.Background(), is, image)
			})
		})
	})
})

var _ = SIGDescribe("Disallow cancel pull image", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var is internalapi.ImageManagerService

	ginkgo.BeforeEach(func() {
		var err error
		_, is, err = getCRIClient()
		framework.ExpectNoError(err)
	})

	ginkgo.Context("with parallel image pull", func() {
		// Configure kubelet to use parallel image pulls
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.It("Should pull image after pod deletion with parallel pull", func(ctx context.Context) {
			// Get a test image
			image := imageutils.GetE2EImage(imageutils.Agnhost)

			// 1. Remove the image before creating the pod
			ginkgo.By("Removing the image before creating the pod")
			err := removeImageWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// Verify image is removed
			exists, err := isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not exist before test"))

			// 2. Create a pod with the image
			ginkgo.By("Creating a pod with the image")
			testpod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "image-pull-deletion-parallel-test",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "test-container",
						Image:           image,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			pod := e2epod.NewPodClient(f).Create(context.Background(), testpod)

			// 3. Waiting for the pod to be scheduled
			ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
			err = waitForPodConditionWithShortPoll(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", time.Minute, func(pod *v1.Pod) (bool, error) {
				return pod.Spec.NodeName != "", nil
			})
			framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

			// 4. Wait for the "Pulling" event and then delete the pod
			ginkgo.By("Waiting for the 'Pulling' event")
			gomega.Eventually(ctx, func(ctx context.Context) string {
				pullingEvent, err := getImagePullAttempts(ctx, f, pod.Name)
				framework.ExpectNoError(err)
				return pullingEvent.Reason
			}).WithTimeout(time.Minute).WithPolling(time.Millisecond * 10).Should(gomega.Equal(kubeletevents.PullingImage))

			ginkgo.By("Deleting the pod")
			err = e2epod.NewPodClient(f).Delete(context.Background(), pod.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			// 5. Wait for 10 seconds and check if the image is pulled
			ginkgo.By("Waiting for 10 seconds")
			time.Sleep(10 * time.Second)

			ginkgo.By("Checking if the image is pulled")
			exists, err = isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// With parallel pull, the image should be pulled after pod deletion
			gomega.Expect(exists).To(gomega.BeTrueBecause("Image should be pulled after pod deletion with parallel pull"))

			// 6. Ensure the image is available after the test
			ginkgo.DeferCleanup(func() error {
				ginkgo.By("Pulling the image back after test")
				return pullImageWithCRI(context.Background(), is, image)
			})
		})
	})

	ginkgo.Context("with serialized image pull", func() {
		// Configure kubelet to use serialized image pulls
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = true
			initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		})
		ginkgo.It("Should pull image after pod deletion with serialized pull", func(ctx context.Context) {
			// Get a test image
			image := imageutils.GetE2EImage(imageutils.Agnhost)

			// 1. Remove the image before creating the pod
			ginkgo.By("Removing the image before creating the pod")
			err := removeImageWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// Verify image is removed
			exists, err := isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)
			gomega.Expect(exists).To(gomega.BeFalseBecause("Image should not exist before test"))

			// 2. Create a pod with the image
			ginkgo.By("Creating a pod with the image")
			testpod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "image-pull-deletion-serial-test",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "test-container",
						Image:           image,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			pod := e2epod.NewPodClient(f).Create(context.Background(), testpod)

			// 3. Waiting for the pod to be scheduled
			ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
			err = waitForPodConditionWithShortPoll(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", time.Minute, func(pod *v1.Pod) (bool, error) {
				return pod.Spec.NodeName != "", nil
			})
			framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

			// 4. Wait for the "Pulling" event and then delete the pod
			ginkgo.By("Waiting for the 'Pulling' event")
			gomega.Eventually(ctx, func(ctx context.Context) string {
				pullingEvent, err := getImagePullAttempts(ctx, f, pod.Name)
				framework.ExpectNoError(err)
				return pullingEvent.Reason
			}).WithTimeout(time.Minute).WithPolling(time.Millisecond * 10).Should(gomega.Equal(kubeletevents.PullingImage))

			ginkgo.By("Deleting the pod")
			err = e2epod.NewPodClient(f).Delete(context.Background(), pod.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			// 5. Wait for 10 seconds and check if the image is pulled
			ginkgo.By("Waiting for 10 seconds")
			time.Sleep(10 * time.Second)

			ginkgo.By("Checking if the image is pulled")
			exists, err = isImageExistWithCRI(context.Background(), is, image)
			framework.ExpectNoError(err)

			// With serialized pull, the image should not be pulled after pod deletion
			gomega.Expect(exists).To(gomega.BeTrueBecause("Image should be pulled after pod deletion with serialized pull"))

			// 6. Ensure the image is available after the test
			ginkgo.DeferCleanup(func() error {
				ginkgo.By("Pulling the image back after test")
				return pullImageWithCRI(context.Background(), is, image)
			})
		})
	})
})

func getPodImagePullDurations(ctx context.Context, f *framework.Framework, testpods []*v1.Pod) (map[string]*pulledStruct, map[string]metav1.Time, map[string]metav1.Time, error) {
	events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, nil, nil, err
	}

	imagePulled := map[string]*pulledStruct{}
	podStartTime := map[string]metav1.Time{}
	podEndTime := map[string]metav1.Time{}

	for _, event := range events.Items {
		if event.Reason == kubeletevents.PulledImage {
			podEndTime[event.InvolvedObject.Name] = event.CreationTimestamp
			for _, testpod := range testpods {
				if event.InvolvedObject.Name == testpod.Name {
					pulled, err := getDurationsFromPulledEventMsg(event.Message)
					if err != nil {
						return nil, nil, nil, err
					}
					imagePulled[testpod.Name] = pulled
					break
				}
			}
		} else if event.Reason == kubeletevents.PullingImage {
			podStartTime[event.InvolvedObject.Name] = event.CreationTimestamp
		}
	}

	return imagePulled, podStartTime, podEndTime, nil
}

// waitForPodConditionWithShortPoll waits for a pod to match the specified condition.
// Compared to the generic WaitForPodCondition, it uses a shorter polling interval for faster response.
func waitForPodConditionWithShortPoll(ctx context.Context, c clientset.Interface, ns, podName, conditionDesc string, timeout time.Duration, condition podCondition) error {
	return framework.Gomega().
		Eventually(ctx, framework.RetryNotFound(framework.GetObject(c.CoreV1().Pods(ns).Get, podName, metav1.GetOptions{}))).
		WithTimeout(timeout).
		WithPolling(time.Millisecond * 50).
		Should(framework.MakeMatcher(func(pod *v1.Pod) (func() string, error) {
			done, err := condition(pod)
			if err != nil {
				return nil, err
			}
			if done {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("expected pod to be %s, got instead:\n%s", conditionDesc, format.Object(pod, 1))
			}, nil
		}))
}

// as pods are created at the same time and image pull will delay 15s, the image pull time should be overlapped
func checkPodPullingOverlap(podStartTime map[string]metav1.Time, podEndTime map[string]metav1.Time, testpods []*v1.Pod) {
	if podStartTime[testpods[0].Name].Time.Before(podStartTime[testpods[1].Name].Time) && podEndTime[testpods[0].Name].Time.Before(podStartTime[testpods[1].Name].Time) {
		framework.Failf("%v pulling time and %v pulling time are not overlapped", testpods[0].Name, testpods[1].Name)
	} else if podStartTime[testpods[0].Name].Time.After(podStartTime[testpods[1].Name].Time) && podStartTime[testpods[0].Name].Time.After(podEndTime[testpods[1].Name].Time) {
		framework.Failf("%v pulling time and %v pulling time are not overlapped", testpods[0].Name, testpods[1].Name)
	}
}

func prepareAndCleanup(ctx context.Context, f *framework.Framework) (testpods []*v1.Pod) {
	// cuda images are > 2Gi and it will reduce the flaky rate
	image1 := imageutils.GetE2EImage(imageutils.AgnhostPrev)
	image2 := imageutils.GetE2EImage(imageutils.Agnhost)
	node := getNodeName(ctx, f)

	testpod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testpod",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "testpod",
				Image:           image1,
				ImagePullPolicy: v1.PullAlways,
			}},
			NodeName:      node,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	testpod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testpod2",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "testpod2",
				Image:           image2,
				ImagePullPolicy: v1.PullAlways,
			}},
			NodeName:      node,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	testpods = []*v1.Pod{testpod, testpod2}

	ginkgo.By("cleanup images")
	for _, pod := range testpods {
		_ = RemoveImage(ctx, pod.Spec.Containers[0].Image)
	}
	return testpods
}

type pulledStruct struct {
	pulledDuration               time.Duration
	pulledIncludeWaitingDuration time.Duration
}

// getDurationsFromPulledEventMsg will parse two durations in the pulled message
// Example msg: `Successfully pulled image \"busybox:1.28\" in 39.356s (49.356s including waiting). Image size: 41901587 bytes.`
func getDurationsFromPulledEventMsg(msg string) (*pulledStruct, error) {
	splits := strings.Split(msg, " ")
	if len(splits) != 13 {
		return nil, fmt.Errorf("pull event message should be spilted to 13: %d", len(splits))
	}
	pulledDuration, err := time.ParseDuration(splits[5])
	if err != nil {
		return nil, err
	}
	// to skip '('
	pulledIncludeWaitingDuration, err := time.ParseDuration(splits[6][1:])
	if err != nil {
		return nil, err
	}
	return &pulledStruct{
		pulledDuration:               pulledDuration,
		pulledIncludeWaitingDuration: pulledIncludeWaitingDuration,
	}, nil
}

func getImagePullAttempts(ctx context.Context, f *framework.Framework, podName string) (v1.Event, error) {
	event := v1.Event{}
	e, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return event, err
	}

	for _, event := range e.Items {
		if event.InvolvedObject.Name == podName && event.Reason == kubeletevents.PullingImage {
			return event, nil
		}
	}
	return event, nil
}

// isImageExistWithCRI checks if the image exists using CRI client
// Returns true if the image exists, false otherwise
func isImageExistWithCRI(ctx context.Context, is internalapi.ImageManagerService, image string) (bool, error) {
	// List all images using CRI client
	resp, err := is.ListImages(ctx, &runtimeapi.ImageFilter{})
	if err != nil {
		return false, fmt.Errorf("failed to list images: %w", err)
	}

	// Check if the image exists in the list
	for _, img := range resp {
		for _, tag := range img.RepoTags {
			if tag == image {
				return true, nil
			}
		}
	}
	return false, nil
}

// removeImageWithCRI removes the specified image using CRI client
// If the image doesn't exist, it returns nil
func removeImageWithCRI(ctx context.Context, is internalapi.ImageManagerService, image string) error {
	// First check if the image exists
	exists, err := isImageExistWithCRI(ctx, is, image)
	if err != nil {
		return err
	}
	if !exists {
		return nil // Image already doesn't exist
	}

	// Remove the image using CRI client
	err = is.RemoveImage(ctx, &runtimeapi.ImageSpec{Image: image})
	if err != nil {
		return fmt.Errorf("failed to remove image: %w", err)
	}
	return nil
}

// pullImageWithCRI pulls the specified image using CRI client
func pullImageWithCRI(ctx context.Context, is internalapi.ImageManagerService, image string) error {
	// Pull the image using CRI client
	_, err := is.PullImage(ctx, &runtimeapi.ImageSpec{Image: image}, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to pull image: %w", err)
	}
	return nil
}
