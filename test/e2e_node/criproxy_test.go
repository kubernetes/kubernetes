//go:build linux

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
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Examples of using CRI proxy
var _ = SIGDescribe(feature.CriProxy, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cri-proxy-example")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Inject a pull image error exception into the CriProxy", func() {
		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
		})

		ginkgo.It("Pod failed to start due to an image pull error.", func(ctx context.Context) {
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
		})
	})

	ginkgo.Context("Image pull backoff", func() {
		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
		})

	})

	framework.Context("Image volume digest error handling", feature.CriProxy, framework.WithFeatureGate(features.ImageVolumeWithDigest), func() {
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

		getImageVolumePod := func() *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "image-vol-test-" + string(uuid.NewUUID()),
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "image-vol-container-" + string(uuid.NewUUID()),
							Image: imageutils.GetPauseImageName(),
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      volumeName,
									MountPath: "/image-volume-" + string(uuid.NewUUID()),
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: volumeName,
							VolumeSource: v1.VolumeSource{
								Image: &v1.ImageVolumeSource{
									Reference:  imageutils.GetPauseImageName(),
									PullPolicy: v1.PullAlways,
								},
							},
						},
					},
				},
			}
		}

		waitForPodContainerStatuses := func(pod *v1.Pod) *v1.Pod {
			ginkgo.By("Waiting for the pod container statuses")

			var err error
			gomega.Eventually(func() []v1.ContainerStatus {
				pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.Background(), pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				return pod.Status.ContainerStatuses
			}).WithPolling(5*time.Second).WithTimeout(2*time.Minute).Should(gomega.HaveLen(1), "couldn't find expected container status")

			return pod
		}

		getVolumeMountStatus := func(pod *v1.Pod) v1.VolumeMountStatus {
			ginkgo.By("Finding the pod volume mount status")

			var volMountStatus *v1.VolumeMountStatus
			containerStatus := pod.Status.ContainerStatuses[0]

			for i := range pod.Status.ContainerStatuses[0].VolumeMounts {
				if containerStatus.VolumeMounts[i].Name == volumeName {
					volMountStatus = &containerStatus.VolumeMounts[i]
					break
				}
			}
			gomega.ExpectWithOffset(1, volMountStatus).ToNot(gomega.BeNil(), "couldn't find expected volume mount status")

			return *volMountStatus
		}

		ginkgo.It("should expect error log when ImageStatus fails for image volume digest", func(ctx context.Context) {
			const imageStatusErrMsg = "mock error message - ImageStatus failed"

			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.ImageStatus {
					return errors.New(imageStatusErrMsg)
				}
				return nil
			})
			framework.ExpectNoError(err)

			pod := getImageVolumePod()
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			pod = waitForPodContainerStatuses(pod)

			volMountStatus := getVolumeMountStatus(pod)

			if volMountStatus.VolumeStatus.Image != nil {
				ginkgo.Fail(fmt.Sprintf("ImageRef should not be set when ImageStatus fails, but got: %s", volMountStatus.VolumeStatus.Image.ImageRef))
			}

			ginkgo.By("Expecting an error when ImageStatus fails")
			gomega.Eventually(func() error {
				return verifyErrorInKubeletLogs(imageStatusErrMsg)
			}).WithPolling(5*time.Second).WithTimeout(20*time.Second).ToNot(gomega.HaveOccurred(), "Could not verify error in kubelet logs")
		})

		ginkgo.It("should expect error log for image volume with empty Image.Image", func(ctx context.Context) {
			// This test verifies error handling when imageSpec.Image is empty (curVolumeMount.Image.Image == "").

			pod := getImageVolumePod()
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			pod = waitForPodContainerStatuses(pod)

			volMountStatus := getVolumeMountStatus(pod)

			if volMountStatus.VolumeStatus.Image != nil {
				ginkgo.Fail(fmt.Sprintf("ImageRef should not be set when ImageStatus fails, but got: %s", volMountStatus.VolumeStatus.Image.ImageRef))
			}

			ginkgo.By("Expecting an error when imageSpec.Image is empty")
			gomega.Eventually(func() error {
				return verifyErrorInKubeletLogs("image was not found")
			}).WithPolling(5*time.Second).WithTimeout(20*time.Second).ToNot(gomega.HaveOccurred(), "Could not verify error in kubelet logs")
		})
	})

	ginkgo.Context("Inject a pull image timeout exception into the CriProxy", func() {
		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
			ginkgo.DeferCleanup(func() error {
				return resetCRIProxyInjector(e2eCriProxy)
			})
		})

		ginkgo.It("Image pull time exceeded 10 seconds", func(ctx context.Context) {
			const delayTime = 10 * time.Second
			err := addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
				if apiName == criproxy.PullImage {
					time.Sleep(10 * time.Second)
				}
				return nil
			})
			framework.ExpectNoError(err)

			pod := e2epod.NewPodClient(f).Create(ctx, newPullImageAlwaysPod())
			podErr := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(podErr)

			imagePullDuration, err := getPodImagePullDuration(ctx, f, pod.Name)
			framework.ExpectNoError(err)

			gomega.Expect(imagePullDuration).To(gomega.BeNumerically(">=", delayTime), "PullImages should take more than 10 seconds")
		})
	})
})

func getFailedToPullImageMsg(ctx context.Context, f *framework.Framework, podName string) (string, error) {
	events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", err
	}

	for _, event := range events.Items {
		if event.Reason == kubeletevents.FailedToPullImage && event.InvolvedObject.Name == podName {
			return event.Message, nil
		}
	}

	return "", fmt.Errorf("failed to find FailedToPullImage event for pod: %s", podName)
}

func getPodImagePullDuration(ctx context.Context, f *framework.Framework, podName string) (time.Duration, error) {
	events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, err
	}

	var startTime, endTime time.Time
	for _, event := range events.Items {
		if event.InvolvedObject.Name == podName {
			switch event.Reason {
			case kubeletevents.PullingImage:
				startTime = event.FirstTimestamp.Time
			case kubeletevents.PulledImage:
				endTime = event.FirstTimestamp.Time
			}
		}
	}

	if startTime.IsZero() || endTime.IsZero() {
		return 0, fmt.Errorf("failed to find both PullingImage and PulledImage events for pod: %s", podName)
	}

	return endTime.Sub(startTime), nil
}

func newPullImageAlwaysPod() *v1.Pod {
	podName := "cri-proxy-test-" + string(uuid.NewUUID())
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image:           imageutils.GetPauseImageName(),
					Name:            podName,
					ImagePullPolicy: v1.PullAlways,
				},
			},
		},
	}
	return pod
}

func verifyErrorInKubeletLogs(errorMsg string) error {
	kubeletLog, err := os.ReadFile(framework.TestContext.ReportDir + "/kubelet.log")
	if err != nil {
		return fmt.Errorf("could not read kubelet logs: %w", err)
	}

	if !strings.Contains(string(kubeletLog), errorMsg) {
		return fmt.Errorf("error message \"%s\" not found in kubelet logs", errorMsg)
	}

	return nil
}
