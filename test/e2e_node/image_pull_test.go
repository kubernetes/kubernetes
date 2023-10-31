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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = SIGDescribe("Pull Image", framework.WithSerial(), nodefeature.MaxParallelImagePull, func() {

	f := framework.NewDefaultFramework("parallel-pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
	httpdNewImage := imageutils.GetE2EImage(imageutils.HttpdNew)
	var testpod, testpod2 *v1.Pod

	ginkgo.Context("parallel image pull with MaxParallelImagePulls=5", func() {

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.BeforeEach(func(ctx context.Context) {
			testpod = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "httpd",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "httpd",
						Image:           httpdImage,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			testpod2 = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "httpd2",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "httpd-new",
						Image:           httpdNewImage,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			ginkgo.By("cleanup images")
			_ = RemoveImage(testpod.Spec.Containers[0].Image)
			_ = RemoveImage(testpod2.Spec.Containers[0].Image)
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("cleanup pods")
			if testpod != nil {
				deletePodSyncByName(ctx, f, testpod.Name)
			}
			if testpod2 != nil {
				deletePodSyncByName(ctx, f, testpod2.Name)
			}
		})

		ginkgo.It("should pull immediately if no more than 5 pods", func(ctx context.Context) {
			node := getNodeName(ctx, f)
			testpod.Spec.NodeName = node
			testpod2.Spec.NodeName = node

			pod := e2epod.NewPodClient(f).Create(ctx, testpod)
			pod2 := e2epod.NewPodClient(f).Create(ctx, testpod2)
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodFailed {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)
			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod2.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodFailed {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)

			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			var httpdPulled, httpdNewPulled pulledStruct
			for _, event := range events.Items {
				var err error
				if event.Reason == kubeletevents.PulledImage {
					if event.InvolvedObject.Name == testpod.Name {
						httpdPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					} else if event.InvolvedObject.Name == testpod2.Name {
						httpdNewPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					}
				}
			}

			// as this is parallel image pulling, the waiting duration should be similar with the pulled duration.
			// use 1.2 as a common ratio
			if float32(httpdNewPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(httpdNewPulled.pulledDuration/time.Millisecond) > 1.2 {
				framework.Failf("the pull duration including waiting %v should be similar with the pulled duration %v",
					httpdNewPulled.pulledIncludeWaitingDuration, httpdNewPulled.pulledDuration)
			}
			if float32(httpdPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(httpdPulled.pulledDuration/time.Millisecond) > 1.2 {
				framework.Failf("the pull duration including waiting %v should be similar with the pulled duration %v",
					httpdPulled.pulledIncludeWaitingDuration, httpdPulled.pulledDuration)
			}
		})

	})
})

var _ = SIGDescribe("Pull Image", framework.WithSerial(), nodefeature.MaxParallelImagePull, func() {

	f := framework.NewDefaultFramework("serialize-pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("serialize image pull", func() {
		// this is the default behavior now.
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = true
			initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		})

		httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
		httpdNewImage := imageutils.GetE2EImage(imageutils.HttpdNew)
		var testpod, testpod2 *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			testpod = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "httpd",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "httpd",
						Image:           httpdImage,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			testpod2 = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "httpd2",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "httpd-new",
						Image:           httpdNewImage,
						ImagePullPolicy: v1.PullAlways,
					}},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}

			ginkgo.By("cleanup images")
			_ = RemoveImage(testpod.Spec.Containers[0].Image)
			_ = RemoveImage(testpod2.Spec.Containers[0].Image)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("cleanup pods")
			if testpod != nil {
				deletePodSyncByName(ctx, f, testpod.Name)
			}
			if testpod2 != nil {
				deletePodSyncByName(ctx, f, testpod2.Name)
			}
		})

		ginkgo.It("should be waiting more", func(ctx context.Context) {

			node := getNodeName(ctx, f)
			testpod.Spec.NodeName = node
			testpod2.Spec.NodeName = node
			pod := e2epod.NewPodClient(f).Create(ctx, testpod)
			pod2 := e2epod.NewPodClient(f).Create(ctx, testpod2)
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodFailed {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)
			err = e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod2.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodFailed {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)

			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			var httpdPulled, httpdNewPulled pulledStruct
			for _, event := range events.Items {
				var err error
				if event.Reason == kubeletevents.PulledImage {
					if event.InvolvedObject.Name == testpod.Name {
						httpdPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					} else if event.InvolvedObject.Name == testpod2.Name {
						httpdNewPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					}
				}
			}

			// as this is serialize image pulling, the waiting duration should be almost double the duration with the pulled duration.
			// use 1.5 as a common ratio to avoid some overlap during pod creation
			if float32(httpdNewPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(httpdNewPulled.pulledDuration/time.Millisecond) < 1.5 &&
				float32(httpdPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(httpdPulled.pulledDuration/time.Millisecond) < 1.5 {
				framework.Failf("At least, one of the pull duration including waiting %v/%v should be similar with the pulled duration %v/%v",
					httpdNewPulled.pulledIncludeWaitingDuration, httpdPulled.pulledIncludeWaitingDuration, httpdNewPulled.pulledDuration, httpdPulled.pulledDuration)
			}

		})

	})
})

type pulledStruct struct {
	pulledDuration               time.Duration
	pulledIncludeWaitingDuration time.Duration
}

// getDurationsFromPulledEventMsg will parse two durations in the pulled message
// Example msg: `Successfully pulled image \"busybox:1.28\" in 39.356s (49.356s including waiting)`
func getDurationsFromPulledEventMsg(msg string) (pulled pulledStruct, err error) {
	splits := strings.Split(msg, " ")
	if len(splits) == 9 {
		pulled.pulledDuration, err = time.ParseDuration(splits[5])
		if err != nil {
			return
		}
		// to skip '('
		pulled.pulledIncludeWaitingDuration, err = time.ParseDuration(splits[6][1:])
		if err != nil {
			return
		}
	} else {
		err = errors.Errorf("pull event message should be spilted to 8: %d", len(splits))
	}
	return
}
