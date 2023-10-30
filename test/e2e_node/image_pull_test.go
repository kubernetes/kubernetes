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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/pkg/errors"
)

var _ = SIGDescribe("Pull Image [Serial] [NodeFeature:MaxParallelImagePull]", func() {
	var pod, pod2 *v1.Pod

	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)
	nginxNewImage := imageutils.GetE2EImage(imageutils.NginxNew)
	nginxPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx",
				Image:           nginxImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	nginxNewPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx-new",
				Image:           nginxNewImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	ginkgo.Context("parallel image pull with MaxParallelImagePulls=5", func() {

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("cleanup images")
			RemoveImage(nginxPodDesc.Spec.Containers[0].Image)
			RemoveImage(nginxNewPodDesc.Spec.Containers[0].Image)
		})

		ginkgo.It("should pull immediately if no more than 5 pods", func(ctx context.Context) {
			node := getNodeName(ctx, f)
			nginxPodDesc.Spec.NodeName = node
			nginxNewPodDesc.Spec.NodeName = node
			pod = e2epod.NewPodClient(f).Create(ctx, nginxPodDesc)
			pod2 = e2epod.NewPodClient(f).Create(ctx, nginxNewPodDesc)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod2.Name, f.Namespace.Name, framework.PodStartTimeout))

			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			var nginxPulled, nginxNewPulled pulledStruct
			for _, event := range events.Items {
				var err error
				if event.Reason == kubeletevents.PulledImage {
					if event.InvolvedObject.Name == nginxPodDesc.Name {
						nginxPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					} else if event.InvolvedObject.Name == nginxNewPodDesc.Name {
						nginxNewPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					}
				}
			}
			deletePodSyncByName(ctx, f, pod.Name)
			deletePodSyncByName(ctx, f, pod2.Name)

			// as this is parallel image pulling, the waiting duration should be similar with the pulled duration.
			// use 1.2 as a common ratio
			if float32(nginxNewPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(nginxNewPulled.pulledDuration/time.Millisecond) > 1.2 {
				framework.Failf("the pull duration including waiting %v should be similar with the pulled duration %v",
					nginxNewPulled.pulledIncludeWaitingDuration, nginxNewPulled.pulledDuration)
			}
			if float32(nginxPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(nginxPulled.pulledDuration/time.Millisecond) > 1.2 {
				framework.Failf("the pull duration including waiting %v should be similar with the pulled duration %v",
					nginxPulled.pulledIncludeWaitingDuration, nginxPulled.pulledDuration)
			}
		})

	})
})

var _ = SIGDescribe("Pull Image [Serial]", func() {

	var pod, pod2 *v1.Pod

	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)
	nginxNewImage := imageutils.GetE2EImage(imageutils.NginxNew)
	nginxPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx",
				Image:           nginxImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	nginxNewPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nginx",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "nginx-new",
				Image:           nginxNewImage,
				ImagePullPolicy: v1.PullAlways,
				Command:         []string{"sh"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	ginkgo.Context("serialize image pull", func() {
		// this is the default behavior now.
		// tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
		// 	initialConfig.SerializeImagePulls = true
		// 	initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		// })

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("cleanup images")
			RemoveImage(nginxPodDesc.Spec.Containers[0].Image)
			RemoveImage(nginxNewPodDesc.Spec.Containers[0].Image)
		})

		ginkgo.It("should be waiting more", func(ctx context.Context) {

			node := getNodeName(ctx, f)
			nginxPodDesc.Spec.NodeName = node
			nginxNewPodDesc.Spec.NodeName = node
			pod = e2epod.NewPodClient(f).Create(ctx, nginxPodDesc)
			pod2 = e2epod.NewPodClient(f).Create(ctx, nginxNewPodDesc)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod2.Name, f.Namespace.Name, framework.PodStartTimeout))

			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			var nginxPulled, nginxNewPulled pulledStruct
			for _, event := range events.Items {
				var err error
				if event.Reason == kubeletevents.PulledImage {
					if event.InvolvedObject.Name == nginxPodDesc.Name {
						nginxPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					} else if event.InvolvedObject.Name == nginxNewPodDesc.Name {
						nginxNewPulled, err = getDurationsFromPulledEventMsg(event.Message)
						framework.ExpectNoError(err)
					}
				}
			}
			deletePodSyncByName(ctx, f, pod.Name)
			deletePodSyncByName(ctx, f, pod2.Name)

			// as this is serialize image pulling, the waiting duration should be almost double the duration with the pulled duration.
			// use 1.5 as a common ratio to avoid some overlap during pod creation
			if float32(nginxNewPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(nginxNewPulled.pulledDuration/time.Millisecond) < 1.5 &&
				float32(nginxPulled.pulledIncludeWaitingDuration/time.Millisecond)/float32(nginxPulled.pulledDuration/time.Millisecond) < 1.5 {
				framework.Failf("At least, one of the pull duration including waiting %v/%v should be similar with the pulled duration %v/%v",
					nginxNewPulled.pulledIncludeWaitingDuration, nginxPulled.pulledIncludeWaitingDuration, nginxNewPulled.pulledDuration, nginxPulled.pulledDuration)
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
