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
)

var _ = SIGDescribe("Pull Image [NodeFeature: MaxParallelImagePull]", func() {

	f := framework.NewDefaultFramework("pull-image-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)
	nginxNewImage := imageutils.GetE2EImage(imageutils.NginxNew)
	nginxPodDesc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nginx",
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
			Name: "nginx",
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

	ginkgo.Context("ParalleImagePull with 2", func() {

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = false
			initialConfig.MaxParallelImagePulls = ptr.To[int32](5)
		})

		ginkgo.It("should pull immediately if no more than 5 pods", func(ctx context.Context) {
			node := getNodeName(ctx, f)
			nginxPodDesc.Spec.NodeName = node
			nginxNewPodDesc.Spec.NodeName = node
			pod := e2epod.NewPodClient(f).Create(ctx, nginxPodDesc)
			pod2 := e2epod.NewPodClient(f).Create(ctx, nginxNewPodDesc)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod2.Name, f.Namespace.Name, framework.PodStartTimeout))

			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			pulledEvents = []pulledStruct{}
			for _, event := range events.Items {
				if event.Reason == kubeletevents.PulledImage {

				}
			}

			// Successfully pulled image \"busybox:1.28\" in 39.356s (39.356s including waiting)",
			// 1. get Pulled event of the pod
			// 2. check the two time is similar? +- 1s
			// 3. for five image pulling, including waiting < pulling + 5s

		})

		ginkgo.It("should be blocked when maxParallelImagePulls is reached", func(ctx context.Context) {

			pod := e2epod.NewPodClient(f).Create(ctx, nginxPodDesc)

			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
			runningPod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// Successfully pulled image \"busybox:1.28\" in 39.356s (39.356s including waiting)",
			// 1. get Pulled event of the pod
			// 2. check the two time is similar? +- 1s
			// 3. for six image pulling, including waiting > pulling + 10s
		})
	})

	ginkgo.Context("SerializeImagePull", func() {

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.SerializeImagePulls = true
			initialConfig.MaxParallelImagePulls = ptr.To[int32](1)
		})

		ginkgo.It("should be waiting more", func(ctx context.Context) {

			pod := e2epod.NewPodClient(f).Create(ctx, nginxPodDesc)

			framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
				f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
			runningPod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// Successfully pulled image \"busybox:1.28\" in 39.356s (39.356s including waiting)",
			// 1. get Pulled event of the pod
			// 2. check the two time is similar? +- 1s
			// 3. for five image pulling, including waiting < pulling + 5s

		})
	})
})

type pulledStruct struct {
	pulledDuration               time.Duration
	pulledIncludeWaitingDuration time.Duration
}
