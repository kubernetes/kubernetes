/*
Copyright 2016 The Kubernetes Authors.

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
	"strconv"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

func getAvailableBytes() uint64 {
	summary, err := getNodeSummary()
	if err != nil {
		framework.ExpectNoError(err)
	}
	return *(summary.Node.Memory.AvailableBytes)
}

func makePod(name string, namespace string, requestedBytes string) *v1.Pod {

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "busybox",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sleep", "30"},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse(requestedBytes),
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

var _ = SIGDescribe("Multiple Pods Addition TEST123", func() {
	f := framework.NewDefaultFramework("multiple-pods-additions")
	podNames := []string{"test1-pod", "test2-pod", "test3-pod"}

	ginkgo.Context("when creating multiple pods ", func() {

		ginkgo.BeforeEach(func() {

		})

		ginkgo.AfterEach(func() {
			for _, podName := range podNames {
				gp := int64(0)
				f.PodClient().DeleteSync(podName, metav1.DeleteOptions{GracePeriodSeconds: &gp}, 2*time.Minute)
			}
		})

		ginkgo.It("should be create pods that reach completion TEST321 [NodeConformance]", func() {
			ginkgo.By("create pods")

			availableBytes := getAvailableBytes()
			requestedBytes := strconv.FormatUint(availableBytes/2, 10)

			for _, podName := range podNames {
				framework.Logf("requested memory %s out of available memory %d for pod %s", requestedBytes, availableBytes, podName)
				pod := makePod(podName, f.Namespace.Name, requestedBytes)
				f.PodClient().Create(pod)
				err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, podName, f.Namespace.Name)
				framework.ExpectNoError(err)
			}

			// ginkgo.By("wait for a pod to be completed")

			// var wg sync.WaitGroup
			// for _, podName := range podNames {
			// 	wg.Add(1)
			// 	go func(podname string) {
			// 		// defer ginkgo.GinkgoRecover()

			// 		wg.Done()
			// 	}(podName)
			// }
			// wg.Wait()

		})
	})

})
