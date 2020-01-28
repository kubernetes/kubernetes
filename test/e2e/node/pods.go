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

package node

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Pods Extended", func() {
	f := framework.NewDefaultFramework("pods")

	framework.KubeDescribe("Delete Grace Period", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		/*
			Release : v1.15
			Testname: Pods, delete grace period
			Description: Create a pod, make sure it is running. Using the http client send a ‘delete’ with gracePeriodSeconds=30. Pod SHOULD get deleted within 30 seconds.
		*/
		ginkgo.It("should be submitted and removed [Flaky]", func() {
			ginkgo.By("creating the pod")
			name := "pod-submit-remove-" + string(uuid.NewUUID())
			value := strconv.Itoa(time.Now().Nanosecond())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						"name": "foo",
						"time": value,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "agnhost",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"pause"},
						},
					},
				},
			}

			ginkgo.By("setting up selector")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			pods, err := podClient.List(options)
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(len(pods.Items), 0)
			options = metav1.ListOptions{
				LabelSelector:   selector.String(),
				ResourceVersion: pods.ListMeta.ResourceVersion,
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(pod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(options)
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(len(pods.Items), 1)

			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(f.WaitForPodRunning(pod.Name))
			// save the running pod
			pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to GET scheduled pod")

			ginkgo.By("deleting the pod gracefully")
			var lastPod v1.Pod
			var statusCode int
			err = f.ClientSet.CoreV1().RESTClient().Delete().AbsPath("/api/v1/namespaces", pod.Namespace, "pods", pod.Name).Param("gracePeriodSeconds", "30").Do(context.TODO()).StatusCode(&statusCode).Into(&lastPod)
			framework.ExpectNoError(err, "failed to use http client to send delete")
			framework.ExpectEqual(statusCode, http.StatusOK, "failed to delete gracefully by client request")

			ginkgo.By("verifying the kubelet observed the termination notice")

			start := time.Now()
			err = wait.Poll(time.Second*5, time.Second*30, func() (bool, error) {
				podList, err := e2ekubelet.GetKubeletPods(f.ClientSet, pod.Spec.NodeName)
				if err != nil {
					framework.Logf("Unable to retrieve kubelet pods for node %v: %v", pod.Spec.NodeName, err)
					return false, nil
				}
				for _, kubeletPod := range podList.Items {
					if pod.Name != kubeletPod.Name || pod.Namespace != kubeletPod.Namespace {
						continue
					}
					if kubeletPod.ObjectMeta.DeletionTimestamp == nil {
						framework.Logf("deletion has not yet been observed")
						return false, nil
					}
					data, _ := json.Marshal(kubeletPod)
					framework.Logf("start=%s, now=%s, kubelet pod: %s", start, time.Now(), string(data))
					return false, nil
				}
				framework.Logf("no pod exists with the name we were looking for, assuming the termination request was observed and completed")
				return true, nil
			})
			framework.ExpectNoError(err, "kubelet never observed the termination notice")

			framework.ExpectNotEqual(lastPod.DeletionTimestamp, nil)
			framework.ExpectNotEqual(lastPod.Spec.TerminationGracePeriodSeconds, 0)

			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(options)
			framework.ExpectNoError(err, "failed to query for pods")
			framework.ExpectEqual(len(pods.Items), 0)

		})
	})

	framework.KubeDescribe("Pods Set QOS Class", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		/*
			Release : v1.9
			Testname: Pods, QOS
			Description:  Create a Pod with CPU and Memory request and limits. Pod status MUST have QOSClass set to PodQOSGuaranteed.
		*/
		framework.ConformanceIt("should be set on Pods with matching resource requests and limits for memory and cpu", func() {
			ginkgo.By("creating the pod")
			name := "pod-qos-class-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						"name": name,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "agnhost",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"pause"},
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(pod)

			ginkgo.By("verifying QOS class is set on the pod")
			pod, err := podClient.Get(name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(pod.Status.QOSClass, v1.PodQOSGuaranteed)
		})
	})
})
