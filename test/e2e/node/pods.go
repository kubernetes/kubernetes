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
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("Pods Extended", func() {
	f := framework.NewDefaultFramework("pods")

	framework.KubeDescribe("Delete Grace Period", func() {
		var podClient *framework.PodClient
		BeforeEach(func() {
			podClient = f.PodClient()
		})
		// TODO: Fix Flaky issue #68066 and then re-add this back into Conformance Suite
		/*
			Release : v1.9
			Testname: Pods, delete grace period
			Description: Create a pod, make sure it is running, create a watch to observe Pod creation. Create a 'kubectl local proxy', capture the port the proxy is listening. Using the http client send a ‘delete’ with gracePeriodSeconds=30. Pod SHOULD get deleted within 30 seconds.
		*/
		It("should be submitted and removed  [Flaky]", func() {
			By("creating the pod")
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
							Name:  "nginx",
							Image: imageutils.GetE2EImage(imageutils.Nginx),
						},
					},
				},
			}

			By("setting up selector")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			pods, err := podClient.List(options)
			Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
			Expect(len(pods.Items)).To(Equal(0))
			options = metav1.ListOptions{
				LabelSelector:   selector.String(),
				ResourceVersion: pods.ListMeta.ResourceVersion,
			}

			By("submitting the pod to kubernetes")
			podClient.Create(pod)

			By("verifying the pod is in kubernetes")
			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(options)
			Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
			Expect(len(pods.Items)).To(Equal(1))

			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(f.WaitForPodRunning(pod.Name))
			// save the running pod
			pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "failed to GET scheduled pod")

			// start local proxy, so we can send graceful deletion over query string, rather than body parameter
			cmd := framework.KubectlCmd("proxy", "-p", "0")
			stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
			Expect(err).NotTo(HaveOccurred(), "failed to start up proxy")
			defer stdout.Close()
			defer stderr.Close()
			defer framework.TryKill(cmd)
			buf := make([]byte, 128)
			var n int
			n, err = stdout.Read(buf)
			Expect(err).NotTo(HaveOccurred(), "failed to read from kubectl proxy stdout")
			output := string(buf[:n])
			proxyRegexp := regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")
			match := proxyRegexp.FindStringSubmatch(output)
			Expect(len(match)).To(Equal(2))
			port, err := strconv.Atoi(match[1])
			Expect(err).NotTo(HaveOccurred(), "failed to convert port into string")

			endpoint := fmt.Sprintf("http://localhost:%d/api/v1/namespaces/%s/pods/%s?gracePeriodSeconds=30", port, pod.Namespace, pod.Name)
			tr := &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			}
			client := &http.Client{Transport: tr}
			req, err := http.NewRequest("DELETE", endpoint, nil)
			Expect(err).NotTo(HaveOccurred(), "failed to create http request")

			By("deleting the pod gracefully")
			rsp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred(), "failed to use http client to send delete")
			Expect(rsp.StatusCode).Should(Equal(http.StatusOK), "failed to delete gracefully by client request")
			var lastPod v1.Pod
			err = json.NewDecoder(rsp.Body).Decode(&lastPod)
			Expect(err).NotTo(HaveOccurred(), "failed to decode graceful termination proxy response")

			defer rsp.Body.Close()

			By("verifying the kubelet observed the termination notice")

			Expect(wait.Poll(time.Second*5, time.Second*30, func() (bool, error) {
				podList, err := framework.GetKubeletPods(f.ClientSet, pod.Spec.NodeName)
				if err != nil {
					framework.Logf("Unable to retrieve kubelet pods for node %v: %v", pod.Spec.NodeName, err)
					return false, nil
				}
				for _, kubeletPod := range podList.Items {
					if pod.Name != kubeletPod.Name {
						continue
					}
					if kubeletPod.ObjectMeta.DeletionTimestamp == nil {
						framework.Logf("deletion has not yet been observed")
						return false, nil
					}
					return false, nil
				}
				framework.Logf("no pod exists with the name we were looking for, assuming the termination request was observed and completed")
				return true, nil
			})).NotTo(HaveOccurred(), "kubelet never observed the termination notice")

			Expect(lastPod.DeletionTimestamp).ToNot(BeNil())
			Expect(lastPod.Spec.TerminationGracePeriodSeconds).ToNot(BeZero())

			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(options)
			Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
			Expect(len(pods.Items)).To(Equal(0))

		})
	})

	framework.KubeDescribe("Pods Set QOS Class", func() {
		var podClient *framework.PodClient
		BeforeEach(func() {
			podClient = f.PodClient()
		})
		/*
			Release : v1.9
			Testname: Pods, QOS
			Description:  Create a Pod with CPU and Memory request and limits. Pos status MUST have QOSClass set to PodQOSGuaranteed.
		*/
		framework.ConformanceIt("should be submitted and removed ", func() {
			By("creating the pod")
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
							Name:  "nginx",
							Image: imageutils.GetE2EImage(imageutils.Nginx),
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

			By("submitting the pod to kubernetes")
			podClient.Create(pod)

			By("verifying QOS class is set on the pod")
			pod, err := podClient.Get(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
			Expect(pod.Status.QOSClass == v1.PodQOSGuaranteed)
		})
	})
})
