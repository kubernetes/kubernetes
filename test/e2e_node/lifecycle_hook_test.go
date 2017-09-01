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

package e2e_node

import (
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	var podClient *framework.PodClient
	const (
		podCheckInterval     = 1 * time.Second
		postStartWaitTimeout = 2 * time.Minute
		preStopWaitTimeout   = 30 * time.Second
	)
	Context("when create a pod with lifecycle hook", func() {
		var targetIP string
		podHandleHookRequest := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-handle-http-request",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pod-handle-http-request",
						Image: "gcr.io/google_containers/netexec:1.7",
						Ports: []v1.ContainerPort{
							{
								ContainerPort: 8080,
								Protocol:      v1.ProtocolTCP,
							},
						},
					},
				},
			},
		}
		BeforeEach(func() {
			podClient = f.PodClient()
			By("create the container to handle the HTTPGet hook request.")
			newPod := podClient.CreateSync(podHandleHookRequest)
			targetIP = newPod.Status.PodIP
		})
		testPodWithHook := func(podWithHook *v1.Pod) {
			By("create the pod with lifecycle hook")
			podClient.CreateSync(podWithHook)
			if podWithHook.Spec.Containers[0].Lifecycle.PostStart != nil {
				By("check poststart hook")
				Eventually(func() error {
					return podClient.MatchContainerOutput(podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[0].Name,
						`GET /echo\?msg=poststart`)
				}, postStartWaitTimeout, podCheckInterval).Should(BeNil())
			}
			By("delete the pod with lifecycle hook")
			podClient.DeleteSync(podWithHook.Name, metav1.NewDeleteOptions(15), framework.DefaultPodDeletionTimeout)
			if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
				By("check prestop hook")
				Eventually(func() error {
					return podClient.MatchContainerOutput(podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[0].Name,
						`GET /echo\?msg=prestop`)
				}, preStopWaitTimeout, podCheckInterval).Should(BeNil())
			}
		}
		It("should execute poststart exec hook properly [Conformance]", func() {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetIP + ":8080/echo?msg=poststart"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-exec-hook", "gcr.io/google_containers/hostexec:1.2", lifecycle)
			testPodWithHook(podWithHook)
		})
		It("should execute prestop exec hook properly [Conformance]", func() {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetIP + ":8080/echo?msg=prestop"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-exec-hook", "gcr.io/google_containers/hostexec:1.2", lifecycle)
			testPodWithHook(podWithHook)
		})
		It("should execute poststart http hook properly [Conformance]", func() {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=poststart",
						Host: targetIP,
						Port: intstr.FromInt(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-http-hook", framework.GetPauseImageNameForHostArch(), lifecycle)
			testPodWithHook(podWithHook)
		})
		It("should execute prestop http hook properly [Conformance]", func() {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=prestop",
						Host: targetIP,
						Port: intstr.FromInt(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-http-hook", framework.GetPauseImageNameForHostArch(), lifecycle)
			testPodWithHook(podWithHook)
		})
	})
})

func getPodWithHook(name string, image string, lifecycle *v1.Lifecycle) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      name,
					Image:     image,
					Lifecycle: lifecycle,
				},
			},
		},
	}
}
