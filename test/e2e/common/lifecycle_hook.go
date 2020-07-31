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

package common

import (
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	var podClient *framework.PodClient
	const (
		podCheckInterval     = 1 * time.Second
		postStartWaitTimeout = 2 * time.Minute
		preStopWaitTimeout   = 30 * time.Second
	)
	ginkgo.Context("when create a pod with lifecycle hook", func() {
		var targetIP, targetURL string
		podHandleHookRequest := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-handle-http-request",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pod-handle-http-request",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"netexec"},
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
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
			ginkgo.By("create the container to handle the HTTPGet hook request.")
			newPod := podClient.CreateSync(podHandleHookRequest)
			targetIP = newPod.Status.PodIP
			targetURL = targetIP
			if strings.Contains(targetIP, ":") {
				targetURL = fmt.Sprintf("[%s]", targetIP)
			}
		})
		testPodWithHook := func(podWithHook *v1.Pod) {
			ginkgo.By("create the pod with lifecycle hook")
			podClient.CreateSync(podWithHook)
			if podWithHook.Spec.Containers[0].Lifecycle.PostStart != nil {
				ginkgo.By("check poststart hook")
				gomega.Eventually(func() error {
					return podClient.MatchContainerOutput(podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[0].Name,
						`GET /echo\?msg=poststart`)
				}, postStartWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
			ginkgo.By("delete the pod with lifecycle hook")
			podClient.DeleteSync(podWithHook.Name, *metav1.NewDeleteOptions(15), framework.DefaultPodDeletionTimeout)
			if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
				ginkgo.By("check prestop hook")
				gomega.Eventually(func() error {
					return podClient.MatchContainerOutput(podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[0].Name,
						`GET /echo\?msg=prestop`)
				}, preStopWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
		}
		/*
			Release: v1.9
			Testname: Pod Lifecycle, post start exec hook
			Description: When a post start handler is specified in the container lifecycle using a 'Exec' action, then the handler MUST be invoked after the start of the container. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a post start that invokes the server pod using ExecAction to validate that the post start is executed.
		*/
		framework.ConformanceIt("should execute poststart exec hook properly [NodeConformance]", func() {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=poststart"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)
			testPodWithHook(podWithHook)
		})
		/*
			Release: v1.9
			Testname: Pod Lifecycle, prestop exec hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'Exec' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a pre-stop that invokes the server pod using ExecAction to validate that the pre-stop is executed.
		*/
		framework.ConformanceIt("should execute prestop exec hook properly [NodeConformance]", func() {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=prestop"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)
			testPodWithHook(podWithHook)
		})
		/*
			Release: v1.9
			Testname: Pod Lifecycle, post start http hook
			Description: When a post start handler is specified in the container lifecycle using a HttpGet action, then the handler MUST be invoked after the start of the container. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a post start that invokes the server pod to validate that the post start is executed.
		*/
		framework.ConformanceIt("should execute poststart http hook properly [NodeConformance]", func() {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=poststart",
						Host: targetIP,
						Port: intstr.FromInt(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-http-hook", imageutils.GetPauseImageName(), lifecycle)
			testPodWithHook(podWithHook)
		})
		/*
			Release: v1.9
			Testname: Pod Lifecycle, prestop http hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'HttpGet' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a pre-stop that invokes the server pod to validate that the pre-stop is executed.
		*/
		framework.ConformanceIt("should execute prestop http hook properly [NodeConformance]", func() {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=prestop",
						Host: targetIP,
						Port: intstr.FromInt(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-http-hook", imageutils.GetPauseImageName(), lifecycle)
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
