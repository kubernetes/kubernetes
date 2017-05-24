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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	var podClient *framework.PodClient
	const (
		podCheckInterval     = 1 * time.Second
		podWaitTimeout       = 2 * time.Minute
		postStartWaitTimeout = 2 * time.Minute
		preStopWaitTimeout   = 30 * time.Second
	)
	Context("when create a pod with lifecycle hook", func() {
		BeforeEach(func() {
			podClient = f.PodClient()
		})

		Context("when it is exec hook", func() {
			var file string
			testPodWithExecHook := func(podWithHook *api.Pod) {
				podCheckHook := getExecHookTestPod("pod-check-hook",
					// Wait until the file is created.
					[]string{"sh", "-c", fmt.Sprintf("while [ ! -e %s ]; do sleep 1; done", file)},
				)
				By("create the pod with lifecycle hook")
				podClient.CreateSync(podWithHook)
				if podWithHook.Spec.Containers[0].Lifecycle.PostStart != nil {
					By("create the hook check pod")
					podClient.Create(podCheckHook)
					By("wait for the hook check pod to success")
					podClient.WaitForSuccess(podCheckHook.Name, postStartWaitTimeout)
				}
				By("delete the pod with lifecycle hook")
				podClient.DeleteSync(podWithHook.Name, api.NewDeleteOptions(15), podWaitTimeout)
				if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
					By("create the hook check pod")
					podClient.Create(podCheckHook)
					By("wait for the prestop check pod to success")
					podClient.WaitForSuccess(podCheckHook.Name, preStopWaitTimeout)
				}
			}

			BeforeEach(func() {
				file = "/tmp/test-" + string(uuid.NewUUID())
			})

			AfterEach(func() {
				By("cleanup the temporary file created in the test.")
				cleanupPod := getExecHookTestPod("pod-clean-up", []string{"rm", file})
				podClient.Create(cleanupPod)
				podClient.WaitForSuccess(cleanupPod.Name, podWaitTimeout)
			})

			It("should execute poststart exec hook properly [Conformance]", func() {
				podWithHook := getExecHookTestPod("pod-with-poststart-exec-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &api.Lifecycle{
					PostStart: &api.Handler{
						Exec: &api.ExecAction{Command: []string{"touch", file}},
					},
				}
				testPodWithExecHook(podWithHook)
			})

			It("should execute prestop exec hook properly [Conformance]", func() {
				podWithHook := getExecHookTestPod("pod-with-prestop-exec-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &api.Lifecycle{
					PreStop: &api.Handler{
						Exec: &api.ExecAction{Command: []string{"touch", file}},
					},
				}
				testPodWithExecHook(podWithHook)
			})
		})

		Context("when it is http hook", func() {
			var targetIP string
			podHandleHookRequest := &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod-handle-http-request",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "pod-handle-http-request",
							Image: "gcr.io/google_containers/netexec:1.7",
							Ports: []api.ContainerPort{
								{
									ContainerPort: 8080,
									Protocol:      api.ProtocolTCP,
								},
							},
						},
					},
				},
			}
			BeforeEach(func() {
				By("create the container to handle the HTTPGet hook request.")
				newPod := podClient.CreateSync(podHandleHookRequest)
				targetIP = newPod.Status.PodIP
			})
			testPodWithHttpHook := func(podWithHook *api.Pod) {
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
				podClient.DeleteSync(podWithHook.Name, api.NewDeleteOptions(15), podWaitTimeout)
				if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
					By("check prestop hook")
					Eventually(func() error {
						return podClient.MatchContainerOutput(podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[0].Name,
							`GET /echo\?msg=prestop`)
					}, preStopWaitTimeout, podCheckInterval).Should(BeNil())
				}
			}
			It("should execute poststart http hook properly [Conformance]", func() {
				podWithHook := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: "pod-with-poststart-http-hook",
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Name:  "pod-with-poststart-http-hook",
								Image: framework.GetPauseImageNameForHostArch(),
								Lifecycle: &api.Lifecycle{
									PostStart: &api.Handler{
										HTTPGet: &api.HTTPGetAction{
											Path: "/echo?msg=poststart",
											Host: targetIP,
											Port: intstr.FromInt(8080),
										},
									},
								},
							},
						},
					},
				}
				testPodWithHttpHook(podWithHook)
			})
			It("should execute prestop http hook properly [Conformance]", func() {
				podWithHook := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: "pod-with-prestop-http-hook",
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Name:  "pod-with-prestop-http-hook",
								Image: framework.GetPauseImageNameForHostArch(),
								Lifecycle: &api.Lifecycle{
									PreStop: &api.Handler{
										HTTPGet: &api.HTTPGetAction{
											Path: "/echo?msg=prestop",
											Host: targetIP,
											Port: intstr.FromInt(8080),
										},
									},
								},
							},
						},
					},
				}
				testPodWithHttpHook(podWithHook)
			})
		})
	})
})

func getExecHookTestPod(name string, cmd []string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  name,
					Image: "gcr.io/google_containers/busybox:1.24",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "tmpfs",
							MountPath: "/tmp",
						},
					},
					Command: cmd,
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name:         "tmpfs",
					VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/tmp"}},
				},
			},
		},
	}
}
