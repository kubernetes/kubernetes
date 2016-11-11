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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	var podClient *framework.PodClient
	var file string
	const podWaitTimeout = 2 * time.Minute

	testPodWithHook := func(podWithHook *v1.Pod) {
		podCheckHook := getLifecycleHookTestPod("pod-check-hook",
			// Wait until the file is created.
			[]string{"sh", "-c", fmt.Sprintf("while [ ! -e %s ]; do sleep 1; done", file)},
		)
		By("create the pod with lifecycle hook")
		podClient.CreateSync(podWithHook)
		if podWithHook.Spec.Containers[0].Lifecycle.PostStart != nil {
			By("create the hook check pod")
			podClient.Create(podCheckHook)
			By("wait for the hook check pod to success")
			podClient.WaitForSuccess(podCheckHook.Name, podWaitTimeout)
		}
		By("delete the pod with lifecycle hook")
		podClient.DeleteSync(podWithHook.Name, v1.NewDeleteOptions(15), podWaitTimeout)
		if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
			By("create the hook check pod")
			podClient.Create(podCheckHook)
			By("wait for the prestop check pod to success")
			podClient.WaitForSuccess(podCheckHook.Name, podWaitTimeout)
		}
	}

	Context("when create a pod with lifecycle hook", func() {
		BeforeEach(func() {
			podClient = f.PodClient()
			file = "/tmp/test-" + string(uuid.NewUUID())
		})

		AfterEach(func() {
			By("cleanup the temporary file created in the test.")
			cleanupPod := getLifecycleHookTestPod("pod-clean-up", []string{"rm", file})
			podClient.Create(cleanupPod)
			podClient.WaitForSuccess(cleanupPod.Name, podWaitTimeout)
		})

		Context("when it is exec hook", func() {
			It("should execute poststart exec hook properly [Conformance]", func() {
				podWithHook := getLifecycleHookTestPod("pod-with-poststart-exec-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
					PostStart: &v1.Handler{
						Exec: &v1.ExecAction{Command: []string{"touch", file}},
					},
				}
				testPodWithHook(podWithHook)
			})

			It("should execute prestop exec hook properly [Conformance]", func() {
				podWithHook := getLifecycleHookTestPod("pod-with-prestop-exec-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
					PreStop: &v1.Handler{
						Exec: &v1.ExecAction{Command: []string{"touch", file}},
					},
				}
				testPodWithHook(podWithHook)
			})
		})

		Context("when it is http hook", func() {
			var targetIP string
			BeforeEach(func() {
				By("cleanup the container to handle the HTTPGet hook request.")
				podHandleHookRequest := getLifecycleHookTestPod("pod-handle-http-request",
					[]string{"sh", "-c",
						// Create test file when receive request on 1234.
						fmt.Sprintf("echo -e \"HTTP/1.1 200 OK\n\" | nc -l -p 1234; touch %s", file),
					},
				)
				podHandleHookRequest.Spec.Containers[0].Ports = []v1.ContainerPort{
					{
						ContainerPort: 1234,
						Protocol:      v1.ProtocolTCP,
					},
				}
				podHandleHookRequest = podClient.CreateSync(podHandleHookRequest)
				targetIP = podHandleHookRequest.Status.PodIP
			})
			It("should execute poststart http hook properly [Conformance]", func() {
				podWithHook := getLifecycleHookTestPod("pod-with-poststart-http-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
					PostStart: &v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Host: targetIP,
							Port: intstr.FromInt(1234),
						},
					},
				}
				testPodWithHook(podWithHook)
			})
			It("should execute prestop http hook properly [Conformance]", func() {
				podWithHook := getLifecycleHookTestPod("pod-with-prestop-http-hook",
					// Block forever
					[]string{"tail", "-f", "/dev/null"},
				)
				podWithHook.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
					PreStop: &v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Host: targetIP,
							Port: intstr.FromInt(1234),
						},
					},
				}
				testPodWithHook(podWithHook)
			})
		})
	})
})

func getLifecycleHookTestPod(name string, cmd []string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: "gcr.io/google_containers/busybox:1.24",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "tmpfs",
							MountPath: "/tmp",
						},
					},
					Command: cmd,
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         "tmpfs",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/tmp"}},
				},
			},
		},
	}
}
