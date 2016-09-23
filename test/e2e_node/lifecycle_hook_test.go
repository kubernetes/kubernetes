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
)

var _ = framework.KubeDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	var podClient *framework.PodClient
	var file string
	const podWaitTimetout = 2 * time.Minute

	testPodWithHook := func(podWithHook *api.Pod) {
		podCheckStart := getTestPod("pod-check-start",
			// Wait until the file is created.
			[]string{"sh", "-c", fmt.Sprintf("while [ ! -e %s ]; do sleep 1; done", file)},
		)
		podCheckStop := getTestPod("pod-check-stop",
			// Wait until the file is deleted.
			[]string{"sh", "-c", fmt.Sprintf("while [ -e %s ]; do sleep 1; done", file)},
		)
		By("create the pod with lifecycle hook")
		podClient.CreateSync(podWithHook)
		By("create the poststart check pod")
		podClient.Create(podCheckStart)
		By("wait for the poststart check pod to success")
		podClient.WaitForSuccess(podCheckStart.Name, podWaitTimetout)
		By("delete the pod with lifecycle hook")
		podClient.DeleteSync(podWithHook.Name, api.NewDeleteOptions(15), podWaitTimetout)
		By("create the prestop check pod")
		podClient.Create(podCheckStop)
		By("wait for the prestop check pod to success")
		podClient.WaitForSuccess(podCheckStop.Name, podWaitTimetout)
	}

	BeforeEach(func() {
		podClient = f.PodClient()
		file = "/tmp/test-" + string(uuid.NewUUID())
	})

	Context("when create a pod with poststart/prestop exec hook", func() {
		It("should execute poststart/prestop exec hook properly", func() {
			podWithHook := getTestPod("pod-with-exec-hook",
				// Block forever
				[]string{"tail", "-f", "/dev/null"},
			)
			podWithHook.Spec.Containers[0].Lifecycle = &api.Lifecycle{
				PostStart: &api.Handler{
					Exec: &api.ExecAction{Command: []string{"touch", file}},
				},
				PreStop: &api.Handler{
					Exec: &api.ExecAction{Command: []string{"rm", file}},
				},
			}
			testPodWithHook(podWithHook)
		})
	})

	Context("when create a pod with poststart/prestop http hook", func() {
		It("should execute poststart/prestop http hook properly", func() {
			podWithHook := getTestPod("pod-with-http-hook",
				[]string{"sh", "-c",
					// create test file when first receive request on 1234 (PostStart Hook),
					// remove test file when second receive request on 4321 (PreStop Hook),
					fmt.Sprintf("echo -e \"HTTP/1.1 200 OK\n\" | nc -l -p 1234; touch %s; "+
						"echo -e \"HTTP/1.1 200 OK\n\" | nc -l -p 4321; rm %s", file, file),
				},
			)
			container := &podWithHook.Spec.Containers[0]
			container.Ports = []api.ContainerPort{
				{
					Name:          "poststart",
					ContainerPort: 1234,
					Protocol:      api.ProtocolTCP,
				},
				{
					Name:          "prestop",
					ContainerPort: 4321,
					Protocol:      api.ProtocolTCP,
				},
			}
			container.Lifecycle = &api.Lifecycle{
				PostStart: &api.Handler{
					// Use port number directly
					HTTPGet: &api.HTTPGetAction{Port: intstr.FromInt(1234)},
				},
				PreStop: &api.Handler{
					// Use port name
					HTTPGet: &api.HTTPGetAction{Port: intstr.FromString("prestop")},
				},
			}
			testPodWithHook(podWithHook)
		})
	})
})

func getTestPod(name string, cmd []string) *api.Pod {
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
