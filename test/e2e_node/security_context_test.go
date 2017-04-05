/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Context("when creating a pod in the host PID namespace", func() {
		makeHostPidPod := func(podName, image string, command []string, hostPID bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					HostPID:       hostPID,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
						},
					},
				},
			}
		}

		nginxPid := ""
		BeforeEach(func() {
			nginxPodName := "nginx-hostpid-" + string(uuid.NewUUID())
			podClient.CreateSync(makeHostPidPod(nginxPodName,
				"gcr.io/google_containers/nginx-slim:0.7",
				nil,
				true,
			))

			output := f.ExecShellInContainer(nginxPodName, nginxPodName,
				"cat /var/run/nginx.pid")
			nginxPid = strings.TrimSpace(output)
		})

		It("should show its pid in the host PID namespace", func() {
			busyboxPodName := "busybox-hostpid-" + string(uuid.NewUUID())
			podClient.Create(makeHostPidPod(busyboxPodName,
				"gcr.io/google_containers/busybox:1.24",
				[]string{"sh", "-c", "pidof nginx"},
				true,
			))
			podClient.WaitForSuccess(busyboxPodName, framework.PodStartTimeout)

			Eventually(func() error {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
				if err != nil {
					return err
				}

				pids := strings.TrimSpace(logs)
				framework.Logf("Got nginx's pid %q from pod %q", pids, busyboxPodName)
				if pids == "" {
					return fmt.Errorf("nginx's pid should be seen by hostpid containers")
				}

				pidSets := sets.NewString(strings.Split(pids, " ")...)
				if !pidSets.Has(nginxPid) {
					return fmt.Errorf("nginx's pid should be seen by hostpid containers")
				}

				return nil
			}, time.Minute, time.Second*4).Should(BeNil())
		})

		It("should not show its pid in the non-hostpid containers", func() {
			busyboxPodName := "busybox-non-hostpid-" + string(uuid.NewUUID())
			podClient.Create(makeHostPidPod(busyboxPodName,
				"gcr.io/google_containers/busybox:1.24",
				[]string{"sh", "-c", "ps -e -o pid"},
				false,
			))
			podClient.WaitForSuccess(busyboxPodName, framework.PodStartTimeout)

			Eventually(func() error {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
				if err != nil {
					return err
				}

				pids := strings.TrimSpace(logs)
				framework.Logf("Got pids %q from busybox pod %q", pids, busyboxPodName)
				if pids == "" {
					return fmt.Errorf("pid list should not be null")
				}

				pidSets := sets.NewString(strings.Split(pids, " ")...)
				if pidSets.Has(nginxPid) {
					return fmt.Errorf("nginx's pid should not be seen by non-hostpid containers")
				}

				return nil
			}, time.Minute, time.Second*4).Should(BeNil())
		})
	})
})
