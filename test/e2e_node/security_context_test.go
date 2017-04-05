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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

	Context("when creating a hostpid pod", func() {
		podName := "busybox-hostpid-" + string(uuid.NewUUID())
		It("it should create the pod in hostPid", func() {
			podClient.CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: v1.RestartPolicyNever,
					HostPID:       true,
					Containers: []v1.Container{
						{
							Image:   "gcr.io/google_containers/busybox:1.24",
							Name:    podName,
							Command: []string{"sh", "-c", "pidof sh; sleep 240"},
						},
					},
				},
			})
			Eventually(func() error {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, podName)
				if err != nil {
					return err
				}

				// the pid of 'sh' should not be 1 if it is in hostPid mode.
				if logs == "1\n" {
					return fmt.Errorf("sh should not be the container's init process")
				}

				return nil
			}, time.Minute, time.Second*4).Should(BeNil())
		})
	})
})
