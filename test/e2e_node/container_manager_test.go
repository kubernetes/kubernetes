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
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet Container Manager", func() {
	f := framework.NewDefaultFramework("kubelet-container-manager")
	var podClient *framework.PodClient

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Describe("oom score adjusting", func() {
		Context("when scheduling a busybox command that always fails in a pod", func() {
			var podName string

			BeforeEach(func() {
				podName = "bin-false" + string(uuid.NewUUID())
				podClient.Create(&api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   ImageRegistry[busyBoxImage],
								Name:    podName,
								Command: []string{"/bin/false"},
							},
						},
					},
				})
			})

			It("should have an error terminated reason", func() {
				Eventually(func() error {
					podData, err := podClient.Get(podName)
					if err != nil {
						return err
					}
					if len(podData.Status.ContainerStatuses) != 1 {
						return fmt.Errorf("expected only one container in the pod %q", podName)
					}
					contTerminatedState := podData.Status.ContainerStatuses[0].State.Terminated
					if contTerminatedState == nil {
						return fmt.Errorf("expected state to be terminated. Got pod status: %+v", podData.Status)
					}
					if contTerminatedState.Reason != "Error" {
						return fmt.Errorf("expected terminated state reason to be error. Got %+v", contTerminatedState)
					}
					return nil
				}, time.Minute, time.Second*4).Should(BeNil())
			})

			It("should be possible to delete", func() {
				err := podClient.Delete(podName, &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %v", err))
			})
		})
	})

})
