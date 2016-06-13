/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet Container Manager", func() {
	f := NewDefaultFramework("kubelet-container-manager")
	Describe("oom score adjusting", func() {
		Context("when scheduling a busybox command that always fails in a pod", func() {
			var podName string

			BeforeEach(func() {
				podName = "bin-false" + string(util.NewUUID())
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      podName,
						Namespace: f.Namespace.Name,
					},
					Spec: api.PodSpec{
						// Force the Pod to schedule to the node without a scheduler running
						NodeName: *nodeName,
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
				}

				_, err := f.Client.Pods(f.Namespace.Name).Create(pod)
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})

			It("should have an error terminated reason", func() {
				Eventually(func() error {
					podData, err := f.Client.Pods(f.Namespace.Name).Get(podName)
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
				err := f.Client.Pods(f.Namespace.Name).Delete(podName, &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %v", err))
			})
		})
	})

})
