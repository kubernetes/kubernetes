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
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Kubelet Container Manager", func() {
	var cl *client.Client
	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})
	Describe("oom score adjusting", func() {
		namespace := "oom-adj"
		Context("when scheduling a busybox command that always fails in a pod", func() {
			var podName string

			BeforeEach(func() {
				podName = "bin-false" + string(util.NewUUID())
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      podName,
						Namespace: namespace,
					},
					Spec: api.PodSpec{
						// Force the Pod to schedule to the node without a scheduler running
						NodeName: *nodeName,
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   "gcr.io/google_containers/busybox:1.24",
								Name:    podName,
								Command: []string{"/bin/false"},
							},
						},
					},
				}

				_, err := cl.Pods(namespace).Create(pod)
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})

			It("it should have an error terminated reason", func() {
				Eventually(func() error {
					podData, err := cl.Pods(namespace).Get(podName)
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

			It("it should be possible to delete", func() {
				err := cl.Pods(namespace).Delete(podName, &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %v", err))
			})

			AfterEach(func() {
				cl.Pods(namespace).Delete(podName, &api.DeleteOptions{})
				Eventually(func() bool {
					_, err := cl.Pods(namespace).Get(podName)
					if err != nil && apierrs.IsNotFound(err) {
						return true
					}
					return false
				}, time.Minute, time.Second*4).Should(BeTrue())
			})

		})
	})

})
