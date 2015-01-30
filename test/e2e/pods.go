/*
Copyright 2014 Google Inc. All rights reserved.

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

package e2e

import (
	"fmt"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Pods", func() {
	var (
		c *client.Client
	)

	BeforeEach(func() {
		c = loadClientOrDie()
	})

	It("should be updated", func() {
		podClient := c.Pods(api.NamespaceDefault)

		By("creating the pod")
		name := "pod-update-" + string(util.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "nginx",
						Image: "dockerfile/nginx",
						Ports: []api.Port{{ContainerPort: 80, HostPort: 8080}},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Path: "/index.html",
									Port: util.NewIntOrStringFromInt(8080),
								},
							},
							InitialDelaySeconds: 30,
						},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		_, err := podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}
		defer func() {
			By("deleting the pod")
			defer GinkgoRecover()
			podClient.Delete(pod.Name)
		}()

		By("waiting for the pod to start running")
		waitForPodRunning(c, pod.Name)

		By("verifying the pod is in kubernetes")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podOut, err := podClient.Get(pod.Name)
		if err != nil {
			Fail(fmt.Sprintf("Failed to get pod: %v", err))
		}

		By("updating the pod")
		value = "time" + value
		pod.Labels["time"] = value
		pod.ResourceVersion = podOut.ResourceVersion
		pod.UID = podOut.UID
		pod, err = podClient.Update(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to update pod: %v", err))
		}

		By("waiting for the updated pod to start running")
		waitForPodRunning(c, pod.Name)

		By("verifying the updated pod is in kubernetes")
		pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		Expect(len(pods.Items)).To(Equal(1))
		fmt.Println("pod update OK")
	})
})
