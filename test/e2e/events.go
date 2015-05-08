/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Events", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should be sent by kubelets and the scheduler about pods scheduling and running", func() {

		podClient := c.Pods(api.NamespaceDefault)

		By("creating the pod")
		name := "send-events-" + string(util.NewUUID())
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
						Name:  "p",
						Image: "gcr.io/google_containers/serve_hostname:1.1",
						Ports: []api.ContainerPort{{ContainerPort: 80}},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		defer func() {
			By("deleting the pod")
			podClient.Delete(pod.Name, nil)
		}()
		if _, err := podClient.Create(pod); err != nil {
			Failf("Failed to create pod: %v", err)
		}

		expectNoError(waitForPodRunning(c, pod.Name))

		By("verifying the pod is in kubernetes")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})), fields.Everything())
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podWithUid, err := podClient.Get(pod.Name)
		if err != nil {
			Failf("Failed to get pod: %v", err)
		}
		fmt.Printf("%+v\n", podWithUid)

		// Check for scheduler event about the pod.
		By("checking for scheduler event about the pod")
		events, err := c.Events(api.NamespaceDefault).List(
			labels.Everything(),
			fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.uid":       string(podWithUid.UID),
				"involvedObject.namespace": api.NamespaceDefault,
				"source":                   "scheduler",
			}.AsSelector(),
		)
		if err != nil {
			Failf("Error while listing events: %v", err)
		}
		Expect(len(events.Items)).ToNot(BeZero(), "scheduler events from running pod")
		fmt.Println("Saw scheduler event for our pod.")

		// Check for kubelet event about the pod.
		By("checking for kubelet event about the pod")
		events, err = c.Events(api.NamespaceDefault).List(
			labels.Everything(),
			fields.Set{
				"involvedObject.uid":       string(podWithUid.UID),
				"involvedObject.kind":      "Pod",
				"involvedObject.namespace": api.NamespaceDefault,
				"source":                   "kubelet",
			}.AsSelector(),
		)
		if err != nil {
			Failf("Error while listing events: %v", err)
		}
		Expect(len(events.Items)).ToNot(BeZero(), "kubelet events from running pod")
		fmt.Println("Saw kubelet event for our pod.")
	})
})
