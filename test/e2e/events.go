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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Events", func() {
	framework := NewFramework("events")

	It("should be sent by kubelets and the scheduler about pods scheduling and running [Conformance]", func() {

		podClient := framework.Client.Pods(framework.Namespace.Name)

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

		expectNoError(framework.WaitForPodRunning(pod.Name))

		By("verifying the pod is in kubernetes")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})), fields.Everything())
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podWithUid, err := podClient.Get(pod.Name)
		if err != nil {
			Failf("Failed to get pod: %v", err)
		}
		fmt.Printf("%+v\n", podWithUid)
		var events *api.EventList
		// Check for scheduler event about the pod.
		By("checking for scheduler event about the pod")
		expectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			events, err := framework.Client.Events(framework.Namespace.Name).List(
				labels.Everything(),
				fields.Set{
					"involvedObject.kind":      "Pod",
					"involvedObject.uid":       string(podWithUid.UID),
					"involvedObject.namespace": framework.Namespace.Name,
					"source":                   "scheduler",
				}.AsSelector(),
			)
			if err != nil {
				return false, err
			}
			if len(events.Items) > 0 {
				fmt.Println("Saw scheduler event for our pod.")
				return true, nil
			}
			return false, nil
		}))
		// Check for kubelet event about the pod.
		By("checking for kubelet event about the pod")
		expectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			events, err = framework.Client.Events(framework.Namespace.Name).List(
				labels.Everything(),
				fields.Set{
					"involvedObject.uid":       string(podWithUid.UID),
					"involvedObject.kind":      "Pod",
					"involvedObject.namespace": framework.Namespace.Name,
					"source":                   "kubelet",
				}.AsSelector(),
			)
			if err != nil {
				return false, err
			}
			if len(events.Items) > 0 {
				fmt.Println("Saw kubelet event for our pod.")
				return true, nil
			}
			return false, nil
		}))
	})
})
