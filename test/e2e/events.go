/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Events", func() {
	f := framework.NewDefaultFramework("events")

	It("should be sent by kubelets and the scheduler about pods scheduling and running [Conformance]", func() {

		podClient := f.Client.Pods(f.Namespace.Name)

		By("creating the pod")
		name := "send-events-" + string(uuid.NewUUID())
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
						Image: "gcr.io/google_containers/serve_hostname:v1.4",
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
			framework.Failf("Failed to create pod: %v", err)
		}

		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := api.ListOptions{LabelSelector: selector}
		pods, err := podClient.List(options)
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podWithUid, err := podClient.Get(pod.Name)
		if err != nil {
			framework.Failf("Failed to get pod: %v", err)
		}
		fmt.Printf("%+v\n", podWithUid)
		var events *api.EventList
		// Check for scheduler event about the pod.
		By("checking for scheduler event about the pod")
		framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.uid":       string(podWithUid.UID),
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   api.DefaultSchedulerName,
			}.AsSelector()
			options := api.ListOptions{FieldSelector: selector}
			events, err := f.Client.Events(f.Namespace.Name).List(options)
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
		framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.uid":       string(podWithUid.UID),
				"involvedObject.kind":      "Pod",
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   "kubelet",
			}.AsSelector()
			options := api.ListOptions{FieldSelector: selector}
			events, err = f.Client.Events(f.Namespace.Name).List(options)
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
