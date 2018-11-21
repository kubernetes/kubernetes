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

package node

import (
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Events", func() {
	f := framework.NewDefaultFramework("events")

	/*
		Release : v1.9
		Testname: Pod events, verify event from Scheduler and Kubelet
		Description: Create a Pod, make sure that the Pod can be queried. Create a event selector for the kind=Pod and the source is the Scheduler. List of the events MUST be at least one. Create a event selector for kind=Pod and the source is the Kubelet. List of the events MUST be at least one. Both Scheduler and Kubelet MUST send events when scheduling and running a Pod.
	*/
	framework.ConformanceIt("should be sent by kubelets and the scheduler about pods scheduling and running ", func() {

		podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)

		By("creating the pod")
		name := "send-events-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "p",
						Image: framework.ServeHostnameImage,
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
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
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podWithUid, err := podClient.Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get pod: %v", err)
		}
		framework.Logf("%+v\n", podWithUid)
		var events *v1.EventList
		// Check for scheduler event about the pod.
		By("checking for scheduler event about the pod")
		framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.uid":       string(podWithUid.UID),
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   v1.DefaultSchedulerName,
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(options)
			if err != nil {
				return false, err
			}
			if len(events.Items) > 0 {
				framework.Logf("Saw scheduler event for our pod.")
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
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).List(options)
			if err != nil {
				return false, err
			}
			if len(events.Items) > 0 {
				framework.Logf("Saw kubelet event for our pod.")
				return true, nil
			}
			return false, nil
		}))
	})
})
