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
	"context"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/onsi/ginkgo"
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

		ginkgo.By("creating the pod")
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
						Args:  []string{"serve-hostname"},
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
					},
				},
			},
		}

		ginkgo.By("submitting the pod to kubernetes")
		defer func() {
			ginkgo.By("deleting the pod")
			podClient.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
		}()
		if _, err := podClient.Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
			framework.Failf("Failed to create pod: %v", err)
		}

		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name))

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(context.TODO(), options)
		framework.ExpectEqual(len(pods.Items), 1)

		ginkgo.By("retrieving the pod")
		podWithUID, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get pod: %v", err)
		}
		framework.Logf("%+v\n", podWithUID)
		var events *v1.EventList
		// Check for scheduler event about the pod.
		ginkgo.By("checking for scheduler event about the pod")
		framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.uid":       string(podWithUID.UID),
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   v1.DefaultSchedulerName,
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), options)
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
		ginkgo.By("checking for kubelet event about the pod")
		framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.uid":       string(podWithUID.UID),
				"involvedObject.kind":      "Pod",
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   "kubelet",
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), options)
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
