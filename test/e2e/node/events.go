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
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Events", func() {
	f := framework.NewDefaultFramework("events")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should be sent by kubelets and the scheduler about pods scheduling and running ", func(ctx context.Context) {

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
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"serve-hostname"},
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
					},
				},
			},
		}

		ginkgo.By("submitting the pod to kubernetes")
		ginkgo.DeferCleanup(func(ctx context.Context) error {
			ginkgo.By("deleting the pod")
			return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
		})
		if _, err := podClient.Create(ctx, pod, metav1.CreateOptions{}); err != nil {
			framework.Failf("Failed to create pod: %v", err)
		}

		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(ctx, options)
		framework.ExpectNoError(err)
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))

		ginkgo.By("retrieving the pod")
		podWithUID, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get pod: %v", err)
		}
		framework.Logf("%+v\n", podWithUID)
		var events *v1.EventList
		// Check for scheduler event about the pod.
		ginkgo.By("checking for scheduler event about the pod")
		framework.ExpectNoError(wait.Poll(framework.Poll, 5*time.Minute, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.uid":       string(podWithUID.UID),
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   v1.DefaultSchedulerName,
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, options)
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
		framework.ExpectNoError(wait.Poll(framework.Poll, 5*time.Minute, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.uid":       string(podWithUID.UID),
				"involvedObject.kind":      "Pod",
				"involvedObject.namespace": f.Namespace.Name,
				"source":                   "kubelet",
			}.AsSelector().String()
			options := metav1.ListOptions{FieldSelector: selector}
			events, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, options)
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
