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

package common

import (
	"fmt"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("InitContainer", func() {
	f := framework.NewDefaultFramework("init-container")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("should invoke init containers on a RestartNever pod", func() {
		framework.SkipIfContainerRuntimeIs("rkt") // #25988

		By("creating the pod")
		name := "pod-init-" + string(uuid.NewUUID())
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
				RestartPolicy: api.RestartPolicyNever,
				InitContainers: []api.Container{
					{
						Name:    "init1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
					{
						Name:    "init2",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
				},
				Containers: []api.Container{
					{
						Name:    "run1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
				},
			},
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(api.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")
		wr := watch.NewRecorder(w)
		event, err := watch.Until(framework.PodStartTimeout, wr, client.PodCompleted)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*api.Pod)

		Expect(endPod.Status.Phase).To(Equal(api.PodSucceeded))
		_, init := api.GetPodCondition(&endPod.Status, api.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(api.ConditionTrue))

		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
		for _, status := range endPod.Status.InitContainerStatuses {
			Expect(status.Ready).To(BeTrue())
			Expect(status.State.Terminated).NotTo(BeNil())
			Expect(status.State.Terminated.ExitCode).To(BeZero())
		}
	})

	It("should invoke init containers on a RestartAlways pod", func() {
		framework.SkipIfContainerRuntimeIs("rkt") // #25988

		By("creating the pod")
		name := "pod-init-" + string(uuid.NewUUID())
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
				InitContainers: []api.Container{
					{
						Name:    "init1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
					{
						Name:    "init2",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
				},
				Containers: []api.Container{
					{
						Name:  "run1",
						Image: framework.GetPauseImageName(f.ClientSet),
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								api.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								api.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(api.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")
		wr := watch.NewRecorder(w)
		event, err := watch.Until(framework.PodStartTimeout, wr, client.PodRunning)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*api.Pod)

		Expect(endPod.Status.Phase).To(Equal(api.PodRunning))
		_, init := api.GetPodCondition(&endPod.Status, api.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(api.ConditionTrue))

		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
		for _, status := range endPod.Status.InitContainerStatuses {
			Expect(status.Ready).To(BeTrue())
			Expect(status.State.Terminated).NotTo(BeNil())
			Expect(status.State.Terminated.ExitCode).To(BeZero())
		}
	})

	It("should not start app containers if init containers fail on a RestartAlways pod", func() {
		framework.SkipIfContainerRuntimeIs("rkt") // #25988

		By("creating the pod")
		name := "pod-init-" + string(uuid.NewUUID())
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
				InitContainers: []api.Container{
					{
						Name:    "init1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/false"},
					},
					{
						Name:    "init2",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
				},
				Containers: []api.Container{
					{
						Name:  "run1",
						Image: framework.GetPauseImageName(f.ClientSet),
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								api.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								api.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(api.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")

		wr := watch.NewRecorder(w)
		event, err := watch.Until(
			framework.PodStartTimeout, wr,
			// check for the first container to fail at least once
			func(evt watch.Event) (bool, error) {
				switch t := evt.Object.(type) {
				case *api.Pod:
					for _, status := range t.Status.ContainerStatuses {
						if status.State.Waiting == nil {
							return false, fmt.Errorf("container %q should not be out of waiting: %#v", status.Name, status)
						}
						if status.State.Waiting.Reason != "PodInitializing" {
							return false, fmt.Errorf("container %q should have reason PodInitializing: %#v", status.Name, status)
						}
					}
					if len(t.Status.InitContainerStatuses) != 2 {
						return false, nil
					}
					status := t.Status.InitContainerStatuses[1]
					if status.State.Waiting == nil {
						return false, fmt.Errorf("second init container should not be out of waiting: %#v", status)
					}
					if status.State.Waiting.Reason != "PodInitializing" {
						return false, fmt.Errorf("second init container should have reason PodInitializing: %#v", status)
					}
					status = t.Status.InitContainerStatuses[0]
					if status.State.Terminated != nil && status.State.Terminated.ExitCode == 0 {
						return false, fmt.Errorf("first init container should have exitCode != 0: %#v", status)
					}
					// continue until we see an attempt to restart the pod
					return status.LastTerminationState.Terminated != nil, nil
				default:
					return false, fmt.Errorf("unexpected object: %#v", t)
				}
			},
			// verify we get two restarts
			func(evt watch.Event) (bool, error) {
				switch t := evt.Object.(type) {
				case *api.Pod:
					status := t.Status.InitContainerStatuses[0]
					if status.RestartCount < 3 {
						return false, nil
					}
					framework.Logf("init container has failed twice: %#v", t)
					// TODO: more conditions
					return true, nil
				default:
					return false, fmt.Errorf("unexpected object: %#v", t)
				}
			},
		)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*api.Pod)

		Expect(endPod.Status.Phase).To(Equal(api.PodPending))
		_, init := api.GetPodCondition(&endPod.Status, api.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(api.ConditionFalse))
		Expect(init.Reason).To(Equal("ContainersNotInitialized"))
		Expect(init.Message).To(Equal("containers with incomplete status: [init1 init2]"))
		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
	})

	It("should not start app containers and fail the pod if init containers fail on a RestartNever pod", func() {
		framework.SkipIfContainerRuntimeIs("rkt") // #25988

		By("creating the pod")
		name := "pod-init-" + string(uuid.NewUUID())
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
				RestartPolicy: api.RestartPolicyNever,
				InitContainers: []api.Container{
					{
						Name:    "init1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
					{
						Name:    "init2",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/false"},
					},
				},
				Containers: []api.Container{
					{
						Name:    "run1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								api.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								api.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		startedPod := podClient.Create(pod)

		w, err := podClient.Watch(api.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")

		wr := watch.NewRecorder(w)
		event, err := watch.Until(
			framework.PodStartTimeout, wr,
			// check for the second container to fail at least once
			func(evt watch.Event) (bool, error) {
				switch t := evt.Object.(type) {
				case *api.Pod:
					for _, status := range t.Status.ContainerStatuses {
						if status.State.Waiting == nil {
							return false, fmt.Errorf("container %q should not be out of waiting: %#v", status.Name, status)
						}
						if status.State.Waiting.Reason != "PodInitializing" {
							return false, fmt.Errorf("container %q should have reason PodInitializing: %#v", status.Name, status)
						}
					}
					if len(t.Status.InitContainerStatuses) != 2 {
						return false, nil
					}
					status := t.Status.InitContainerStatuses[0]
					if status.State.Terminated == nil {
						if status.State.Waiting != nil && status.State.Waiting.Reason != "PodInitializing" {
							return false, fmt.Errorf("second init container should have reason PodInitializing: %#v", status)
						}
						return false, nil
					}
					if status.State.Terminated != nil && status.State.Terminated.ExitCode != 0 {
						return false, fmt.Errorf("first init container should have exitCode != 0: %#v", status)
					}
					status = t.Status.InitContainerStatuses[1]
					if status.State.Terminated == nil {
						return false, nil
					}
					if status.State.Terminated.ExitCode == 0 {
						return false, fmt.Errorf("second init container should have failed: %#v", status)
					}
					return true, nil
				default:
					return false, fmt.Errorf("unexpected object: %#v", t)
				}
			},
			client.PodCompleted,
		)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*api.Pod)

		Expect(endPod.Status.Phase).To(Equal(api.PodFailed))
		_, init := api.GetPodCondition(&endPod.Status, api.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(api.ConditionFalse))
		Expect(init.Reason).To(Equal("ContainersNotInitialized"))
		Expect(init.Message).To(Equal("containers with incomplete status: [init2]"))
		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
		Expect(endPod.Status.ContainerStatuses[0].State.Waiting).ToNot(BeNil())
	})
})
