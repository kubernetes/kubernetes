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

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/pkg/util/uuid"
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
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
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
				Containers: []v1.Container{
					{
						Name:    "run1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
					},
				},
			},
		}
		if err := podutil.SetInitContainersAnnotations(pod); err != nil {
			Expect(err).To(BeNil())
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(v1.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")
		wr := watch.NewRecorder(w)
		event, err := watch.Until(framework.PodStartTimeout, wr, conditions.PodCompleted)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*v1.Pod)
		if err := podutil.SetInitContainersAndStatuses(endPod); err != nil {
			Expect(err).To(BeNil())
		}

		Expect(endPod.Status.Phase).To(Equal(v1.PodSucceeded))
		_, init := v1.GetPodCondition(&endPod.Status, v1.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(v1.ConditionTrue))

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
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
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
				Containers: []v1.Container{
					{
						Name:  "run1",
						Image: framework.GetPauseImageName(f.ClientSet),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								v1.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		if err := podutil.SetInitContainersAnnotations(pod); err != nil {
			Expect(err).To(BeNil())
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(v1.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")
		wr := watch.NewRecorder(w)
		event, err := watch.Until(framework.PodStartTimeout, wr, conditions.PodRunning)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*v1.Pod)

		Expect(endPod.Status.Phase).To(Equal(v1.PodRunning))
		_, init := v1.GetPodCondition(&endPod.Status, v1.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(v1.ConditionTrue))
		if err := podutil.SetInitContainersAndStatuses(endPod); err != nil {
			Expect(err).To(BeNil())
		}

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

		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
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
				Containers: []v1.Container{
					{
						Name:  "run1",
						Image: framework.GetPauseImageName(f.ClientSet),
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								v1.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		if err := podutil.SetInitContainersAnnotations(pod); err != nil {
			Expect(err).To(BeNil())
		}
		startedPod := podClient.Create(pod)
		w, err := podClient.Watch(v1.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")

		wr := watch.NewRecorder(w)
		event, err := watch.Until(
			framework.PodStartTimeout, wr,
			// check for the first container to fail at least once
			func(evt watch.Event) (bool, error) {
				switch t := evt.Object.(type) {
				case *v1.Pod:
					if err := podutil.SetInitContainersAndStatuses(t); err != nil {
						Expect(err).To(BeNil())
					}
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
				case *v1.Pod:
					if err := podutil.SetInitContainersAndStatuses(t); err != nil {
						Expect(err).To(BeNil())
					}
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
		endPod := event.Object.(*v1.Pod)
		if err := podutil.SetInitContainersAndStatuses(endPod); err != nil {
			Expect(err).To(BeNil())
		}

		Expect(endPod.Status.Phase).To(Equal(v1.PodPending))
		_, init := v1.GetPodCondition(&endPod.Status, v1.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(v1.ConditionFalse))
		Expect(init.Reason).To(Equal("ContainersNotInitialized"))
		Expect(init.Message).To(Equal("containers with incomplete status: [init1 init2]"))
		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
	})

	It("should not start app containers and fail the pod if init containers fail on a RestartNever pod", func() {
		framework.SkipIfContainerRuntimeIs("rkt") // #25988

		By("creating the pod")
		name := "pod-init-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
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
				Containers: []v1.Container{
					{
						Name:    "run1",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/true"},
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
								v1.ResourceMemory: *resource.NewQuantity(30*1024*1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		}
		if err := podutil.SetInitContainersAnnotations(pod); err != nil {
			Expect(err).To(BeNil())
		}
		startedPod := podClient.Create(pod)

		w, err := podClient.Watch(v1.SingleObject(startedPod.ObjectMeta))
		Expect(err).NotTo(HaveOccurred(), "error watching a pod")

		wr := watch.NewRecorder(w)
		event, err := watch.Until(
			framework.PodStartTimeout, wr,
			// check for the second container to fail at least once
			func(evt watch.Event) (bool, error) {
				switch t := evt.Object.(type) {
				case *v1.Pod:
					if err := podutil.SetInitContainersAndStatuses(t); err != nil {
						Expect(err).To(BeNil())
					}
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
			conditions.PodCompleted,
		)
		Expect(err).To(BeNil())
		framework.CheckInvariants(wr.Events(), framework.ContainerInitInvariant)
		endPod := event.Object.(*v1.Pod)

		Expect(endPod.Status.Phase).To(Equal(v1.PodFailed))
		_, init := v1.GetPodCondition(&endPod.Status, v1.PodInitialized)
		Expect(init).NotTo(BeNil())
		Expect(init.Status).To(Equal(v1.ConditionFalse))
		Expect(init.Reason).To(Equal("ContainersNotInitialized"))
		Expect(init.Message).To(Equal("containers with incomplete status: [init2]"))
		Expect(len(endPod.Status.InitContainerStatuses)).To(Equal(2))
		Expect(endPod.Status.ContainerStatuses[0].State.Waiting).ToNot(BeNil())
	})
})
