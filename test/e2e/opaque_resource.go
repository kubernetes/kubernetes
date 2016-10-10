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

package e2e

import (
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/types"
)

var opaqueResName api.ResourceName

var _ = framework.KubeDescribe("Opaque resources [Feature:OpaqueResources]", func() {
	f := framework.NewDefaultFramework("opaque-resource")
	opaqueResName = api.OpaqueIntResourceName("foo")
	var node *api.Node

	// Custom matcher to validate pod schedulability.
	beScheduled := func() GomegaMatcher { return &scheduledMatcher{f} }
	notBeScheduled := func() GomegaMatcher { return &notScheduledMatcher{f} }

	BeforeEach(func() {
		node = refresh(f, node)
		removeOpaqueRes(f, node)
		node = refresh(f, node)
	})

	It("should should patch node capacity and verify allocatable is updated following kubelet sync.", func() {
		By("Verifying the advertised capacity and allocatable do not contain opaque resource 'foo'")
		Expect(node.Status.Capacity).NotTo(HaveKey(opaqueResName))
		Expect(node.Status.Allocatable).NotTo(HaveKey(opaqueResName))

		By("Patching node capacity to add an opaque integer resource")
		// Use the client to patch node capacity, adding opaque "foo" resource.
		patch := []byte(`[{"op": "add", "path": "/status/capacity/pod.alpha.kubernetes.io~1opaque-int-resource-foo", "value": "5"}]`)
		f.Client.Patch(api.JSONPatchType).Resource("nodes").Name(node.Name).SubResource("status").Body(patch).Do()

		By("Watching node status to verify the advertised opaque resource is eventually present in capacity")
		awaitNodePredicate(f.Client, node, func(n *api.Node) bool {
			rQuant, found := n.Status.Capacity[opaqueResName]
			return found && rQuant.MilliValue() == int64(5000)
		})

		By("Watching node status to verify the advertised opaque resource is eventually present in allocatable")
		awaitNodePredicate(f.Client, node, func(n *api.Node) bool {
			rQuant := n.Status.Allocatable[opaqueResName]
			return rQuant.MilliValue() == int64(5000)
		})
	})

	Context("With opaque resources already advertised", func() {
		BeforeEach(func() {
			node = refresh(f, node)
			// Use the client to patch node capacity, adding opaque "foo" resource.
			patch := []byte(`[{"op": "add", "path": "/status/capacity/pod.alpha.kubernetes.io~1opaque-int-resource-foo", "value": "5"}]`)
			f.Client.Patch(api.JSONPatchType).Resource("nodes").Name(node.Name).SubResource("status").Body(patch).Do()
			awaitNodePredicate(f.Client, node, func(n *api.Node) bool {
				rQuant, found := n.Status.Allocatable[opaqueResName]
				return found && rQuant.MilliValue() == int64(5000)
			})
			node = refresh(f, node)
		})

		It("should not break pods that do not consume opaque integer resources.", func() {
			By("Verifying that a vanilla pod can still be scheduled")
			requests := api.ResourceList{api.ResourceCPU: resource.MustParse("0.1")}
			limits := api.ResourceList{api.ResourceCPU: resource.MustParse("0.2")}
			pod, err := f.Client.Pods(f.Namespace.Name).Create(newTestPod(f, "without-oir", requests, limits))
			Expect(err).NotTo(HaveOccurred())
			Expect(pod).To(beScheduled())
		})

		It("should schedule pods that do consume opaque integer resources.", func() {
			By("Creating a pod that requires less of the opaque resource than is allocatable on a node.")
			requests := api.ResourceList{
				api.ResourceCPU: resource.MustParse("0.1"),
				opaqueResName:   resource.MustParse("1"),
			}
			limits := api.ResourceList{
				api.ResourceCPU: resource.MustParse("0.2"),
				opaqueResName:   resource.MustParse("2"),
			}
			pod, err := f.Client.Pods(f.Namespace.Name).Create(newTestPod(f, "min-oir", requests, limits))
			Expect(err).NotTo(HaveOccurred())
			Expect(pod).To(beScheduled())
		})

		It("should not schedule pods that exceed the available amount of opaque integer resource.", func() {
			By("Creating a pod that requires more of the opaque resource than is allocatable on any node")
			requests := api.ResourceList{opaqueResName: resource.MustParse("6")}
			limits := api.ResourceList{}
			pod, err := f.Client.Pods(f.Namespace.Name).Create(newTestPod(f, "over-max-oir", requests, limits))
			Expect(err).NotTo(HaveOccurred())
			// TODO(CD): Watch scheduler events instead to catch an indication of
			//           insufficient resources or other unschedulable condition
			//           instead of waiting for this to time out.
			Expect(pod).To(notBeScheduled())
		})

		It("should account opaque integer resources in pods with multiple containers.", func() {
			By("Creating a pod with two containers that together require less of the opaque resource than is allocatable on a node")
			requests := api.ResourceList{opaqueResName: resource.MustParse("1")}
			limits := api.ResourceList{}
			// This pod consumes 2 "foo" resources.
			pod := &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "mult-container-oir",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "pause",
							Image: framework.GetPauseImageName(f.Client),
							Resources: api.ResourceRequirements{
								Requests: requests,
								Limits:   limits,
							},
						},
						{
							Name:  "pause-sidecar",
							Image: framework.GetPauseImageName(f.Client),
							Resources: api.ResourceRequirements{
								Requests: requests,
								Limits:   limits,
							},
						},
					},
				},
			}
			pod, err := f.Client.Pods(f.Namespace.Name).Create(pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(pod).To(beScheduled())

			By("Creating a pod with two containers that together require more of the opaque resource than is allotable on any node")
			requests = api.ResourceList{opaqueResName: resource.MustParse("3")}
			limits = api.ResourceList{}
			// This pod consumes 6 "foo" resources.
			pod = &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "mult-container-over-max-oir",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "pause",
							Image: framework.GetPauseImageName(f.Client),
							Resources: api.ResourceRequirements{
								Requests: requests,
								Limits:   limits,
							},
						},
						{
							Name:  "pause-sidecar",
							Image: framework.GetPauseImageName(f.Client),
							Resources: api.ResourceRequirements{
								Requests: requests,
								Limits:   limits,
							},
						},
					},
				},
			}
			pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
			Expect(err).NotTo(HaveOccurred())
			// TODO(CD): Watch scheduler events instead to catch an indication of
			//           insufficient resources or other unschedulable condition
			//           instead of waiting for this to time out.
			Expect(pod).To(notBeScheduled())
		})
	})
})

// Returns the current version of the supplied node. If `old` is nil, returns
// any non-master node.
func refresh(f *framework.Framework, old *api.Node) *api.Node {
	if old == nil {
		// Priming invocation; select the first non-master node.
		nodes, err := f.Client.Nodes().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		for _, n := range nodes.Items {
			if !system.IsMasterNode(&n) {
				return &n
			}
		}
		Fail("unable to select a non-master node")
		return nil
	}

	// Get the node that has the same name as the argument.
	n, err := f.Client.Nodes().Get(old.Name)
	Expect(err).NotTo(HaveOccurred())
	return n
}

// Waits for the supplied predicate to become true for the supplied node.
func awaitNodePredicate(c *unversioned.Client, node *api.Node, p func(*api.Node) bool) {
	// Allocatable is updated by the kubelet on next sync following the above
	// patch operation. Wait up 20s polling every 2s (default interval: 10s).
	timeout := "20s"
	pollInterval := "2s"
	Eventually(func() bool {
		n, e := c.Nodes().Get(node.Name)
		if e != nil {
			return false
		}
		return p(n)
	}, timeout, pollInterval).Should(BeTrue())
}

// Removes the "foo" resources from the supplied node.
func removeOpaqueRes(f *framework.Framework, n *api.Node) {
	// Use the client to patch node capacity, removing foo resource.
	patch := []byte(`[{"op": "remove", "path": "/status/capacity/pod.alpha.kubernetes.io~1opaque-int-resource-foo"}]`)
	f.Client.Patch(api.JSONPatchType).Resource("nodes").Name(n.Name).SubResource("status").Body(patch).Do()
	awaitNodePredicate(f.Client, n, func(n *api.Node) bool {
		_, exists := n.Status.Allocatable[opaqueResName]
		return !exists
	})
}

type scheduledMatcher struct{ f *framework.Framework }

func (m *scheduledMatcher) Match(actual interface{}) (success bool, err error) {
	err = m.f.WaitForPodRunning(actual.(*api.Pod).Name)
	success = err == nil
	return
}

func (m *scheduledMatcher) FailureMessage(actual interface{}) (message string) {
	pod := actual.(*api.Pod)
	podJSON, runningPodJSON := podErrorJSON(m.f, pod)
	return fmt.Sprintf("Expected pod [%s] to be scheduled\n%v\n\nRunning pods are:\n\n%s", pod.Name, string(podJSON), string(runningPodJSON))
}

func (m *scheduledMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	pod := actual.(*api.Pod)
	podJSON, runningPodJSON := podErrorJSON(m.f, pod)
	return fmt.Sprintf("Expected pod [%s] not to be scheduled\n%v\n\nRunning pods are:\n\n%s", pod.Name, string(podJSON), string(runningPodJSON))
}

type notScheduledMatcher struct{ f *framework.Framework }

func (m *notScheduledMatcher) Match(actual interface{}) (success bool, err error) {
	pod := actual.(*api.Pod)
	// Wait up 1 minute polling every 5 seconds.
	timeout := time.Minute
	interval := 5 * time.Second
	err = wait.Poll(interval, timeout, func() (done bool, err error) {
		p, err := m.f.Client.Pods(m.f.Namespace.Name).Get(pod.Name)
		if err != nil {
			return false, nil
		}
		// Return true if the unschedulable condition is present.
		for _, cond := range p.Status.Conditions {
			if cond.Reason == "Unschedulable" {
				return true, nil
			}
		}
		return false, nil
	})
	return err == nil, err
}

func (m *notScheduledMatcher) FailureMessage(actual interface{}) (message string) {
	pod := actual.(*api.Pod)
	podJSON, runningPodJSON := podErrorJSON(m.f, pod)
	return fmt.Sprintf("Expected pod [%s] not to be scheduled\n%v\n\nRunning pods are:\n\n%s", pod.Name, string(podJSON), string(runningPodJSON))
}

func (m *notScheduledMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	pod := actual.(*api.Pod)
	podJSON, runningPodJSON := podErrorJSON(m.f, pod)
	return fmt.Sprintf("Expected pod [%s] to be scheduled\n%v\n\nRunning pods are:\n\n%s", pod.Name, string(podJSON), string(runningPodJSON))
}

func podErrorJSON(f *framework.Framework, pod *api.Pod) (podJSON []byte, runningPodJSON []byte) {
	podJSON, err := json.MarshalIndent(pod, "", "  ")
	runningPodJSON = []byte{}
	allPods, err := f.Client.Pods(f.Namespace.Name).List(api.ListOptions{})
	if err == nil {
		runningPodJSON, _ = json.MarshalIndent(allPods.Items, "", "  ")
	}
	return
}
