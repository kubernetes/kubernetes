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

package e2e_node

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("MemoryEvictionSimple [Slow] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("eviction-test-simple")

	Context("when there is memory pressure", func() {
		AfterEach(func() {
			// Check available memory after condition disappears, just in case:
			// Wait for available memory to decrease to a reasonable level before ending the test.
			// This helps prevent interference with tests that start immediately after this one.
			By("waiting for available memory to decrease to a reasonable level before ending the test.")
			Eventually(func() error {
				memoryLimit := getMemoryLimit(f)
				summary, err := getNodeSummary()
				if err != nil {
					return err
				}
				if summary.Node.Memory.AvailableBytes == nil {
					return fmt.Errorf("summary.Node.Memory.AvailableBytes was nil, cannot get memory stats.")
				}
				avail := int64(*summary.Node.Memory.AvailableBytes)

				halflimit := memoryLimit.Value() / 2

				// Wait for at least half of memory limit to be available
				if avail >= halflimit {
					return nil
				}
				return fmt.Errorf("current available memory is: %d bytes. Expected at least %d bytes available.", avail, halflimit)
			}, 5*time.Minute, 15*time.Second).Should(BeNil())
		})
		var burstable *api.Pod
		It("should contain node memory pressure condition)", func() {
			memoryLimit := getMemoryLimit(f)
			By("creating a memory hogging pod.")
			burstable = createMemhogPod(f, "burstable-", "burstable", api.ResourceRequirements{
				Limits: api.ResourceList{
					"cpu": resource.MustParse("100m"),
					// Set the memory limit to 80% of machine capacity to induce memory pressure.
					// TODO: Fetch kubelet configuration and set this value based on eviction thresholds.
					"memory": memoryLimit,
				},
			}, true)

			By("polling the node condition and wait till memory pressure ")
			Eventually(func() error {
				nodeList, err := f.Client.Nodes().List(api.ListOptions{})
				if err != nil {
					return fmt.Errorf("tried to get node list but got error: %v", err)
				}
				// Assuming that there is only one node, because this is a node e2e test.
				if len(nodeList.Items) != 1 {
					return fmt.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
				}
				node := nodeList.Items[0]

				_, pressure := api.GetNodeCondition(&node.Status, api.NodeMemoryPressure)
				if pressure == nil || pressure.Status != api.ConditionTrue {
					return fmt.Errorf("node is not reporting memory pressure condition. Current Conditions: %+v", node.Status.Conditions)
				}
				return nil
			}, 5*time.Minute, 5*time.Second).Should(BeNil())
		})
		It("should drop the node memory pressure condition", func() {
			By("deleting the memory hog pod.")
			graceperiod := int64(1)
			err := f.PodClient().Delete(burstable.Name, &api.DeleteOptions{GracePeriodSeconds: &graceperiod})
			Expect(err).To(BeNil(), fmt.Sprintf("Failed to delete memory hogging pod: %v", err))

			// Wait for the memory pressure condition to disappear from the node status before continuing.
			Eventually(func() error {
				nodeList, err := f.Client.Nodes().List(api.ListOptions{})
				if err != nil {
					return fmt.Errorf("tried to get node list but got error: %v", err)
				}
				// Assuming that there is only one node, because this is a node e2e test.
				if len(nodeList.Items) != 1 {
					return fmt.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
				}
				node := nodeList.Items[0]
				_, pressure := api.GetNodeCondition(&node.Status, api.NodeMemoryPressure)
				if pressure != nil && pressure.Status == api.ConditionTrue {
					return fmt.Errorf("node is still reporting memory pressure condition: %s", pressure)
				}
				return nil
			}, 5*time.Minute, 15*time.Second).Should(BeNil())
		})
		It("should admit a best effort pod", func() {
			// Finally, try starting a new pod and wait for it to be scheduled and running.
			// This is the final check to try to prevent interference with subsequent tests.
			podName := "admit-best-effort-pod"
			f.PodClient().CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image: ImageRegistry[pauseImage],
							Name:  podName,
						},
					},
				},
			})
			// Pod admitted. Memory eviction feature test ends.
		})
	})
})

func getMemoryLimit(f *framework.Framework) resource.Quantity {
	By("getting the memory limits of the current node.")
	nodeList, err := f.Client.Nodes().List(api.ListOptions{})
	Expect(err).To(BeNil(), fmt.Sprintf("Failed to get list of nodes: %v", err))
	// Assuming that there is only one node, because this is a node e2e test.
	Expect(len(nodeList.Items)).To(Equal(1), fmt.Sprintf("Expected one node: %v", nodeList.Items))
	node := nodeList.Items[0]
	memoryLimit := node.Status.Capacity[api.ResourceMemory]
	Expect(memoryLimit.Value() > 0).To(BeTrue(), fmt.Sprintf("Expected memory limit to be greater than 0: %+v", node.Status))
	return memoryLimit
}
