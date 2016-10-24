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

func Logf(format string, args ...interface{}) {
	fmt.Fprintf(GinkgoWriter, "INFO: "+format+"\n", args...)
}

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("MemoryEvictionAdmitAfterEvict [FLAKY] [Slow] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("eviction-test-admit")

	Context("when there is memory pressure", func() {
		It("should evict pods", func() {
			By("creating a best effort pod")
			besteffort := createMemhogPod(f, "besteffort-", "besteffort", api.ResourceRequirements{}, false)
			By("observing memory pressure on the node")
			Eventually(func() error {
				node, err := f.Client.Nodes().Get(besteffort.Spec.NodeName)
				if err != nil {
					return fmt.Errorf("tried to get node, but got error: %v", err)
				}
				_, pressure := api.GetNodeCondition(&node.Status, api.NodeMemoryPressure)
				if pressure != nil && pressure.Status == api.ConditionFalse {
					return fmt.Errorf("node is still not reporting memory pressure condition: %s", pressure)
				}
				return nil
			}, 5*time.Minute, 15*time.Second).Should(BeNil())
			By("observing the pod eviction")
			Eventually(func() error {
				pod, err := f.Client.Pods(f.Namespace.Name).Get(besteffort.Name)
				if err != nil {
					return err
				}
				if pod.Status.Phase == api.PodFailed {
					return nil
				}
				return fmt.Errorf("pod has not yet been evicted")
			}, 60*time.Minute, 5*time.Second).Should(BeNil())
			By("attempting to create a new best effort pod")
			podName := "admit-best-effort-pod"
			f.PodClient().Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image: framework.GetPauseImageNameForHostArch(),
							Name:  podName,
						},
					},
				},
			})
			By("observing the pod is not admitted to the kubelet")
			Eventually(func() error {
				pod, err := f.Client.Pods(f.Namespace.Name).Get(podName)
				if err != nil {
					return err
				}
				if pod.Status.Phase == api.PodFailed {
					return nil
				}
				return fmt.Errorf("pod has not yet been evicted")
			}, 60*time.Minute, 5*time.Second).Should(BeNil())
			By("observing memory pressure on the node is relieved")
			Eventually(func() error {
				node, err := f.Client.Nodes().Get(besteffort.Spec.NodeName)
				if err != nil {
					return fmt.Errorf("tried to get node, but got error: %v", err)
				}
				_, pressure := api.GetNodeCondition(&node.Status, api.NodeMemoryPressure)
				if pressure != nil && pressure.Status == api.ConditionTrue {
					return fmt.Errorf("node is still reporting memory pressure condition: %s", pressure)
				}
				return nil
			}, 5*time.Minute, 15*time.Second).Should(BeNil())

			// Finally, try starting a new pod and wait for it to be scheduled and running.
			// This is the final check to try to prevent interference with subsequent tests.
			podName = "admit-best-effort-pod"
			f.PodClient().CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image: framework.GetPauseImageNameForHostArch(),
							Name:  podName,
						},
					},
				},
			})
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
