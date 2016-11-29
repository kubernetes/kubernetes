/*
Copyright 2015 The Kubernetes Authors.

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

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	mb = 1024 * 1024
	gb = 1024 * mb

	// TODO(madhusudancs): find a way to query kubelet's disk space manager to obtain this value. 256MB
	// is the default that is set today. This test might break if the default value changes. This value
	// can be configured by setting the "low-diskspace-threshold-mb" flag while starting a kubelet.
	// However, kubelets are started as part of the cluster start up, once, before any e2e test is run,
	// and remain unchanged until all the tests are run and the cluster is brought down. Changing the
	// flag value affects all the e2e tests. So we are hard-coding this value for now.
	lowDiskSpaceThreshold uint64 = 256 * mb

	nodeOODTimeOut = 5 * time.Minute

	numNodeOODPods = 3
)

// Plan:
// 1. Fill disk space on all nodes except one. One node is left out so that we can schedule pods
//    on that node. Arbitrarily choose that node to be node with index 0.  This makes this a disruptive test.
// 2. Get the CPU capacity on unfilled node.
// 3. Divide the available CPU into one less than the number of pods we want to schedule. We want
//    to schedule 3 pods, so divide CPU capacity by 2.
// 4. Request the divided CPU for each pod.
// 5. Observe that 2 of the pods schedule onto the node whose disk is not full, and the remaining
//    pod stays pending and does not schedule onto the nodes whose disks are full nor the node
//    with the other two pods, since there is not enough free CPU capacity there.
// 6. Recover disk space from one of the nodes whose disk space was previously filled. Arbritrarily
//    choose that node to be node with index 1.
// 7. Observe that the pod in pending status schedules on that node.
//
// Flaky issue #20015.  We have no clear path for how to test this functionality in a non-flaky way.
var _ = framework.KubeDescribe("NodeOutOfDisk [Serial] [Flaky] [Disruptive]", func() {
	var c clientset.Interface
	var unfilledNodeName, recoveredNodeName string
	f := framework.NewDefaultFramework("node-outofdisk")

	BeforeEach(func() {
		c = f.ClientSet

		nodelist := framework.GetReadySchedulableNodesOrDie(c)

		// Skip this test on small clusters.  No need to fail since it is not a use
		// case that any cluster of small size needs to support.
		framework.SkipUnlessNodeCountIsAtLeast(2)

		unfilledNodeName = nodelist.Items[0].Name
		for _, node := range nodelist.Items[1:] {
			fillDiskSpace(c, &node)
		}
	})

	AfterEach(func() {

		nodelist := framework.GetReadySchedulableNodesOrDie(c)
		Expect(len(nodelist.Items)).ToNot(BeZero())
		for _, node := range nodelist.Items {
			if unfilledNodeName == node.Name || recoveredNodeName == node.Name {
				continue
			}
			recoverDiskSpace(c, &node)
		}
	})

	It("runs out of disk space", func() {
		unfilledNode, err := c.Core().Nodes().Get(unfilledNodeName)
		framework.ExpectNoError(err)

		By(fmt.Sprintf("Calculating CPU availability on node %s", unfilledNode.Name))
		milliCpu, err := availCpu(c, unfilledNode)
		framework.ExpectNoError(err)

		// Per pod CPU should be just enough to fit only (numNodeOODPods - 1) pods on the given
		// node. We compute this value by dividing the available CPU capacity on the node by
		// (numNodeOODPods - 1) and subtracting ϵ from it. We arbitrarily choose ϵ to be 1%
		// of the available CPU per pod, i.e. 0.01 * milliCpu/(numNodeOODPods-1). Instead of
		// subtracting 1% from the value, we directly use 0.99 as the multiplier.
		podCPU := int64(float64(milliCpu/(numNodeOODPods-1)) * 0.99)

		ns := f.Namespace.Name
		podClient := c.Core().Pods(ns)

		By("Creating pods and waiting for all but one pods to be scheduled")

		for i := 0; i < numNodeOODPods-1; i++ {
			name := fmt.Sprintf("pod-node-outofdisk-%d", i)
			createOutOfDiskPod(c, ns, name, podCPU)

			framework.ExpectNoError(f.WaitForPodRunning(name))
			pod, err := podClient.Get(name)
			framework.ExpectNoError(err)
			Expect(pod.Spec.NodeName).To(Equal(unfilledNodeName))
		}

		pendingPodName := fmt.Sprintf("pod-node-outofdisk-%d", numNodeOODPods-1)
		createOutOfDiskPod(c, ns, pendingPodName, podCPU)

		By(fmt.Sprintf("Finding a failed scheduler event for pod %s", pendingPodName))
		wait.Poll(2*time.Second, 5*time.Minute, func() (bool, error) {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      pendingPodName,
				"involvedObject.namespace": ns,
				"source":                   v1.DefaultSchedulerName,
				"reason":                   "FailedScheduling",
			}.AsSelector().String()
			options := v1.ListOptions{FieldSelector: selector}
			schedEvents, err := c.Core().Events(ns).List(options)
			framework.ExpectNoError(err)

			if len(schedEvents.Items) > 0 {
				return true, nil
			} else {
				return false, nil
			}
		})

		nodelist := framework.GetReadySchedulableNodesOrDie(c)
		Expect(len(nodelist.Items)).To(BeNumerically(">", 1))

		nodeToRecover := nodelist.Items[1]
		Expect(nodeToRecover.Name).ToNot(Equal(unfilledNodeName))

		recoverDiskSpace(c, &nodeToRecover)
		recoveredNodeName = nodeToRecover.Name

		By(fmt.Sprintf("Verifying that pod %s schedules on node %s", pendingPodName, recoveredNodeName))
		framework.ExpectNoError(f.WaitForPodRunning(pendingPodName))
		pendingPod, err := podClient.Get(pendingPodName)
		framework.ExpectNoError(err)
		Expect(pendingPod.Spec.NodeName).To(Equal(recoveredNodeName))
	})
})

// createOutOfDiskPod creates a pod in the given namespace with the requested amount of CPU.
func createOutOfDiskPod(c clientset.Interface, ns, name string, milliCPU int64) {
	podClient := c.Core().Pods(ns)

	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: framework.GetPauseImageName(c),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							// Request enough CPU to fit only two pods on a given node.
							v1.ResourceCPU: *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
						},
					},
				},
			},
		},
	}

	_, err := podClient.Create(pod)
	framework.ExpectNoError(err)
}

// availCpu calculates the available CPU on a given node by subtracting the CPU requested by
// all the pods from the total available CPU capacity on the node.
func availCpu(c clientset.Interface, node *v1.Node) (int64, error) {
	podClient := c.Core().Pods(v1.NamespaceAll)

	selector := fields.Set{"spec.nodeName": node.Name}.AsSelector().String()
	options := v1.ListOptions{FieldSelector: selector}
	pods, err := podClient.List(options)
	if err != nil {
		return 0, fmt.Errorf("failed to retrieve all the pods on node %s: %v", node.Name, err)
	}
	avail := node.Status.Capacity.Cpu().MilliValue()
	for _, pod := range pods.Items {
		for _, cont := range pod.Spec.Containers {
			avail -= cont.Resources.Requests.Cpu().MilliValue()
		}
	}
	return avail, nil
}

// availSize returns the available disk space on a given node by querying node stats which
// is in turn obtained internally from cadvisor.
func availSize(c clientset.Interface, node *v1.Node) (uint64, error) {
	statsResource := fmt.Sprintf("api/v1/proxy/nodes/%s/stats/", node.Name)
	framework.Logf("Querying stats for node %s using url %s", node.Name, statsResource)
	res, err := c.Core().RESTClient().Get().AbsPath(statsResource).Timeout(time.Minute).Do().Raw()
	if err != nil {
		return 0, fmt.Errorf("error querying cAdvisor API: %v", err)
	}
	ci := cadvisorapi.ContainerInfo{}
	err = json.Unmarshal(res, &ci)
	if err != nil {
		return 0, fmt.Errorf("couldn't unmarshal container info: %v", err)
	}
	return ci.Stats[len(ci.Stats)-1].Filesystem[0].Available, nil
}

// fillDiskSpace fills the available disk space on a given node by creating a large file. The disk
// space on the node is filled in such a way that the available space after filling the disk is just
// below the lowDiskSpaceThreshold mark.
func fillDiskSpace(c clientset.Interface, node *v1.Node) {
	avail, err := availSize(c, node)
	framework.ExpectNoError(err, "Node %s: couldn't obtain available disk size %v", node.Name, err)

	fillSize := (avail - lowDiskSpaceThreshold + (100 * mb))

	framework.Logf("Node %s: disk space available %d bytes", node.Name, avail)
	By(fmt.Sprintf("Node %s: creating a file of size %d bytes to fill the available disk space", node.Name, fillSize))

	cmd := fmt.Sprintf("fallocate -l %d test.img", fillSize)
	framework.ExpectNoError(framework.IssueSSHCommand(cmd, framework.TestContext.Provider, node))

	ood := framework.WaitForNodeToBe(c, node.Name, v1.NodeOutOfDisk, true, nodeOODTimeOut)
	Expect(ood).To(BeTrue(), "Node %s did not run out of disk within %v", node.Name, nodeOODTimeOut)

	avail, err = availSize(c, node)
	framework.Logf("Node %s: disk space available %d bytes", node.Name, avail)
	Expect(avail < lowDiskSpaceThreshold).To(BeTrue())
}

// recoverDiskSpace recovers disk space, filled by creating a large file, on a given node.
func recoverDiskSpace(c clientset.Interface, node *v1.Node) {
	By(fmt.Sprintf("Recovering disk space on node %s", node.Name))
	cmd := "rm -f test.img"
	framework.ExpectNoError(framework.IssueSSHCommand(cmd, framework.TestContext.Provider, node))

	ood := framework.WaitForNodeToBe(c, node.Name, v1.NodeOutOfDisk, false, nodeOODTimeOut)
	Expect(ood).To(BeTrue(), "Node %s's out of disk condition status did not change to false within %v", node.Name, nodeOODTimeOut)
}
