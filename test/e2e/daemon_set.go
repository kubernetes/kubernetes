/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Daemon set", func() {
	f := &Framework{BaseName: "daemonsets"}

	BeforeEach(func() {
		f.beforeEach()
		err := clearNodeLabels(f.Client)
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		err := clearNodeLabels(f.Client)
		Expect(err).NotTo(HaveOccurred())
		f.afterEach()
	})

	It("should launch a daemon pod on every node of the cluster", func() {
		testDaemonSets(f)
	})
})

func clearNodeLabels(c *client.Client) error {
	nodeClient := c.Nodes()
	nodeList, err := nodeClient.List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for _, node := range nodeList.Items {
		if len(node.Labels) != 0 {
			node.Labels = map[string]string{}
			newNode, err := nodeClient.Update(&node)
			if err != nil {
				return err
			} else if len(newNode.Labels) != 0 {
				return fmt.Errorf("Could not make node labels nil.")
			}
		}
	}
	return nil
}

func checkDaemonPodOnNodes(f *Framework, selector map[string]string, nodeNames []string) func() (bool, error) {
	return func() (bool, error) {
		podList, err := f.Client.Pods(f.Namespace.Name).List(labels.Set(selector).AsSelector(), fields.Everything())
		if err != nil {
			return false, nil
		}
		pods := podList.Items

		nodesToPodCount := make(map[string]int)
		for _, pod := range pods {
			nodesToPodCount[pod.Spec.NodeName] += 1
		}

		// Ensure that exactly 1 pod is running on all nodes in nodeNames.
		for _, nodeName := range nodeNames {
			if nodesToPodCount[nodeName] != 1 {
				return false, nil
			}
		}

		// Ensure that sizes of the lists are the same. We've verified that every element of nodeNames is in
		// nodesToPodCount, so verifying the lengths are equal ensures that there aren't pods running on any
		// other nodes.
		return len(nodesToPodCount) == len(nodeNames), nil
	}
}

func checkRunningOnAllNodes(f *Framework, selector map[string]string) func() (bool, error) {
	return func() (bool, error) {
		nodeList, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			return false, nil
		}
		nodeNames := make([]string, 0)
		for _, node := range nodeList.Items {
			nodeNames = append(nodeNames, node.Name)
		}
		return checkDaemonPodOnNodes(f, selector, nodeNames)()
	}
}

func checkRunningOnNoNodes(f *Framework, selector map[string]string) func() (bool, error) {
	return checkDaemonPodOnNodes(f, selector, make([]string, 0))
}

func testDaemonSets(f *Framework) {
	ns := f.Namespace.Name
	c := f.Client
	simpleDSName := "simple-daemon-set"
	image := "gcr.io/google_containers/serve_hostname:1.1"
	label := map[string]string{"name": simpleDSName}
	retryTimeout := 1 * time.Minute
	retryInterval := 5 * time.Second

	Logf("Creating simple daemon set %s", simpleDSName)
	_, err := c.DaemonSets(ns).Create(&experimental.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name: simpleDSName,
		},
		Spec: experimental.DaemonSetSpec{
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: label,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  simpleDSName,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())

	By("Check that daemon pods launch on every node of the cluster.")
	Expect(err).NotTo(HaveOccurred())
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnAllNodes(f, label))
	Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

	By("Stop a daemon pod, check that the daemon pod is revived.")
	podClient := c.Pods(ns)

	podList, err := podClient.List(labels.Set(label).AsSelector(), fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	Expect(len(podList.Items)).To(BeNumerically(">", 0))
	pod := podList.Items[0]
	err = podClient.Delete(pod.Name, nil)
	Expect(err).NotTo(HaveOccurred())
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnAllNodes(f, label))
	Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to revive")

	complexDSName := "complex-daemon-set"
	complexLabel := map[string]string{"name": complexDSName}
	nodeSelector := map[string]string{"color": "blue"}
	Logf("Creating daemon with a node selector %s", complexDSName)
	_, err = c.DaemonSets(ns).Create(&experimental.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name: complexDSName,
		},
		Spec: experimental.DaemonSetSpec{
			Selector: complexLabel,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: complexLabel,
				},
				Spec: api.PodSpec{
					NodeSelector: nodeSelector,
					Containers: []api.Container{
						{
							Name:  complexDSName,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())

	By("Initially, daemon pods should not be running on any nodes.")
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnNoNodes(f, complexLabel))
	Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on no nodes")

	By("Change label of node, check that daemon pod is launched.")
	nodeClient := c.Nodes()
	nodeList, err := nodeClient.List(labels.Everything(), fields.Everything())
	Expect(len(nodeList.Items)).To(BeNumerically(">", 0))
	nodeList.Items[0].Labels = nodeSelector
	newNode, err := nodeClient.Update(&nodeList.Items[0])
	Expect(err).NotTo(HaveOccurred())
	Expect(len(newNode.Labels)).To(Equal(1))
	err = wait.Poll(retryInterval, retryTimeout, checkDaemonPodOnNodes(f, complexLabel, []string{newNode.Name}))
	Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on new nodes")

	By("remove the node selector and wait for")
	newNode, err = nodeClient.Get(newNode.Name)
	Expect(err).NotTo(HaveOccurred(), "error getting node")
	newNode.Labels = map[string]string{}
	newNode, err = nodeClient.Update(newNode)
	Expect(err).NotTo(HaveOccurred())
	Expect(wait.Poll(retryInterval, retryTimeout, checkRunningOnNoNodes(f, complexLabel))).
		NotTo(HaveOccurred(), "error waiting for daemon pod to not be running on nodes")
}
