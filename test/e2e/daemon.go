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
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Daemon", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
		err = clearNodeLabels(c)
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		err := clearNodeLabels(c)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should launch a daemon pod on every node of the cluster", func() {
		testDaemons(c)
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

func checkDaemonPodOnNodes(c *client.Client, selector map[string]string, nodeNames []string) func() (bool, error) {
	// Don't return an error, because returning an error will abort wait.Poll, but
	// if there's an error, we want to try getting the daemon again.
	return func() (bool, error) {
		// Get list of pods satisfying selector.
		podList, err := c.Pods(api.NamespaceDefault).List(labels.Set(selector).AsSelector(), fields.Everything())
		if err != nil {
			return false, nil
		}
		pods := podList.Items

		// Get a map of node names to number of daemon pods running on the node.
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

func checkRunningOnAllNodes(c *client.Client, selector map[string]string) func() (bool, error) {
	// Don't return an error, because returning an error will abort wait.Poll, but
	// if there's an error, we want to try getting the daemon again.
	return func() (bool, error) {
		nodeList, err := c.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			return false, nil
		}
		nodeNames := make([]string, 0)
		for _, node := range nodeList.Items {
			nodeNames = append(nodeNames, node.Name)
		}
		return checkDaemonPodOnNodes(c, selector, nodeNames)()
	}
}

func checkRunningOnNoNodes(c *client.Client, selector map[string]string) func() (bool, error) {
	return checkDaemonPodOnNodes(c, selector, make([]string, 0))
}

func testDaemons(c *client.Client) {
	ns := api.NamespaceDefault
	simpleDaemonName := "simple-daemon"
	image := "gcr.io/google_containers/serve_hostname:1.1"
	label := map[string]string{"name": simpleDaemonName}
	retryTimeout := 1 * time.Minute
	retryInterval := 5 * time.Second

	By(fmt.Sprintf("Creating simple daemon %s", simpleDaemonName))
	simpleDaemon, err := c.Daemons(ns).Create(&api.Daemon{
		ObjectMeta: api.ObjectMeta{
			Name: simpleDaemonName,
		},
		Spec: api.DaemonSpec{
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: label,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  simpleDaemonName,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		By(fmt.Sprintf("Check that reaper kills all daemon pods for %s", simpleDaemon.Name))
		daemonReaper, err := kubectl.ReaperFor("Daemon", c)
		Expect(err).NotTo(HaveOccurred())
		_, err = daemonReaper.Stop(ns, simpleDaemon.Name, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		err = wait.Poll(retryInterval, retryTimeout, checkRunningOnNoNodes(c, label))
		Expect(err).NotTo(HaveOccurred())
	}()

	By("Check that daemon pods launch on every node of the cluster.")
	Expect(err).NotTo(HaveOccurred())
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnAllNodes(c, label))
	Expect(err).NotTo(HaveOccurred())

	By("Stop a daemon pod, check that the daemon pod is revived.")
	podClient := c.Pods(api.NamespaceDefault)
	podList, err := podClient.List(labels.Set(label).AsSelector(), fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	Expect(len(podList.Items)).To(BeNumerically(">", 0))
	pod := podList.Items[0]
	err = podClient.Delete(pod.Name, nil)
	Expect(err).NotTo(HaveOccurred())
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnAllNodes(c, label))
	Expect(err).NotTo(HaveOccurred())

	complexDaemonName := "complex-daemon"
	complexLabel := map[string]string{"name": complexDaemonName}
	nodeSelector := map[string]string{"color": "blue"}
	By(fmt.Sprintf("Creating daemon with a node selector %s", complexDaemonName))
	complexDaemon, err := c.Daemons(ns).Create(&api.Daemon{
		ObjectMeta: api.ObjectMeta{
			Name: complexDaemonName,
		},
		Spec: api.DaemonSpec{
			Selector: complexLabel,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: complexLabel,
				},
				Spec: api.PodSpec{
					NodeSelector: nodeSelector,
					Containers: []api.Container{
						{
							Name:  complexDaemonName,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		By(fmt.Sprintf("Check that reaper kills all daemon pods for %s", complexDaemon.Name))
		daemonReaper, err := kubectl.ReaperFor("Daemon", c)
		Expect(err).NotTo(HaveOccurred())
		_, err = daemonReaper.Stop(ns, complexDaemon.Name, 0, nil)
		Expect(err).NotTo(HaveOccurred())
		err = wait.Poll(retryInterval, retryTimeout, checkRunningOnNoNodes(c, complexLabel))
		Expect(err).NotTo(HaveOccurred())
	}()

	By(fmt.Sprintf("Initially, daemon pods should not be running on any nodes."))
	err = wait.Poll(retryInterval, retryTimeout, checkRunningOnNoNodes(c, complexLabel))
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Change label of node, check that daemon pod is launched."))
	nodeClient := c.Nodes()
	nodeList, err := nodeClient.List(labels.Everything(), fields.Everything())
	Expect(len(nodeList.Items)).To(BeNumerically(">", 0))
	nodeList.Items[0].Labels = nodeSelector
	newNode, err := nodeClient.Update(&nodeList.Items[0])
	Expect(err).NotTo(HaveOccurred())
	Expect(len(newNode.Labels)).To(Equal(1))
	err = wait.Poll(retryInterval, retryTimeout, checkDaemonPodOnNodes(c, complexLabel, []string{newNode.Name}))
	Expect(err).NotTo(HaveOccurred())
}
