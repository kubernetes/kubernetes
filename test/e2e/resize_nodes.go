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
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func resizeNodeInstanceGroup(size int) error {
	// TODO: make this hit the compute API directly instread of shelling out to gcloud.
	output, err := exec.Command("gcloud", "preview", "managed-instance-groups", "--project="+testContext.CloudConfig.ProjectID, "--zone="+testContext.CloudConfig.Zone,
		"resize", testContext.CloudConfig.NodeInstanceGroup, fmt.Sprintf("--new-size=%v", size)).CombinedOutput()
	if err != nil {
		Logf("Failed to resize node instance group: %v", string(output))
	}
	return err
}

func nodeInstanceGroupSize() (int, error) {
	// TODO: make this hit the compute API directly instread of shelling out to gcloud.
	output, err := exec.Command("gcloud", "preview", "managed-instance-groups", "--project="+testContext.CloudConfig.ProjectID,
		"--zone="+testContext.CloudConfig.Zone, "describe", testContext.CloudConfig.NodeInstanceGroup).CombinedOutput()
	if err != nil {
		return -1, err
	}
	pattern := "currentSize: "
	i := strings.Index(string(output), pattern)
	if i == -1 {
		return -1, fmt.Errorf("could not find '%s' in the output '%s'", pattern, output)
	}
	truncated := output[i+len(pattern):]
	j := strings.Index(string(truncated), "\n")
	if j == -1 {
		return -1, fmt.Errorf("could not find new line in the truncated output '%s'", truncated)
	}

	currentSize, err := strconv.Atoi(string(truncated[:j]))
	if err != nil {
		return -1, err
	}
	return currentSize, nil
}

func waitForNodeInstanceGroupSize(size int) error {
	for start := time.Now(); time.Since(start) < 4*time.Minute; time.Sleep(5 * time.Second) {
		currentSize, err := nodeInstanceGroupSize()
		if err != nil {
			Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != size {
			Logf("Waiting for node istance group size %d, current size %d", size, currentSize)
			continue
		}
		Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting for node instance group size to be %d", size)
}

func waitForClusterSize(c *client.Client, size int) error {
	for start := time.Now(); time.Since(start) < 4*time.Minute; time.Sleep(20 * time.Second) {
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}
		if len(nodes.Items) == size {
			Logf("Cluster has reached the desired size %d", size)
			return nil
		}
		Logf("Waiting for cluster size %d, current size %d", size, len(nodes.Items))
	}
	return fmt.Errorf("timeout waiting for cluster size to be %d", size)
}

func newServiceWithNameSelector(name string) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "test-service",
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": name,
			},
			Ports: []api.ServicePort{{
				Port:       9376,
				TargetPort: util.NewIntOrStringFromInt(9376),
			}},
		},
	}
}

func createServiceWithNameSelector(c *client.Client, ns, name string) error {
	_, err := c.Services(ns).Create(newServiceWithNameSelector(name))
	return err
}

func newReplicationControllerWithNameSelector(name string, replicas int, image string) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}
}

func createServeHostnameReplicationController(c *client.Client, ns, name string, replicas int) (*api.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.ReplicationControllers(ns).Create(newReplicationControllerWithNameSelector(name, replicas, "gcr.io/google_containers/serve_hostname:1.1"))
}

func resizeReplicationController(c *client.Client, ns, name string, replicas int) error {
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		return err
	}
	rc.Spec.Replicas = replicas
	_, err = c.ReplicationControllers(rc.Namespace).Update(rc)
	return err
}

func waitForPodsCreated(c *client.Client, ns, name string, replicas int) (*api.PodList, error) {
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		pods, err := c.Pods(ns).List(label, fields.Everything())
		if err != nil {
			return nil, err
		}

		Logf("Controller %s: Found %d pods out of %d", name, len(pods.Items), replicas)
		if len(pods.Items) == replicas {
			return pods, nil
		}
	}
	return nil, fmt.Errorf("controller %s: Gave up waiting for %d pods to come up", name, replicas)
}

func waitForPodsRunning(c *client.Client, pods *api.PodList) []error {
	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	By("ensuring each pod is running")
	e := []error{}
	for _, pod := range pods.Items {
		// TODO: make waiting parallel.
		err := waitForPodRunningInNamespace(c, pod.Name, pod.Namespace)
		if err != nil {
			e = append(e, err)
		}
	}
	return e
}

func verifyPodsResponding(c *client.Client, ns, name string, pods *api.PodList) error {
	By("trying to dial each unique pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return wait.Poll(retryInterval, retryTimeout, podResponseChecker{c, ns, label, name, pods}.checkAllResponses)
}

func waitForPodsCreatedRunningResponding(c *client.Client, ns, name string, replicas int) error {
	pods, err := waitForPodsCreated(c, ns, name, replicas)
	if err != nil {
		return err
	}
	e := waitForPodsRunning(c, pods)
	if len(e) > 0 {
		return fmt.Errorf("Failed to wait for pods running: %v", e)
	}
	err = verifyPodsResponding(c, ns, name, pods)
	if err != nil {
		return err
	}
	return nil
}

var _ = Describe("Nodes", func() {
	supportedProviders := []string{"gce"}
	var testName string
	var c *client.Client
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		testingNs, err := createTestingNS("resize-nodes", c)
		ns = testingNs.Name
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By(fmt.Sprintf("destroying namespace for this suite %s", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete namespace '%s', %v", ns, err)
		}
	})

	Describe("Resize", func() {
		BeforeEach(func() {
			if !providerIs(supportedProviders...) {
				Failf("Nodes.Resize test is only supported for providers %v (not %s). You can avoid this failure by using ginkgo.skip=Nodes.Resize in your environment.",
					supportedProviders, testContext.Provider)
			}
		})

		AfterEach(func() {
			if !providerIs(supportedProviders...) {
				return
			}
			By("restoring the original node instance group size")
			if err := resizeNodeInstanceGroup(testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := waitForNodeInstanceGroupSize(testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := waitForClusterSize(c, testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original cluster size: %v", err)
			}
		})

		testName = "should be able to delete nodes."
		It(testName, func() {
			Logf("starting test %s", testName)

			if testContext.CloudConfig.NumNodes < 2 {
				Failf("Failing test %s as it requires at lease 2 nodes (not %d)", testName, testContext.CloudConfig.NumNodes)
				return
			}

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker containter kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			replicas := testContext.CloudConfig.NumNodes
			createServeHostnameReplicationController(c, ns, name, replicas)
			err := waitForPodsCreatedRunningResponding(c, ns, name, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("decreasing cluster size to %d", replicas-1))
			err = resizeNodeInstanceGroup(replicas - 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForNodeInstanceGroupSize(replicas - 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForClusterSize(c, replicas-1)
			Expect(err).NotTo(HaveOccurred())

			By("verifying whether the pods from the removed node are recreated")
			err = waitForPodsCreatedRunningResponding(c, ns, name, replicas)
			Expect(err).NotTo(HaveOccurred())
		})

		testName = "should be able to add nodes."
		It(testName, func() {
			Logf("starting test %s", testName)

			if testContext.CloudConfig.NumNodes < 2 {
				Failf("Failing test %s as it requires at lease 2 nodes (not %d)", testName, testContext.CloudConfig.NumNodes)
				return
			}

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker containter kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-add-node"
			createServiceWithNameSelector(c, ns, name)
			replicas := testContext.CloudConfig.NumNodes
			createServeHostnameReplicationController(c, ns, name, replicas)
			err := waitForPodsCreatedRunningResponding(c, ns, name, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing cluster size to %d", replicas+1))
			err = resizeNodeInstanceGroup(replicas + 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForNodeInstanceGroupSize(replicas + 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForClusterSize(c, replicas+1)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", replicas+1))
			err = resizeReplicationController(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForPodsCreatedRunningResponding(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("Network", func() {
		BeforeEach(func() {
			if !providerIs(supportedProviders...) {
				Failf("Nodes.Network test is only supported for providers %v (not %s). You can avoid this failure by using ginkgo.skip=Nodes.Network in your environment.",
					supportedProviders, testContext.Provider)
			}
		})

		testName = "should survive network partition."
		It(testName, func() {
			if testContext.CloudConfig.NumNodes < 2 {
				By(fmt.Sprintf("skipping %s test, which requires at lease 2 nodes (not %d)",
					testName, testContext.CloudConfig.NumNodes))
				return
			}

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker containter kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-net"
			createServiceWithNameSelector(c, ns, name)
			replicas := testContext.CloudConfig.NumNodes
			createServeHostnameReplicationController(c, ns, name, replicas)
			err := waitForPodsCreatedRunningResponding(c, ns, name, replicas)
			Expect(err).NotTo(HaveOccurred())

			By("cause network partition on one node")
			nodelist, err := c.Nodes().List(labels.Everything(), fields.Everything())
			Expect(err).NotTo(HaveOccurred())
			node := nodelist.Items[0]
			pod, err := waitForRCPodOnNode(c, ns, name, node.Name)
			Expect(err).NotTo(HaveOccurred())
			Logf("Getting external IP address for %s", name)
			host := ""
			for _, a := range node.Status.Addresses {
				if a.Type == api.NodeExternalIP {
					host = a.Address + ":22"
					break
				}
			}
			Logf("Setting network partition on %s", node.Name)
			dropCmd := fmt.Sprintf("sudo iptables -I OUTPUT 1 -d %s -j DROP", testContext.CloudConfig.MasterName)
			if _, _, code, err := SSH(dropCmd, host, testContext.Provider); code != 0 || err != nil {
				Failf("Expected 0 exit code and nil error when running %s on %s, got %d and %v",
					dropCmd, node, code, err)
			}

			Logf("Waiting for node %s to be not ready", node.Name)
			waitForNodeToBe(c, node.Name, false, 2*time.Minute)

			Logf("Waiting for pod %s to be removed", pod.Name)
			waitForRCPodToDisappear(c, ns, name, pod.Name)

			By("verifying whether the pod from the partitioned node is recreated")
			err = waitForPodsCreatedRunningResponding(c, ns, name, replicas)
			Expect(err).NotTo(HaveOccurred())

			By("remove network partition")
			undropCmd := "sudo iptables --delete OUTPUT 1"
			if _, _, code, err := SSH(undropCmd, host, testContext.Provider); code != 0 || err != nil {
				Failf("Expected 0 exit code and nil error when running %s on %s, got %d and %v",
					undropCmd, node, code, err)
			}

			Logf("Waiting for node %s to be ready", node.Name)
			waitForNodeToBe(c, node.Name, true, 2*time.Minute)

			By("verify wheter new pods can be created on the re-attached node")
			err = resizeReplicationController(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForPodsCreatedRunningResponding(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			_, err = waitForRCPodOnNode(c, ns, name, node.Name)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
