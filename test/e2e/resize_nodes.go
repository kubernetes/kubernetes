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
	"regexp"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
)

const (
	serveHostnameImage        = "gcr.io/google_containers/serve_hostname:1.1"
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
)

func resizeGroup(size int) error {
	if testContext.Provider == "gce" || testContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "resize",
			testContext.CloudConfig.NodeInstanceGroup, fmt.Sprintf("--size=%v", size),
			"--project="+testContext.CloudConfig.ProjectID, "--zone="+testContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			Logf("Failed to resize node instance group: %v", string(output))
		}
		return err
	} else {
		// Supported by aws
		instanceGroups, ok := testContext.CloudConfig.Provider.(aws_cloud.InstanceGroups)
		if !ok {
			return fmt.Errorf("Provider does not support InstanceGroups")
		}
		return instanceGroups.ResizeInstanceGroup(testContext.CloudConfig.NodeInstanceGroup, size)
	}
}

func groupSize() (int, error) {
	if testContext.Provider == "gce" || testContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", testContext.CloudConfig.NodeInstanceGroup, "--project="+testContext.CloudConfig.ProjectID,
			"--zone="+testContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return -1, err
		}
		re := regexp.MustCompile("RUNNING")
		return len(re.FindAllString(string(output), -1)), nil
	} else {
		// Supported by aws
		instanceGroups, ok := testContext.CloudConfig.Provider.(aws_cloud.InstanceGroups)
		if !ok {
			return -1, fmt.Errorf("provider does not support InstanceGroups")
		}
		instanceGroup, err := instanceGroups.DescribeInstanceGroup(testContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", testContext.CloudConfig.NodeInstanceGroup)
		}
		return instanceGroup.CurrentSize()
	}
}

func waitForGroupSize(size int) error {
	timeout := 10 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		currentSize, err := groupSize()
		if err != nil {
			Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != size {
			Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting %v for node instance group size to be %d", timeout, size)
}

func svcByName(name string) *api.Service {
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

func newSVCByName(c *client.Client, ns, name string) error {
	_, err := c.Services(ns).Create(svcByName(name))
	return err
}

func podOnNode(podName, nodeName string, image string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"name": podName,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  podName,
					Image: image,
					Ports: []api.ContainerPort{{ContainerPort: 9376}},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: api.RestartPolicyNever,
		},
	}
}

func newPodOnNode(c *client.Client, namespace, podName, nodeName string) error {
	pod, err := c.Pods(namespace).Create(podOnNode(podName, nodeName, serveHostnameImage))
	if err == nil {
		Logf("Created pod %s on node %s", pod.ObjectMeta.Name, nodeName)
	} else {
		Logf("Failed to create pod %s on node %s: %v", podName, nodeName, err)
	}
	return err
}

func rcByName(name string, replicas int, image string, labels map[string]string) *api.ReplicationController {
	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
	})
}

func rcByNamePort(name string, replicas int, image string, port int, labels map[string]string) *api.ReplicationController {
	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
		Ports: []api.ContainerPort{{ContainerPort: port}},
	})
}

func rcByNameContainer(name string, replicas int, image string, labels map[string]string, c api.Container) *api.ReplicationController {
	// Add "name": name to the labels, overwriting if it exists.
	labels["name"] = name
	return &api.ReplicationController{
		TypeMeta: api.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: latest.Version,
		},
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
					Labels: labels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{c},
				},
			},
		},
	}
}

// newRCByName creates a replication controller with a selector by name of name.
func newRCByName(c *client.Client, ns, name string, replicas int) (*api.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.ReplicationControllers(ns).Create(rcByNamePort(
		name, replicas, serveHostnameImage, 9376, map[string]string{}))
}

func resizeRC(c *client.Client, ns, name string, replicas int) error {
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		return err
	}
	rc.Spec.Replicas = replicas
	_, err = c.ReplicationControllers(rc.Namespace).Update(rc)
	return err
}

func podsCreated(c *client.Client, ns, name string, replicas int) (*api.PodList, error) {
	timeout := time.Minute
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		pods, err := c.Pods(ns).List(label, fields.Everything())
		if err != nil {
			return nil, err
		}

		created := []api.Pod{}
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp != nil {
				continue
			}
			created = append(created, pod)
		}
		Logf("Pod name %s: Found %d pods out of %d", name, len(created), replicas)

		if len(created) == replicas {
			pods.Items = created
			return pods, nil
		}
	}
	return nil, fmt.Errorf("Pod name %s: Gave up waiting %v for %d pods to come up", name, timeout, replicas)
}

func podsRunning(c *client.Client, pods *api.PodList) []error {
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

func verifyPods(c *client.Client, ns, name string, wantName bool, replicas int) error {
	pods, err := podsCreated(c, ns, name, replicas)
	if err != nil {
		return err
	}
	e := podsRunning(c, pods)
	if len(e) > 0 {
		return fmt.Errorf("failed to wait for pods running: %v", e)
	}
	err = podsResponding(c, ns, name, wantName, pods)
	if err != nil {
		return fmt.Errorf("failed to wait for pods responding: %v", err)
	}
	return nil
}

// Blocks outgoing network traffic on 'node'. Then verifies that 'podNameToDisappear',
// that belongs to replication controller 'rcName', really disappeared.
// Finally, it checks that the replication controller recreates the
// pods on another node and that now the number of replicas is equal 'replicas'.
// At the end (even in case of errors), the network traffic is brought back to normal.
// This function executes commands on a node so it will work only for some
// environments.
func performTemporaryNetworkFailure(c *client.Client, ns, rcName string, replicas int, podNameToDisappear string, node *api.Node) {
	Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == api.NodeExternalIP {
			host = a.Address + ":22"
			break
		}
	}
	if host == "" {
		Failf("Couldn't get the external IP of host %s with addresses %v", node.Name, node.Status.Addresses)
	}

	By(fmt.Sprintf("block network traffic from node %s to the master", node.Name))
	master := ""
	switch testContext.Provider {
	case "gce":
		// TODO(#10085): The use of MasterName will cause iptables to do a DNS
		// lookup to resolve the name to an IP address, which will slow down the
		// test and cause it to fail if DNS is absent or broken. Use the
		// internal IP address instead (i.e. NOT the one in testContext.Host).
		master = testContext.CloudConfig.MasterName
	case "gke":
		master = strings.TrimPrefix(testContext.Host, "https://")
	case "aws":
		// TODO(justinsb): Avoid hardcoding this.
		master = "172.20.0.9"
	default:
		Failf("This test is not supported for provider %s and should be disabled", testContext.Provider)
	}
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump DROP", master)
	defer func() {
		// This code will execute even if setting the iptables rule failed.
		// It is on purpose because we may have an error even if the new rule
		// had been inserted. (yes, we could look at the error code and ssh error
		// separately, but I prefer to stay on the safe side).

		By(fmt.Sprintf("Unblock network traffic from node %s to the master", node.Name))
		undropCmd := fmt.Sprintf("sudo iptables --delete %s", iptablesRule)
		// Undrop command may fail if the rule has never been created.
		// In such case we just lose 30 seconds, but the cluster is healthy.
		// But if the rule had been created and removing it failed, the node is broken and
		// not coming back. Subsequent tests will run or fewer nodes (some of the tests
		// may fail). Manual intervention is required in such case (recreating the
		// cluster solves the problem too).
		err := wait.Poll(time.Millisecond*100, time.Second*30, func() (bool, error) {
			_, _, code, err := SSHVerbose(undropCmd, host, testContext.Provider)
			if code == 0 && err == nil {
				return true, nil
			} else {
				Logf("Expected 0 exit code and nil error when running '%s' on %s, got %d and %v",
					undropCmd, node.Name, code, err)
				return false, nil
			}
		})
		if err != nil {
			Failf("Failed to remove the iptable DROP rule. Manual intervention is "+
				"required on node %s: remove rule %s, if exists", node.Name, iptablesRule)
		}
	}()

	Logf("Waiting %v to ensure node %s is ready before beginning test...", resizeNodeReadyTimeout, node.Name)
	if !waitForNodeToBe(c, node.Name, true, resizeNodeReadyTimeout) {
		Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
	}

	// The command will block all outgoing network traffic from the node to the master
	// When multi-master is implemented, this test will have to be improved to block
	// network traffic to all masters.
	// We could also block network traffic from the master(s) to this node,
	// but blocking it one way is sufficient for this test.
	dropCmd := fmt.Sprintf("sudo iptables --insert %s", iptablesRule)
	if _, _, code, err := SSHVerbose(dropCmd, host, testContext.Provider); code != 0 || err != nil {
		Failf("Expected 0 exit code and nil error when running %s on %s, got %d and %v",
			dropCmd, node.Name, code, err)
	}

	Logf("Waiting %v for node %s to be not ready after simulated network failure", resizeNodeNotReadyTimeout, node.Name)
	if !waitForNodeToBe(c, node.Name, false, resizeNodeNotReadyTimeout) {
		Failf("Node %s did not become not-ready within %v", node.Name, resizeNodeNotReadyTimeout)
	}

	Logf("Waiting for pod %s to be removed", podNameToDisappear)
	err := waitForRCPodToDisappear(c, ns, rcName, podNameToDisappear)
	Expect(err).NotTo(HaveOccurred())

	By("verifying whether the pod from the unreachable node is recreated")
	err = verifyPods(c, ns, rcName, true, replicas)
	Expect(err).NotTo(HaveOccurred())

	// network traffic is unblocked in a deferred function
}

var _ = Describe("Nodes", func() {
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
		By("checking whether all nodes are healthy")
		if err := allNodesReady(c, time.Minute); err != nil {
			Failf("Not all nodes are ready: %v", err)
		}
		By(fmt.Sprintf("destroying namespace for this suite %s", ns))
		if err := deleteNS(c, ns); err != nil {
			Failf("Couldn't delete namespace '%s', %v", ns, err)
		}
		if err := deleteTestingNS(c); err != nil {
			Failf("Couldn't delete testing namespaces '%s', %v", ns, err)
		}
	})

	Describe("Resize", func() {
		var skipped bool

		BeforeEach(func() {
			skipped = true
			SkipUnlessProviderIs("gce", "gke", "aws")
			SkipUnlessNodeCountIsAtLeast(2)
			skipped = false
		})

		AfterEach(func() {
			if skipped {
				return
			}

			By("restoring the original node instance group size")
			if err := resizeGroup(testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := waitForGroupSize(testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := waitForClusterSize(c, testContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				Failf("Couldn't restore the original cluster size: %v", err)
			}
		})

		It("should be able to delete nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			replicas := testContext.CloudConfig.NumNodes
			newRCByName(c, ns, name, replicas)
			err := verifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("decreasing cluster size to %d", replicas-1))
			err = resizeGroup(replicas - 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForGroupSize(replicas - 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForClusterSize(c, replicas-1, 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By("verifying whether the pods from the removed node are recreated")
			err = verifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())
		})

		// TODO: Bug here - testName is not correct
		It("should be able to add nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-add-node"
			newSVCByName(c, ns, name)
			replicas := testContext.CloudConfig.NumNodes
			newRCByName(c, ns, name, replicas)
			err := verifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing cluster size to %d", replicas+1))
			err = resizeGroup(replicas + 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForGroupSize(replicas + 1)
			Expect(err).NotTo(HaveOccurred())
			err = waitForClusterSize(c, replicas+1, 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", replicas+1))
			err = resizeRC(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = verifyPods(c, ns, name, true, replicas+1)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("Network", func() {
		Context("when a minion node becomes unreachable", func() {
			BeforeEach(func() {
				SkipUnlessProviderIs("gce", "gke", "aws")
				SkipUnlessNodeCountIsAtLeast(2)
			})

			// TODO marekbiskup 2015-06-19 #10085
			// This test has nothing to do with resizing nodes so it should be moved elsewhere.
			// Two things are tested here:
			// 1. pods from a uncontactable nodes are rescheduled
			// 2. when a node joins the cluster, it can host new pods.
			// Factor out the cases into two separate tests.
			It("[replication controller] recreates pods scheduled on the unreachable minion node "+
				"AND allows scheduling of pods on a minion after it rejoins the cluster", func() {

				// Create a replication controller for a service that serves its hostname.
				// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
				name := "my-hostname-net"
				newSVCByName(c, ns, name)
				replicas := testContext.CloudConfig.NumNodes
				newRCByName(c, ns, name, replicas)
				err := verifyPods(c, ns, name, true, replicas)
				Expect(err).NotTo(HaveOccurred(), "Each pod should start running and responding")

				By("choose a node with at least one pod - we will block some network traffic on this node")
				label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
				pods, err := c.Pods(ns).List(label, fields.Everything()) // list pods after all have been scheduled
				Expect(err).NotTo(HaveOccurred())
				nodeName := pods.Items[0].Spec.NodeName

				node, err := c.Nodes().Get(nodeName)
				Expect(err).NotTo(HaveOccurred())

				By(fmt.Sprintf("block network traffic from node %s", node.Name))
				performTemporaryNetworkFailure(c, ns, name, replicas, pods.Items[0].Name, node)
				Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
				if !waitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
					Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
				}

				// sleep a bit, to allow Watch in NodeController to catch up.
				time.Sleep(5 * time.Second)

				By("verify whether new pods can be created on the re-attached node")
				// increasing the RC size is not a valid way to test this
				// since we have no guarantees the pod will be scheduled on our node.
				additionalPod := "additionalpod"
				err = newPodOnNode(c, ns, additionalPod, node.Name)
				Expect(err).NotTo(HaveOccurred())
				err = verifyPods(c, ns, additionalPod, true, 1)
				Expect(err).NotTo(HaveOccurred())

				// verify that it is really on the requested node
				{
					pod, err := c.Pods(ns).Get(additionalPod)
					Expect(err).NotTo(HaveOccurred())
					if pod.Spec.NodeName != node.Name {
						Logf("Pod %s found on invalid node: %s instead of %s", pod.Name, pod.Spec.NodeName, node.Name)
					}
				}
			})
		})
	})
})
