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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/client/cache"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	controllerframework "k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	serveHostnameImage        = "gcr.io/google_containers/serve_hostname:1.1"
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
	nodeReadinessTimeout      = 3 * time.Minute
	podNotReadyTimeout        = 1 * time.Minute
	podReadyTimeout           = 2 * time.Minute
	testPort                  = 9376
)

func resizeGroup(size int) error {
	if testContext.ReportDir != "" {
		CoreDump(testContext.ReportDir)
		defer CoreDump(testContext.ReportDir)
	}
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
	} else if testContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		return awscloud.ResizeInstanceGroup(client, testContext.CloudConfig.NodeInstanceGroup, size)
	} else {
		return fmt.Errorf("Provider does not support InstanceGroups")
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
	} else if testContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		instanceGroup, err := awscloud.DescribeInstanceGroup(client, testContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", testContext.CloudConfig.NodeInstanceGroup)
		}
		return instanceGroup.CurrentSize()
	} else {
		return -1, fmt.Errorf("provider does not support InstanceGroups")
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

func svcByName(name string, port int) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Selector: map[string]string{
				"name": name,
			},
			Ports: []api.ServicePort{{
				Port:       port,
				TargetPort: intstr.FromInt(port),
			}},
		},
	}
}

func newSVCByName(c *client.Client, ns, name string) error {
	_, err := c.Services(ns).Create(svcByName(name, testPort))
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

func rcByNamePort(name string, replicas int, image string, port int, protocol api.Protocol, labels map[string]string) *api.ReplicationController {
	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
		Ports: []api.ContainerPort{{ContainerPort: port, Protocol: protocol}},
	})
}

func rcByNameContainer(name string, replicas int, image string, labels map[string]string, c api.Container) *api.ReplicationController {
	// Add "name": name to the labels, overwriting if it exists.
	labels["name"] = name
	gracePeriod := int64(0)
	return &api.ReplicationController{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
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
					Containers:                    []api.Container{c},
					TerminationGracePeriodSeconds: &gracePeriod,
				},
			},
		},
	}
}

// newRCByName creates a replication controller with a selector by name of name.
func newRCByName(c *client.Client, ns, name string, replicas int) (*api.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.ReplicationControllers(ns).Create(rcByNamePort(
		name, replicas, serveHostnameImage, 9376, api.ProtocolTCP, map[string]string{}))
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

func getMaster(c *client.Client) string {
	master := ""
	switch testContext.Provider {
	case "gce":
		eps, err := c.Endpoints(api.NamespaceDefault).Get("kubernetes")
		if err != nil {
			Failf("Fail to get kubernetes endpoinds: %v", err)
		}
		if len(eps.Subsets) != 1 || len(eps.Subsets[0].Addresses) != 1 {
			Failf("There are more than 1 endpoints for kubernetes service: %+v", eps)
		}
		master = eps.Subsets[0].Addresses[0].IP
	case "gke":
		master = strings.TrimPrefix(testContext.Host, "https://")
	case "aws":
		// TODO(justinsb): Avoid hardcoding this.
		master = "172.20.0.9"
	default:
		Failf("This test is not supported for provider %s and should be disabled", testContext.Provider)
	}
	return master
}

// Return node external IP concatenated with port 22 for ssh
// e.g. 1.2.3.4:22
func getNodeExternalIP(node *api.Node) string {
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
	return host
}

// Blocks outgoing network traffic on 'node'. Then verifies that 'podNameToDisappear',
// that belongs to replication controller 'rcName', really disappeared.
// Finally, it checks that the replication controller recreates the
// pods on another node and that now the number of replicas is equal 'replicas'.
// At the end (even in case of errors), the network traffic is brought back to normal.
// This function executes commands on a node so it will work only for some
// environments.
func performTemporaryNetworkFailure(c *client.Client, ns, rcName string, replicas int, podNameToDisappear string, node *api.Node) {
	host := getNodeExternalIP(node)
	master := getMaster(c)
	By(fmt.Sprintf("block network traffic from node %s to the master", node.Name))
	defer func() {
		// This code will execute even if setting the iptables rule failed.
		// It is on purpose because we may have an error even if the new rule
		// had been inserted. (yes, we could look at the error code and ssh error
		// separately, but I prefer to stay on the safe side).
		By(fmt.Sprintf("Unblock network traffic from node %s to the master", node.Name))
		unblockNetwork(host, master)
	}()

	Logf("Waiting %v to ensure node %s is ready before beginning test...", resizeNodeReadyTimeout, node.Name)
	if !waitForNodeToBe(c, node.Name, api.NodeReady, true, resizeNodeReadyTimeout) {
		Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
	}
	blockNetwork(host, master)

	Logf("Waiting %v for node %s to be not ready after simulated network failure", resizeNodeNotReadyTimeout, node.Name)
	if !waitForNodeToBe(c, node.Name, api.NodeReady, false, resizeNodeNotReadyTimeout) {
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

func expectNodeReadiness(isReady bool, newNode chan *api.Node) {
	timeout := false
	expected := false
	timer := time.After(nodeReadinessTimeout)
	for !expected && !timeout {
		select {
		case n := <-newNode:
			if isNodeConditionSetAsExpected(n, api.NodeReady, isReady) {
				expected = true
			} else {
				Logf("Observed node ready status is NOT %v as expected", isReady)
			}
		case <-timer:
			timeout = true
		}
	}
	if !expected {
		Failf("Failed to observe node ready status change to %v", isReady)
	}
}

var _ = Describe("Nodes [Disruptive]", func() {
	framework := NewDefaultFramework("resize-nodes")
	var systemPodsNo int
	var c *client.Client
	var ns string
	BeforeEach(func() {
		c = framework.Client
		ns = framework.Namespace.Name
		systemPods, err := c.Pods(api.NamespaceSystem).List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = len(systemPods.Items)
	})

	// Slow issue #13323 (8 min)
	Describe("Resize [Slow]", func() {
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
			// In GKE, our current tunneling setup has the potential to hold on to a broken tunnel (from a
			// rebooted/deleted node) for up to 5 minutes before all tunnels are dropped and recreated.
			// Most tests make use of some proxy feature to verify functionality. So, if a reboot test runs
			// right before a test that tries to get logs, for example, we may get unlucky and try to use a
			// closed tunnel to a node that was recently rebooted. There's no good way to poll for proxies
			// being closed, so we sleep.
			//
			// TODO(cjcullen) reduce this sleep (#19314)
			if providerIs("gke") {
				By("waiting 5 minutes for all dead tunnels to be dropped")
				time.Sleep(5 * time.Minute)
			}
			if err := waitForGroupSize(testContext.CloudConfig.NumNodes); err != nil {
				Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := waitForClusterSize(c, testContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			By("waiting for system pods to successfully restart")

			err := waitForPodsRunningReady(api.NamespaceSystem, systemPodsNo, podReadyBeforeTimeout)
			Expect(err).NotTo(HaveOccurred())
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
		Context("when a node becomes unreachable", func() {
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
			It("[replication controller] recreates pods scheduled on the unreachable node "+
				"AND allows scheduling of pods on a node after it rejoins the cluster", func() {

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
				options := api.ListOptions{LabelSelector: label}
				pods, err := c.Pods(ns).List(options) // list pods after all have been scheduled
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

			// What happens in this test:
			// 	Network traffic from a node to master is cut off to simulate network partition
			// Expect to observe:
			// 1. Node is marked NotReady after timeout by nodecontroller (40seconds)
			// 2. All pods on node are marked NotReady shortly after #1
			// 3. Node and pods return to Ready after connectivivty recovers
			It("All pods on the unreachable node should be marked as NotReady upon the node turn NotReady "+
				"AND all pods should be mark back to Ready when the node get back to Ready before pod eviction timeout", func() {
				By("choose a node - we will block all network traffic on this node")
				var podOpts api.ListOptions
				nodeOpts := api.ListOptions{}
				nodes, err := c.Nodes().List(nodeOpts)
				Expect(err).NotTo(HaveOccurred())
				filterNodes(nodes, func(node api.Node) bool {
					if !isNodeConditionSetAsExpected(&node, api.NodeReady, true) {
						return false
					}
					podOpts = api.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name)}
					pods, err := c.Pods(api.NamespaceAll).List(podOpts)
					if err != nil || len(pods.Items) <= 0 {
						return false
					}
					return true
				})
				if len(nodes.Items) <= 0 {
					Failf("No eligible node were found: %d", len(nodes.Items))
				}
				node := nodes.Items[0]
				podOpts = api.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name)}
				if err = waitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, podRunningReady); err != nil {
					Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
				}

				By("Set up watch on node status")
				nodeSelector := fields.OneTermEqualSelector("metadata.name", node.Name)
				stopCh := make(chan struct{})
				newNode := make(chan *api.Node)
				var controller *controllerframework.Controller
				_, controller = controllerframework.NewInformer(
					&cache.ListWatch{
						ListFunc: func(options api.ListOptions) (runtime.Object, error) {
							options.FieldSelector = nodeSelector
							return framework.Client.Nodes().List(options)
						},
						WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector
							return framework.Client.Nodes().Watch(options)
						},
					},
					&api.Node{},
					0,
					controllerframework.ResourceEventHandlerFuncs{
						UpdateFunc: func(oldObj, newObj interface{}) {
							n, ok := newObj.(*api.Node)
							Expect(ok).To(Equal(true))
							newNode <- n

						},
					},
				)

				defer func() {
					// Will not explicitly close newNode channel here due to
					// race condition where stopCh and newNode are closed but informer onUpdate still executes.
					close(stopCh)
				}()
				go controller.Run(stopCh)

				By(fmt.Sprintf("Block traffic from node %s to the master", node.Name))
				host := getNodeExternalIP(&node)
				master := getMaster(c)
				defer func() {
					By(fmt.Sprintf("Unblock traffic from node %s to the master", node.Name))
					unblockNetwork(host, master)

					if CurrentGinkgoTestDescription().Failed {
						return
					}

					By("Expect to observe node and pod status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
					if err = waitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, podRunningReady); err != nil {
						Failf("Pods on node %s did not become ready and running within %v: %v", node.Name, podReadyTimeout, err)
					}
				}()

				blockNetwork(host, master)

				By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				if err = waitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, podNotReady); err != nil {
					Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}
			})
		})
	})
})
