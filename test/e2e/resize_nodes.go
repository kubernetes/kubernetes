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
	"k8s.io/kubernetes/test/e2e/framework"

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
	serveHostnameImage        = "gcr.io/google_containers/serve_hostname:v1.4"
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
	nodeReadinessTimeout      = 3 * time.Minute
	podNotReadyTimeout        = 1 * time.Minute
	podReadyTimeout           = 2 * time.Minute
	testPort                  = 9376
)

func ResizeGroup(group string, size int32) error {
	if framework.TestContext.ReportDir != "" {
		framework.CoreDump(framework.TestContext.ReportDir)
		defer framework.CoreDump(framework.TestContext.ReportDir)
	}
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "resize",
			group, fmt.Sprintf("--size=%v", size),
			"--project="+framework.TestContext.CloudConfig.ProjectID, "--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			framework.Logf("Failed to resize node instance group: %v", string(output))
		}
		return err
	} else if framework.TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		return awscloud.ResizeInstanceGroup(client, group, int(size))
	} else {
		return fmt.Errorf("Provider does not support InstanceGroups")
	}
}

func GetGroupNodes(group string) ([]string, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return nil, err
		}
		re := regexp.MustCompile(".*RUNNING")
		lines := re.FindAllString(string(output), -1)
		for i, line := range lines {
			lines[i] = line[:strings.Index(line, " ")]
		}
		return lines, nil
	} else {
		return nil, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func GroupSize(group string) (int, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return -1, err
		}
		re := regexp.MustCompile("RUNNING")
		return len(re.FindAllString(string(output), -1)), nil
	} else if framework.TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		instanceGroup, err := awscloud.DescribeInstanceGroup(client, group)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", group)
		}
		return instanceGroup.CurrentSize()
	} else {
		return -1, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func WaitForGroupSize(group string, size int32) error {
	timeout := 10 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		currentSize, err := GroupSize(group)
		if err != nil {
			framework.Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != int(size) {
			framework.Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		framework.Logf("Node instance group has reached the desired size %d", size)
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
				Port:       int32(port),
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
		framework.Logf("Created pod %s on node %s", pod.ObjectMeta.Name, nodeName)
	} else {
		framework.Logf("Failed to create pod %s on node %s: %v", podName, nodeName, err)
	}
	return err
}

func rcByName(name string, replicas int32, image string, labels map[string]string) *api.ReplicationController {
	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
	})
}

func rcByNamePort(name string, replicas int32, image string, port int, protocol api.Protocol, labels map[string]string) *api.ReplicationController {
	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
		Ports: []api.ContainerPort{{ContainerPort: int32(port), Protocol: protocol}},
	})
}

func rcByNameContainer(name string, replicas int32, image string, labels map[string]string, c api.Container) *api.ReplicationController {
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
func newRCByName(c *client.Client, ns, name string, replicas int32) (*api.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.ReplicationControllers(ns).Create(rcByNamePort(
		name, replicas, serveHostnameImage, 9376, api.ProtocolTCP, map[string]string{}))
}

func resizeRC(c *client.Client, ns, name string, replicas int32) error {
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
	switch framework.TestContext.Provider {
	case "gce":
		eps, err := c.Endpoints(api.NamespaceDefault).Get("kubernetes")
		if err != nil {
			framework.Failf("Fail to get kubernetes endpoinds: %v", err)
		}
		if len(eps.Subsets) != 1 || len(eps.Subsets[0].Addresses) != 1 {
			framework.Failf("There are more than 1 endpoints for kubernetes service: %+v", eps)
		}
		master = eps.Subsets[0].Addresses[0].IP
	case "gke":
		master = strings.TrimPrefix(framework.TestContext.Host, "https://")
	case "aws":
		// TODO(justinsb): Avoid hardcoding this.
		master = "172.20.0.9"
	default:
		framework.Failf("This test is not supported for provider %s and should be disabled", framework.TestContext.Provider)
	}
	return master
}

// Return node external IP concatenated with port 22 for ssh
// e.g. 1.2.3.4:22
func getNodeExternalIP(node *api.Node) string {
	framework.Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == api.NodeExternalIP {
			host = a.Address + ":22"
			break
		}
	}
	if host == "" {
		framework.Failf("Couldn't get the external IP of host %s with addresses %v", node.Name, node.Status.Addresses)
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
func performTemporaryNetworkFailure(c *client.Client, ns, rcName string, replicas int32, podNameToDisappear string, node *api.Node) {
	host := getNodeExternalIP(node)
	master := getMaster(c)
	By(fmt.Sprintf("block network traffic from node %s to the master", node.Name))
	defer func() {
		// This code will execute even if setting the iptables rule failed.
		// It is on purpose because we may have an error even if the new rule
		// had been inserted. (yes, we could look at the error code and ssh error
		// separately, but I prefer to stay on the safe side).
		By(fmt.Sprintf("Unblock network traffic from node %s to the master", node.Name))
		framework.UnblockNetwork(host, master)
	}()

	framework.Logf("Waiting %v to ensure node %s is ready before beginning test...", resizeNodeReadyTimeout, node.Name)
	if !framework.WaitForNodeToBe(c, node.Name, api.NodeReady, true, resizeNodeReadyTimeout) {
		framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
	}
	framework.BlockNetwork(host, master)

	framework.Logf("Waiting %v for node %s to be not ready after simulated network failure", resizeNodeNotReadyTimeout, node.Name)
	if !framework.WaitForNodeToBe(c, node.Name, api.NodeReady, false, resizeNodeNotReadyTimeout) {
		framework.Failf("Node %s did not become not-ready within %v", node.Name, resizeNodeNotReadyTimeout)
	}

	framework.Logf("Waiting for pod %s to be removed", podNameToDisappear)
	err := framework.WaitForRCPodToDisappear(c, ns, rcName, podNameToDisappear)
	Expect(err).NotTo(HaveOccurred())

	By("verifying whether the pod from the unreachable node is recreated")
	err = framework.VerifyPods(c, ns, rcName, true, replicas)
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
			if framework.IsNodeConditionSetAsExpected(n, api.NodeReady, isReady) {
				expected = true
			} else {
				framework.Logf("Observed node ready status is NOT %v as expected", isReady)
			}
		case <-timer:
			timeout = true
		}
	}
	if !expected {
		framework.Failf("Failed to observe node ready status change to %v", isReady)
	}
}

var _ = framework.KubeDescribe("Nodes [Disruptive]", func() {
	f := framework.NewDefaultFramework("resize-nodes")
	var systemPodsNo int32
	var c *client.Client
	var ns string
	ignoreLabels := framework.ImagePullerLabels
	var group string

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
		systemPods, err := framework.GetPodsInNamespace(c, ns, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = int32(len(systemPods))
		if strings.Index(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") >= 0 {
			framework.Failf("Test dose not support cluster setup with more than one MIG: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
		} else {
			group = framework.TestContext.CloudConfig.NodeInstanceGroup
		}
	})

	// Slow issue #13323 (8 min)
	framework.KubeDescribe("Resize [Slow]", func() {
		var skipped bool

		BeforeEach(func() {
			skipped = true
			framework.SkipUnlessProviderIs("gce", "gke", "aws")
			framework.SkipUnlessNodeCountIsAtLeast(2)
			skipped = false
		})

		AfterEach(func() {
			if skipped {
				return
			}

			By("restoring the original node instance group size")
			if err := ResizeGroup(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}
			// In GKE, our current tunneling setup has the potential to hold on to a broken tunnel (from a
			// rebooted/deleted node) for up to 5 minutes before all tunnels are dropped and recreated.
			// Most tests make use of some proxy feature to verify functionality. So, if a reboot test runs
			// right before a test that tries to get logs, for example, we may get unlucky and try to use a
			// closed tunnel to a node that was recently rebooted. There's no good way to framework.Poll for proxies
			// being closed, so we sleep.
			//
			// TODO(cjcullen) reduce this sleep (#19314)
			if framework.ProviderIs("gke") {
				By("waiting 5 minutes for all dead tunnels to be dropped")
				time.Sleep(5 * time.Minute)
			}
			if err := WaitForGroupSize(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := framework.WaitForClusterSize(c, framework.TestContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				framework.Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			By("waiting for system pods to successfully restart")
			err := framework.WaitForPodsRunningReady(c, api.NamespaceSystem, systemPodsNo, framework.PodReadyBeforeTimeout, ignoreLabels)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for image prepulling pods to complete")
			framework.WaitForPodsSuccess(c, api.NamespaceSystem, framework.ImagePullerLabels, imagePrePullingTimeout)
		})

		It("should be able to delete nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			newRCByName(c, ns, name, replicas)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("decreasing cluster size to %d", replicas-1))
			err = ResizeGroup(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = WaitForGroupSize(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForClusterSize(c, int(replicas-1), 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By("verifying whether the pods from the removed node are recreated")
			err = framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())
		})

		// TODO: Bug here - testName is not correct
		It("should be able to add nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-add-node"
			newSVCByName(c, ns, name)
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			newRCByName(c, ns, name, replicas)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing cluster size to %d", replicas+1))
			err = ResizeGroup(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = WaitForGroupSize(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForClusterSize(c, int(replicas+1), 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", replicas+1))
			err = resizeRC(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.VerifyPods(c, ns, name, true, replicas+1)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	framework.KubeDescribe("Network", func() {
		Context("when a node becomes unreachable", func() {
			BeforeEach(func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				framework.SkipUnlessNodeCountIsAtLeast(2)
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
				replicas := int32(framework.TestContext.CloudConfig.NumNodes)
				newRCByName(c, ns, name, replicas)
				err := framework.VerifyPods(c, ns, name, true, replicas)
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
				framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
				if !framework.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
					framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
				}

				// sleep a bit, to allow Watch in NodeController to catch up.
				time.Sleep(5 * time.Second)

				By("verify whether new pods can be created on the re-attached node")
				// increasing the RC size is not a valid way to test this
				// since we have no guarantees the pod will be scheduled on our node.
				additionalPod := "additionalpod"
				err = newPodOnNode(c, ns, additionalPod, node.Name)
				Expect(err).NotTo(HaveOccurred())
				err = framework.VerifyPods(c, ns, additionalPod, true, 1)
				Expect(err).NotTo(HaveOccurred())

				// verify that it is really on the requested node
				{
					pod, err := c.Pods(ns).Get(additionalPod)
					Expect(err).NotTo(HaveOccurred())
					if pod.Spec.NodeName != node.Name {
						framework.Logf("Pod %s found on invalid node: %s instead of %s", pod.Name, pod.Spec.NodeName, node.Name)
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
				framework.FilterNodes(nodes, func(node api.Node) bool {
					if !framework.IsNodeConditionSetAsExpected(&node, api.NodeReady, true) {
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
					framework.Failf("No eligible node were found: %d", len(nodes.Items))
				}
				node := nodes.Items[0]
				podOpts = api.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name)}
				if err = framework.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, framework.PodRunningReady); err != nil {
					framework.Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
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
							return f.Client.Nodes().List(options)
						},
						WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector
							return f.Client.Nodes().Watch(options)
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
					framework.UnblockNetwork(host, master)

					if CurrentGinkgoTestDescription().Failed {
						return
					}

					By("Expect to observe node and pod status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
					if err = framework.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, framework.PodRunningReady); err != nil {
						framework.Failf("Pods on node %s did not become ready and running within %v: %v", node.Name, podReadyTimeout, err)
					}
				}()

				framework.BlockNetwork(host, master)

				By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				if err = framework.WaitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, framework.PodNotReady); err != nil {
					framework.Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}
			})
		})
	})
})
