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
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"

	"k8s.io/client-go/tools/cache"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	nodepkg "k8s.io/kubernetes/pkg/controller/node"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	timeout                = 60 * time.Second
	podReadyTimeout        = 2 * time.Minute
	podNotReadyTimeout     = 1 * time.Minute
	nodeReadinessTimeout   = 3 * time.Minute
	resizeNodeReadyTimeout = 2 * time.Minute
)

func expectNodeReadiness(isReady bool, newNode chan *v1.Node) {
	timeout := false
	expected := false
	timer := time.After(nodeReadinessTimeout)
	for !expected && !timeout {
		select {
		case n := <-newNode:
			if framework.IsNodeConditionSetAsExpected(n, v1.NodeReady, isReady) {
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

func podOnNode(podName, nodeName string, image string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"name": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  podName,
					Image: image,
					Ports: []v1.ContainerPort{{ContainerPort: 9376}},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func newPodOnNode(c clientset.Interface, namespace, podName, nodeName string) error {
	pod, err := c.Core().Pods(namespace).Create(podOnNode(podName, nodeName, framework.ServeHostnameImage))
	if err == nil {
		framework.Logf("Created pod %s on node %s", pod.ObjectMeta.Name, nodeName)
	} else {
		framework.Logf("Failed to create pod %s on node %s: %v", podName, nodeName, err)
	}
	return err
}

var _ = framework.KubeDescribe("[sig-apps] Network Partition [Disruptive] [Slow]", func() {
	f := framework.NewDefaultFramework("network-partition")
	var systemPodsNo int32
	var c clientset.Interface
	var ns string
	ignoreLabels := framework.ImagePullerLabels
	var group string

	BeforeEach(func() {
		c = f.ClientSet
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

	framework.KubeDescribe("Pods", func() {
		Context("should return to running and ready state after network partition is healed", func() {
			BeforeEach(func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				framework.SkipUnlessNodeCountIsAtLeast(2)
			})

			// What happens in this test:
			//	Network traffic from a node to master is cut off to simulate network partition
			// Expect to observe:
			// 1. Node is marked NotReady after timeout by nodecontroller (40seconds)
			// 2. All pods on node are marked NotReady shortly after #1
			// 3. Node and pods return to Ready after connectivivty recovers
			It("All pods on the unreachable node should be marked as NotReady upon the node turn NotReady "+
				"AND all pods should be mark back to Ready when the node get back to Ready before pod eviction timeout", func() {
				By("choose a node - we will block all network traffic on this node")
				var podOpts metav1.ListOptions
				nodeOpts := metav1.ListOptions{}
				nodes, err := c.Core().Nodes().List(nodeOpts)
				Expect(err).NotTo(HaveOccurred())
				framework.FilterNodes(nodes, func(node v1.Node) bool {
					if !framework.IsNodeConditionSetAsExpected(&node, v1.NodeReady, true) {
						return false
					}
					podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
					pods, err := c.Core().Pods(metav1.NamespaceAll).List(podOpts)
					if err != nil || len(pods.Items) <= 0 {
						return false
					}
					return true
				})
				if len(nodes.Items) <= 0 {
					framework.Failf("No eligible node were found: %d", len(nodes.Items))
				}
				node := nodes.Items[0]
				podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
				if err = framework.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReady); err != nil {
					framework.Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
				}

				By("Set up watch on node status")
				nodeSelector := fields.OneTermEqualSelector("metadata.name", node.Name)
				stopCh := make(chan struct{})
				newNode := make(chan *v1.Node)
				var controller cache.Controller
				_, controller = cache.NewInformer(
					&cache.ListWatch{
						ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
							options.FieldSelector = nodeSelector.String()
							obj, err := f.ClientSet.Core().Nodes().List(options)
							return runtime.Object(obj), err
						},
						WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector.String()
							return f.ClientSet.Core().Nodes().Watch(options)
						},
					},
					&v1.Node{},
					0,
					cache.ResourceEventHandlerFuncs{
						UpdateFunc: func(oldObj, newObj interface{}) {
							n, ok := newObj.(*v1.Node)
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
				host := framework.GetNodeExternalIP(&node)
				master := framework.GetMasterAddress(c)
				defer func() {
					By(fmt.Sprintf("Unblock traffic from node %s to the master", node.Name))
					framework.UnblockNetwork(host, master)

					if CurrentGinkgoTestDescription().Failed {
						return
					}

					By("Expect to observe node and pod status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
					if err = framework.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReady); err != nil {
						framework.Failf("Pods on node %s did not become ready and running within %v: %v", node.Name, podReadyTimeout, err)
					}
				}()

				framework.BlockNetwork(host, master)

				By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				if err = framework.WaitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, testutils.PodNotReady); err != nil {
					framework.Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}
			})
		})
	})

	framework.KubeDescribe("[ReplicationController]", func() {
		It("should recreate pods scheduled on the unreachable node "+
			"AND allow scheduling of pods on a node after it rejoins the cluster", func() {

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-net"
			common.NewSVCByName(c, ns, name)
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			common.NewRCByName(c, ns, name, replicas, nil)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred(), "Each pod should start running and responding")

			By("choose a node with at least one pod - we will block some network traffic on this node")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.Core().Pods(ns).List(options) // list pods after all have been scheduled
			Expect(err).NotTo(HaveOccurred())
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.Core().Nodes().Get(nodeName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())

			// This creates a temporary network partition, verifies that 'podNameToDisappear',
			// that belongs to replication controller 'rcName', really disappeared (because its
			// grace period is set to 0).
			// Finally, it checks that the replication controller recreates the
			// pods on another node and that now the number of replicas is equal 'replicas'.
			By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			framework.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := framework.WaitForRCPodToDisappear(c, ns, name, pods.Items[0].Name)
				Expect(err).NotTo(HaveOccurred())

				By("verifying whether the pod from the unreachable node is recreated")
				err = framework.VerifyPods(c, ns, name, true, replicas)
				Expect(err).NotTo(HaveOccurred())
			})

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
				pod, err := c.Core().Pods(ns).Get(additionalPod, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())
				if pod.Spec.NodeName != node.Name {
					framework.Logf("Pod %s found on invalid node: %s instead of %s", pod.Name, pod.Spec.NodeName, node.Name)
				}
			}
		})

		It("should eagerly create replacement pod during network partition when termination grace is non-zero", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-net"
			gracePeriod := int64(30)

			common.NewSVCByName(c, ns, name)
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			common.NewRCByName(c, ns, name, replicas, &gracePeriod)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred(), "Each pod should start running and responding")

			By("choose a node with at least one pod - we will block some network traffic on this node")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.Core().Pods(ns).List(options) // list pods after all have been scheduled
			Expect(err).NotTo(HaveOccurred())
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.Core().Nodes().Get(nodeName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())

			// This creates a temporary network partition, verifies that 'podNameToDisappear',
			// that belongs to replication controller 'rcName', did not disappear (because its
			// grace period is set to 30).
			// Finally, it checks that the replication controller recreates the
			// pods on another node and that now the number of replicas is equal 'replicas + 1'.
			By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			framework.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := framework.WaitForRCPodToDisappear(c, ns, name, pods.Items[0].Name)
				Expect(err).To(Equal(wait.ErrWaitTimeout), "Pod was not deleted during network partition.")

				By(fmt.Sprintf("verifying that there are %v running pods during partition", replicas))
				_, err = framework.PodsCreated(c, ns, name, replicas)
				Expect(err).NotTo(HaveOccurred())
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !framework.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}
		})
	})

	framework.KubeDescribe("[StatefulSet]", func() {
		psName := "ss"
		labels := map[string]string{
			"foo": "bar",
		}
		headlessSvcName := "test"

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("creating service " + headlessSvcName + " in namespace " + f.Namespace.Name)
			headlessService := framework.CreateServiceSpec(headlessSvcName, "", true, labels)
			_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(headlessService)
			framework.ExpectNoError(err)
			c = f.ClientSet
			ns = f.Namespace.Name
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all stateful set in ns %v", ns)
			framework.DeleteAllStatefulSets(c, ns)
		})

		It("should come back up if node goes down [Slow] [Disruptive]", func() {
			petMounts := []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts := []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ps := framework.NewStatefulSet(psName, ns, headlessSvcName, 3, petMounts, podMounts, labels)
			_, err := c.AppsV1beta1().StatefulSets(ns).Create(ps)
			Expect(err).NotTo(HaveOccurred())

			pst := framework.NewStatefulSetTester(c)

			nn := framework.TestContext.CloudConfig.NumNodes
			nodeNames, err := framework.CheckNodesReady(f.ClientSet, framework.NodeReadyInitialTimeout, nn)
			framework.ExpectNoError(err)
			common.RestartNodes(f.ClientSet, nodeNames)

			By("waiting for pods to be running again")
			pst.WaitForRunningAndReady(*ps.Spec.Replicas, ps)
		})

		It("should not reschedule stateful pods if there is a network partition [Slow] [Disruptive]", func() {
			ps := framework.NewStatefulSet(psName, ns, headlessSvcName, 3, []v1.VolumeMount{}, []v1.VolumeMount{}, labels)
			_, err := c.AppsV1beta1().StatefulSets(ns).Create(ps)
			Expect(err).NotTo(HaveOccurred())

			pst := framework.NewStatefulSetTester(c)
			pst.WaitForRunningAndReady(*ps.Spec.Replicas, ps)

			pod := pst.GetPodList(ps).Items[0]
			node, err := c.Core().Nodes().Get(pod.Spec.NodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// Blocks outgoing network traffic on 'node'. Then verifies that 'podNameToDisappear',
			// that belongs to StatefulSet 'statefulSetName', **does not** disappear due to forced deletion from the apiserver.
			// The grace period on the stateful pods is set to a value > 0.
			framework.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Checking that the NodeController does not force delete stateful pods %v", pod.Name)
				err := framework.WaitTimeoutForPodNoLongerRunningInNamespace(c, pod.Name, ns, 10*time.Minute)
				Expect(err).To(Equal(wait.ErrWaitTimeout), "Pod was not deleted during network partition.")
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !framework.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}

			By("waiting for pods to be running again")
			pst.WaitForRunningAndReady(*ps.Spec.Replicas, ps)
		})
	})

	framework.KubeDescribe("[Job]", func() {
		It("should create new pods when node is partitioned", func() {
			parallelism := int32(2)
			completions := int32(4)

			job := framework.NewTestJob("notTerminate", "network-partition", v1.RestartPolicyNever,
				parallelism, completions)
			job, err := framework.CreateJob(f.ClientSet, f.Namespace.Name, job)
			Expect(err).NotTo(HaveOccurred())
			label := labels.SelectorFromSet(labels.Set(map[string]string{framework.JobSelectorKey: job.Name}))

			By(fmt.Sprintf("verifying that there are now %v running pods", parallelism))
			_, err = framework.PodsCreatedByLabel(c, ns, job.Name, parallelism, label)
			Expect(err).NotTo(HaveOccurred())

			By("choose a node with at least one pod - we will block some network traffic on this node")
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.Core().Pods(ns).List(options) // list pods after all have been scheduled
			Expect(err).NotTo(HaveOccurred())
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.Core().Nodes().Get(nodeName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())

			// This creates a temporary network partition, verifies that the job has 'parallelism' number of
			// running pods after the node-controller detects node unreachable.
			By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			framework.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := framework.WaitForPodToDisappear(c, ns, pods.Items[0].Name, label, 20*time.Second, 10*time.Minute)
				Expect(err).To(Equal(wait.ErrWaitTimeout), "Pod was not deleted during network partition.")

				By(fmt.Sprintf("verifying that there are now %v running pods", parallelism))
				_, err = framework.PodsCreatedByLabel(c, ns, job.Name, parallelism, label)
				Expect(err).NotTo(HaveOccurred())
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !framework.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}
		})
	})

	framework.KubeDescribe("Pods", func() {
		Context("should be evicted from unready Node", func() {
			BeforeEach(func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				framework.SkipUnlessNodeCountIsAtLeast(2)
			})

			// What happens in this test:
			//	Network traffic from a node to master is cut off to simulate network partition
			// Expect to observe:
			// 1. Node is marked NotReady after timeout by nodecontroller (40seconds)
			// 2. All pods on node are marked NotReady shortly after #1
			// 3. After enough time passess all Pods are evicted from the given Node
			It("[Feature:TaintEviction] All pods on the unreachable node should be marked as NotReady upon the node turn NotReady "+
				"AND all pods should be evicted after eviction timeout passes", func() {
				By("choose a node - we will block all network traffic on this node")
				var podOpts metav1.ListOptions
				nodes := framework.GetReadySchedulableNodesOrDie(c)
				framework.FilterNodes(nodes, func(node v1.Node) bool {
					if !framework.IsNodeConditionSetAsExpected(&node, v1.NodeReady, true) {
						return false
					}
					podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
					pods, err := c.Core().Pods(metav1.NamespaceAll).List(podOpts)
					if err != nil || len(pods.Items) <= 0 {
						return false
					}
					return true
				})
				if len(nodes.Items) <= 0 {
					framework.Failf("No eligible node were found: %d", len(nodes.Items))
				}
				node := nodes.Items[0]
				podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
				if err := framework.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReadyOrSucceeded); err != nil {
					framework.Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
				}
				pods, err := c.Core().Pods(metav1.NamespaceAll).List(podOpts)
				framework.ExpectNoError(err)
				podTolerationTimes := map[string]time.Duration{}
				// This test doesn't add tolerations by itself, but because they may be present in the cluster
				// it needs to account for that.
				for _, pod := range pods.Items {
					namespacedName := fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)
					tolerations := pod.Spec.Tolerations
					framework.ExpectNoError(err)
					for _, toleration := range tolerations {
						if toleration.ToleratesTaint(nodepkg.UnreachableTaintTemplate) {
							if toleration.TolerationSeconds != nil {
								podTolerationTimes[namespacedName] = time.Duration(*toleration.TolerationSeconds) * time.Second
								break
							} else {
								podTolerationTimes[namespacedName] = -1
							}
						}
					}
					if _, ok := podTolerationTimes[namespacedName]; !ok {
						podTolerationTimes[namespacedName] = 0
					}
				}
				neverEvictedPods := []string{}
				maxTolerationTime := time.Duration(0)
				for podName, tolerationTime := range podTolerationTimes {
					if tolerationTime < 0 {
						neverEvictedPods = append(neverEvictedPods, podName)
					} else {
						if tolerationTime > maxTolerationTime {
							maxTolerationTime = tolerationTime
						}
					}
				}
				framework.Logf(
					"Only %v should be running after partition. Maximum TolerationSeconds among other Pods is %v",
					neverEvictedPods,
					maxTolerationTime,
				)

				By("Set up watch on node status")
				nodeSelector := fields.OneTermEqualSelector("metadata.name", node.Name)
				stopCh := make(chan struct{})
				newNode := make(chan *v1.Node)
				var controller cache.Controller
				_, controller = cache.NewInformer(
					&cache.ListWatch{
						ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
							options.FieldSelector = nodeSelector.String()
							obj, err := f.ClientSet.Core().Nodes().List(options)
							return runtime.Object(obj), err
						},
						WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector.String()
							return f.ClientSet.Core().Nodes().Watch(options)
						},
					},
					&v1.Node{},
					0,
					cache.ResourceEventHandlerFuncs{
						UpdateFunc: func(oldObj, newObj interface{}) {
							n, ok := newObj.(*v1.Node)
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
				host := framework.GetNodeExternalIP(&node)
				master := framework.GetMasterAddress(c)
				defer func() {
					By(fmt.Sprintf("Unblock traffic from node %s to the master", node.Name))
					framework.UnblockNetwork(host, master)

					if CurrentGinkgoTestDescription().Failed {
						return
					}

					By("Expect to observe node status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
				}()

				framework.BlockNetwork(host, master)

				By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				framework.ExpectNoError(wait.Poll(1*time.Second, timeout, func() (bool, error) {
					return framework.NodeHasTaint(c, node.Name, nodepkg.UnreachableTaintTemplate)
				}))
				if err = framework.WaitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, testutils.PodNotReady); err != nil {
					framework.Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}

				sleepTime := maxTolerationTime + 20*time.Second
				By(fmt.Sprintf("Sleeping for %v and checking if all Pods were evicted", sleepTime))
				time.Sleep(sleepTime)
				pods, err = c.Core().Pods(v1.NamespaceAll).List(podOpts)
				framework.ExpectNoError(err)
				seenRunning := []string{}
				for _, pod := range pods.Items {
					namespacedName := fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)
					shouldBeTerminating := true
					for _, neverEvictedPod := range neverEvictedPods {
						if neverEvictedPod == namespacedName {
							shouldBeTerminating = false
						}
					}
					if pod.DeletionTimestamp == nil {
						seenRunning = append(seenRunning, namespacedName)
						if shouldBeTerminating {
							framework.Failf("Pod %v should have been deleted but was seen running", namespacedName)
						}
					}
				}

				for _, neverEvictedPod := range neverEvictedPods {
					running := false
					for _, runningPod := range seenRunning {
						if runningPod == neverEvictedPod {
							running = true
							break
						}
					}
					if !running {
						framework.Failf("Pod %v was evicted even though it shouldn't", neverEvictedPod)
					}
				}
			})
		})
	})
})
