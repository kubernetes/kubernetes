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

package apps

import (
	"context"
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

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	nodepkg "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ejob "k8s.io/kubernetes/test/e2e/framework/job"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo"
)

const (
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
			if e2enode.IsConditionSetAsExpected(n, v1.NodeReady, isReady) {
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
					Args:  []string{"serve-hostname"},
					Ports: []v1.ContainerPort{{ContainerPort: 9376}},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func newPodOnNode(c clientset.Interface, namespace, podName, nodeName string) error {
	pod, err := c.CoreV1().Pods(namespace).Create(context.TODO(), podOnNode(podName, nodeName, framework.ServeHostnameImage), metav1.CreateOptions{})
	if err == nil {
		framework.Logf("Created pod %s on node %s", pod.ObjectMeta.Name, nodeName)
	} else {
		framework.Logf("Failed to create pod %s on node %s: %v", podName, nodeName, err)
	}
	return err
}

var _ = SIGDescribe("Network Partition [Disruptive] [Slow]", func() {
	f := framework.NewDefaultFramework("network-partition")
	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		_, err := e2epod.GetPodsInNamespace(c, ns, map[string]string{})
		framework.ExpectNoError(err)

		// TODO(foxish): Re-enable testing on gce after kubernetes#56787 is fixed.
		e2eskipper.SkipUnlessProviderIs("gke", "aws")
		if strings.Contains(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			framework.Failf("Test dose not support cluster setup with more than one MIG: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
		}
	})

	framework.KubeDescribe("Pods", func() {
		ginkgo.Context("should return to running and ready state after network partition is healed", func() {
			ginkgo.BeforeEach(func() {
				e2eskipper.SkipUnlessNodeCountIsAtLeast(2)
				e2eskipper.SkipUnlessSSHKeyPresent()
			})

			// What happens in this test:
			//	Network traffic from a node to master is cut off to simulate network partition
			// Expect to observe:
			// 1. Node is marked NotReady after timeout by nodecontroller (40seconds)
			// 2. All pods on node are marked NotReady shortly after #1
			// 3. Node and pods return to Ready after connectivity recovers
			ginkgo.It("All pods on the unreachable node should be marked as NotReady upon the node turn NotReady "+
				"AND all pods should be mark back to Ready when the node get back to Ready before pod eviction timeout", func() {
				ginkgo.By("choose a node - we will block all network traffic on this node")
				var podOpts metav1.ListOptions
				nodeOpts := metav1.ListOptions{}
				nodes, err := c.CoreV1().Nodes().List(context.TODO(), nodeOpts)
				framework.ExpectNoError(err)
				e2enode.Filter(nodes, func(node v1.Node) bool {
					if !e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true) {
						return false
					}
					podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
					pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), podOpts)
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
				if err = e2epod.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReady); err != nil {
					framework.Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
				}

				ginkgo.By("Set up watch on node status")
				nodeSelector := fields.OneTermEqualSelector("metadata.name", node.Name)
				stopCh := make(chan struct{})
				newNode := make(chan *v1.Node)
				var controller cache.Controller
				_, controller = cache.NewInformer(
					&cache.ListWatch{
						ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
							options.FieldSelector = nodeSelector.String()
							obj, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), options)
							return runtime.Object(obj), err
						},
						WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector.String()
							return f.ClientSet.CoreV1().Nodes().Watch(context.TODO(), options)
						},
					},
					&v1.Node{},
					0,
					cache.ResourceEventHandlerFuncs{
						UpdateFunc: func(oldObj, newObj interface{}) {
							n, ok := newObj.(*v1.Node)
							framework.ExpectEqual(ok, true)
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

				ginkgo.By(fmt.Sprintf("Block traffic from node %s to the control plane", node.Name))
				host, err := e2enode.GetExternalIP(&node)
				framework.ExpectNoError(err)
				controlPlaneAddresses := framework.GetControlPlaneAddresses(c)
				defer func() {
					ginkgo.By(fmt.Sprintf("Unblock traffic from node %s to the control plane", node.Name))
					for _, instanceAddress := range controlPlaneAddresses {
						e2enetwork.UnblockNetwork(host, instanceAddress)
					}

					if ginkgo.CurrentGinkgoTestDescription().Failed {
						return
					}

					ginkgo.By("Expect to observe node and pod status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
					if err = e2epod.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReady); err != nil {
						framework.Failf("Pods on node %s did not become ready and running within %v: %v", node.Name, podReadyTimeout, err)
					}
				}()

				for _, instanceAddress := range controlPlaneAddresses {
					e2enetwork.BlockNetwork(host, instanceAddress)
				}

				ginkgo.By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				if err = e2epod.WaitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, testutils.PodNotReady); err != nil {
					framework.Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}
			})
		})
	})

	framework.KubeDescribe("[ReplicationController]", func() {
		ginkgo.It("should recreate pods scheduled on the unreachable node "+
			"AND allow scheduling of pods on a node after it rejoins the cluster", func() {
			e2eskipper.SkipUnlessSSHKeyPresent()

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-net"
			common.NewSVCByName(c, ns, name)
			numNodes, err := e2enode.TotalRegistered(f.ClientSet)
			framework.ExpectNoError(err)
			replicas := int32(numNodes)
			common.NewRCByName(c, ns, name, replicas, nil, nil)
			err = e2epod.VerifyPods(c, ns, name, true, replicas)
			framework.ExpectNoError(err, "Each pod should start running and responding")

			ginkgo.By("choose a node with at least one pod - we will block some network traffic on this node")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options) // list pods after all have been scheduled
			framework.ExpectNoError(err)
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// This creates a temporary network partition, verifies that 'podNameToDisappear',
			// that belongs to replication controller 'rcName', really disappeared (because its
			// grace period is set to 0).
			// Finally, it checks that the replication controller recreates the
			// pods on another node and that now the number of replicas is equal 'replicas'.
			ginkgo.By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			e2enetwork.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := waitForRCPodToDisappear(c, ns, name, pods.Items[0].Name)
				framework.ExpectNoError(err)

				ginkgo.By("verifying whether the pod from the unreachable node is recreated")
				err = e2epod.VerifyPods(c, ns, name, true, replicas)
				framework.ExpectNoError(err)
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !e2enode.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}

			// sleep a bit, to allow Watch in NodeController to catch up.
			time.Sleep(5 * time.Second)

			ginkgo.By("verify whether new pods can be created on the re-attached node")
			// increasing the RC size is not a valid way to test this
			// since we have no guarantees the pod will be scheduled on our node.
			additionalPod := "additionalpod"
			err = newPodOnNode(c, ns, additionalPod, node.Name)
			framework.ExpectNoError(err)
			err = e2epod.VerifyPods(c, ns, additionalPod, true, 1)
			framework.ExpectNoError(err)

			// verify that it is really on the requested node
			{
				pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), additionalPod, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if pod.Spec.NodeName != node.Name {
					framework.Logf("Pod %s found on invalid node: %s instead of %s", pod.Name, pod.Spec.NodeName, node.Name)
				}
			}
		})

		ginkgo.It("should eagerly create replacement pod during network partition when termination grace is non-zero", func() {
			e2eskipper.SkipUnlessSSHKeyPresent()

			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-net"
			gracePeriod := int64(30)

			common.NewSVCByName(c, ns, name)
			numNodes, err := e2enode.TotalRegistered(f.ClientSet)
			framework.ExpectNoError(err)
			replicas := int32(numNodes)
			common.NewRCByName(c, ns, name, replicas, &gracePeriod, []string{"serve-hostname"})
			err = e2epod.VerifyPods(c, ns, name, true, replicas)
			framework.ExpectNoError(err, "Each pod should start running and responding")

			ginkgo.By("choose a node with at least one pod - we will block some network traffic on this node")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options) // list pods after all have been scheduled
			framework.ExpectNoError(err)
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// This creates a temporary network partition, verifies that 'podNameToDisappear',
			// that belongs to replication controller 'rcName', did not disappear (because its
			// grace period is set to 30).
			// Finally, it checks that the replication controller recreates the
			// pods on another node and that now the number of replicas is equal 'replicas + 1'.
			ginkgo.By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			e2enetwork.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := waitForRCPodToDisappear(c, ns, name, pods.Items[0].Name)
				framework.ExpectEqual(err, wait.ErrWaitTimeout, "Pod was not deleted during network partition.")

				ginkgo.By(fmt.Sprintf("verifying that there are %v running pods during partition", replicas))
				_, err = e2epod.PodsCreated(c, ns, name, replicas)
				framework.ExpectNoError(err)
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !e2enode.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
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

		ginkgo.BeforeEach(func() {
			// TODO(foxish): Re-enable testing on gce after kubernetes#56787 is fixed.
			e2eskipper.SkipUnlessProviderIs("gke")
			ginkgo.By("creating service " + headlessSvcName + " in namespace " + f.Namespace.Name)
			headlessService := e2eservice.CreateServiceSpec(headlessSvcName, "", true, labels)
			_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), headlessService, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			c = f.ClientSet
			ns = f.Namespace.Name
		})

		ginkgo.AfterEach(func() {
			if ginkgo.CurrentGinkgoTestDescription().Failed {
				framework.DumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all stateful set in ns %v", ns)
			e2estatefulset.DeleteAllStatefulSets(c, ns)
		})

		ginkgo.It("should come back up if node goes down [Slow] [Disruptive]", func() {
			petMounts := []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts := []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ps := e2estatefulset.NewStatefulSet(psName, ns, headlessSvcName, 3, petMounts, podMounts, labels)
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ps, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			nn, err := e2enode.TotalRegistered(f.ClientSet)
			framework.ExpectNoError(err)
			nodes, err := e2enode.CheckReady(f.ClientSet, nn, framework.NodeReadyInitialTimeout)
			framework.ExpectNoError(err)
			common.RestartNodes(f.ClientSet, nodes)

			ginkgo.By("waiting for pods to be running again")
			e2estatefulset.WaitForRunningAndReady(c, *ps.Spec.Replicas, ps)
		})

		ginkgo.It("should not reschedule stateful pods if there is a network partition [Slow] [Disruptive]", func() {
			e2eskipper.SkipUnlessSSHKeyPresent()

			ps := e2estatefulset.NewStatefulSet(psName, ns, headlessSvcName, 3, []v1.VolumeMount{}, []v1.VolumeMount{}, labels)
			_, err := c.AppsV1().StatefulSets(ns).Create(context.TODO(), ps, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			e2estatefulset.WaitForRunningAndReady(c, *ps.Spec.Replicas, ps)

			pod := e2estatefulset.GetPodList(c, ps).Items[0]
			node, err := c.CoreV1().Nodes().Get(context.TODO(), pod.Spec.NodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// Blocks outgoing network traffic on 'node'. Then verifies that 'podNameToDisappear',
			// that belongs to StatefulSet 'statefulSetName', **does not** disappear due to forced deletion from the apiserver.
			// The grace period on the stateful pods is set to a value > 0.
			e2enetwork.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Checking that the NodeController does not force delete stateful pods %v", pod.Name)
				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(c, pod.Name, ns, 10*time.Minute)
				framework.ExpectEqual(err, wait.ErrWaitTimeout, "Pod was not deleted during network partition.")
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !e2enode.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}

			ginkgo.By("waiting for pods to be running again")
			e2estatefulset.WaitForRunningAndReady(c, *ps.Spec.Replicas, ps)
		})
	})

	framework.KubeDescribe("[Job]", func() {
		ginkgo.It("should create new pods when node is partitioned", func() {
			e2eskipper.SkipUnlessSSHKeyPresent()

			parallelism := int32(2)
			completions := int32(4)
			backoffLimit := int32(6) // default value

			job := e2ejob.NewTestJob("notTerminate", "network-partition", v1.RestartPolicyNever,
				parallelism, completions, nil, backoffLimit)
			job, err := e2ejob.CreateJob(f.ClientSet, f.Namespace.Name, job)
			framework.ExpectNoError(err)
			label := labels.SelectorFromSet(labels.Set(map[string]string{e2ejob.JobSelectorKey: job.Name}))

			ginkgo.By(fmt.Sprintf("verifying that there are now %v running pods", parallelism))
			_, err = e2epod.PodsCreatedByLabel(c, ns, job.Name, parallelism, label)
			framework.ExpectNoError(err)

			ginkgo.By("choose a node with at least one pod - we will block some network traffic on this node")
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options) // list pods after all have been scheduled
			framework.ExpectNoError(err)
			nodeName := pods.Items[0].Spec.NodeName

			node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// This creates a temporary network partition, verifies that the job has 'parallelism' number of
			// running pods after the node-controller detects node unreachable.
			ginkgo.By(fmt.Sprintf("blocking network traffic from node %s", node.Name))
			e2enetwork.TestUnderTemporaryNetworkFailure(c, ns, node, func() {
				framework.Logf("Waiting for pod %s to be removed", pods.Items[0].Name)
				err := e2epod.WaitForPodToDisappear(c, ns, pods.Items[0].Name, label, 20*time.Second, 10*time.Minute)
				framework.ExpectEqual(err, wait.ErrWaitTimeout, "Pod was not deleted during network partition.")

				ginkgo.By(fmt.Sprintf("verifying that there are now %v running pods", parallelism))
				_, err = e2epod.PodsCreatedByLabel(c, ns, job.Name, parallelism, label)
				framework.ExpectNoError(err)
			})

			framework.Logf("Waiting %v for node %s to be ready once temporary network failure ends", resizeNodeReadyTimeout, node.Name)
			if !e2enode.WaitForNodeToBeReady(c, node.Name, resizeNodeReadyTimeout) {
				framework.Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
			}
		})
	})

	framework.KubeDescribe("Pods", func() {
		ginkgo.Context("should be evicted from unready Node", func() {
			ginkgo.BeforeEach(func() {
				e2eskipper.SkipUnlessNodeCountIsAtLeast(2)
			})

			// What happens in this test:
			//	Network traffic from a node to master is cut off to simulate network partition
			// Expect to observe:
			// 1. Node is marked NotReady after timeout by nodecontroller (40seconds)
			// 2. All pods on node are marked NotReady shortly after #1
			// 3. After enough time passess all Pods are evicted from the given Node
			ginkgo.It("[Feature:TaintEviction] All pods on the unreachable node should be marked as NotReady upon the node turn NotReady "+
				"AND all pods should be evicted after eviction timeout passes", func() {
				e2eskipper.SkipUnlessSSHKeyPresent()
				ginkgo.By("choose a node - we will block all network traffic on this node")
				var podOpts metav1.ListOptions
				nodes, err := e2enode.GetReadySchedulableNodes(c)
				framework.ExpectNoError(err)
				e2enode.Filter(nodes, func(node v1.Node) bool {
					if !e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true) {
						return false
					}
					podOpts = metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, node.Name).String()}
					pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), podOpts)
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
				if err := e2epod.WaitForMatchPodsCondition(c, podOpts, "Running and Ready", podReadyTimeout, testutils.PodRunningReadyOrSucceeded); err != nil {
					framework.Failf("Pods on node %s are not ready and running within %v: %v", node.Name, podReadyTimeout, err)
				}
				pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), podOpts)
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

				ginkgo.By("Set up watch on node status")
				nodeSelector := fields.OneTermEqualSelector("metadata.name", node.Name)
				stopCh := make(chan struct{})
				newNode := make(chan *v1.Node)
				var controller cache.Controller
				_, controller = cache.NewInformer(
					&cache.ListWatch{
						ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
							options.FieldSelector = nodeSelector.String()
							obj, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), options)
							return runtime.Object(obj), err
						},
						WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
							options.FieldSelector = nodeSelector.String()
							return f.ClientSet.CoreV1().Nodes().Watch(context.TODO(), options)
						},
					},
					&v1.Node{},
					0,
					cache.ResourceEventHandlerFuncs{
						UpdateFunc: func(oldObj, newObj interface{}) {
							n, ok := newObj.(*v1.Node)
							framework.ExpectEqual(ok, true)
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

				ginkgo.By(fmt.Sprintf("Block traffic from node %s to the control plane", node.Name))
				host, err := e2enode.GetExternalIP(&node)
				framework.ExpectNoError(err)
				controlPlaneAddresses := framework.GetControlPlaneAddresses(c)
				defer func() {
					ginkgo.By(fmt.Sprintf("Unblock traffic from node %s to the control plane", node.Name))
					for _, instanceAddress := range controlPlaneAddresses {
						e2enetwork.UnblockNetwork(host, instanceAddress)
					}

					if ginkgo.CurrentGinkgoTestDescription().Failed {
						return
					}

					ginkgo.By("Expect to observe node status change from NotReady to Ready after network connectivity recovers")
					expectNodeReadiness(true, newNode)
				}()

				for _, instanceAddress := range controlPlaneAddresses {
					e2enetwork.BlockNetwork(host, instanceAddress)
				}

				ginkgo.By("Expect to observe node and pod status change from Ready to NotReady after network partition")
				expectNodeReadiness(false, newNode)
				framework.ExpectNoError(wait.Poll(1*time.Second, timeout, func() (bool, error) {
					return framework.NodeHasTaint(c, node.Name, nodepkg.UnreachableTaintTemplate)
				}))
				if err = e2epod.WaitForMatchPodsCondition(c, podOpts, "NotReady", podNotReadyTimeout, testutils.PodNotReady); err != nil {
					framework.Failf("Pods on node %s did not become NotReady within %v: %v", node.Name, podNotReadyTimeout, err)
				}

				sleepTime := maxTolerationTime + 20*time.Second
				ginkgo.By(fmt.Sprintf("Sleeping for %v and checking if all Pods were evicted", sleepTime))
				time.Sleep(sleepTime)
				pods, err = c.CoreV1().Pods(v1.NamespaceAll).List(context.TODO(), podOpts)
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

// waitForRCPodToDisappear returns nil if the pod from the given replication controller (described by rcName) no longer exists.
// In case of failure or too long waiting time, an error is returned.
func waitForRCPodToDisappear(c clientset.Interface, ns, rcName, podName string) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	// NodeController evicts pod after 5 minutes, so we need timeout greater than that to observe effects.
	// The grace period must be set to 0 on the pod for it to be deleted during the partition.
	// Otherwise, it goes to the 'Terminating' state till the kubelet confirms deletion.
	return e2epod.WaitForPodToDisappear(c, ns, podName, label, 20*time.Second, 10*time.Minute)
}
