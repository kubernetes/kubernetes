/*
Copyright 2024 The Kubernetes Authors.

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

package windows

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	metrics "k8s.io/metrics/pkg/client/clientset/versioned"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// It can take 10-15 seconds for node memory-pressure taint to show up on the node
	// so we'll wait 45 seconds for the taint to show up so the e2e test case can catch
	// it and the wait for the taint to be removed so other serial/slow tests can run
	// against the same node.
	waitForNodeMemoryPressureTaintDelayDuration = 45 * time.Second

	// eviction pod namespace base name
	evictionPodNamespaceBaseName = "eviction-test-windows"
)

var _ = sigDescribe(feature.Windows, "Eviction", framework.WithSerial(), framework.WithSlow(), framework.WithDisruptive(), (func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework(evictionPodNamespaceBaseName)
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// This test will first find a Windows node memory-pressure hard-eviction enabled.
	// The test will then schedule a pod that requests and consumes 500Mi of memory and then
	// another pod that will consume the rest of the node's memory.
	// The test will then verify that the second pod gets evicted and then the node again becomes
	// ready for schedule after the second pod gets evicted.
	ginkgo.It("should evict a pod when a node experiences memory pressure", func(ctx context.Context) {
		framework.Logf("Looking for a Windows node with memory-pressure eviction enabled")
		selector := labels.Set{"kubernetes.io/os": "windows"}.AsSelector()
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{
			LabelSelector: selector.String(),
		})
		framework.ExpectNoError(err)

		var node *v1.Node
		var nodeMem nodeMemory
		for _, n := range nodeList.Items {
			nm := getNodeMemory(ctx, f, n)
			if nm.hardEviction.Value() != 0 {
				framework.Logf("Using node %s", n.Name)
				node = &n
				nodeMem = nm
				break
			}
		}

		if node == nil {
			e2eskipper.Skipf("No Windows nodes with hard memory-pressure eviction found")
		}

		framework.Logf("Node %q capacity: %v Mi", node.Name, nodeMem.capacity.Value()/(1024*1024))
		framework.Logf("Node %q hard eviction threshold: %v Mi", node.Name, nodeMem.hardEviction.Value()/(1024*1024))
		framework.Logf("Available memory before eviction: %v Mi", (nodeMem.capacity.Value()-nodeMem.hardEviction.Value())/(1024*1024))

		err = waitForMemoryPressureTaintRemoval(ctx, f, node.Name, 10*time.Minute)
		framework.ExpectNoError(err, "Timed out waiting for memory-pressure taint to be removed from node %q", node.Name)

		cleanupImagePullerPods(ctx, f)

		ginkgo.DeferCleanup(f.DeleteNamespace, f.Namespace.Name)

		ginkgo.By("Scheduling a pod that requests and consumes 500Mi of Memory")

		pod1 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pod1",
						Image: imageutils.GetE2EImage(imageutils.ResourceConsumer),
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: *resource.NewQuantity(500*1024*1024, resource.BinarySI),
							},
						},
						Command: []string{
							"/bin/testlimit.exe",
							"-accepteula",
							"-d",
							"100Mb",
							"-e",
							"5",
							"20000s",
							"-c",
							"5"},
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
				NodeName: node.Name,
			},
		}
		pod1, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod1)
		framework.ExpectNoError(err)

		ginkgo.By("Scheduling another pod will consume the rest of the node's memory")
		chunks := int((nodeMem.capacity.Value()-nodeMem.hardEviction.Value())/(300*1024*1024) + 3)
		framework.Logf("Pod2 will request approximately %v Mi total memory (%d chunks × 300Mi)", chunks*300, chunks)

		pod2 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod2",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pod2",
						Image: imageutils.GetE2EImage(imageutils.ResourceConsumer),
						Command: []string{
							"/bin/testlimit.exe",
							"-accepteula",
							"-d",
							"300Mb",
							"-e",
							"1",
							"20000s",
							"-c",
							strconv.Itoa(chunks)},
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
				NodeName: node.Name,
			},
		}
		pod2, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		if ginkgo.CurrentSpecReport().Failed() {
			logNodeMemoryDebugInfo(ctx, f, node.Name, f.Namespace.Name, "pod2")
		}

		ginkgo.By("Waiting for pod2 to start running")
		err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod2)
		framework.ExpectNoError(err)

		framework.Logf("Waiting for pod2 to be evicted")

		gomega.Eventually(ctx, func() error {
			// Get updated node info
			node, err = f.ClientSet.CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("failed to get node: %w", err)
			}

			// Log node memory pressure condition and taints
			for _, cond := range node.Status.Conditions {
				if cond.Type == v1.NodeMemoryPressure {
					framework.Logf("Node condition: MemoryPressure = %v (Reason: %s, Message: %s)",
						cond.Status, cond.Reason, cond.Message)
				}
			}
			for _, taint := range node.Spec.Taints {
				framework.Logf("Node %q has taint %q (Effect: %q)", node.Name, taint.Key, taint.Effect)
			}

			// Check for eviction events in the namespace
			events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			if err != nil {
				return fmt.Errorf("failed to list events: %w", err)
			}

			evicted := false
			for _, e := range events.Items {
				if e.Reason == "Evicted" {
					if e.InvolvedObject.Name == pod2.Name {
						framework.Logf("Eviction event for pod2: %q", e.Message)
						evicted = true
					} else {
						framework.Logf("Eviction event for other pod %q: %q", e.InvolvedObject.Name, e.Message)
					}
				}
				if e.InvolvedObject.Name == pod2.Name {
					framework.Logf("Event for pod2: Type=%s, Reason=%s, Message=%q", e.Type, e.Reason, e.Message)
				}
			}

			if evicted {
				return nil
			}
			return fmt.Errorf("pod2 not evicted yet; still waiting")
		}).WithTimeout(10*time.Minute).WithPolling(10*time.Second).Should(gomega.Succeed(), "pod2 should eventually be evicted")

		ginkgo.By("Waiting for node.kubernetes.io/memory-pressure taint to be removed")
		// ensure e2e test framework catches the memory-pressure taint
		time.Sleep(waitForNodeMemoryPressureTaintDelayDuration)
		// wait for node.kubernetes.io/memory-pressure=NoSchedule to be removed so other tests can run
		err = e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 10*time.Minute)
		framework.ExpectNoError(err)
	})
}))

// Delete img-puller pods if they exist because eviction manager keeps selecting them for eviction first
// Note we cannot just delete the namespace because a deferred cleanup task tries to delete the ns if
// image pre-pulling was enabled.
func cleanupImagePullerPods(ctx context.Context, f *framework.Framework) {
	nsList, err := f.ClientSet.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	for _, ns := range nsList.Items {
		if strings.Contains(ns.Name, "img-puller") {
			framework.Logf("Deleting pods in namespace %s", ns.Name)
			podList, err := f.ClientSet.CoreV1().Pods(ns.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, pod := range podList.Items {
				framework.Logf("  Deleting pod %s", pod.Name)
				err = f.ClientSet.CoreV1().Pods(ns.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err)
			}
			break
		}
	}
}

func waitForMemoryPressureTaintRemoval(ctx context.Context, f *framework.Framework, nodeName string, timeout time.Duration) error {
	framework.Logf("Waiting for memory-pressure taint to be removed from node %q", nodeName)
	return wait.PollUntilContextTimeout(ctx, 10*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get node %q: %v", nodeName, err)
			return false, err
		}

		// Log node conditions
		for _, cond := range node.Status.Conditions {
			if cond.Type == v1.NodeMemoryPressure {
				framework.Logf("Node condition: MemoryPressure = %v (reason: %s, message: %s)", cond.Status, cond.Reason, cond.Message)
			}
		}

		// Log taints
		hasTaint := false
		for _, taint := range node.Spec.Taints {
			if taint.Key == v1.TaintNodeMemoryPressure && taint.Effect == v1.TaintEffectNoSchedule {
				framework.Logf("Node %q still has memory-pressure taint (Effect: %s, TimeAdded: %v)", nodeName, taint.Effect, taint.TimeAdded)
				hasTaint = true
			}
		}

		// Log all pods on the node
		podList, err := f.ClientSet.CoreV1().Pods("").List(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
		})
		if err != nil {
			framework.Logf("Failed to list pods on node %q: %v", nodeName, err)
		} else {
			for _, pod := range podList.Items {
				framework.Logf("Pod %q in ns %q phase: %s", pod.Name, pod.Namespace, pod.Status.Phase)
			}
		}

		if !hasTaint {
			framework.Logf("Memory-pressure taint has been removed from node %q", nodeName)
			return true, nil
		}
		return false, nil
	})
}

func logNodeMemoryDebugInfo(ctx context.Context, f *framework.Framework, nodeName string, namespace string, podName string) {
	framework.Logf("========== DEBUG INFO FOR NODE: %q ==========", nodeName)

	// Node description
	node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		framework.Logf("Error fetching node %q: %v", nodeName, err)
	} else {
		for _, cond := range node.Status.Conditions {
			if cond.Type == v1.NodeMemoryPressure {
				framework.Logf("Node condition - MemoryPressure: %v (Reason: %s, Message: %s, LastHeartbeat: %s)",
					cond.Status, cond.Reason, cond.Message, cond.LastHeartbeatTime)
			}
		}
		for _, taint := range node.Spec.Taints {
			if taint.Key == v1.TaintNodeMemoryPressure {
				framework.Logf("Node taint - memory-pressure: Effect=%s, TimeAdded=%v", taint.Effect, taint.TimeAdded)
			}
		}
	}

	// Pods on node
	pods, err := f.ClientSet.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
	})
	if err != nil {
		framework.Logf("Error listing pods on node %q: %v", nodeName, err)
	} else {
		for _, p := range pods.Items {
			framework.Logf("Pod %q in ns %q is in phase: %s", p.Name, p.Namespace, p.Status.Phase)
		}
	}

	metricsClient, err := metrics.NewForConfig(f.ClientConfig())
	if err == nil {
		// Node metrics
		nodeMetrics, err := metricsClient.MetricsV1beta1().NodeMetricses().Get(ctx, nodeName, metav1.GetOptions{})
		if err == nil {
			memUsage := nodeMetrics.Usage.Memory().ScaledValue(resource.Mega)
			framework.Logf("Node %q is using %d Mi of memory", nodeName, memUsage)
		}

		// Pod metrics
		podMetricsList, err := metricsClient.MetricsV1beta1().PodMetricses(namespace).List(ctx, metav1.ListOptions{})
		if err == nil {
			for _, podMetrics := range podMetricsList.Items {
				if podMetrics.Name == podName {
					var totalMem int64
					for _, c := range podMetrics.Containers {
						totalMem += c.Usage.Memory().ScaledValue(resource.Mega)
					}
					framework.Logf("Pod %q is using %d Mi of memory", podMetrics.Name, totalMem)
				}
			}
		}
	}

	// Events in pod's namespace
	events, err := f.ClientSet.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, e := range events.Items {
			if strings.Contains(e.Message, podName) || e.InvolvedObject.Name == podName {
				framework.Logf("Event for pod %q: Type=%s, Reason=%s, Message=%q", podName, e.Type, e.Reason, e.Message)
			}
		}
	}

	framework.Logf("========== END DEBUG INFO FOR NODE: %q ==========", nodeName)
}
