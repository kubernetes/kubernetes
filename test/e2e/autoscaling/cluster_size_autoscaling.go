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

package autoscaling

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	scaleUpTimeout = 5 * time.Minute
)

// WaitForClusterSizeFunc waits until the cluster size matches the given function.
func WaitForClusterSizeFunc(ctx context.Context, c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration) error {
	return WaitForClusterSizeFuncWithUnready(ctx, c, sizeFunc, timeout, 0)
}

// WaitForClusterSizeFuncWithUnready waits until the cluster size matches the given function and assumes some unready nodes.
func WaitForClusterSizeFuncWithUnready(ctx context.Context, c clientset.Interface, sizeFunc func(int) bool, timeout time.Duration, expectedUnready int) error {
	for start := time.Now(); time.Since(start) < timeout && ctx.Err() == nil; time.Sleep(20 * time.Second) {
		nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			klog.Warningf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		e2enode.Filter(nodes, func(node v1.Node) bool {
			return e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true)
		})
		numReady := len(nodes.Items)

		if numNodes == numReady+expectedUnready && sizeFunc(numNodes) {
			klog.Infof("Cluster has reached the desired size. Current size %d, not ready nodes %d", numNodes, numNodes-numReady)
			return nil
		}
		klog.Infof("Waiting for cluster with func, current size %d, not ready nodes %d", numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for appropriate cluster size", timeout)
}

func waitForCaPodsReadyInNamespace(ctx context.Context, f *framework.Framework, c clientset.Interface, tolerateUnreadyCount int) error {
	var notready []string
	for start := time.Now(); time.Now().Before(start.Add(scaleUpTimeout)) && ctx.Err() == nil; time.Sleep(20 * time.Second) {
		pods, err := c.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return fmt.Errorf("failed to get pods: %w", err)
		}
		notready = make([]string, 0)
		for _, pod := range pods.Items {
			ready := false
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.PodReady && c.Status == v1.ConditionTrue {
					ready = true
				}
			}
			// Failed pods in this context generally mean that they have been
			// double scheduled onto a node, but then failed a constraint check.
			if pod.Status.Phase == v1.PodFailed {
				klog.Warningf("Pod has failed: %v", pod)
			}
			if !ready && pod.Status.Phase != v1.PodFailed {
				notready = append(notready, pod.Name)
			}
		}
		if len(notready) <= tolerateUnreadyCount {
			klog.Infof("sufficient number of pods ready. Tolerating %d unready", tolerateUnreadyCount)
			return nil
		}
		klog.Infof("Too many pods are not ready yet: %v", notready)
	}
	klog.Info("Timeout on waiting for pods being ready")
	klog.Info(e2ekubectl.RunKubectlOrDie(f.Namespace.Name, "get", "pods", "-o", "json", "--all-namespaces"))
	klog.Info(e2ekubectl.RunKubectlOrDie(f.Namespace.Name, "get", "nodes", "-o", "json"))

	// Some pods are still not running.
	return fmt.Errorf("too many pods are still not running: %v", notready)
}

func waitForAllCaPodsReadyInNamespace(ctx context.Context, f *framework.Framework, c clientset.Interface) error {
	return waitForCaPodsReadyInNamespace(ctx, f, c, 0)
}

// Create an RC running a given number of pods with anti-affinity
func runAntiAffinityPods(ctx context.Context, f *framework.Framework, namespace string, pods int, id string, podLabels, antiAffinityLabels map[string]string) error {
	config := &testutils.RCConfig{
		Affinity:  buildAntiAffinity(antiAffinityLabels),
		Client:    f.ClientSet,
		Name:      id,
		Namespace: namespace,
		Timeout:   scaleUpTimeout,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  pods,
		Labels:    podLabels,
	}
	err := e2erc.RunRC(ctx, *config)
	if err != nil {
		return err
	}
	_, err = f.ClientSet.CoreV1().ReplicationControllers(namespace).Get(ctx, id, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func buildAntiAffinity(labels map[string]string) *v1.Affinity {
	return &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchLabels: labels,
					},
					TopologyKey: "kubernetes.io/hostname",
				},
			},
		},
	}
}

// Increase cluster size by creating pods with anti-affinity.
// Returns a function that removes the pods.
// Adds the same to deferred cleanup in case the function was not called.
func increaseClusterSizeWithTimeout(ctx context.Context, f *framework.Framework, c clientset.Interface, targetNodeCount int, timeout time.Duration) func() error {
	labels := map[string]string{
		"anti-affinity": "yes",
	}
	framework.ExpectNoError(runAntiAffinityPods(ctx, f, f.Namespace.Name, targetNodeCount, "increase-size-pod", labels, labels))
	cleanupFunc := func() error {
		return e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, f.Namespace.Name, "increase-size-pod")
	}

	ginkgo.DeferCleanup(func(ctx context.Context) {
		klog.Infof("Cleaning up RC and pods if test did not clean them up")
		err := cleanupFunc()
		klog.Infof("Error during cleanup: %v", err)
	})

	// Verify that cluster size is increased
	framework.ExpectNoError(WaitForClusterSizeFuncWithUnready(ctx, f.ClientSet,
		func(size int) bool { return size >= targetNodeCount }, timeout, 0))
	framework.ExpectNoError(waitForAllCaPodsReadyInNamespace(ctx, f, c))
	return cleanupFunc
}

func increaseClusterSize(ctx context.Context, f *framework.Framework, c clientset.Interface, targetNodeCount int) func() error {
	return increaseClusterSizeWithTimeout(ctx, f, c, targetNodeCount, scaleUpTimeout)
}
