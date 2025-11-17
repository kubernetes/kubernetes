/*
Copyright 2021 The Kubernetes Authors.

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

package daemonset

import (
	"context"
	"fmt"
	"k8s.io/klog/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubectl/pkg/util/podutils"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/format"
)

func NewDaemonSet(dsName, image string, labels map[string]string, volumes []v1.Volume, mounts []v1.VolumeMount, ports []v1.ContainerPort, args ...string) *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   dsName,
			Labels: labels,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "app",
							Image:           image,
							Args:            args,
							Ports:           ports,
							VolumeMounts:    mounts,
							SecurityContext: &v1.SecurityContext{},
						},
					},
					SecurityContext: &v1.PodSecurityContext{},
					Volumes:         volumes,
				},
			},
		},
	}
}

func CheckRunningOnAllNodes(ctx context.Context, f *framework.Framework, ds *appsv1.DaemonSet) (bool, error) {
	nodeNames := SchedulableNodes(ctx, f.ClientSet, ds)
	return CheckDaemonPodOnNodes(f, ds, nodeNames)(ctx)
}

// CheckPresentOnNodes will check that the daemonset will be present on at least the given number of
// schedulable nodes.
func CheckPresentOnNodes(ctx context.Context, c clientset.Interface, ds *appsv1.DaemonSet, ns string, numNodes int) (bool, error) {
	nodeNames := SchedulableNodes(ctx, c, ds)
	if len(nodeNames) < numNodes {
		return false, nil
	}
	return checkDaemonPodStateOnNodes(ctx, c, ds, ns, nodeNames, func(pod *v1.Pod) bool {
		return pod.Status.Phase != v1.PodPending
	})
}

func SchedulableNodes(ctx context.Context, c clientset.Interface, ds *appsv1.DaemonSet) []string {
	logger := klog.FromContext(ctx)
	nodeList, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		shouldRun, _ := daemon.NodeShouldRunDaemonPod(logger, &node, ds)
		if !shouldRun {
			framework.Logf("DaemonSet pods can't tolerate node %s with taints %+v, skip checking this node", node.Name, node.Spec.Taints)
			continue
		}
		nodeNames = append(nodeNames, node.Name)
	}
	return nodeNames
}

func CheckDaemonPodOnNodes(f *framework.Framework, ds *appsv1.DaemonSet, nodeNames []string) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		return checkDaemonPodStateOnNodes(ctx, f.ClientSet, ds, f.Namespace.Name, nodeNames, func(pod *v1.Pod) bool {
			return podutils.IsPodAvailable(pod, ds.Spec.MinReadySeconds, metav1.Now())
		})
	}
}

func checkDaemonPodStateOnNodes(ctx context.Context, c clientset.Interface, ds *appsv1.DaemonSet, ns string, nodeNames []string, stateChecker func(*v1.Pod) bool) (bool, error) {
	podList, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		framework.Logf("could not get the pod list: %v", err)
		return false, nil
	}
	pods := podList.Items

	nodesToPodCount := make(map[string]int)
	for _, pod := range pods {
		if !metav1.IsControlledBy(&pod, ds) {
			continue
		}
		if pod.DeletionTimestamp != nil {
			continue
		}
		if stateChecker(&pod) {
			nodesToPodCount[pod.Spec.NodeName]++
		}
	}
	framework.Logf("Number of nodes with available pods controlled by daemonset %s: %d", ds.Name, len(nodesToPodCount))

	// Ensure that exactly 1 pod is running on all nodes in nodeNames.
	for _, nodeName := range nodeNames {
		if nodesToPodCount[nodeName] != 1 {
			framework.Logf("Node %s is running %d daemon pod, expected 1", nodeName, nodesToPodCount[nodeName])
			return false, nil
		}
	}

	framework.Logf("Number of running nodes: %d, number of available pods: %d in daemonset %s", len(nodeNames), len(nodesToPodCount), ds.Name)
	// Ensure that sizes of the lists are the same. We've verified that every element of nodeNames is in
	// nodesToPodCount, so verifying the lengths are equal ensures that there aren't pods running on any
	// other nodes.
	return len(nodesToPodCount) == len(nodeNames), nil
}

// CheckDaemonStatus ensures that eventually the daemon set has the desired
// number of pods scheduled and ready. It returns a descriptive error if that
// state is not reached in the amount of time it takes to start
// pods. f.Timeouts.PodStart can be changed to influence that timeout.
func CheckDaemonStatus(ctx context.Context, f *framework.Framework, dsName string) error {
	return framework.Gomega().Eventually(ctx, framework.GetObject(f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).Get, dsName, metav1.GetOptions{})).
		WithTimeout(f.Timeouts.PodStart).
		Should(framework.MakeMatcher(func(ds *appsv1.DaemonSet) (failure func() string, err error) {
			desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
			if desired == scheduled && scheduled == ready {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("Expected daemon set to reach state where all desired pods are scheduled and ready. Got instead DesiredScheduled: %d, CurrentScheduled: %d, Ready: %d\n%s", desired, scheduled, ready, format.Object(ds, 1))
			}, nil
		}))
}
