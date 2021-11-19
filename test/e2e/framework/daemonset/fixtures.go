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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/test/e2e/framework"
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

func CheckRunningOnAllNodes(f *framework.Framework, ds *appsv1.DaemonSet) (bool, error) {
	nodeNames := SchedulableNodes(f.ClientSet, ds)
	return CheckDaemonPodOnNodes(f, ds, nodeNames)()
}

// CheckPresentOnNodes will check that the daemonset will be present on at least the given number of
// schedulable nodes.
func CheckPresentOnNodes(c clientset.Interface, ds *appsv1.DaemonSet, ns string, numNodes int) (bool, error) {
	nodeNames := SchedulableNodes(c, ds)
	if len(nodeNames) < numNodes {
		return false, nil
	}
	return checkDaemonPodStateOnNodes(c, ds, ns, nodeNames, func(pod *v1.Pod) bool {
		return pod.Status.Phase != v1.PodPending
	})
}

func SchedulableNodes(c clientset.Interface, ds *appsv1.DaemonSet) []string {
	nodeList, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err)
	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		if !canScheduleOnNode(node, ds) {
			framework.Logf("DaemonSet pods can't tolerate node %s with taints %+v, skip checking this node", node.Name, node.Spec.Taints)
			continue
		}
		nodeNames = append(nodeNames, node.Name)
	}
	return nodeNames
}

// canScheduleOnNode checks if a given DaemonSet can schedule pods on the given node
func canScheduleOnNode(node v1.Node, ds *appsv1.DaemonSet) bool {
	newPod := daemon.NewPod(ds, node.Name)
	fitsNodeName, fitsNodeAffinity, fitsTaints := daemon.Predicates(newPod, &node, node.Spec.Taints)
	return fitsNodeName && fitsNodeAffinity && fitsTaints
}

func CheckDaemonPodOnNodes(f *framework.Framework, ds *appsv1.DaemonSet, nodeNames []string) func() (bool, error) {
	return func() (bool, error) {
		return checkDaemonPodStateOnNodes(f.ClientSet, ds, f.Namespace.Name, nodeNames, func(pod *v1.Pod) bool {
			return podutil.IsPodAvailable(pod, ds.Spec.MinReadySeconds, metav1.Now())
		})
	}
}

func checkDaemonPodStateOnNodes(c clientset.Interface, ds *appsv1.DaemonSet, ns string, nodeNames []string, stateChecker func(*v1.Pod) bool) (bool, error) {
	podList, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
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

func CheckDaemonStatus(f *framework.Framework, dsName string) error {
	ds, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).Get(context.TODO(), dsName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
	if desired != scheduled && desired != ready {
		return fmt.Errorf("error in daemon status. DesiredScheduled: %d, CurrentScheduled: %d, Ready: %d", desired, scheduled, ready)
	}
	return nil
}
