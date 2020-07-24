/*
Copyright 2017 The Kubernetes Authors.

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

package upgrades

import (
	"context"
	"github.com/onsi/ginkgo"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"
)

// DaemonSetUpgradeTest tests that a DaemonSet is running before and after
// a cluster upgrade.
type DaemonSetUpgradeTest struct {
	daemonSet *appsv1.DaemonSet
}

// Name returns the tracking name of the test.
func (DaemonSetUpgradeTest) Name() string { return "[sig-apps] daemonset-upgrade" }

// Setup creates a DaemonSet and verifies that it's running
func (t *DaemonSetUpgradeTest) Setup(f *framework.Framework) {
	daemonSetName := "ds1"
	labelSet := map[string]string{"ds-name": daemonSetName}
	image := framework.ServeHostnameImage

	ns := f.Namespace

	t.daemonSet = &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      daemonSetName,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labelSet,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labelSet,
				},
				Spec: v1.PodSpec{
					Tolerations: []v1.Toleration{
						{Operator: v1.TolerationOpExists},
					},
					Containers: []v1.Container{
						{
							Name:            daemonSetName,
							Image:           image,
							Args:            []string{"serve-hostname"},
							Ports:           []v1.ContainerPort{{ContainerPort: 9376}},
							SecurityContext: &v1.SecurityContext{},
						},
					},
				},
			},
		},
	}

	ginkgo.By("Creating a DaemonSet")
	var err error
	if t.daemonSet, err = f.ClientSet.AppsV1().DaemonSets(ns.Name).Create(context.TODO(), t.daemonSet, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test DaemonSet %s: %v", t.daemonSet.Name, err)
	}

	ginkgo.By("Waiting for DaemonSet pods to become ready")
	err = wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		return checkRunningOnAllNodes(f, t.daemonSet.Namespace, t.daemonSet.Labels)
	})
	framework.ExpectNoError(err)

	ginkgo.By("Validating the DaemonSet after creation")
	t.validateRunningDaemonSet(f)
}

// Test waits until the upgrade has completed and then verifies that the DaemonSet
// is still running
func (t *DaemonSetUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	ginkgo.By("Waiting for upgradet to complete before re-validating DaemonSet")
	<-done

	ginkgo.By("validating the DaemonSet is still running after upgrade")
	t.validateRunningDaemonSet(f)
}

// Teardown cleans up any remaining resources.
func (t *DaemonSetUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func (t *DaemonSetUpgradeTest) validateRunningDaemonSet(f *framework.Framework) {
	ginkgo.By("confirming the DaemonSet pods are running on all expected nodes")
	res, err := checkRunningOnAllNodes(f, t.daemonSet.Namespace, t.daemonSet.Labels)
	framework.ExpectNoError(err)
	if !res {
		framework.Failf("expected DaemonSet pod to be running on all nodes, it was not")
	}

	// DaemonSet resource itself should be good
	ginkgo.By("confirming the DaemonSet resource is in a good state")
	res, err = checkDaemonStatus(f, t.daemonSet.Namespace, t.daemonSet.Name)
	framework.ExpectNoError(err)
	if !res {
		framework.Failf("expected DaemonSet to be in a good state, it was not")
	}
}

func checkRunningOnAllNodes(f *framework.Framework, namespace string, selector map[string]string) (bool, error) {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, err
	}

	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		if len(node.Spec.Taints) != 0 {
			framework.Logf("Ignore taints %v on Node %v for DaemonSet Pod.", node.Spec.Taints, node.Name)
		}
		// DaemonSet Pods are expected to run on all the nodes in e2e.
		nodeNames = append(nodeNames, node.Name)
	}

	return checkDaemonPodOnNodes(f, namespace, selector, nodeNames)
}

func checkDaemonPodOnNodes(f *framework.Framework, namespace string, labelSet map[string]string, nodeNames []string) (bool, error) {
	selector := labels.Set(labelSet).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := f.ClientSet.CoreV1().Pods(namespace).List(context.TODO(), options)
	if err != nil {
		return false, err
	}
	pods := podList.Items

	nodesToPodCount := make(map[string]int)
	for _, pod := range pods {
		if controller.IsPodActive(&pod) {
			framework.Logf("Pod name: %v\t Node Name: %v", pod.Name, pod.Spec.NodeName)
			nodesToPodCount[pod.Spec.NodeName]++
		}
	}
	framework.Logf("nodesToPodCount: %v", nodesToPodCount)

	// Ensure that exactly 1 pod is running on all nodes in nodeNames.
	for _, nodeName := range nodeNames {
		if nodesToPodCount[nodeName] != 1 {
			return false, nil
		}
	}

	// Ensure that sizes of the lists are the same. We've verified that every element of nodeNames is in
	// nodesToPodCount, so verifying the lengths are equal ensures that there aren't pods running on any
	// other nodes.
	return len(nodesToPodCount) == len(nodeNames), nil
}

func checkDaemonStatus(f *framework.Framework, namespace string, dsName string) (bool, error) {
	ds, err := f.ClientSet.AppsV1().DaemonSets(namespace).Get(context.TODO(), dsName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}

	desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
	if desired != scheduled && desired != ready {
		return false, nil
	}

	return true, nil
}
