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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller"

	"k8s.io/kubernetes/pkg/api/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// DaemonSetUpgradeTest tests that a DaemonSet is running before and after
// a cluster upgrade.
type DaemonSetUpgradeTest struct {
	daemonSet *extensions.DaemonSet
}

func (DaemonSetUpgradeTest) Name() string { return "daemonset-upgrade" }

// Setup creates a DaemonSet and verifies that it's running
func (t *DaemonSetUpgradeTest) Setup(f *framework.Framework) {
	daemonSetName := "ds1"
	labelSet := map[string]string{"ds-name": daemonSetName}
	image := "gcr.io/google_containers/serve_hostname:v1.4"

	ns := f.Namespace

	t.daemonSet = &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      daemonSetName,
		},
		Spec: extensions.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labelSet,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  daemonSetName,
							Image: image,
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}

	By("Creating a DaemonSet")
	var err error
	if t.daemonSet, err = f.ClientSet.Extensions().DaemonSets(ns.Name).Create(t.daemonSet); err != nil {
		framework.Failf("unable to create test DaemonSet %s: %v", t.daemonSet.Name, err)
	}

	By("Waiting for DaemonSet pods to become ready")
	err = wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		return checkRunningOnAllNodes(f, t.daemonSet.Namespace, t.daemonSet.Labels)
	})
	framework.ExpectNoError(err)

	By("Validating the DaemonSet after creation")
	t.validateRunningDaemonSet(f)
}

// Test waits until the upgrade has completed and then verifies that the DaemonSet
// is still running
func (t *DaemonSetUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	By("Waiting for upgradet to complete before re-validating DaemonSet")
	<-done

	By("validating the DaemonSet is still running after upgrade")
	t.validateRunningDaemonSet(f)
}

// Teardown cleans up any remaining resources.
func (t *DaemonSetUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func (t *DaemonSetUpgradeTest) validateRunningDaemonSet(f *framework.Framework) {
	By("confirming the DaemonSet pods are running on all expected nodes")
	res, err := checkRunningOnAllNodes(f, t.daemonSet.Namespace, t.daemonSet.Labels)
	framework.ExpectNoError(err)
	if !res {
		framework.Failf("expected DaemonSet pod to be running on all nodes, it was not")
	}

	// DaemonSet resource itself should be good
	By("confirming the DaemonSet resource is in a good state")
	res, err = checkDaemonStatus(f, t.daemonSet.Namespace, t.daemonSet.Name)
	framework.ExpectNoError(err)
	if !res {
		framework.Failf("expected DaemonSet to be in a good state, it was not")
	}
}

func checkRunningOnAllNodes(f *framework.Framework, namespace string, selector map[string]string) (bool, error) {
	nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return false, err
	}

	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		if len(node.Spec.Taints) == 0 {
			nodeNames = append(nodeNames, node.Name)
		} else {
			framework.Logf("Node %v not expected to have DaemonSet pod, has taints %v", node.Name, node.Spec.Taints)
		}
	}

	return checkDaemonPodOnNodes(f, namespace, selector, nodeNames)
}

func checkDaemonPodOnNodes(f *framework.Framework, namespace string, labelSet map[string]string, nodeNames []string) (bool, error) {
	selector := labels.Set(labelSet).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := f.ClientSet.Core().Pods(namespace).List(options)
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
	ds, err := f.ClientSet.ExtensionsV1beta1().DaemonSets(namespace).Get(dsName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}

	desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
	if desired != scheduled && desired != ready {
		return false, nil
	}

	return true, nil
}
