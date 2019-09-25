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
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"

	"github.com/onsi/ginkgo"
)

const (
	defaultTestTimeout   = time.Duration(5 * time.Minute)
	clusterAddonLabelKey = "k8s-app"
	clusterComponentKey  = "component"
	kubeProxyLabelName   = "kube-proxy"
)

// KubeProxyUpgradeTest tests kube-proxy static pods -> DaemonSet upgrade path.
type KubeProxyUpgradeTest struct {
}

// Name returns the tracking name of the test.
func (KubeProxyUpgradeTest) Name() string { return "[sig-network] kube-proxy-upgrade" }

// Setup verifies kube-proxy static pods is running before upgrade.
func (t *KubeProxyUpgradeTest) Setup(f *framework.Framework) {
	ginkgo.By("Waiting for kube-proxy static pods running and ready")
	err := waitForKubeProxyStaticPodsRunning(f.ClientSet)
	framework.ExpectNoError(err)
}

// Test validates if kube-proxy is migrated from static pods to DaemonSet.
func (t *KubeProxyUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	c := f.ClientSet

	// Block until upgrade is done.
	ginkgo.By("Waiting for upgrade to finish")
	<-done

	ginkgo.By("Waiting for kube-proxy static pods disappear")
	err := waitForKubeProxyStaticPodsDisappear(c)
	framework.ExpectNoError(err)

	ginkgo.By("Waiting for kube-proxy DaemonSet running and ready")
	err = waitForKubeProxyDaemonSetRunning(c)
	framework.ExpectNoError(err)
}

// Teardown does nothing.
func (t *KubeProxyUpgradeTest) Teardown(f *framework.Framework) {
}

// KubeProxyDowngradeTest tests kube-proxy DaemonSet -> static pods downgrade path.
type KubeProxyDowngradeTest struct {
}

// Name returns the tracking name of the test.
func (KubeProxyDowngradeTest) Name() string { return "[sig-network] kube-proxy-downgrade" }

// Setup verifies kube-proxy DaemonSet is running before upgrade.
func (t *KubeProxyDowngradeTest) Setup(f *framework.Framework) {
	ginkgo.By("Waiting for kube-proxy DaemonSet running and ready")
	err := waitForKubeProxyDaemonSetRunning(f.ClientSet)
	framework.ExpectNoError(err)
}

// Test validates if kube-proxy is migrated from DaemonSet to static pods.
func (t *KubeProxyDowngradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	c := f.ClientSet

	// Block until upgrade is done.
	ginkgo.By("Waiting for upgrade to finish")
	<-done

	ginkgo.By("Waiting for kube-proxy DaemonSet disappear")
	err := waitForKubeProxyDaemonSetDisappear(c)
	framework.ExpectNoError(err)

	ginkgo.By("Waiting for kube-proxy static pods running and ready")
	err = waitForKubeProxyStaticPodsRunning(c)
	framework.ExpectNoError(err)
}

// Teardown does nothing.
func (t *KubeProxyDowngradeTest) Teardown(f *framework.Framework) {
}

func waitForKubeProxyStaticPodsRunning(c clientset.Interface) error {
	framework.Logf("Waiting up to %v for kube-proxy static pods running", defaultTestTimeout)

	condition := func() (bool, error) {
		pods, err := getKubeProxyStaticPods(c)
		if err != nil {
			framework.Logf("Failed to get kube-proxy static pods: %v", err)
			return false, nil
		}

		nodes, err := e2enode.GetReadySchedulableNodes(c)
		if err != nil {
			framework.Logf("Failed to get nodes: %v", err)
			return false, nil
		}

		numberSchedulableNodes := len(nodes.Items)
		numberkubeProxyPods := 0
		for _, pod := range pods.Items {
			if pod.Status.Phase == v1.PodRunning {
				numberkubeProxyPods = numberkubeProxyPods + 1
			}
		}
		if numberkubeProxyPods != numberSchedulableNodes {
			framework.Logf("Expect %v kube-proxy static pods running, got %v running, %v in total", numberSchedulableNodes, numberkubeProxyPods, len(pods.Items))
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollImmediate(5*time.Second, defaultTestTimeout, condition); err != nil {
		return fmt.Errorf("error waiting for kube-proxy static pods running: %v", err)
	}
	return nil
}

func waitForKubeProxyStaticPodsDisappear(c clientset.Interface) error {
	framework.Logf("Waiting up to %v for kube-proxy static pods disappear", defaultTestTimeout)

	condition := func() (bool, error) {
		pods, err := getKubeProxyStaticPods(c)
		if err != nil {
			framework.Logf("Failed to get kube-proxy static pods: %v", err)
			return false, nil
		}

		if len(pods.Items) != 0 {
			framework.Logf("Expect kube-proxy static pods to disappear, got %v pods", len(pods.Items))
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollImmediate(5*time.Second, defaultTestTimeout, condition); err != nil {
		return fmt.Errorf("error waiting for kube-proxy static pods disappear: %v", err)
	}
	return nil
}

func waitForKubeProxyDaemonSetRunning(c clientset.Interface) error {
	framework.Logf("Waiting up to %v for kube-proxy DaemonSet running", defaultTestTimeout)

	condition := func() (bool, error) {
		daemonSets, err := getKubeProxyDaemonSet(c)
		if err != nil {
			framework.Logf("Failed to get kube-proxy DaemonSet: %v", err)
			return false, nil
		}

		if len(daemonSets.Items) != 1 {
			framework.Logf("Expect only one kube-proxy DaemonSet, got %v", len(daemonSets.Items))
			return false, nil
		}

		nodes, err := e2enode.GetReadySchedulableNodes(c)
		if err != nil {
			framework.Logf("Failed to get nodes: %v", err)
			return false, nil
		}

		numberSchedulableNodes := len(nodes.Items)
		numberkubeProxyPods := int(daemonSets.Items[0].Status.NumberAvailable)
		if numberkubeProxyPods != numberSchedulableNodes {
			framework.Logf("Expect %v kube-proxy DaemonSet pods running, got %v", numberSchedulableNodes, numberkubeProxyPods)
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollImmediate(5*time.Second, defaultTestTimeout, condition); err != nil {
		return fmt.Errorf("error waiting for kube-proxy DaemonSet running: %v", err)
	}
	return nil
}

func waitForKubeProxyDaemonSetDisappear(c clientset.Interface) error {
	framework.Logf("Waiting up to %v for kube-proxy DaemonSet disappear", defaultTestTimeout)

	condition := func() (bool, error) {
		daemonSets, err := getKubeProxyDaemonSet(c)
		if err != nil {
			framework.Logf("Failed to get kube-proxy DaemonSet: %v", err)
			return false, nil
		}

		if len(daemonSets.Items) != 0 {
			framework.Logf("Expect kube-proxy DaemonSet to disappear, got %v DaemonSet", len(daemonSets.Items))
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollImmediate(5*time.Second, defaultTestTimeout, condition); err != nil {
		return fmt.Errorf("error waiting for kube-proxy DaemonSet disappear: %v", err)
	}
	return nil
}

func getKubeProxyStaticPods(c clientset.Interface) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{clusterComponentKey: kubeProxyLabelName}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	return c.CoreV1().Pods(metav1.NamespaceSystem).List(listOpts)
}

func getKubeProxyDaemonSet(c clientset.Interface) (*appsv1.DaemonSetList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{clusterAddonLabelKey: kubeProxyLabelName}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	return c.AppsV1().DaemonSets(metav1.NamespaceSystem).List(listOpts)
}
