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

package apps

import (
	"context"
	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
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

	t.daemonSet = e2edaemonset.NewDaemonSet(daemonSetName, image, labelSet, nil, nil, []v1.ContainerPort{{ContainerPort: 9376}}, "serve-hostname")
	t.daemonSet.Spec.Template.Spec.Tolerations = []v1.Toleration{
		{Operator: v1.TolerationOpExists},
	}

	ginkgo.By("Creating a DaemonSet")
	var err error
	if t.daemonSet, err = f.ClientSet.AppsV1().DaemonSets(ns.Name).Create(context.TODO(), t.daemonSet, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test DaemonSet %s: %v", t.daemonSet.Name, err)
	}

	ginkgo.By("Waiting for DaemonSet pods to become ready")
	err = wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		return e2edaemonset.CheckRunningOnAllNodes(f, t.daemonSet)
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
	res, err := e2edaemonset.CheckRunningOnAllNodes(f, t.daemonSet)
	framework.ExpectNoError(err)
	if !res {
		framework.Failf("expected DaemonSet pod to be running on all nodes, it was not")
	}

	// DaemonSet resource itself should be good
	ginkgo.By("confirming the DaemonSet resource is in a good state")
	err = e2edaemonset.CheckDaemonStatus(f, t.daemonSet.Name)
	framework.ExpectNoError(err)
}
