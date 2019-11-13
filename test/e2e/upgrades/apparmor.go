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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2esecurity "k8s.io/kubernetes/test/e2e/framework/security"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
)

// AppArmorUpgradeTest tests that AppArmor profiles are enforced & usable across upgrades.
type AppArmorUpgradeTest struct {
	pod *v1.Pod
}

// Name returns the tracking name of the test.
func (AppArmorUpgradeTest) Name() string { return "apparmor-upgrade" }

// Skip returns true when this test can be skipped.
func (AppArmorUpgradeTest) Skip(upgCtx UpgradeContext) bool {
	supportedImages := make(map[string]bool)
	for _, d := range framework.AppArmorDistros {
		supportedImages[d] = true
	}

	for _, vCtx := range upgCtx.Versions {
		if !supportedImages[vCtx.NodeImage] {
			return true
		}
	}
	return false
}

// Setup creates a secret and then verifies that a pod can consume it.
func (t *AppArmorUpgradeTest) Setup(f *framework.Framework) {
	ginkgo.By("Loading AppArmor profiles to nodes")
	e2esecurity.LoadAppArmorProfiles(f.Namespace.Name, f.ClientSet)

	// Create the initial test pod.
	ginkgo.By("Creating a long-running AppArmor enabled pod.")
	t.pod = e2esecurity.CreateAppArmorTestPod(f.Namespace.Name, f.ClientSet, f.PodClient(), false, false)

	// Verify initial state.
	t.verifyNodesAppArmorEnabled(f)
	t.verifyNewPodSucceeds(f)
}

// Test waits for the upgrade to complete, and then verifies that a
// pod can still consume the secret.
func (t *AppArmorUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	if upgrade == MasterUpgrade {
		t.verifyPodStillUp(f)
	}
	t.verifyNodesAppArmorEnabled(f)
	t.verifyNewPodSucceeds(f)
}

// Teardown cleans up any remaining resources.
func (t *AppArmorUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
	ginkgo.By("Logging container failures")
	e2ekubectl.LogFailedContainers(f.ClientSet, f.Namespace.Name, framework.Logf)
}

func (t *AppArmorUpgradeTest) verifyPodStillUp(f *framework.Framework) {
	ginkgo.By("Verifying an AppArmor profile is continuously enforced for a pod")
	pod, err := f.PodClient().Get(t.pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Should be able to get pod")
	framework.ExpectEqual(pod.Status.Phase, v1.PodRunning, "Pod should stay running")
	gomega.Expect(pod.Status.ContainerStatuses[0].State.Running).NotTo(gomega.BeNil(), "Container should be running")
	gomega.Expect(pod.Status.ContainerStatuses[0].RestartCount).To(gomega.BeZero(), "Container should not need to be restarted")
}

func (t *AppArmorUpgradeTest) verifyNewPodSucceeds(f *framework.Framework) {
	ginkgo.By("Verifying an AppArmor profile is enforced for a new pod")
	e2esecurity.CreateAppArmorTestPod(f.Namespace.Name, f.ClientSet, f.PodClient(), false, true)
}

func (t *AppArmorUpgradeTest) verifyNodesAppArmorEnabled(f *framework.Framework) {
	ginkgo.By("Verifying nodes are AppArmor enabled")
	nodes, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to list nodes")
	for _, node := range nodes.Items {
		gomega.Expect(node.Status.Conditions).To(gstruct.MatchElements(conditionType, gstruct.IgnoreExtras, gstruct.Elements{
			"Ready": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Message": gomega.ContainSubstring("AppArmor enabled"),
			}),
		}))
	}
}

func conditionType(condition interface{}) string {
	return string(condition.(v1.NodeCondition).Type)
}
