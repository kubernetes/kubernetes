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
	api "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
)

// AppArmorUpgradeTest tests that AppArmor profiles are enforced & usable across upgrades.
type AppArmorUpgradeTest struct {
	pod *api.Pod
}

func (AppArmorUpgradeTest) Name() string { return "apparmor-upgrade" }

func (AppArmorUpgradeTest) Skip(upgCtx UpgradeContext) bool {
	supportedImages := make(map[string]bool)
	for _, d := range common.AppArmorDistros {
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
	By("Loading AppArmor profiles to nodes")
	common.LoadAppArmorProfiles(f)

	// Create the initial test pod.
	By("Creating a long-running AppArmor enabled pod.")
	t.pod = common.CreateAppArmorTestPod(f, false)

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
	By("Logging container failures")
	framework.LogFailedContainers(f.ClientSet, f.Namespace.Name, framework.Logf)
}

func (t *AppArmorUpgradeTest) verifyPodStillUp(f *framework.Framework) {
	By("Verifying an AppArmor profile is continuously enforced for a pod")
	pod, err := f.PodClient().Get(t.pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Should be able to get pod")
	Expect(pod.Status.Phase).To(Equal(api.PodRunning), "Pod should stay running")
	Expect(pod.Status.ContainerStatuses[0].State.Running).NotTo(BeNil(), "Container should be running")
	Expect(pod.Status.ContainerStatuses[0].RestartCount).To(BeZero(), "Container should not need to be restarted")
}

func (t *AppArmorUpgradeTest) verifyNewPodSucceeds(f *framework.Framework) {
	By("Verifying an AppArmor profile is enforced for a new pod")
	common.CreateAppArmorTestPod(f, true)
}

func (t *AppArmorUpgradeTest) verifyNodesAppArmorEnabled(f *framework.Framework) {
	By("Verifying nodes are AppArmor enabled")
	nodes, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "Failed to list nodes")
	for _, node := range nodes.Items {
		Expect(node.Status.Conditions).To(gstruct.MatchElements(conditionType, gstruct.IgnoreExtras, gstruct.Elements{
			"Ready": gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Message": ContainSubstring("AppArmor enabled"),
			}),
		}))
	}
}

func conditionType(condition interface{}) string {
	return string(condition.(api.NodeCondition).Type)
}
