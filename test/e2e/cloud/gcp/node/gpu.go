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

package node

import (
	"k8s.io/kubernetes/test/e2e/cloud/gcp/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/upgrades/node"
	"k8s.io/kubernetes/test/utils/junit"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var upgradeTests = []upgrades.Test{
	&node.NvidiaGPUUpgradeTest{},
}

var _ = SIGDescribe("gpu Upgrade [Feature:GPUUpgrade]", func() {
	f := framework.NewDefaultFramework("gpu-upgrade")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should NOT disrupt gpu pod [Feature:GPUMasterUpgrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU master upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-master-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)

			upgradeFunc := common.ControlPlaneUpgradeFunc(f, upgCtx, gpuUpgradeTest, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})
	ginkgo.Describe("cluster upgrade", func() {
		ginkgo.It("should be able to run gpu pod after upgrade [Feature:GPUClusterUpgrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)

			upgradeFunc := common.ClusterUpgradeFunc(f, upgCtx, gpuUpgradeTest, nil, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
	ginkgo.Describe("cluster downgrade", func() {
		ginkgo.It("should be able to run gpu pod after downgrade [Feature:GPUClusterDowngrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster downgrade"}
			gpuDowngradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuDowngradeTest)

			upgradeFunc := common.ClusterDowngradeFunc(f, upgCtx, gpuDowngradeTest, nil, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})
