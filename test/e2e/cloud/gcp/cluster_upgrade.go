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

package gcp

import (
	"k8s.io/kubernetes/test/e2e/cloud/gcp/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/upgrades/apps"
	"k8s.io/kubernetes/test/e2e/upgrades/autoscaling"
	"k8s.io/kubernetes/test/e2e/upgrades/network"
	"k8s.io/kubernetes/test/e2e/upgrades/node"
	"k8s.io/kubernetes/test/e2e/upgrades/storage"
	"k8s.io/kubernetes/test/utils/junit"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// TODO: Those tests should be splitted by SIG and moved to SIG-owned directories,
//   however that involves also splitting the actual upgrade jobs too.
//   Figure out the eventual solution for it.
var upgradeTests = []upgrades.Test{
	&apps.DaemonSetUpgradeTest{},
	&apps.DeploymentUpgradeTest{},
	&apps.JobUpgradeTest{},
	&apps.ReplicaSetUpgradeTest{},
	&apps.StatefulSetUpgradeTest{},
	&autoscaling.HPAUpgradeTest{},
	&network.ServiceUpgradeTest{},
	&node.AppArmorUpgradeTest{},
	&node.ConfigMapUpgradeTest{},
	&node.SecretUpgradeTest{},
	&storage.PersistentVolumeUpgradeTest{},
	&storage.VolumeModeDowngradeTest{},
}

var _ = SIGDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	// Create the frameworks here because we can only create them
	// in a "Describe".
	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:MasterUpgrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Master upgrade"}
			masterUpgradeTest := &junit.TestCase{
				Name:      "[sig-cloud-provider-gcp] master-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, masterUpgradeTest)

			upgradeFunc := common.ControlPlaneUpgradeFunc(f, upgCtx, masterUpgradeTest, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})

	ginkgo.Describe("cluster upgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster upgrade"}
			clusterUpgradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterUpgradeTest)

			upgradeFunc := common.ClusterUpgradeFunc(f, upgCtx, clusterUpgradeTest, nil, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("Downgrade [Feature:Downgrade]", func() {
	f := framework.NewDefaultFramework("cluster-downgrade")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	ginkgo.Describe("cluster downgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:ClusterDowngrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster downgrade"}
			clusterDowngradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterDowngradeTest)

			upgradeFunc := common.ClusterDowngradeFunc(f, upgCtx, clusterDowngradeTest, nil, nil)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})
