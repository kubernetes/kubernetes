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
	"context"

	"k8s.io/kubernetes/test/e2e/cloud/gcp/common"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
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

// TODO: Those tests should be split by SIG and moved to SIG-owned directories,
//
//	however that involves also splitting the actual upgrade jobs too.
//	Figure out the eventual solution for it.
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

var _ = SIGDescribe("Upgrade", feature.Upgrade, func() {
	f := framework.NewDefaultFramework("cluster-upgrade")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	// Create the frameworks here because we can only create them
	// in a "Describe".
	ginkgo.Describe("master upgrade", func() {
		f.It("should maintain a functioning cluster", feature.MasterUpgrade, func(ctx context.Context) {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Master upgrade"}
			masterUpgradeTest := &junit.TestCase{
				Name:      "[sig-cloud-provider-gcp] master-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, masterUpgradeTest)

			upgradeFunc := common.ControlPlaneUpgradeFunc(f, upgCtx, masterUpgradeTest, nil)
			upgrades.RunUpgradeSuite(ctx, upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})

	ginkgo.Describe("cluster upgrade", func() {
		f.It("should maintain a functioning cluster", feature.ClusterUpgrade, func(ctx context.Context) {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster upgrade"}
			clusterUpgradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterUpgradeTest)

			upgradeFunc := common.ClusterUpgradeFunc(f, upgCtx, clusterUpgradeTest, nil, nil)
			upgrades.RunUpgradeSuite(ctx, upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("Downgrade", feature.Downgrade, func() {
	f := framework.NewDefaultFramework("cluster-downgrade")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	ginkgo.Describe("cluster downgrade", func() {
		f.It("should maintain a functioning cluster", feature.ClusterDowngrade, func(ctx context.Context) {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster downgrade"}
			clusterDowngradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterDowngradeTest)

			upgradeFunc := common.ClusterDowngradeFunc(f, upgCtx, clusterDowngradeTest, nil, nil)
			upgrades.RunUpgradeSuite(ctx, upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})
