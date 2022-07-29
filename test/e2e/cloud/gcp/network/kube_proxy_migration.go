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

package network

import (
	"fmt"

	"k8s.io/kubernetes/test/e2e/cloud/gcp/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/upgrades/network"
	"k8s.io/kubernetes/test/utils/junit"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var upgradeTests = []upgrades.Test{
	&network.KubeProxyUpgradeTest{},
	&network.ServiceUpgradeTest{},
}

var downgradeTests = []upgrades.Test{
	&network.KubeProxyDowngradeTest{},
	&network.ServiceUpgradeTest{},
}

func kubeProxyDaemonSetExtraEnvs(enableKubeProxyDaemonSet bool) []string {
	return []string{fmt.Sprintf("KUBE_PROXY_DAEMONSET=%v", enableKubeProxyDaemonSet)}
}

var _ = SIGDescribe("kube-proxy migration [Feature:KubeProxyDaemonSetMigration]", func() {
	f := framework.NewDefaultFramework("kube-proxy-ds-migration")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	upgradeTestFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)
	downgradeTestsFrameworks := upgrades.CreateUpgradeFrameworks(downgradeTests)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	ginkgo.Describe("Upgrade kube-proxy from static pods to a DaemonSet", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetUpgrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy upgrade"}
			kubeProxyUpgradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyUpgradeTest)

			extraEnvs := kubeProxyDaemonSetExtraEnvs(true)
			upgradeFunc := common.ClusterUpgradeFunc(f, upgCtx, kubeProxyUpgradeTest, extraEnvs, extraEnvs)
			upgrades.RunUpgradeSuite(upgCtx, upgradeTests, upgradeTestFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})

	ginkgo.Describe("Downgrade kube-proxy from a DaemonSet to static pods", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetDowngrade]", func() {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy downgrade"}
			kubeProxyDowngradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-downgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyDowngradeTest)

			extraEnvs := kubeProxyDaemonSetExtraEnvs(false)
			upgradeFunc := common.ClusterDowngradeFunc(f, upgCtx, kubeProxyDowngradeTest, extraEnvs, extraEnvs)
			upgrades.RunUpgradeSuite(upgCtx, downgradeTests, downgradeTestsFrameworks, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})
