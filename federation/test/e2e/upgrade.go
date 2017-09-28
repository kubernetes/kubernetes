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

package e2e

import (
	fedframework "k8s.io/kubernetes/federation/test/e2e/framework"
	"k8s.io/kubernetes/federation/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var upgradeTests = upgrades.SimpleUpgradeTests()

var _ = framework.KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-upgrade")

	framework.KubeDescribe("Federation Control Plane upgrade", func() {
		It("should maintain a functioning federation [Feature:FCPUpgrade]", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			cm := chaosmonkey.New(func() {
				federationControlPlaneUpgrade(f)
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.FCPUpgrade,
				})
			}
			cm.Do()
		})
	})

	framework.KubeDescribe("Federated clusters upgrade", func() {
		It("should maintain a functioning federation [Feature:FederatedClustersUpgrade]", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			cm := chaosmonkey.New(func() {
				federatedClustersUpgrade(f)
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.FederatedClustersUpgrade,
				})
			}
			cm.Do()
		})
	})

	framework.KubeDescribe("FCP upgrade followed by federated clusters upgrade", func() {
		It("should maintain a functioning federation [Feature:FCPUpgradeFollowedByFederatedClustersUpgrade]", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			cm := chaosmonkey.New(func() {
				federationControlPlaneUpgrade(f)
				federatedClustersUpgrade(f)
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.FCPUpgradeFollowedByFederatedClustersUpgrade,
				})
			}
			cm.Do()
		})
	})

	framework.KubeDescribe("Federated clusters upgrade followed by FCP upgrade", func() {
		It("should maintain a functioning federation [Feature:FederatedClustersUpgradeFollowedByFCPUpgrade]", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			cm := chaosmonkey.New(func() {
				federatedClustersUpgrade(f)
				federationControlPlaneUpgrade(f)
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.FederatedClustersUpgradeFollowedByFCPUpgrade,
				})
			}
			cm.Do()
		})
	})
})

type chaosMonkeyAdapter struct {
	test        upgrades.Test
	framework   *fedframework.Framework
	upgradeType upgrades.FederationUpgradeType
}

func (cma *chaosMonkeyAdapter) Setup() {
	cma.test.Setup(cma.framework)
}

func (cma *chaosMonkeyAdapter) Test(stopCh <-chan struct{}) {
	cma.test.Test(cma.framework, stopCh, cma.upgradeType)
}

func (cma *chaosMonkeyAdapter) Teardown() {
	cma.test.Teardown(cma.framework)
}

func federationControlPlaneUpgrade(f *fedframework.Framework) {
	federationVersion, err := framework.RealVersion(framework.TestContext.FederationUpgradeTarget)
	framework.ExpectNoError(err)
	framework.ExpectNoError(fedframework.FederationControlPlaneUpgrade(federationVersion))
	framework.ExpectNoError(fedframework.CheckFederationVersion(f.FederationClientset, federationVersion))
}

func federatedClustersUpgrade(f *fedframework.Framework) {
	k8sVersion, err := framework.RealVersion(framework.TestContext.UpgradeTarget)
	framework.ExpectNoError(err)
	clusters := f.GetRegisteredClusters()
	for _, cluster := range clusters {
		framework.ExpectNoError(fedframework.MasterUpgrade(cluster.Name, k8sVersion))
		framework.ExpectNoError(framework.CheckMasterVersion(cluster.Clientset, k8sVersion))

		// TODO: Need to add Node upgrade. Add once this framework is stable
	}
}
