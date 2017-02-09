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

package e2e

import (
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"

	. "github.com/onsi/ginkgo"
)

var etcdUpgradeTests = []upgrades.Test{
	&upgrades.ServiceUpgradeTest{},
	&upgrades.SecretUpgradeTest{},
}

var _ = framework.KubeDescribe("etcd Upgrade [Feature:EtcdUpgrade]", func() {
	f := framework.NewDefaultFramework("etcd-upgrade")

	framework.KubeDescribe("etcd upgrade", func() {
		It("should maintain a functioning cluster", func() {
			cm := chaosmonkey.New(func() {
				framework.ExpectNoError(framework.EtcdUpgrade(framework.TestContext.EtcdUpgradeStorage, framework.TestContext.EtcdUpgradeVersion))
				// check the etcd version
			})
			for _, t := range etcdUpgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.EtcdUpgrade,
				})
			}

			cm.Do()
		})
	})
})
