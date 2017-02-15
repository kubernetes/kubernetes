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

package upgrades

/*

Use this test in order to test any upgrade tests you're writing without having
to actually go through the upgrade process. This test performs a null-op for
the upgrade. To run this test (assuming you have a cluster set up and a built
e2e test binary):

go run hack/e2e.go -- -v --test --test_args="--ginkgo.focus=\[Feature:MockUpgrade\]"

*/

import (
	"time"

	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var upgradeTests = []Test{
	&ServiceUpgradeTest{},
	&SecretUpgradeTest{},
	&DeploymentUpgradeTest{},
	&ConfigMapUpgradeTest{},
	&JobUpgradeTest{},
	&DaemonSetUpgradeTest{},
}

var _ = framework.KubeDescribe("MockUpgrade [Feature:MockUpgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	framework.KubeDescribe("mock upgrade", func() {
		It("should maintain a functioning cluster [Feature:MockUpgrade]", func() {
			Skip("Don't run me by default.")
			cm := chaosmonkey.New(func() { time.Sleep(30 * time.Second) })
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: MasterUpgrade,
				})
			}

			cm.Do()
		})
	})
})

type chaosMonkeyAdapter struct {
	test        Test
	framework   *framework.Framework
	upgradeType UpgradeType
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
