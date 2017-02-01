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
	"fmt"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"

	. "github.com/onsi/ginkgo"
)

var upgradeTests = []upgrades.Test{
	&upgrades.ServiceUpgradeTest{},
	&upgrades.SecretUpgradeTest{},
	&upgrades.DeploymentUpgradeTest{},
}

var _ = framework.KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	framework.KubeDescribe("master upgrade", func() {
		It("should maintain a functioning cluster [Feature:MasterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.MasterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.ClientSet, v))
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.MasterUpgrade,
				})
			}

			cm.Do()
		})
	})

	framework.KubeDescribe("node upgrade", func() {
		It("should maintain a functioning cluster [Feature:NodeUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.NodeUpgrade(f, v, framework.TestContext.UpgradeImage))
				framework.ExpectNoError(checkNodesVersions(f.ClientSet, v))
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.NodeUpgrade,
				})
			}
			cm.Do()
		})
	})

	framework.KubeDescribe("cluster upgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.MasterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.ClientSet, v))
				framework.ExpectNoError(framework.NodeUpgrade(f, v, framework.TestContext.UpgradeImage))
				framework.ExpectNoError(checkNodesVersions(f.ClientSet, v))
			})
			for _, t := range upgradeTests {
				cm.RegisterInterface(&chaosMonkeyAdapter{
					test:        t,
					framework:   f,
					upgradeType: upgrades.ClusterUpgrade,
				})
			}
			cm.Do()
		})
	})
})

type chaosMonkeyAdapter struct {
	test        upgrades.Test
	framework   *framework.Framework
	upgradeType upgrades.UpgradeType
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

// realVersion turns a version constant s into a version string deployable on
// GKE.  See hack/get-build.sh for more information.
func realVersion(s string) (string, error) {
	framework.Logf(fmt.Sprintf("Getting real version for %q", s))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/get-build.sh"), "-v", s)
	if err != nil {
		return v, err
	}
	framework.Logf("Version for %q is %q", s, v)
	return strings.TrimPrefix(strings.TrimSpace(v), "v"), nil
}

func checkMasterVersion(c clientset.Interface, want string) error {
	framework.Logf("Checking master version")
	v, err := c.Discovery().ServerVersion()
	if err != nil {
		return fmt.Errorf("checkMasterVersion() couldn't get the master version: %v", err)
	}
	// We do prefix trimming and then matching because:
	// want looks like:  0.19.3-815-g50e67d4
	// got  looks like: v0.19.3-815-g50e67d4034e858-dirty
	got := strings.TrimPrefix(v.GitVersion, "v")
	if !strings.HasPrefix(got, want) {
		return fmt.Errorf("master had kube-apiserver version %s which does not start with %s",
			got, want)
	}
	framework.Logf("Master is at version %s", want)
	return nil
}

func checkNodesVersions(cs clientset.Interface, want string) error {
	l := framework.GetReadySchedulableNodesOrDie(cs)
	for _, n := range l.Items {
		// We do prefix trimming and then matching because:
		// want   looks like:  0.19.3-815-g50e67d4
		// kv/kvp look  like: v0.19.3-815-g50e67d4034e858-dirty
		kv, kpv := strings.TrimPrefix(n.Status.NodeInfo.KubeletVersion, "v"),
			strings.TrimPrefix(n.Status.NodeInfo.KubeProxyVersion, "v")
		if !strings.HasPrefix(kv, want) {
			return fmt.Errorf("node %s had kubelet version %s which does not start with %s",
				n.ObjectMeta.Name, kv, want)
		}
		if !strings.HasPrefix(kpv, want) {
			return fmt.Errorf("node %s had kube-proxy version %s which does not start with %s",
				n.ObjectMeta.Name, kpv, want)
		}
	}
	return nil
}
