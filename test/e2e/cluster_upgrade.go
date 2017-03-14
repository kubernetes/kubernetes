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
	"k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"

	. "github.com/onsi/ginkgo"
)

var upgradeTests = []upgrades.Test{
	&upgrades.ServiceUpgradeTest{},
	&upgrades.SecretUpgradeTest{},
	&upgrades.StatefulSetUpgradeTest{},
	&upgrades.DeploymentUpgradeTest{},
	&upgrades.JobUpgradeTest{},
	&upgrades.ConfigMapUpgradeTest{},
	&upgrades.HPAUpgradeTest{},
	&upgrades.PersistentVolumeUpgradeTest{},
	&upgrades.DaemonSetUpgradeTest{},
	&upgrades.IngressUpgradeTest{},
	&upgrades.AppArmorUpgradeTest{},
}

var _ = framework.KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := map[string]*framework.Framework{}
	for _, t := range upgradeTests {
		testFrameworks[t.Name()] = framework.NewDefaultFramework(t.Name())
	}

	framework.KubeDescribe("master upgrade", func() {
		It("should maintain a functioning cluster [Feature:MasterUpgrade]", func() {
			target, err := framework.RealVersion(framework.TestContext.UpgradeTarget)
			framework.ExpectNoError(err)

			current, err := f.ClientSet.Discovery().ServerVersion()
			framework.ExpectNoError(err)

			versions, err := parseVersions(target, current.String())
			framework.ExpectNoError(err)

			cm := chaosmonkey.New(func() {
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			})
			for _, t := range upgradeTests {
				cma := chaosMonkeyAdapter{
					test:        t,
					framework:   testFrameworks[t.Name()],
					upgradeType: upgrades.MasterUpgrade,
					versions:    versions,
				}
				cm.Register(cma.Test)
			}

			cm.Do()
		})
	})

	framework.KubeDescribe("node upgrade", func() {
		It("should maintain a functioning cluster [Feature:NodeUpgrade]", func() {
			target, err := framework.RealVersion(framework.TestContext.UpgradeTarget)
			framework.ExpectNoError(err)

			current, err := f.ClientSet.Discovery().ServerVersion()
			framework.ExpectNoError(err)

			versions, err := parseVersions(target, current.String())
			framework.ExpectNoError(err)

			cm := chaosmonkey.New(func() {
				framework.ExpectNoError(framework.NodeUpgrade(f, target, framework.TestContext.UpgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			})
			for _, t := range upgradeTests {
				cma := chaosMonkeyAdapter{
					test:        t,
					framework:   testFrameworks[t.Name()],
					upgradeType: upgrades.NodeUpgrade,
					versions:    versions,
				}
				cm.Register(cma.Test)
			}
			cm.Do()
		})
	})

	framework.KubeDescribe("cluster upgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			target, err := framework.RealVersion(framework.TestContext.UpgradeTarget)
			framework.ExpectNoError(err)

			current, err := f.ClientSet.Discovery().ServerVersion()
			framework.ExpectNoError(err)

			versions, err := parseVersions(target, current.String())
			framework.ExpectNoError(err)

			cm := chaosmonkey.New(func() {
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
				framework.ExpectNoError(framework.NodeUpgrade(f, target, framework.TestContext.UpgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			})
			for _, t := range upgradeTests {
				cma := chaosMonkeyAdapter{
					test:        t,
					framework:   testFrameworks[t.Name()],
					upgradeType: upgrades.ClusterUpgrade,
					versions:    versions,
				}
				cm.Register(cma.Test)
			}
			cm.Do()
		})
	})
})

var _ = framework.KubeDescribe("Downgrade [Feature:Downgrade]", func() {
	f := framework.NewDefaultFramework("cluster-downgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := map[string]*framework.Framework{}
	for _, t := range upgradeTests {
		testFrameworks[t.Name()] = framework.NewDefaultFramework(t.Name())
	}

	framework.KubeDescribe("cluster downgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterDowngrade]", func() {
			target, err := framework.RealVersion(framework.TestContext.UpgradeTarget)
			framework.ExpectNoError(err)

			current, err := f.ClientSet.Discovery().ServerVersion()
			framework.ExpectNoError(err)

			versions, err := parseVersions(target, current.String())
			framework.ExpectNoError(err)

			cm := chaosmonkey.New(func() {
				// Yes this really is a downgrade. And nodes must downgrade first.
				framework.ExpectNoError(framework.NodeUpgrade(f, target, framework.TestContext.UpgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			})
			for _, t := range upgradeTests {
				cma := chaosMonkeyAdapter{
					test:        t,
					framework:   testFrameworks[t.Name()],
					upgradeType: upgrades.ClusterUpgrade,
					versions:    versions,
				}
				cm.Register(cma.Test)
			}
			cm.Do()
		})
	})
})

var _ = framework.KubeDescribe("etcd Upgrade [Feature:EtcdUpgrade]", func() {
	f := framework.NewDefaultFramework("etc-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := map[string]*framework.Framework{}
	for _, t := range upgradeTests {
		testFrameworks[t.Name()] = framework.NewDefaultFramework(t.Name())
	}

	framework.KubeDescribe("etcd upgrade", func() {
		It("should maintain a functioning cluster", func() {
			current, err := f.ClientSet.Discovery().ServerVersion()
			framework.ExpectNoError(err)

			versions, err := parseVersions(current.String())
			framework.ExpectNoError(err)

			cm := chaosmonkey.New(func() {
				framework.ExpectNoError(framework.EtcdUpgrade(framework.TestContext.EtcdUpgradeStorage, framework.TestContext.EtcdUpgradeVersion))
				// TODO(mml): verify the etcd version
			})
			for _, t := range upgradeTests {
				cma := chaosMonkeyAdapter{
					test:        t,
					framework:   testFrameworks[t.Name()],
					upgradeType: upgrades.EtcdUpgrade,
					versions:    versions,
				}
				cm.Register(cma.Test)
			}

			cm.Do()
		})
	})
})

type chaosMonkeyAdapter struct {
	test        upgrades.Test
	framework   *framework.Framework
	upgradeType upgrades.UpgradeType
	versions    []version.Version
}

func (cma *chaosMonkeyAdapter) Test(sem *chaosmonkey.Semaphore) {
	if skippable, ok := cma.test.(upgrades.VersionSkippable); ok && skippable.SkipVersions(cma.versions...) {
		By("skipping test " + cma.test.Name())
		sem.Ready()
		return
	}

	cma.test.Setup(cma.framework)
	defer cma.test.Teardown(cma.framework)
	sem.Ready()
	cma.test.Test(cma.framework, sem.StopCh, cma.upgradeType)
}

func parseVersions(vs ...string) ([]version.Version, error) {
	var ret []version.Version
	for _, v := range vs {
		semver, err := version.ParseSemantic(v)
		if err != nil {
			return nil, err
		}
		ret = append(ret, *semver)
	}
	return ret, nil
}
