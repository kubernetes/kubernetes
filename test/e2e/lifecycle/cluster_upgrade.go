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

package lifecycle

import (
	"encoding/xml"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	"k8s.io/kubernetes/test/e2e/upgrades"
	apps "k8s.io/kubernetes/test/e2e/upgrades/apps"
	"k8s.io/kubernetes/test/e2e/upgrades/storage"
	"k8s.io/kubernetes/test/utils/junit"

	. "github.com/onsi/ginkgo"
)

var (
	upgradeTarget = flag.String("upgrade-target", "ci/latest", "Version to upgrade to (e.g. 'release/stable', 'release/latest', 'ci/latest', '0.19.1', '0.19.1-669-gabac8c8') if doing an upgrade test.")
	upgradeImage  = flag.String("upgrade-image", "", "Image to upgrade to (e.g. 'container_vm' or 'gci') if doing an upgrade test.")
)

var upgradeTests = []upgrades.Test{
	&upgrades.ServiceUpgradeTest{},
	&upgrades.SecretUpgradeTest{},
	&apps.ReplicaSetUpgradeTest{},
	&apps.StatefulSetUpgradeTest{},
	&apps.DeploymentUpgradeTest{},
	&apps.JobUpgradeTest{},
	&upgrades.ConfigMapUpgradeTest{},
	&upgrades.HPAUpgradeTest{},
	&storage.PersistentVolumeUpgradeTest{},
	&apps.DaemonSetUpgradeTest{},
	&upgrades.IngressUpgradeTest{},
	&upgrades.AppArmorUpgradeTest{},
	&storage.VolumeModeDowngradeTest{},
}

var gpuUpgradeTests = []upgrades.Test{
	&upgrades.NvidiaGPUUpgradeTest{},
}

var statefulsetUpgradeTests = []upgrades.Test{
	&upgrades.MySqlUpgradeTest{},
	&upgrades.EtcdUpgradeTest{},
	&upgrades.CassandraUpgradeTest{},
}

var kubeProxyUpgradeTests = []upgrades.Test{
	&upgrades.KubeProxyUpgradeTest{},
	&upgrades.ServiceUpgradeTest{},
	&upgrades.IngressUpgradeTest{},
}

var kubeProxyDowngradeTests = []upgrades.Test{
	&upgrades.KubeProxyDowngradeTest{},
	&upgrades.ServiceUpgradeTest{},
	&upgrades.IngressUpgradeTest{},
}

// Forcefully swap ingress image.
var ingressUpgradeTests = []upgrades.Test{
	&upgrades.IngressUpgradeTest{},
}

var _ = SIGDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(upgradeTests)
	Describe("master upgrade", func() {
		It("should maintain a functioning cluster [Feature:MasterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Master upgrade"}
			masterUpgradeTest := &junit.TestCase{
				Name:      "[sig-cluster-lifecycle] master-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, masterUpgradeTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, masterUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			}
			runUpgradeSuite(f, upgradeTests, testFrameworks, testSuite, upgCtx, upgrades.MasterUpgrade, upgradeFunc)
		})
	})

	Describe("node upgrade", func() {
		It("should maintain a functioning cluster [Feature:NodeUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Node upgrade"}
			nodeUpgradeTest := &junit.TestCase{
				Name:      "node-upgrade",
				Classname: "upgrade_tests",
			}

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, nodeUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			}
			runUpgradeSuite(f, upgradeTests, testFrameworks, testSuite, upgCtx, upgrades.NodeUpgrade, upgradeFunc)
		})
	})

	Describe("cluster upgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster upgrade"}
			clusterUpgradeTest := &junit.TestCase{Name: "[sig-cluster-lifecycle] cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterUpgradeTest)
			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, clusterUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			}
			runUpgradeSuite(f, upgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("Downgrade [Feature:Downgrade]", func() {
	f := framework.NewDefaultFramework("cluster-downgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(upgradeTests)

	Describe("cluster downgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster downgrade"}
			clusterDowngradeTest := &junit.TestCase{Name: "[sig-cluster-lifecycle] cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterDowngradeTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, clusterDowngradeTest)
				// Yes this really is a downgrade. And nodes must downgrade first.
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			}
			runUpgradeSuite(f, upgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("etcd Upgrade [Feature:EtcdUpgrade]", func() {
	f := framework.NewDefaultFramework("etc-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(upgradeTests)
	Describe("etcd upgrade", func() {
		It("should maintain a functioning cluster", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), "")
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Etcd upgrade"}
			etcdTest := &junit.TestCase{Name: "[sig-cluster-lifecycle] etcd-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, etcdTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, etcdTest)
				framework.ExpectNoError(framework.EtcdUpgrade(framework.TestContext.EtcdUpgradeStorage, framework.TestContext.EtcdUpgradeVersion))
			}
			runUpgradeSuite(f, upgradeTests, testFrameworks, testSuite, upgCtx, upgrades.EtcdUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("ingress Upgrade [Feature:IngressUpgrade]", func() {
	f := framework.NewDefaultFramework("ingress-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(ingressUpgradeTests)
	Describe("ingress upgrade", func() {
		It("should maintain a functioning ingress", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), "")
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "ingress upgrade"}
			ingressTest := &junit.TestCase{Name: "[sig-networking] ingress-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, ingressTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, ingressTest)
				framework.ExpectNoError(framework.IngressUpgrade(true))
			}
			runUpgradeSuite(f, ingressUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.IngressUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("ingress Downgrade [Feature:IngressDowngrade]", func() {
	f := framework.NewDefaultFramework("ingress-downgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(ingressUpgradeTests)
	Describe("ingress downgrade", func() {
		It("should maintain a functioning ingress", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), "")
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "ingress downgrade"}
			ingressTest := &junit.TestCase{Name: "[sig-networking] ingress-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, ingressTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, ingressTest)
				framework.ExpectNoError(framework.IngressUpgrade(false))
			}
			runUpgradeSuite(f, ingressUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.IngressUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("gpu Upgrade [Feature:GPUUpgrade]", func() {
	f := framework.NewDefaultFramework("gpu-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(gpuUpgradeTests)
	Describe("master upgrade", func() {
		It("should NOT disrupt gpu pod [Feature:GPUMasterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU master upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-master-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)
			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, gpuUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			}
			runUpgradeSuite(f, gpuUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.MasterUpgrade, upgradeFunc)
		})
	})
	Describe("cluster upgrade", func() {
		It("should be able to run gpu pod after upgrade [Feature:GPUClusterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)
			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, gpuUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			}
			runUpgradeSuite(f, gpuUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
	Describe("cluster downgrade", func() {
		It("should be able to run gpu pod after downgrade [Feature:GPUClusterDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster downgrade"}
			gpuDowngradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuDowngradeTest)
			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, gpuDowngradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			}
			runUpgradeSuite(f, gpuUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = Describe("[sig-apps] stateful Upgrade [Feature:StatefulUpgrade]", func() {
	f := framework.NewDefaultFramework("stateful-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	testFrameworks := createUpgradeFrameworks(statefulsetUpgradeTests)
	framework.KubeDescribe("stateful upgrade", func() {
		It("should maintain a functioning cluster", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Stateful upgrade"}
			statefulUpgradeTest := &junit.TestCase{Name: "[sig-apps] stateful-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, statefulUpgradeTest)
			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, statefulUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgrade(target))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
				framework.ExpectNoError(framework.NodeUpgrade(f, target, *upgradeImage))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			}
			runUpgradeSuite(f, statefulsetUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = SIGDescribe("kube-proxy migration [Feature:KubeProxyDaemonSetMigration]", func() {
	f := framework.NewDefaultFramework("kube-proxy-ds-migration")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
	})

	Describe("Upgrade kube-proxy from static pods to a DaemonSet", func() {
		testFrameworks := createUpgradeFrameworks(kubeProxyUpgradeTests)

		It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy upgrade"}
			kubeProxyUpgradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyUpgradeTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, kubeProxyUpgradeTest)
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.MasterUpgradeGCEWithKubeProxyDaemonSet(target, true))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
				framework.ExpectNoError(framework.NodeUpgradeGCEWithKubeProxyDaemonSet(f, target, *upgradeImage, true))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
			}
			runUpgradeSuite(f, kubeProxyUpgradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})

	Describe("Downgrade kube-proxy from a DaemonSet to static pods", func() {
		testFrameworks := createUpgradeFrameworks(kubeProxyDowngradeTests)

		It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy downgrade"}
			kubeProxyDowngradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-downgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyDowngradeTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, kubeProxyDowngradeTest)
				// Yes this really is a downgrade. And nodes must downgrade first.
				target := upgCtx.Versions[1].Version.String()
				framework.ExpectNoError(framework.NodeUpgradeGCEWithKubeProxyDaemonSet(f, target, *upgradeImage, false))
				framework.ExpectNoError(framework.CheckNodesVersions(f.ClientSet, target))
				framework.ExpectNoError(framework.MasterUpgradeGCEWithKubeProxyDaemonSet(target, false))
				framework.ExpectNoError(framework.CheckMasterVersion(f.ClientSet, target))
			}
			runUpgradeSuite(f, kubeProxyDowngradeTests, testFrameworks, testSuite, upgCtx, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

type chaosMonkeyAdapter struct {
	test        upgrades.Test
	testReport  *junit.TestCase
	framework   *framework.Framework
	upgradeType upgrades.UpgradeType
	upgCtx      upgrades.UpgradeContext
}

func (cma *chaosMonkeyAdapter) Test(sem *chaosmonkey.Semaphore) {
	start := time.Now()
	var once sync.Once
	ready := func() {
		once.Do(func() {
			sem.Ready()
		})
	}
	defer finalizeUpgradeTest(start, cma.testReport)
	defer ready()
	if skippable, ok := cma.test.(upgrades.Skippable); ok && skippable.Skip(cma.upgCtx) {
		By("skipping test " + cma.test.Name())
		cma.testReport.Skipped = "skipping test " + cma.test.Name()
		return
	}

	defer cma.test.Teardown(cma.framework)
	cma.test.Setup(cma.framework)
	ready()
	cma.test.Test(cma.framework, sem.StopCh, cma.upgradeType)
}

func finalizeUpgradeTest(start time.Time, tc *junit.TestCase) {
	tc.Time = time.Since(start).Seconds()
	r := recover()
	if r == nil {
		return
	}

	switch r := r.(type) {
	case ginkgowrapper.FailurePanic:
		tc.Failures = []*junit.Failure{
			{
				Message: r.Message,
				Type:    "Failure",
				Value:   fmt.Sprintf("%s\n\n%s", r.Message, r.FullStackTrace),
			},
		}
	case ginkgowrapper.SkipPanic:
		tc.Skipped = fmt.Sprintf("%s:%d %q", r.Filename, r.Line, r.Message)
	default:
		tc.Errors = []*junit.Error{
			{
				Message: fmt.Sprintf("%v", r),
				Type:    "Panic",
				Value:   fmt.Sprintf("%v", r),
			},
		}
	}
}

func createUpgradeFrameworks(tests []upgrades.Test) map[string]*framework.Framework {
	nsFilter := regexp.MustCompile("[^[:word:]-]+") // match anything that's not a word character or hyphen
	testFrameworks := map[string]*framework.Framework{}
	for _, t := range tests {
		ns := nsFilter.ReplaceAllString(t.Name(), "-") // and replace with a single hyphen
		ns = strings.Trim(ns, "-")
		testFrameworks[t.Name()] = framework.NewDefaultFramework(ns)
	}
	return testFrameworks
}

func runUpgradeSuite(
	f *framework.Framework,
	tests []upgrades.Test,
	testFrameworks map[string]*framework.Framework,
	testSuite *junit.TestSuite,
	upgCtx *upgrades.UpgradeContext,
	upgradeType upgrades.UpgradeType,
	upgradeFunc func(),
) {
	upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget)
	framework.ExpectNoError(err)

	cm := chaosmonkey.New(upgradeFunc)
	for _, t := range tests {
		testCase := &junit.TestCase{
			Name:      t.Name(),
			Classname: "upgrade_tests",
		}
		testSuite.TestCases = append(testSuite.TestCases, testCase)
		cma := chaosMonkeyAdapter{
			test:        t,
			testReport:  testCase,
			framework:   testFrameworks[t.Name()],
			upgradeType: upgradeType,
			upgCtx:      *upgCtx,
		}
		cm.Register(cma.Test)
	}

	start := time.Now()
	defer func() {
		testSuite.Update()
		testSuite.Time = time.Since(start).Seconds()
		if framework.TestContext.ReportDir != "" {
			fname := filepath.Join(framework.TestContext.ReportDir, fmt.Sprintf("junit_%supgrades.xml", framework.TestContext.ReportPrefix))
			f, err := os.Create(fname)
			if err != nil {
				return
			}
			defer f.Close()
			xml.NewEncoder(f).Encode(testSuite)
		}
	}()
	cm.Do()
}

func getUpgradeContext(c discovery.DiscoveryInterface, upgradeTarget string) (*upgrades.UpgradeContext, error) {
	current, err := c.ServerVersion()
	if err != nil {
		return nil, err
	}

	curVer, err := version.ParseSemantic(current.String())
	if err != nil {
		return nil, err
	}

	upgCtx := &upgrades.UpgradeContext{
		Versions: []upgrades.VersionContext{
			{
				Version:   *curVer,
				NodeImage: framework.TestContext.NodeOSDistro,
			},
		},
	}

	if len(upgradeTarget) == 0 {
		return upgCtx, nil
	}

	next, err := framework.RealVersion(upgradeTarget)
	if err != nil {
		return nil, err
	}

	nextVer, err := version.ParseSemantic(next)
	if err != nil {
		return nil, err
	}

	upgCtx.Versions = append(upgCtx.Versions, upgrades.VersionContext{
		Version:   *nextVer,
		NodeImage: *upgradeImage,
	})

	return upgCtx, nil
}
