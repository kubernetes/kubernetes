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
	"encoding/xml"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eginkgowrapper "k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/upgrades/apps"
	"k8s.io/kubernetes/test/e2e/upgrades/auth"
	"k8s.io/kubernetes/test/e2e/upgrades/autoscaling"
	"k8s.io/kubernetes/test/e2e/upgrades/network"
	"k8s.io/kubernetes/test/e2e/upgrades/node"
	"k8s.io/kubernetes/test/e2e/upgrades/storage"
	"k8s.io/kubernetes/test/utils/junit"

	"github.com/onsi/ginkgo"
)

var (
	upgradeTarget = e2econfig.Flags.String("upgrade-target", "ci/latest", "Version to upgrade to (e.g. 'release/stable', 'release/latest', 'ci/latest', '0.19.1', '0.19.1-669-gabac8c8') if doing an upgrade test.")
	upgradeImage  = e2econfig.Flags.String("upgrade-image", "", "Image to upgrade to (e.g. 'container_vm' or 'gci') if doing an upgrade test.")
)

const etcdImage = "3.4.9-1"

var upgradeTests = []upgrades.Test{
	&network.ServiceUpgradeTest{},
	&node.SecretUpgradeTest{},
	&apps.ReplicaSetUpgradeTest{},
	&apps.StatefulSetUpgradeTest{},
	&apps.DeploymentUpgradeTest{},
	&apps.JobUpgradeTest{},
	&node.ConfigMapUpgradeTest{},
	&autoscaling.HPAUpgradeTest{},
	&storage.PersistentVolumeUpgradeTest{},
	&apps.DaemonSetUpgradeTest{},
	&node.AppArmorUpgradeTest{},
	&storage.VolumeModeDowngradeTest{},
}

var gpuUpgradeTests = []upgrades.Test{
	&node.NvidiaGPUUpgradeTest{},
}

var statefulsetUpgradeTests = []upgrades.Test{
	&apps.MySQLUpgradeTest{},
	&apps.EtcdUpgradeTest{},
	&apps.CassandraUpgradeTest{},
}

var kubeProxyUpgradeTests = []upgrades.Test{
	&network.KubeProxyUpgradeTest{},
	&network.ServiceUpgradeTest{},
}

var kubeProxyDowngradeTests = []upgrades.Test{
	&network.KubeProxyDowngradeTest{},
	&network.ServiceUpgradeTest{},
}

var serviceaccountAdmissionControllerMigrationTests = []upgrades.Test{
	&auth.ServiceAccountAdmissionControllerMigrationTest{},
}

func kubeProxyDaemonSetExtraEnvs(enableKubeProxyDaemonSet bool) []string {
	return []string{fmt.Sprintf("KUBE_PROXY_DAEMONSET=%v", enableKubeProxyDaemonSet)}
}

// TODO(#98326): Split the test by SIGs, move to appropriate directories and use SIGDescribe.
var _ = ginkgo.Describe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	// Create the frameworks here because we can only create them
	// in a "Describe".
	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:MasterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Master upgrade"}
			masterUpgradeTest := &junit.TestCase{
				Name:      "[sig-cloud-provider-gcp] master-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, masterUpgradeTest, nil)

			upgradeFunc := ControlPlaneUpgradeFunc(f, upgCtx, masterUpgradeTest, nil)
			runUpgradeSuite(upgCtx, upgradeTests, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})

	ginkgo.Describe("cluster upgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster upgrade"}
			clusterUpgradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterUpgradeTest)

			upgradeFunc := ClusterUpgradeFunc(f, upgCtx, clusterUpgradeTest, nil, nil)
			runUpgradeSuite(upgCtx, upgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

// TODO(#98326): Split the test by SIGs, move to appropriate directories and use SIGDescribe.
var _ = ginkgo.Describe("Downgrade [Feature:Downgrade]", func() {
	f := framework.NewDefaultFramework("cluster-downgrade")

	ginkgo.Describe("cluster downgrade", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:ClusterDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Cluster downgrade"}
			clusterDowngradeTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, clusterDowngradeTest)

			upgradeFunc := ClusterDowngradeFunc(f, upgCtx, clusterDowngradeTest, nil, nil)
			runUpgradeSuite(upgCtx, upgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

var _ = ginkgo.Describe("etcd Upgrade [Feature:EtcdUpgrade]", func() {
	f := framework.NewDefaultFramework("etc-upgrade")

	ginkgo.Describe("etcd upgrade", func() {
		ginkgo.It("should maintain a functioning cluster", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Etcd upgrade"}
			etcdTest := &junit.TestCase{Name: "[sig-cloud-provider-gcp] etcd-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, etcdTest)

			upgradeFunc := func() {
				start := time.Now()
				defer finalizeUpgradeTest(start, etcdTest)
				framework.ExpectNoError(framework.EtcdUpgrade(framework.TestContext.EtcdUpgradeStorage, framework.TestContext.EtcdUpgradeVersion))
			}
			runUpgradeSuite(upgCtx, upgradeTests, testSuite, upgrades.EtcdUpgrade, upgradeFunc)
		})
	})
})

// TODO(#98326): Split the test by SIGs, move to appropriate directories and use SIGDescribe.
var _ = ginkgo.Describe("gpu Upgrade [Feature:GPUUpgrade]", func() {
	f := framework.NewDefaultFramework("gpu-upgrade")

	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should NOT disrupt gpu pod [Feature:GPUMasterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU master upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-master-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)

			upgradeFunc := ControlPlaneUpgradeFunc(f, upgCtx, gpuUpgradeTest, nil)
			runUpgradeSuite(upgCtx, gpuUpgradeTests, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})
	ginkgo.Describe("cluster upgrade", func() {
		ginkgo.It("should be able to run gpu pod after upgrade [Feature:GPUClusterUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster upgrade"}
			gpuUpgradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuUpgradeTest)

			upgradeFunc := ClusterUpgradeFunc(f, upgCtx, gpuUpgradeTest, nil, nil)
			runUpgradeSuite(upgCtx, gpuUpgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
	ginkgo.Describe("cluster downgrade", func() {
		ginkgo.It("should be able to run gpu pod after downgrade [Feature:GPUClusterDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "GPU cluster downgrade"}
			gpuDowngradeTest := &junit.TestCase{Name: "[sig-node] gpu-cluster-downgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, gpuDowngradeTest)

			upgradeFunc := ClusterDowngradeFunc(f, upgCtx, gpuDowngradeTest, nil, nil)
			runUpgradeSuite(upgCtx, gpuUpgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

// TODO(#98326): Move the test to sig-aps dir and use SIGDescribe.
var _ = ginkgo.Describe("[sig-apps] stateful Upgrade [Feature:StatefulUpgrade]", func() {
	f := framework.NewDefaultFramework("stateful-upgrade")

	ginkgo.Describe("stateful upgrade", func() {
		ginkgo.It("should maintain a functioning cluster", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "Stateful upgrade"}
			statefulUpgradeTest := &junit.TestCase{Name: "[sig-apps] stateful-upgrade", Classname: "upgrade_tests"}
			testSuite.TestCases = append(testSuite.TestCases, statefulUpgradeTest)

			upgradeFunc := ClusterUpgradeFunc(f, upgCtx, statefulUpgradeTest, nil, nil)
			runUpgradeSuite(upgCtx, statefulsetUpgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

// TODO(#98326): Move the test to sig-network dir and use SIGDescribe.
var _ = ginkgo.Describe("kube-proxy migration [Feature:KubeProxyDaemonSetMigration]", func() {
	f := framework.NewDefaultFramework("kube-proxy-ds-migration")

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	ginkgo.Describe("Upgrade kube-proxy from static pods to a DaemonSet", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetUpgrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy upgrade"}
			kubeProxyUpgradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-upgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyUpgradeTest)

			extraEnvs := kubeProxyDaemonSetExtraEnvs(true)
			upgradeFunc := ClusterUpgradeFunc(f, upgCtx, kubeProxyUpgradeTest, extraEnvs, extraEnvs)
			runUpgradeSuite(upgCtx, kubeProxyUpgradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})

	ginkgo.Describe("Downgrade kube-proxy from a DaemonSet to static pods", func() {
		ginkgo.It("should maintain a functioning cluster [Feature:KubeProxyDaemonSetDowngrade]", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "kube-proxy downgrade"}
			kubeProxyDowngradeTest := &junit.TestCase{
				Name:      "kube-proxy-ds-downgrade",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, kubeProxyDowngradeTest)

			extraEnvs := kubeProxyDaemonSetExtraEnvs(false)
			upgradeFunc := ClusterDowngradeFunc(f, upgCtx, kubeProxyDowngradeTest, extraEnvs, extraEnvs)
			runUpgradeSuite(upgCtx, kubeProxyDowngradeTests, testSuite, upgrades.ClusterUpgrade, upgradeFunc)
		})
	})
})

// TODO(#98326): Move the test to sig-auth dir and use SIGDescribe.
var _ = ginkgo.Describe("[sig-auth] ServiceAccount admission controller migration [Feature:BoundServiceAccountTokenVolume]", func() {
	f := framework.NewDefaultFramework("serviceaccount-admission-controller-migration")

	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should maintain a functioning cluster", func() {
			upgCtx, err := getUpgradeContext(f.ClientSet.Discovery(), *upgradeTarget, *upgradeImage)
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "ServiceAccount admission controller migration"}
			serviceaccountAdmissionControllerMigrationTest := &junit.TestCase{
				Name:      "[sig-auth] serviceaccount-admission-controller-migration",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, serviceaccountAdmissionControllerMigrationTest)

			extraEnvs := []string{"KUBE_FEATURE_GATES=BoundServiceAccountTokenVolume=true"}
			upgradeFunc := ControlPlaneUpgradeFunc(f, upgCtx, serviceaccountAdmissionControllerMigrationTest, extraEnvs)
			runUpgradeSuite(upgCtx, serviceaccountAdmissionControllerMigrationTests, testSuite, upgrades.MasterUpgrade, upgradeFunc)
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
		ginkgo.By("skipping test " + cma.test.Name())
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
	case e2eginkgowrapper.FailurePanic:
		tc.Failures = []*junit.Failure{
			{
				Message: r.Message,
				Type:    "Failure",
				Value:   fmt.Sprintf("%s\n\n%s", r.Message, r.FullStackTrace),
			},
		}
	case e2eskipper.SkipPanic:
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
	upgCtx *upgrades.UpgradeContext,
	tests []upgrades.Test,
	testSuite *junit.TestSuite,
	upgradeType upgrades.UpgradeType,
	upgradeFunc func(),
) {
	testFrameworks := createUpgradeFrameworks(tests)

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

func getUpgradeContext(c discovery.DiscoveryInterface, upgradeTarget, upgradeImage string) (*upgrades.UpgradeContext, error) {
	current, err := c.ServerVersion()
	if err != nil {
		return nil, err
	}

	curVer, err := utilversion.ParseSemantic(current.String())
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

	next, err := realVersion(upgradeTarget)
	if err != nil {
		return nil, err
	}

	nextVer, err := utilversion.ParseSemantic(next)
	if err != nil {
		return nil, err
	}

	upgCtx.Versions = append(upgCtx.Versions, upgrades.VersionContext{
		Version:   *nextVer,
		NodeImage: upgradeImage,
	})

	return upgCtx, nil
}

// realVersion turns a version constants into a version string deployable on
// GKE.  See hack/get-build.sh for more information.
func realVersion(s string) (string, error) {
	framework.Logf("Getting real version for %q", s)
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/get-build.sh"), "-v", s)
	if err != nil {
		return v, fmt.Errorf("error getting real version for %q: %v", s, err)
	}
	framework.Logf("Version for %q is %q", s, v)
	return strings.TrimPrefix(strings.TrimSpace(v), "v"), nil
}
