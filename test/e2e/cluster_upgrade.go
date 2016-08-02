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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// TODO(mikedanese): Add setup, validate, and teardown for:
//  - secrets
//  - volumes
//  - persistent volumes
var _ = framework.KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	framework.KubeDescribe("master upgrade", func() {
		It("should maintain responsive services [Feature:MasterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.MasterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})

	framework.KubeDescribe("node upgrade", func() {
		It("should maintain a functioning cluster [Feature:NodeUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.NodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceUpBeforeAndAfter(f, sem)
			})
			cm.Do()
		})

		It("should maintain responsive services [Feature:ExperimentalNodeUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.NodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})

	framework.KubeDescribe("cluster upgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.MasterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
				framework.ExpectNoError(framework.NodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceUpBeforeAndAfter(f, sem)
			})
			cm.Do()
		})

		It("should maintain responsive services [Feature:ExperimentalClusterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(framework.MasterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
				framework.ExpectNoError(framework.NodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})
})

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

func testServiceUpBeforeAndAfter(f *framework.Framework, sem *chaosmonkey.Semaphore) {
	testService(f, sem, false)
}

func testServiceRemainsUp(f *framework.Framework, sem *chaosmonkey.Semaphore) {
	testService(f, sem, true)
}

// testService is a helper for testServiceUpBeforeAndAfter and testServiceRemainsUp with a flag for testDuringDisruption
//
// TODO(ihmccreery) remove this abstraction once testServiceUpBeforeAndAfter is no longer needed, because node upgrades
// maintain a responsive service.
func testService(f *framework.Framework, sem *chaosmonkey.Semaphore, testDuringDisruption bool) {
	// Setup
	serviceName := "service-test"

	jig := NewServiceTestJig(f.Client, serviceName)
	// nodeIP := pickNodeIP(jig.Client) // for later

	By("creating a TCP service " + serviceName + " with type=LoadBalancer in namespace " + f.Namespace.Name)
	// TODO it's weird that we have to do this and then wait WaitForLoadBalancer which changes
	// tcpService.
	tcpService := jig.CreateTCPServiceOrFail(f.Namespace.Name, func(s *api.Service) {
		s.Spec.Type = api.ServiceTypeLoadBalancer
	})
	tcpService = jig.WaitForLoadBalancerOrFail(f.Namespace.Name, tcpService.Name, loadBalancerCreateTimeoutDefault)
	jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)

	// Get info to hit it with
	tcpIngressIP := getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
	svcPort := int(tcpService.Spec.Ports[0].Port)

	By("creating pod to be part of service " + serviceName)
	// TODO newRCTemplate only allows for the creation of one replica... that probably won't
	// work so well.
	jig.RunOrFail(f.Namespace.Name, nil)

	// Hit it once before considering ourselves ready
	By("hitting the pod through the service's LoadBalancer")
	jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeoutDefault)

	sem.Ready()

	if testDuringDisruption {
		// Continuous validation
		wait.Until(func() {
			By("hitting the pod through the service's LoadBalancer")
			jig.TestReachableHTTP(tcpIngressIP, svcPort, framework.Poll)
		}, framework.Poll, sem.StopCh)
	} else {
		// Block until chaosmonkey is done
		By("waiting for upgrade to finish without checking if service remains up")
		<-sem.StopCh
	}

	// Sanity check and hit it once more
	By("hitting the pod through the service's LoadBalancer")
	jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeoutDefault)
	jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)
}

func checkMasterVersion(c *client.Client, want string) error {
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

func checkNodesVersions(c *client.Client, want string) error {
	l := framework.GetReadySchedulableNodesOrDie(c)
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
