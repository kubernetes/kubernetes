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

package upgrades

import (
	"fmt"
	"net/http"
	"path/filepath"

	. "github.com/onsi/ginkgo"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// IngressUpgradeTest adapts the Ingress e2e for upgrade testing
type IngressUpgradeTest struct {
	gceController *framework.GCEIngressController
	jig           *framework.IngressTestJig
	httpClient    *http.Client
	ip            string
	ipName        string
}

func (IngressUpgradeTest) Name() string { return "ingress-upgrade" }

// Setup creates a GLBC, allocates an ip, and an ingress resource,
// then waits for a successful connectivity check to the ip.
func (t *IngressUpgradeTest) Setup(f *framework.Framework) {
	framework.SkipUnlessProviderIs("gce", "gke")

	// jig handles all Kubernetes testing logic
	jig := framework.NewIngressTestJig(f.ClientSet)

	ns := f.Namespace

	// gceController handles all cloud testing logic
	gceController := &framework.GCEIngressController{
		Ns:     ns.Name,
		Client: jig.Client,
		Cloud:  framework.TestContext.CloudConfig,
	}
	gceController.Init()

	t.gceController = gceController
	t.jig = jig
	t.httpClient = framework.BuildInsecureClient(framework.IngressReqTimeout)

	// Allocate a static-ip for the Ingress, this IP is cleaned up via CleanupGCEIngressController
	t.ipName = fmt.Sprintf("%s-static-ip", ns.Name)
	t.ip = t.gceController.CreateStaticIP(t.ipName)

	// Create a working basic Ingress
	By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", t.ipName, t.ip))
	jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "static-ip"), ns.Name, map[string]string{
		"kubernetes.io/ingress.global-static-ip-name": t.ipName,
		"kubernetes.io/ingress.allow-http":            "false",
	})

	By("waiting for Ingress to come up with ip: " + t.ip)
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/", t.ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, t.httpClient, false))
}

// Test waits for the upgrade to complete, and then verifies
// with a connectvity check to the loadbalancer ip.
func (t *IngressUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	switch upgrade {
	case MasterUpgrade:
		// Restarting the ingress controller shouldn't disrupt a steady state
		// Ingress. Restarting the ingress controller and deleting ingresses
		// while it's down will leak cloud resources, because the ingress
		// controller doesn't checkpoint to disk.
		t.verify(f, done, true)
	default:
		// Currently ingress gets disrupted across node upgrade, because endpoints
		// get killed and we don't have any guarantees that 2 nodes don't overlap
		// their upgrades (even on cloud platforms like GCE, because VM level
		// rolling upgrades are not Kubernetes aware).
		t.verify(f, done, false)
	}
}

// Teardown cleans up any remaining resources.
func (t *IngressUpgradeTest) Teardown(f *framework.Framework) {
	if CurrentGinkgoTestDescription().Failed {
		framework.DescribeIng(t.gceController.Ns)
	}
	if t.jig.Ingress != nil {
		By("Deleting ingress")
		t.jig.TryDeleteIngress()
	} else {
		By("No ingress created, no cleanup necessary")
	}

	By("Cleaning up cloud resources")
	framework.CleanupGCEIngressController(t.gceController)
}

func (t *IngressUpgradeTest) verify(f *framework.Framework, done <-chan struct{}, testDuringDisruption bool) {
	if testDuringDisruption {
		By("continuously hitting the Ingress IP")
		wait.Until(func() {
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/", t.ip), "", framework.LoadBalancerPollTimeout, t.jig.PollInterval, t.httpClient, false))
		}, t.jig.PollInterval, done)
	} else {
		By("waiting for upgrade to finish without checking if Ingress remains up")
		<-done
	}
	By("hitting the Ingress IP " + t.ip)
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/", t.ip), "", framework.LoadBalancerPollTimeout, t.jig.PollInterval, t.httpClient, false))
}
