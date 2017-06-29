/*
Copyright 2015 The Kubernetes Authors.

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
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	rbacv1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Loadbalancing: L7", func() {
	defer GinkgoRecover()
	var (
		ns               string
		jig              *framework.IngressTestJig
		conformanceTests []framework.IngressConformanceTests
	)
	f := framework.NewDefaultFramework("ingress")

	BeforeEach(func() {
		f.BeforeEach()
		jig = framework.NewIngressTestJig(f.ClientSet)
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		framework.BindClusterRole(jig.Client.Rbac(), "cluster-admin", f.Namespace.Name,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})

		err := framework.WaitForAuthorizationUpdate(jig.Client.AuthorizationV1beta1(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			"", "create", schema.GroupResource{Resource: "pods"}, true)
		framework.ExpectNoError(err)
	})

	// Before enabling this loadbalancer test in any other test list you must
	// make sure the associated project has enough quota. At the time of this
	// writing a GCE project is allowed 3 backend services by default. This
	// test requires at least 5.
	//
	// Slow by design ~10m for each "It" block dominated by loadbalancer setup time
	// TODO: write similar tests for nginx, haproxy and AWS Ingress.
	framework.KubeDescribe("GCE [Slow] [Feature:Ingress]", func() {
		var gceController *framework.GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &framework.GCEIngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			gceController.Init()
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()

			By("Cleaning up cloud resources")
			framework.CleanupGCEIngressController(gceController)
		})

		It("should conform to Ingress spec", func() {
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns)
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(true)
			}
		})

		It("should create ingress with given static-ip", func() {
			// ip released when the rest of lb resources are deleted in CleanupGCEIngressController
			ip := gceController.CreateStaticIP(ns)
			By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ns, ip))

			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "static-ip"), ns, map[string]string{
				"kubernetes.io/ingress.global-static-ip-name": ns,
				"kubernetes.io/ingress.allow-http":            "false",
			})

			By("waiting for Ingress to come up with ip: " + ip)
			httpClient := framework.BuildInsecureClient(framework.IngressReqTimeout)
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, false))

			By("should reject HTTP traffic")
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("http://%v/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, true))

			By("should have correct firewall rule for ingress")
			fw := gceController.GetFirewallRule()
			expFw := jig.ConstructFirewallForIngress(gceController)
			// Passed the last argument as `true` to verify the backend ports is a subset
			// of the allowed ports in firewall rule, given there may be other existing
			// ingress resources and backends we are not aware of.
			Expect(framework.VerifyFirewallRule(fw, expFw, gceController.Cloud.Network, true)).NotTo(HaveOccurred())

			// TODO: uncomment the restart test once we have a way to synchronize
			// and know that the controller has resumed watching. If we delete
			// the ingress before the controller is ready we will leak.
			// By("restaring glbc")
			// restarter := NewRestartConfig(
			//	 framework.GetMasterHost(), "glbc", glbcHealthzPort, restartPollInterval, restartTimeout)
			// restarter.restart()
			// By("should continue serving on provided static-ip for 30 seconds")
			// framework.ExpectNoError(jig.verifyURL(fmt.Sprintf("https://%v/", ip), "", 30, 1*time.Second, httpClient))
		})

		// TODO: Implement a multizone e2e that verifies traffic reaches each
		// zone based on pod labels.
	})

	// Time: borderline 5m, slow by design
	framework.KubeDescribe("[Slow] Nginx", func() {
		var nginxController *framework.NginxIngressController

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing nginx controller")
			jig.Class = "nginx"
			nginxController = &framework.NginxIngressController{Ns: ns, Client: jig.Client}

			// TODO: This test may fail on other platforms. We can simply skip it
			// but we want to allow easy testing where a user might've hand
			// configured firewalls.
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(framework.GcloudComputeResourceCreate("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID, "--allow", "tcp:80,tcp:443", "--network", framework.TestContext.CloudConfig.Network))
			} else {
				framework.Logf("WARNING: Not running on GCE/GKE, cannot create firewall rules for :80, :443. Assuming traffic can reach the external ips of all nodes in cluster on those ports.")
			}

			nginxController.Init()
		})

		AfterEach(func() {
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(framework.GcloudComputeResourceDelete("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID))
			}
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()
		})

		It("should conform to Ingress spec", func() {
			// Poll more frequently to reduce e2e completion time.
			// This test runs in presubmit.
			jig.PollInterval = 5 * time.Second
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns)
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(false)
			}
		})
	})
})
