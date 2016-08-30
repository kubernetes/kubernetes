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

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	// parent path to yaml test manifests.
	ingressManifestPath = "test/e2e/testing-manifests/ingress"

	// timeout on a single http request.
	reqTimeout = 10 * time.Second

	// healthz port used to verify glbc restarted correctly on the master.
	glbcHealthzPort = 8086

	// On average it takes ~6 minutes for a single backend to come online in GCE.
	lbPollTimeout = 15 * time.Minute

	// General cloud resource poll timeout (eg: create static ip, firewall etc)
	cloudResourcePollTimeout = 5 * time.Minute

	// Time required by the loadbalancer to cleanup, proportional to numApps/Ing.
	lbCleanupTimeout = 5 * time.Minute
	lbPollInterval   = 30 * time.Second

	// Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMap = "ingress-uid"
	uidKey       = "uid"

	// GCE only allows names < 64 characters, and the loadbalancer controller inserts
	// a single character of padding.
	nameLenLimit = 62
)

var _ = framework.KubeDescribe("Loadbalancing: L7 [Feature:Ingress]", func() {
	defer GinkgoRecover()
	var (
		ns               string
		jig              *testJig
		conformanceTests []conformanceTests
	)
	f := framework.NewDefaultFramework("ingress")

	BeforeEach(func() {
		f.BeforeEach()
		jig = newTestJig(f.Client)
		ns = f.Namespace.Name
	})

	// Before enabling this loadbalancer test in any other test list you must
	// make sure the associated project has enough quota. At the time of this
	// writing a GCE project is allowed 3 backend services by default. This
	// test requires at least 5.
	//
	// Slow by design ~10m for each "It" block dominated by loadbalancer setup time
	// TODO: write similar tests for nginx, haproxy and AWS Ingress.
	framework.KubeDescribe("GCE [Slow] [Feature: Ingress]", func() {
		var gceController *GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &GCEIngressController{ns: ns, Project: framework.TestContext.CloudConfig.ProjectID, c: jig.client}
			gceController.init()
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				describeIng(ns)
			}
			if jig.ing == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.deleteIngress()

			By("Cleaning up cloud resources")
			cleanupGCE(gceController)
		})

		It("should conform to Ingress spec", func() {
			conformanceTests = createComformanceTests(jig, ns)
			for _, t := range conformanceTests {
				By(t.entryLog)
				t.execute()
				By(t.exitLog)
				jig.waitForIngress()
			}
		})

		It("shoud create ingress with given static-ip ", func() {
			ip := gceController.staticIP(ns)
			By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ns, ip))

			jig.createIngress(filepath.Join(ingressManifestPath, "static-ip"), ns, map[string]string{
				"kubernetes.io/ingress.global-static-ip-name": ns,
				"kubernetes.io/ingress.allow-http":            "false",
			})

			By("waiting for Ingress to come up with ip: " + ip)
			httpClient := buildInsecureClient(reqTimeout)
			ExpectNoError(pollURL(fmt.Sprintf("https://%v/", ip), "", lbPollTimeout, httpClient, false))

			By("should reject HTTP traffic")
			ExpectNoError(pollURL(fmt.Sprintf("http://%v/", ip), "", lbPollTimeout, httpClient, true))

			// TODO: uncomment the restart test once we have a way to synchronize
			// and know that the controller has resumed watching. If we delete
			// the ingress before the controller is ready we will leak.
			// By("restaring glbc")
			// restarter := NewRestartConfig(
			//	 framework.GetMasterHost(), "glbc", glbcHealthzPort, restartPollInterval, restartTimeout)
			// restarter.restart()
			// By("should continue serving on provided static-ip for 30 seconds")
			// ExpectNoError(jig.verifyURL(fmt.Sprintf("https://%v/", ip), "", 30, 1*time.Second, httpClient))
		})

		// TODO: Implement a multizone e2e that verifies traffic reaches each
		// zone based on pod labels.
	})

	// Time: borderline 5m, slow by design
	framework.KubeDescribe("Nginx [Slow] [Feature: Ingress]", func() {
		var nginxController *NginxIngressController

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing nginx controller")
			jig.class = "nginx"
			nginxController = &NginxIngressController{ns: ns, c: jig.client}

			// TODO: This test may fail on other platforms. We can simply skip it
			// but we want to allow easy testing where a user might've hand
			// configured firewalls.
			if framework.ProviderIs("gce", "gke") {
				ExpectNoError(gcloudCreate("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID, "--allow", "tcp:80,tcp:443", "--network", framework.TestContext.CloudConfig.Network))
			} else {
				framework.Logf("WARNING: Not running on GCE/GKE, cannot create firewall rules for :80, :443. Assuming traffic can reach the external ips of all nodes in cluster on those ports.")
			}

			nginxController.init()
		})

		AfterEach(func() {
			if framework.ProviderIs("gce", "gke") {
				ExpectNoError(gcloudDelete("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID))
			}
			if CurrentGinkgoTestDescription().Failed {
				describeIng(ns)
			}
			if jig.ing == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.deleteIngress()
		})

		It("should conform to Ingress spec", func() {
			conformanceTests = createComformanceTests(jig, ns)
			for _, t := range conformanceTests {
				By(t.entryLog)
				t.execute()
				By(t.exitLog)
				jig.waitForIngress()
			}
		})
	})
})
