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

package windows

import (
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const dnsTestPodHostName = "dns-querier-1"
const dnsTestServiceName = "dns-test-service"

var _ = SIGDescribe("DNS", func() {
	f := framework.NewDefaultFramework("dns")

	/*It("should provide DNS for pods for Hostname and Subdomain", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test-hostname-attribute": "true",
		}
		serviceName := "dns-test-service-2"
		podHostname := "dns-querier-2"
		headlessService := framework.CreateServiceSpec(serviceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred(), "failed to create headless service: %s", serviceName)

		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(headlessService.Name, nil)
		}()

		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", podHostname, serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		hostNames := []string{hostFQDN, podHostname}
		namesToResolve := []string{hostFQDN}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostNames, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostNames, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod1.ObjectMeta.Labels = testServiceSelector
		pod1.Spec.Hostname = podHostname
		pod1.Spec.Subdomain = serviceName

		validateDNSResults(f, pod1, append(wheezyFileNames, jessieFileNames...))
	})*/

	It("should support configurable pod resolv.conf", func() {
		By("Preparing a test DNS service with injected DNS names...")
		testInjectedIP := "1.1.1.1"
		testSearchPath := "resolv.conf.local"

		By("Creating a pod with dnsPolicy=None and customized dnsConfig...")
		testUtilsPod := generateDNSUtilsPod()
		testUtilsPod.Spec.DNSPolicy = v1.DNSNone
		testNdotsValue := "2"
		testUtilsPod.Spec.DNSConfig = &v1.PodDNSConfig{
			Nameservers: []string{testInjectedIP},
			Searches:    []string{testSearchPath},
			Options: []v1.PodDNSConfigOption{
				{
					Name:  "ndots",
					Value: &testNdotsValue,
				},
			},
		}
		testUtilsPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(testUtilsPod)
		Expect(err).NotTo(HaveOccurred(), "failed to create pod: %s", testUtilsPod.Name)
		framework.Logf("Created pod %v", testUtilsPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testUtilsPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(testUtilsPod.Name, metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("Failed to delete pod %s: %v", testUtilsPod.Name, err)
			}
		}()
		Expect(f.WaitForPodRunning(testUtilsPod.Name)).NotTo(HaveOccurred(), "failed to wait for pod %s to be running", testUtilsPod.Name)

		By("Verifying customized DNS option is configured on pod...")
		// TODO: Figure out a better way other than checking the actual resolv,conf file.
		cmd := []string{"cat", "/etc/resolv.conf"}
		stdout, stderr, err := f.ExecWithOptions(framework.ExecOptions{
			Command:       cmd,
			Namespace:     f.Namespace.Name,
			PodName:       testUtilsPod.Name,
			ContainerName: "util",
			CaptureStdout: true,
			CaptureStderr: true,
		})
		Expect(err).NotTo(HaveOccurred(), "failed to examine resolv,conf file on pod, stdout: %v, stderr: %v, err: %v", stdout, stderr, err)
		if !strings.Contains(stdout, "ndots:2") {
			framework.Failf("customized DNS options not found in resolv.conf, got: %s", stdout)
		}
		// TODO: Add more test cases for other DNSPolicies.
	})
})
