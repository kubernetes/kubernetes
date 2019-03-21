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

package network

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const dnsTestPodHostName = "dns-querier-1"
const dnsTestServiceName = "dns-test-service"

var _ = SIGDescribe("DNS", func() {
	f := framework.NewDefaultFramework("dns")

	/*
		Release : v1.9
		Testname: DNS, cluster
		Description: When a Pod is created, the pod MUST be able to resolve cluster dns entries such as kubernetes.default via DNS.
	*/
	framework.ConformanceIt("should provide DNS for the cluster ", func() {
		// All the names we need to be able to resolve.
		// TODO: Spin up a separate test service and test that dns works for that service.
		// NOTE: This only contains the FQDN and the Host name, for testing partial name, see the test below
		namesToResolve := []string{
			fmt.Sprintf("kubernetes.default.svc.%s", framework.TestContext.ClusterDNSDomain),
		}
		// Added due to #8512. This is critical for GCE and GKE deployments.
		if framework.ProviderIs("gce", "gke") {
			namesToResolve = append(namesToResolve, "google.com")
			// Windows containers do not have a route to the GCE
			// metadata server by default.
			if !framework.NodeOSDistroIs("windows") {
				namesToResolve = append(namesToResolve, "metadata")
			}
		}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should resolve DNS of partial qualified names for the cluster ", func() {
		// All the names we need to be able to resolve.
		// TODO: Spin up a separate test service and test that dns works for that service.
		namesToResolve := []string{
			"kubernetes.default",
			"kubernetes.default.svc",
		}
		// Added due to #8512. This is critical for GCE and GKE deployments.
		if framework.ProviderIs("gce", "gke") {
			namesToResolve = append(namesToResolve, "google.com")
			// Windows containers do not have a route to the GCE
			// metadata server by default.
			if !framework.NodeOSDistroIs("windows") {
				namesToResolve = append(namesToResolve, "metadata")
			}
		}
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.cluster.local", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostEntries, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostEntries, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release : v1.14
		Testname: DNS, cluster
		Description: When a Pod is created, the pod MUST be able to resolve cluster dns entries such as kubernetes.default via /etc/hosts.
	*/
	framework.ConformanceIt("should provide /etc/hosts entries for the cluster [LinuxOnly]", func() {
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(nil, hostEntries, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(nil, hostEntries, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes /etc/hosts and exposes the results by HTTP.
		By("creating a pod to probe /etc/hosts")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release : v1.9
		Testname: DNS, services
		Description: When a headless service is created, the service MUST be able to resolve all the required service endpoints. When the service is created, any pod in the same namespace must be able to resolve the service by all of the expected DNS names.
	*/
	framework.ConformanceIt("should provide DNS for services ", func() {
		// NOTE: This only contains the FQDN and the Host name, for testing partial name, see the test below
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := framework.CreateServiceSpec(dnsTestServiceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred(), "failed to create headless service: %s", dnsTestServiceName)
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(headlessService.Name, nil)
		}()

		regularServiceName := "test-service-2"
		regularService := framework.CreateServiceSpec(regularServiceName, "", false, testServiceSelector)
		regularService, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(regularService)
		Expect(err).NotTo(HaveOccurred(), "failed to create regular service: %s", regularServiceName)

		defer func() {
			By("deleting the test service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(regularService.Name, nil)
		}()

		// All the names we need to be able to resolve.
		// TODO: Create more endpoints and ensure that multiple A records are returned
		// for headless service.
		namesToResolve := []string{
			fmt.Sprintf("%s.%s.svc.cluster.local", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc.cluster.local", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc.cluster.local", regularService.Name, f.Namespace.Name),
		}

		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should resolve DNS of partial qualified names for services ", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := framework.CreateServiceSpec(dnsTestServiceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred(), "failed to create headless service: %s", dnsTestServiceName)
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(headlessService.Name, nil)
		}()

		regularServiceName := "test-service-2"
		regularService := framework.CreateServiceSpec(regularServiceName, "", false, testServiceSelector)
		regularService, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(regularService)
		Expect(err).NotTo(HaveOccurred(), "failed to create regular service: %s", regularServiceName)
		defer func() {
			By("deleting the test service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(regularService.Name, nil)
		}()

		// All the names we need to be able to resolve.
		// TODO: Create more endpoints and ensure that multiple A records are returned
		// for headless service.
		namesToResolve := []string{
			fmt.Sprintf("%s", headlessService.Name),
			fmt.Sprintf("%s.%s", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", regularService.Name, f.Namespace.Name),
		}

		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for pods for Hostname and Subdomain", func() {
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
	})

	It("should provide DNS for ExternalName services", func() {
		// Create a test ExternalName service.
		By("Creating a test externalName service")
		serviceName := "dns-test-service-3"
		externalNameService := framework.CreateServiceSpec(serviceName, "foo.example.com", false, nil)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(externalNameService)
		Expect(err).NotTo(HaveOccurred(), "failed to create ExternalName service: %s", serviceName)

		defer func() {
			By("deleting the test externalName service")
			defer GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(externalNameService.Name, nil)
		}()
		hostFQDN := fmt.Sprintf("%s.%s.svc.%s", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		wheezyProbeCmd, wheezyFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		validateTargetedProbeOutput(f, pod1, []string{wheezyFileName, jessieFileName}, "foo.example.com.")

		// Test changing the externalName field
		By("changing the externalName to bar.example.com")
		_, err = framework.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.ExternalName = "bar.example.com"
		})
		Expect(err).NotTo(HaveOccurred(), "failed to change externalName of service: %s", serviceName)
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a second pod to probe DNS")
		pod2 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		validateTargetedProbeOutput(f, pod2, []string{wheezyFileName, jessieFileName}, "bar.example.com.")

		// Test changing type from ExternalName to ClusterIP
		By("changing the service to type=ClusterIP")
		_, err = framework.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
			}
		})
		Expect(err).NotTo(HaveOccurred(), "failed to change service type to ClusterIP for service: %s", serviceName)
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, "A", "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, "A", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a third pod to probe DNS")
		pod3 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		svc, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Get(externalNameService.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to get service: %s", externalNameService.Name)

		validateTargetedProbeOutput(f, pod3, []string{wheezyFileName, jessieFileName}, svc.Spec.ClusterIP)
	})

	It("should support configurable pod resolv.conf", func() {
		By("Preparing a test DNS service with injected DNS names...")
		testInjectedIP := "1.1.1.1"
		testDNSNameShort := "notexistname"
		testSearchPath := "resolv.conf.local"
		testDNSNameFull := fmt.Sprintf("%s.%s", testDNSNameShort, testSearchPath)

		testServerPod := generateDNSServerPod(map[string]string{
			testDNSNameFull: testInjectedIP,
		})
		testServerPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(testServerPod)
		Expect(err).NotTo(HaveOccurred(), "failed to create pod: %s", testServerPod.Name)
		framework.Logf("Created pod %v", testServerPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testServerPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(testServerPod.Name, metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("Failed to delete pod %s: %v", testServerPod.Name, err)
			}
		}()
		Expect(f.WaitForPodRunning(testServerPod.Name)).NotTo(HaveOccurred(), "failed to wait for pod %s to be running", testServerPod.Name)

		// Retrieve server pod IP.
		testServerPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(testServerPod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to get pod %v", testServerPod.Name)
		testServerIP := testServerPod.Status.PodIP
		framework.Logf("testServerIP is %s", testServerIP)

		By("Creating a pod with dnsPolicy=None and customized dnsConfig...")
		testUtilsPod := generateDNSUtilsPod()
		testUtilsPod.Spec.DNSPolicy = v1.DNSNone
		testNdotsValue := "2"
		testUtilsPod.Spec.DNSConfig = &v1.PodDNSConfig{
			Nameservers: []string{testServerIP},
			Searches:    []string{testSearchPath},
			Options: []v1.PodDNSConfigOption{
				{
					Name:  "ndots",
					Value: &testNdotsValue,
				},
			},
		}
		testUtilsPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(testUtilsPod)
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

		By("Verifying customized name server and search path are working...")
		// Do dig on not-exist-dns-name and see if the injected DNS record is returned.
		// This verifies both:
		// - Custom search path is appended.
		// - DNS query is sent to the specified server.
		cmd = []string{"/usr/bin/dig", "+short", "+search", testDNSNameShort}
		digFunc := func() (bool, error) {
			stdout, stderr, err := f.ExecWithOptions(framework.ExecOptions{
				Command:       cmd,
				Namespace:     f.Namespace.Name,
				PodName:       testUtilsPod.Name,
				ContainerName: "util",
				CaptureStdout: true,
				CaptureStderr: true,
			})
			if err != nil {
				framework.Logf("Failed to execute dig command, stdout:%v, stderr: %v, err: %v", stdout, stderr, err)
				return false, nil
			}
			res := strings.Split(stdout, "\n")
			if len(res) != 1 || res[0] != testInjectedIP {
				framework.Logf("Expect command `%v` to return %s, got: %v", cmd, testInjectedIP, res)
				return false, nil
			}
			return true, nil
		}
		err = wait.PollImmediate(5*time.Second, 3*time.Minute, digFunc)
		Expect(err).NotTo(HaveOccurred(), "failed to verify customized name server and search path")

		// TODO: Add more test cases for other DNSPolicies.
	})

})
