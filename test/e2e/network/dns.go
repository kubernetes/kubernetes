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
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

const dnsTestPodHostName = "dns-querier-1"
const dnsTestServiceName = "dns-test-service"

var _ = SIGDescribe("DNS", func() {
	f := framework.NewDefaultFramework("dns")

	/*
		Release: v1.9
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
		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	// Added due to #8512. This is critical for GCE and GKE deployments.
	ginkgo.It("should provide DNS for the cluster [Provider:GCE]", func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		namesToResolve := []string{"google.com"}
		// Windows containers do not have a route to the GCE metadata server by default.
		if !framework.NodeOSDistroIs("windows") {
			namesToResolve = append(namesToResolve, "metadata")
		}

		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	// [LinuxOnly]: As Windows currently does not support resolving PQDNs.
	ginkgo.It("should resolve DNS of partial qualified names for the cluster [LinuxOnly]", func() {
		// All the names we need to be able to resolve.
		namesToResolve := []string{
			"kubernetes.default",
			"kubernetes.default.svc",
		}
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostEntries, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostEntries, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.14
		Testname: DNS, cluster
		Description: When a Pod is created, the pod MUST be able to resolve cluster dns entries such as kubernetes.default via /etc/hosts.
	*/
	framework.ConformanceIt("should provide /etc/hosts entries for the cluster [LinuxOnly]", func() {
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(nil, hostEntries, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(nil, hostEntries, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes /etc/hosts and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe /etc/hosts")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.9
		Testname: DNS, services
		Description: When a headless service is created, the service MUST be able to resolve all the required service endpoints. When the service is created, any pod in the same namespace must be able to resolve the service by all of the expected DNS names.
	*/
	framework.ConformanceIt("should provide DNS for services ", func() {
		// NOTE: This only contains the FQDN and the Host name, for testing partial name, see the test below
		// Create a test headless service.
		ginkgo.By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := e2eservice.CreateServiceSpec(dnsTestServiceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), headlessService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create headless service: %s", dnsTestServiceName)
		defer func() {
			ginkgo.By("deleting the test headless service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), headlessService.Name, metav1.DeleteOptions{})
		}()

		regularServiceName := "test-service-2"
		regularService := e2eservice.CreateServiceSpec(regularServiceName, "", false, testServiceSelector)
		regularService, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), regularService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create regular service: %s", regularServiceName)

		defer func() {
			ginkgo.By("deleting the test service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), regularService.Name, metav1.DeleteOptions{})
		}()

		// All the names we need to be able to resolve.
		// TODO: Create more endpoints and ensure that multiple A records are returned
		// for headless service.
		namesToResolve := []string{
			fmt.Sprintf("%s.%s.svc.%s", headlessService.Name, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			fmt.Sprintf("_http._tcp.%s.%s.svc.%s", headlessService.Name, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			fmt.Sprintf("_http._tcp.%s.%s.svc.%s", regularService.Name, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
		}

		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.17
		Testname: DNS, PQDN for services
		Description: Create a headless service and normal service. Both the services MUST be able to resolve partial qualified DNS entries of their service endpoints by serving A records and SRV records.
		[LinuxOnly]: As Windows currently does not support resolving PQDNs.
	*/
	framework.ConformanceIt("should resolve DNS of partial qualified names for services [LinuxOnly]", func() {
		// Create a test headless service.
		ginkgo.By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := e2eservice.CreateServiceSpec(dnsTestServiceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), headlessService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create headless service: %s", dnsTestServiceName)
		defer func() {
			ginkgo.By("deleting the test headless service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), headlessService.Name, metav1.DeleteOptions{})
		}()

		regularServiceName := "test-service-2"
		regularService := e2eservice.CreateServiceSpec(regularServiceName, "", false, testServiceSelector)
		regularService, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), regularService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create regular service: %s", regularServiceName)
		defer func() {
			ginkgo.By("deleting the test service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), regularService.Name, metav1.DeleteOptions{})
		}()

		// All the names we need to be able to resolve.
		// for headless service.
		namesToResolve := []string{
			headlessService.Name,
			fmt.Sprintf("%s.%s", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", regularService.Name, f.Namespace.Name),
		}

		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.15
		Testname: DNS, resolve the hostname
		Description: Create a headless service with label. Create a Pod with label to match service's label, with hostname and a subdomain same as service name.
		Pod MUST be able to resolve its fully qualified domain name as well as hostname by serving an A record at that name.
	*/
	framework.ConformanceIt("should provide DNS for pods for Hostname [LinuxOnly]", func() {
		// Create a test headless service.
		ginkgo.By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test-hostname-attribute": "true",
		}
		serviceName := "dns-test-service-2"
		podHostname := "dns-querier-2"
		headlessService := e2eservice.CreateServiceSpec(serviceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), headlessService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create headless service: %s", serviceName)

		defer func() {
			ginkgo.By("deleting the test headless service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), headlessService.Name, metav1.DeleteOptions{})
		}()

		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", podHostname, serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		hostNames := []string{hostFQDN, podHostname}
		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(nil, hostNames, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(nil, hostNames, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod1.ObjectMeta.Labels = testServiceSelector
		pod1.Spec.Hostname = podHostname
		pod1.Spec.Subdomain = serviceName

		validateDNSResults(f, pod1, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.15
		Testname: DNS, resolve the subdomain
		Description: Create a headless service with label. Create a Pod with label to match service's label, with hostname and a subdomain same as service name.
		Pod MUST be able to resolve its fully qualified domain name as well as subdomain by serving an A record at that name.
	*/
	framework.ConformanceIt("should provide DNS for pods for Subdomain", func() {
		// Create a test headless service.
		ginkgo.By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test-hostname-attribute": "true",
		}
		serviceName := "dns-test-service-2"
		podHostname := "dns-querier-2"
		headlessService := e2eservice.CreateServiceSpec(serviceName, "", true, testServiceSelector)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), headlessService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create headless service: %s", serviceName)

		defer func() {
			ginkgo.By("deleting the test headless service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), headlessService.Name, metav1.DeleteOptions{})
		}()

		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", podHostname, serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		subdomain := fmt.Sprintf("%s.%s.svc.%s", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		namesToResolve := []string{hostFQDN, subdomain}
		// TODO: Validate both IPv4 and IPv6 families for dual-stack
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, "", "wheezy", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, "", "jessie", f.Namespace.Name, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterIsIPv6())
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)
		pod1.ObjectMeta.Labels = testServiceSelector
		pod1.Spec.Hostname = podHostname
		pod1.Spec.Subdomain = serviceName

		validateDNSResults(f, pod1, append(wheezyFileNames, jessieFileNames...))
	})

	/*
		Release: v1.15
		Testname: DNS, for ExternalName Services
		Description: Create a service with externalName. Pod MUST be able to resolve the address for this service via CNAME. When externalName of this service is changed, Pod MUST resolve to new DNS entry for the service.
		Change the service type from externalName to ClusterIP, Pod MUST resolve DNS to the service by serving A records.
	*/
	framework.ConformanceIt("should provide DNS for ExternalName services", func() {
		// Create a test ExternalName service.
		ginkgo.By("Creating a test externalName service")
		serviceName := "dns-test-service-3"
		externalNameService := e2eservice.CreateServiceSpec(serviceName, "foo.example.com", false, nil)
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), externalNameService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create ExternalName service: %s", serviceName)

		defer func() {
			ginkgo.By("deleting the test externalName service")
			defer ginkgo.GinkgoRecover()
			f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), externalNameService.Name, metav1.DeleteOptions{})
		}()
		hostFQDN := fmt.Sprintf("%s.%s.svc.%s", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		wheezyProbeCmd, wheezyFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		validateTargetedProbeOutput(f, pod1, []string{wheezyFileName, jessieFileName}, "foo.example.com.")

		// Test changing the externalName field
		ginkgo.By("changing the externalName to bar.example.com")
		_, err = e2eservice.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.ExternalName = "bar.example.com"
		})
		framework.ExpectNoError(err, "failed to change externalName of service: %s", serviceName)
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a second pod to probe DNS")
		pod2 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		validateTargetedProbeOutput(f, pod2, []string{wheezyFileName, jessieFileName}, "bar.example.com.")

		// Test changing type from ExternalName to ClusterIP
		ginkgo.By("changing the service to type=ClusterIP")
		_, err = e2eservice.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
			}
		})
		framework.ExpectNoError(err, "failed to change service type to ClusterIP for service: %s", serviceName)
		targetRecord := "A"
		if framework.TestContext.ClusterIsIPv6() {
			targetRecord = "AAAA"
		}
		// TODO: For dual stack we can run from here two createTargetedProbeCommand()
		// one looking for an A record and another one for an AAAA record
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, targetRecord, "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, targetRecord, "jessie")
		ginkgo.By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		ginkgo.By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		ginkgo.By("creating a third pod to probe DNS")
		pod3 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, dnsTestPodHostName, dnsTestServiceName)

		svc, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Get(context.TODO(), externalNameService.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get service: %s", externalNameService.Name)

		validateTargetedProbeOutput(f, pod3, []string{wheezyFileName, jessieFileName}, svc.Spec.ClusterIP)
	})

	/*
		Release: v1.17
		Testname: DNS, custom dnsConfig
		Description: Create a Pod with DNSPolicy as None and custom DNS configuration, specifying nameservers and search path entries.
		Pod creation MUST be successful and provided DNS configuration MUST be configured in the Pod.
	*/
	framework.ConformanceIt("should support configurable pod DNS nameservers", func() {
		ginkgo.By("Creating a pod with dnsPolicy=None and customized dnsConfig...")
		testServerIP := "1.1.1.1"
		testSearchPath := "resolv.conf.local"
		testAgnhostPod := e2epod.NewAgnhostPod(f.Namespace.Name, "test-dns-nameservers", nil, nil, nil)
		testAgnhostPod.Spec.DNSPolicy = v1.DNSNone
		testAgnhostPod.Spec.DNSConfig = &v1.PodDNSConfig{
			Nameservers: []string{testServerIP},
			Searches:    []string{testSearchPath},
		}
		testAgnhostPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), testAgnhostPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod: %s", testAgnhostPod.Name)
		framework.Logf("Created pod %v", testAgnhostPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testAgnhostPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), testAgnhostPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("ginkgo.Failed to delete pod %s: %v", testAgnhostPod.Name, err)
			}
		}()
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, testAgnhostPod.Name, f.Namespace.Name, framework.PodStartTimeout)
		framework.ExpectNoError(err, "failed to wait for pod %s to be running", testAgnhostPod.Name)

		runCommand := func(arg string) string {
			cmd := []string{"/agnhost", arg}
			stdout, stderr, err := f.ExecWithOptions(framework.ExecOptions{
				Command:       cmd,
				Namespace:     f.Namespace.Name,
				PodName:       testAgnhostPod.Name,
				ContainerName: testAgnhostPod.Spec.Containers[0].Name,
				CaptureStdout: true,
				CaptureStderr: true,
			})
			framework.ExpectNoError(err, "failed to run command '/agnhost %s' on pod, stdout: %v, stderr: %v, err: %v", arg, stdout, stderr, err)
			return stdout
		}

		ginkgo.By("Verifying customized DNS suffix list is configured on pod...")
		stdout := runCommand("dns-suffix")
		if !strings.Contains(stdout, testSearchPath) {
			framework.Failf("customized DNS suffix list not found configured in pod, expected to contain: %s, got: %s", testSearchPath, stdout)
		}

		ginkgo.By("Verifying customized DNS server is configured on pod...")
		stdout = runCommand("dns-server-list")
		if !strings.Contains(stdout, testServerIP) {
			framework.Failf("customized DNS server not found in configured in pod, expected to contain: %s, got: %s", testServerIP, stdout)
		}
	})

	ginkgo.It("should support configurable pod resolv.conf", func() {
		ginkgo.By("Preparing a test DNS service with injected DNS names...")
		testInjectedIP := "1.1.1.1"
		testDNSNameShort := "notexistname"
		testSearchPath := "resolv.conf.local"
		testDNSNameFull := fmt.Sprintf("%s.%s", testDNSNameShort, testSearchPath)

		corednsConfig := generateCoreDNSConfigmap(f.Namespace.Name, map[string]string{
			testDNSNameFull: testInjectedIP,
		})
		corednsConfig, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), corednsConfig, metav1.CreateOptions{})
		framework.ExpectNoError(err, "unable to create test configMap %s", corednsConfig.Name)

		defer func() {
			framework.Logf("Deleting configmap %s...", corednsConfig.Name)
			err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), corednsConfig.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete configmap %s: %v", corednsConfig.Name)
		}()

		testServerPod := generateCoreDNSServerPod(corednsConfig)
		testServerPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), testServerPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod: %s", testServerPod.Name)
		framework.Logf("Created pod %v", testServerPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testServerPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), testServerPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("ginkgo.Failed to delete pod %s: %v", testServerPod.Name, err)
			}
		}()
		err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, testServerPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "failed to wait for pod %s to be running", testServerPod.Name)

		// Retrieve server pod IP.
		testServerPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), testServerPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %v", testServerPod.Name)
		testServerIP := testServerPod.Status.PodIP
		framework.Logf("testServerIP is %s", testServerIP)

		ginkgo.By("Creating a pod with dnsPolicy=None and customized dnsConfig...")
		testUtilsPod := e2epod.NewAgnhostPod(f.Namespace.Name, "e2e-dns-utils", nil, nil, nil)
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
		testUtilsPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), testUtilsPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod: %s", testUtilsPod.Name)
		framework.Logf("Created pod %v", testUtilsPod)
		defer func() {
			framework.Logf("Deleting pod %s...", testUtilsPod.Name)
			if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), testUtilsPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("ginkgo.Failed to delete pod %s: %v", testUtilsPod.Name, err)
			}
		}()
		err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, testUtilsPod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "failed to wait for pod %s to be running", testUtilsPod.Name)

		ginkgo.By("Verifying customized DNS option is configured on pod...")
		// TODO: Figure out a better way other than checking the actual resolv,conf file.
		cmd := []string{"cat", "/etc/resolv.conf"}
		stdout, stderr, err := f.ExecWithOptions(framework.ExecOptions{
			Command:       cmd,
			Namespace:     f.Namespace.Name,
			PodName:       testUtilsPod.Name,
			ContainerName: testUtilsPod.Spec.Containers[0].Name,
			CaptureStdout: true,
			CaptureStderr: true,
		})
		framework.ExpectNoError(err, "failed to examine resolv,conf file on pod, stdout: %v, stderr: %v, err: %v", stdout, stderr, err)
		if !strings.Contains(stdout, "ndots:2") {
			framework.Failf("customized DNS options not found in resolv.conf, got: %s", stdout)
		}

		ginkgo.By("Verifying customized name server and search path are working...")
		// Do dig on not-exist-dns-name and see if the injected DNS record is returned.
		// This verifies both:
		// - Custom search path is appended.
		// - DNS query is sent to the specified server.
		cmd = []string{"dig", "+short", "+search", testDNSNameShort}
		digFunc := func() (bool, error) {
			stdout, stderr, err := f.ExecWithOptions(framework.ExecOptions{
				Command:       cmd,
				Namespace:     f.Namespace.Name,
				PodName:       testUtilsPod.Name,
				ContainerName: testUtilsPod.Spec.Containers[0].Name,
				CaptureStdout: true,
				CaptureStderr: true,
			})
			if err != nil {
				framework.Logf("ginkgo.Failed to execute dig command, stdout:%v, stderr: %v, err: %v", stdout, stderr, err)
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
		framework.ExpectNoError(err, "failed to verify customized name server and search path")

		// TODO: Add more test cases for other DNSPolicies.
	})

})
