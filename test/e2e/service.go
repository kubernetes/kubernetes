/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/intstr"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Maximum time a kube-proxy daemon on a node is allowed to not
// notice a Service update, such as type=NodePort.
// TODO: This timeout should be O(10s), observed values are O(1m), 5m is very
// liberal. Fix tracked in #20567.
const kubeProxyLagTimeout = 5 * time.Minute

// Maximum time a load balancer is allowed to not respond after creation.
const loadBalancerLagTimeoutDefault = 2 * time.Minute

// On AWS there is a delay between ELB creation and serving traffic;
// a few minutes is typical, so use 10m.
const loadBalancerLagTimeoutAWS = 10 * time.Minute

// How long to wait for a load balancer to be created/modified.
//TODO: once support ticket 21807001 is resolved, reduce this timeout back to something reasonable
const loadBalancerCreateTimeout = 20 * time.Minute

// This should match whatever the default/configured range is
var ServiceNodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

var _ = framework.KubeDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")

	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = framework.LoadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	// TODO: We get coverage of TCP/UDP and multi-port services through the DNS test. We should have a simpler test for multi-port TCP here.

	It("should provide secure master service [Conformance]", func() {
		_, err := c.Services(api.NamespaceDefault).Get("kubernetes")
		Expect(err).NotTo(HaveOccurred())
	})

	It("should serve a basic endpoint from pods [Conformance]", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "endpoint-test2"
		ns := f.Namespace.Name
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}

		By("creating service " + serviceName + " in namespace " + ns)
		defer func() {
			err := c.Services(ns).Delete(serviceName)
			Expect(err).NotTo(HaveOccurred())
		}()

		service := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: serviceName,
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{{
					Port:       80,
					TargetPort: intstr.FromInt(80),
				}},
			},
		}
		_, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())

		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{})

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := c.Pods(ns).Delete(name, nil)
				Expect(err).NotTo(HaveOccurred())
			}
		}()

		name1 := "pod1"
		name2 := "pod2"

		createPodOrFail(c, ns, name1, labels, []api.ContainerPort{{ContainerPort: 80}})
		names[name1] = true
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{name1: {80}})

		createPodOrFail(c, ns, name2, labels, []api.ContainerPort{{ContainerPort: 80}})
		names[name2] = true
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{name1: {80}, name2: {80}})

		deletePodOrFail(c, ns, name1)
		delete(names, name1)
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{name2: {80}})

		deletePodOrFail(c, ns, name2)
		delete(names, name2)
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{})
	})

	It("should serve multiport endpoints from pods [Conformance]", func() {
		// TODO: use the ServiceTestJig here
		// repacking functionality is intentionally not tested here - it's better to test it in an integration test.
		serviceName := "multi-endpoint-test"
		ns := f.Namespace.Name

		defer func() {
			err := c.Services(ns).Delete(serviceName)
			Expect(err).NotTo(HaveOccurred())
		}()

		labels := map[string]string{"foo": "bar"}

		svc1port := "svc1"
		svc2port := "svc2"

		By("creating service " + serviceName + " in namespace " + ns)
		service := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: serviceName,
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{
					{
						Name:       "portname1",
						Port:       80,
						TargetPort: intstr.FromString(svc1port),
					},
					{
						Name:       "portname2",
						Port:       81,
						TargetPort: intstr.FromString(svc2port),
					},
				},
			},
		}
		_, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())
		port1 := 100
		port2 := 101
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{})

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := c.Pods(ns).Delete(name, nil)
				Expect(err).NotTo(HaveOccurred())
			}
		}()

		containerPorts1 := []api.ContainerPort{
			{
				Name:          svc1port,
				ContainerPort: int32(port1),
			},
		}
		containerPorts2 := []api.ContainerPort{
			{
				Name:          svc2port,
				ContainerPort: int32(port2),
			},
		}

		podname1 := "pod1"
		podname2 := "pod2"

		createPodOrFail(c, ns, podname1, labels, containerPorts1)
		names[podname1] = true
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{podname1: {port1}})

		createPodOrFail(c, ns, podname2, labels, containerPorts2)
		names[podname2] = true
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{podname1: {port1}, podname2: {port2}})

		deletePodOrFail(c, ns, podname1)
		delete(names, podname1)
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{podname2: {port2}})

		deletePodOrFail(c, ns, podname2)
		delete(names, podname2)
		validateEndpointsOrFail(c, ns, serviceName, PortsByPodName{})
	})

	It("should be able to up and down services", func() {
		// TODO: use the ServiceTestJig here
		// this test uses framework.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		ns := f.Namespace.Name
		numPods, servicePort := 3, 80

		By("creating service1 in namespace " + ns)
		podNames1, svc1IP, err := startServeHostnameService(c, ns, "service1", servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())
		By("creating service2 in namespace " + ns)
		podNames2, svc2IP, err := startServeHostnameService(c, ns, "service2", servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		hosts, err := framework.NodeSSHHosts(c)
		Expect(err).NotTo(HaveOccurred())
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		By("verifying service1 is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))

		By("verifying service2 is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))

		// Stop service 1 and make sure it is gone.
		By("stopping service1")
		framework.ExpectNoError(stopServeHostnameService(c, ns, "service1"))

		By("verifying service1 is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(c, host, svc1IP, servicePort))
		By("verifying service2 is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))

		// Start another service and verify both are up.
		By("creating service3 in namespace " + ns)
		podNames3, svc3IP, err := startServeHostnameService(c, ns, "service3", servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		if svc2IP == svc3IP {
			framework.Failf("service IPs conflict: %v", svc2IP)
		}

		By("verifying service2 is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))

		By("verifying service3 is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames3, svc3IP, servicePort))
	})

	It("should work after restarting kube-proxy [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		framework.SkipUnlessProviderIs("gce", "gke")

		ns := f.Namespace.Name
		numPods, servicePort := 3, 80

		svc1 := "service1"
		svc2 := "service2"

		defer func() { framework.ExpectNoError(stopServeHostnameService(c, ns, svc1)) }()
		podNames1, svc1IP, err := startServeHostnameService(c, ns, svc1, servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		defer func() { framework.ExpectNoError(stopServeHostnameService(c, ns, svc2)) }()
		podNames2, svc2IP, err := startServeHostnameService(c, ns, svc2, servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}

		hosts, err := framework.NodeSSHHosts(c)
		Expect(err).NotTo(HaveOccurred())
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))

		By("Restarting kube-proxy")
		if err := framework.RestartKubeProxy(host); err != nil {
			framework.Failf("error restarting kube-proxy: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))

		By("Removing iptable rules")
		result, err := framework.SSH(`
			sudo iptables -t nat -F KUBE-SERVICES || true;
			sudo iptables -t nat -F KUBE-PORTALS-HOST || true;
			sudo iptables -t nat -F KUBE-PORTALS-CONTAINER || true`, host, framework.TestContext.Provider)
		if err != nil || result.Code != 0 {
			framework.LogSSHResult(result)
			framework.Failf("couldn't remove iptable rules: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))
	})

	It("should work after restarting apiserver [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		// TODO: framework.RestartApiserver doesn't work in GKE - fix it and reenable this test.
		framework.SkipUnlessProviderIs("gce")

		ns := f.Namespace.Name
		numPods, servicePort := 3, 80

		defer func() { framework.ExpectNoError(stopServeHostnameService(c, ns, "service1")) }()
		podNames1, svc1IP, err := startServeHostnameService(c, ns, "service1", servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		hosts, err := framework.NodeSSHHosts(c)
		Expect(err).NotTo(HaveOccurred())
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))

		// Restart apiserver
		if err := framework.RestartApiserver(); err != nil {
			framework.Failf("error restarting apiserver: %v", err)
		}
		if err := framework.WaitForApiserverUp(c); err != nil {
			framework.Failf("error while waiting for apiserver up: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))

		// Create a new service and check if it's not reusing IP.
		defer func() { framework.ExpectNoError(stopServeHostnameService(c, ns, "service2")) }()
		podNames2, svc2IP, err := startServeHostnameService(c, ns, "service2", servicePort, numPods)
		Expect(err).NotTo(HaveOccurred())

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(c, ns, host, podNames2, svc2IP, servicePort))
	})

	// TODO: Run this test against the userspace proxy and nodes
	// configured with a default deny firewall to validate that the
	// proxy whitelists NodePort traffic.
	It("should be able to create a functioning NodePort service", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name

		jig := NewServiceTestJig(c, serviceName)
		nodeIP := pickNodeIP(jig.Client) // for later

		By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		service := jig.CreateTCPServiceOrFail(ns, func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeNodePort
		})
		jig.SanityCheckService(service, api.ServiceTypeNodePort)
		nodePort := int(service.Spec.Ports[0].NodePort)

		By("creating pod to be part of service " + serviceName)
		jig.RunOrFail(ns, nil)

		By("hitting the pod through the service's NodePort")
		jig.TestReachableHTTP(nodeIP, nodePort, kubeProxyLagTimeout)

		By("verifying the node port is locked")
		hostExec := framework.LaunchHostExecPod(f.Client, f.Namespace.Name, "hostexec")
		// Even if the node-ip:node-port check above passed, this hostexec pod
		// might fall on a node with a laggy kube-proxy.
		cmd := fmt.Sprintf(`for i in $(seq 1 300); do if ss -ant46 'sport = :%d' | grep ^LISTEN; then exit 0; fi; sleep 1; done; exit 1`, nodePort)
		stdout, err := framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
		if err != nil {
			framework.Failf("expected node port %d to be in use, stdout: %v. err: %v", nodePort, stdout, err)
		}
	})

	It("should be able to change the type and ports of a service [Slow]", func() {
		// requires cloud load-balancer support
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerSupportsUDP := !framework.ProviderIs("aws")

		loadBalancerLagTimeout := loadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = loadBalancerLagTimeoutAWS
		}

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		By("creating a second namespace")
		namespacePtr, err := f.CreateNamespace("services", nil)
		Expect(err).NotTo(HaveOccurred())
		ns2 := namespacePtr.Name // LB2 in ns2 on UDP
		framework.Logf("namespace for UDP test: %s", ns2)

		jig := NewServiceTestJig(c, serviceName)
		nodeIP := pickNodeIP(jig.Client) // for later

		// Test TCP and UDP Services.  Services with the same name in different
		// namespaces should get different node ports and load balancers.

		By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpService := jig.CreateTCPServiceOrFail(ns1, nil)
		jig.SanityCheckService(tcpService, api.ServiceTypeClusterIP)

		By("creating a UDP service " + serviceName + " with type=ClusterIP in namespace " + ns2)
		udpService := jig.CreateUDPServiceOrFail(ns2, nil)
		jig.SanityCheckService(udpService, api.ServiceTypeClusterIP)

		By("verifying that TCP and UDP use the same port")
		if tcpService.Spec.Ports[0].Port != udpService.Spec.Ports[0].Port {
			framework.Failf("expected to use the same port for TCP and UDP")
		}
		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port (TCP and UDP): %d", svcPort)

		By("creating a pod to be part of the TCP service " + serviceName)
		jig.RunOrFail(ns1, nil)

		By("creating a pod to be part of the UDP service " + serviceName)
		jig.RunOrFail(ns2, nil)

		// Change the services to NodePort.

		By("changing the TCP service to type=NodePort")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *api.Service) {
			s.Spec.Type = api.ServiceTypeNodePort
		})
		jig.SanityCheckService(tcpService, api.ServiceTypeNodePort)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		framework.Logf("TCP node port: %d", tcpNodePort)

		By("changing the UDP service to type=NodePort")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *api.Service) {
			s.Spec.Type = api.ServiceTypeNodePort
		})
		jig.SanityCheckService(udpService, api.ServiceTypeNodePort)
		udpNodePort := int(udpService.Spec.Ports[0].NodePort)
		framework.Logf("UDP node port: %d", udpNodePort)

		By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, kubeProxyLagTimeout)

		By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, kubeProxyLagTimeout)

		// Change the services to LoadBalancer.

		requestedIP := ""
		staticIPName := ""
		if framework.ProviderIs("gce", "gke") {
			By("creating a static load balancer IP")
			staticIPName = fmt.Sprintf("e2e-external-lb-test-%s", framework.RunId)
			requestedIP, err = createGCEStaticIP(staticIPName)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				if staticIPName != "" {
					// Release GCE static IP - this is not kube-managed and will not be automatically released.
					if err := deleteGCEStaticIP(staticIPName); err != nil {
						framework.Logf("failed to release static IP %s: %v", staticIPName, err)
					}
				}
			}()
			framework.Logf("Allocated static load balancer IP: %s", requestedIP)
		}

		By("changing the TCP service to type=LoadBalancer")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *api.Service) {
			s.Spec.LoadBalancerIP = requestedIP // will be "" if not applicable
			s.Spec.Type = api.ServiceTypeLoadBalancer
		})

		if loadBalancerSupportsUDP {
			By("changing the UDP service to type=LoadBalancer")
			udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
			})
		}

		By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService = jig.WaitForLoadBalancerOrFail(ns1, tcpService.Name)
		jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", tcpNodePort, tcpService.Spec.Ports[0].NodePort)
		}
		if requestedIP != "" && getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != requestedIP {
			framework.Failf("unexpected TCP Status.LoadBalancer.Ingress (expected %s, got %s)", requestedIP, getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		tcpIngressIP := getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

		if framework.ProviderIs("gce", "gke") {
			// Do this as early as possible, which overrides the `defer` above.
			// This is mostly out of fear of leaking the IP in a timeout case
			// (as of this writing we're not 100% sure where the leaks are
			// coming from, so this is first-aid rather than surgery).
			By("demoting the static IP to ephemeral")
			if staticIPName != "" {
				// Deleting it after it is attached "demotes" it to an
				// ephemeral IP, which can be auto-released.
				if err := deleteGCEStaticIP(staticIPName); err != nil {
					framework.Failf("failed to release static IP %s: %v", staticIPName, err)
				}
				staticIPName = ""
			}
		}

		var udpIngressIP string
		if loadBalancerSupportsUDP {
			By("waiting for the UDP service to have a load balancer")
			// 2nd one should be faster since they ran in parallel.
			udpService = jig.WaitForLoadBalancerOrFail(ns2, udpService.Name)
			jig.SanityCheckService(udpService, api.ServiceTypeLoadBalancer)
			if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
				framework.Failf("UDP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", udpNodePort, udpService.Spec.Ports[0].NodePort)
			}
			udpIngressIP = getIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
			framework.Logf("UDP load balancer: %s", udpIngressIP)

			By("verifying that TCP and UDP use different load balancers")
			if tcpIngressIP == udpIngressIP {
				framework.Failf("Load balancers are not different: %s", getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
			}
		}

		By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, kubeProxyLagTimeout)

		By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, kubeProxyLagTimeout)

		By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' node ports.

		By("changing the TCP service's NodePort")
		tcpService = jig.ChangeServiceNodePortOrFail(ns1, tcpService.Name, tcpNodePort)
		jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)
		tcpNodePortOld := tcpNodePort
		tcpNodePort = int(tcpService.Spec.Ports[0].NodePort)
		if tcpNodePort == tcpNodePortOld {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) did not change", tcpNodePort)
		}
		if getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("TCP node port: %d", tcpNodePort)

		By("changing the UDP service's NodePort")
		udpService = jig.ChangeServiceNodePortOrFail(ns2, udpService.Name, udpNodePort)
		if loadBalancerSupportsUDP {
			jig.SanityCheckService(udpService, api.ServiceTypeLoadBalancer)
		} else {
			jig.SanityCheckService(udpService, api.ServiceTypeNodePort)
		}
		udpNodePortOld := udpNodePort
		udpNodePort = int(udpService.Spec.Ports[0].NodePort)
		if udpNodePort == udpNodePortOld {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) did not change", udpNodePort)
		}
		if loadBalancerSupportsUDP && getIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, getIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("UDP node port: %d", udpNodePort)

		By("hitting the TCP service's new NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, kubeProxyLagTimeout)

		By("hitting the UDP service's new NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, kubeProxyLagTimeout)

		By("checking the old TCP NodePort is closed")
		jig.TestNotReachableHTTP(nodeIP, tcpNodePortOld, kubeProxyLagTimeout)

		By("checking the old UDP NodePort is closed")
		jig.TestNotReachableUDP(nodeIP, udpNodePortOld, kubeProxyLagTimeout)

		By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' main ports.

		By("changing the TCP service's port")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *api.Service) {
			s.Spec.Ports[0].Port++
		})
		jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)
		svcPortOld := svcPort
		svcPort = int(tcpService.Spec.Ports[0].Port)
		if svcPort == svcPortOld {
			framework.Failf("TCP Spec.Ports[0].Port (%d) did not change", svcPort)
		}
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) changed", tcpService.Spec.Ports[0].NodePort)
		}
		if getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}

		By("changing the UDP service's port")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *api.Service) {
			s.Spec.Ports[0].Port++
		})
		if loadBalancerSupportsUDP {
			jig.SanityCheckService(udpService, api.ServiceTypeLoadBalancer)
		} else {
			jig.SanityCheckService(udpService, api.ServiceTypeNodePort)
		}
		if int(udpService.Spec.Ports[0].Port) != svcPort {
			framework.Failf("UDP Spec.Ports[0].Port (%d) did not change", udpService.Spec.Ports[0].Port)
		}
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) changed", udpService.Spec.Ports[0].NodePort)
		}
		if loadBalancerSupportsUDP && getIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, getIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}

		framework.Logf("service port (TCP and UDP): %d", svcPort)

		By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, kubeProxyLagTimeout)

		By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, kubeProxyLagTimeout)

		By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout) // this may actually recreate the LB

		if loadBalancerSupportsUDP {
			By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout) // this may actually recreate the LB)
		}

		// Change the services back to ClusterIP.

		By("changing TCP service back to type=ClusterIP")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *api.Service) {
			s.Spec.Type = api.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		// Wait for the load balancer to be destroyed asynchronously
		tcpService = jig.WaitForLoadBalancerDestroyOrFail(ns1, tcpService.Name, tcpIngressIP, svcPort)
		jig.SanityCheckService(tcpService, api.ServiceTypeClusterIP)

		By("changing UDP service back to type=ClusterIP")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *api.Service) {
			s.Spec.Type = api.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		if loadBalancerSupportsUDP {
			// Wait for the load balancer to be destroyed asynchronously
			udpService = jig.WaitForLoadBalancerDestroyOrFail(ns2, udpService.Name, udpIngressIP, svcPort)
			jig.SanityCheckService(udpService, api.ServiceTypeClusterIP)
		}

		By("checking the TCP NodePort is closed")
		jig.TestNotReachableHTTP(nodeIP, tcpNodePort, kubeProxyLagTimeout)

		By("checking the UDP NodePort is closed")
		jig.TestNotReachableUDP(nodeIP, udpNodePort, kubeProxyLagTimeout)

		By("checking the TCP LoadBalancer is closed")
		jig.TestNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			By("checking the UDP LoadBalancer is closed")
			jig.TestNotReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}
	})

	It("should prevent NodePort collisions", func() {
		// TODO: use the ServiceTestJig here
		baseName := "nodeport-collision-"
		serviceName1 := baseName + "1"
		serviceName2 := baseName + "2"
		ns := f.Namespace.Name

		t := NewServerTest(c, ns, serviceName1)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		By("creating service " + serviceName1 + " with type NodePort in namespace " + ns)
		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort
		result, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if result.Spec.Type != api.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", result)
		}
		if len(result.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", result)
		}
		port := result.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", result)
		}

		By("creating service " + serviceName2 + " with conflicting NodePort")
		service2 := t.BuildServiceSpec()
		service2.Name = serviceName2
		service2.Spec.Type = api.ServiceTypeNodePort
		service2.Spec.Ports[0].NodePort = port.NodePort
		result2, err := t.CreateService(service2)
		if err == nil {
			framework.Failf("Created service with conflicting NodePort: %v", result2)
		}
		expectedErr := fmt.Sprintf("%d.*port is already allocated", port.NodePort)
		Expect(fmt.Sprintf("%v", err)).To(MatchRegexp(expectedErr))

		By("deleting service " + serviceName1 + " to release NodePort")
		err = t.DeleteService(serviceName1)
		Expect(err).NotTo(HaveOccurred())

		By("creating service " + serviceName2 + " with no-longer-conflicting NodePort")
		_, err = t.CreateService(service2)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should check NodePort out-of-range", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-range-test"
		ns := f.Namespace.Name

		t := NewServerTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !ServiceNodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}

		outOfRangeNodePort := 0
		rand.Seed(time.Now().UTC().UnixNano())
		for {
			outOfRangeNodePort = 1 + rand.Intn(65535)
			if !ServiceNodePortRange.Contains(outOfRangeNodePort) {
				break
			}
		}
		By(fmt.Sprintf("changing service "+serviceName+" to out-of-range NodePort %d", outOfRangeNodePort))
		result, err := updateService(c, ns, serviceName, func(s *api.Service) {
			s.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		})
		if err == nil {
			framework.Failf("failed to prevent update of service with out-of-range NodePort: %v", result)
		}
		expectedErr := fmt.Sprintf("%d.*port is not in the valid range", outOfRangeNodePort)
		Expect(fmt.Sprintf("%v", err)).To(MatchRegexp(expectedErr))

		By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("creating service "+serviceName+" with out-of-range NodePort %d", outOfRangeNodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		service, err = t.CreateService(service)
		if err == nil {
			framework.Failf("failed to prevent create of service with out-of-range NodePort (%d): %v", outOfRangeNodePort, service)
		}
		Expect(fmt.Sprintf("%v", err)).To(MatchRegexp(expectedErr))
	})

	It("should release NodePorts on delete", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-reuse"
		ns := f.Namespace.Name

		t := NewServerTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !ServiceNodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}
		nodePort := port.NodePort

		By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		Expect(err).NotTo(HaveOccurred())

		hostExec := framework.LaunchHostExecPod(f.Client, f.Namespace.Name, "hostexec")
		cmd := fmt.Sprintf(`! ss -ant46 'sport = :%d' | tail -n +2 | grep LISTEN`, nodePort)
		var stdout string
		if pollErr := wait.PollImmediate(framework.Poll, kubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				framework.Logf("expected node port (%d) to not be in use, stdout: %v", nodePort, stdout)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected node port (%d) to not be in use in %v, stdout: %v", nodePort, kubeProxyLagTimeout, stdout)
		}

		By(fmt.Sprintf("creating service "+serviceName+" with same NodePort %d", nodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = nodePort
		service, err = t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())
	})
})

// updateService fetches a service, calls the update function on it,
// and then attempts to send the updated service. It retries up to 2
// times in the face of timeouts and conflicts.
func updateService(c *client.Client, namespace, serviceName string, update func(*api.Service)) (*api.Service, error) {
	var service *api.Service
	var err error
	for i := 0; i < 3; i++ {
		service, err = c.Services(namespace).Get(serviceName)
		if err != nil {
			return service, err
		}

		update(service)

		service, err = c.Services(namespace).Update(service)

		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			return service, err
		}
	}
	return service, err
}

func getContainerPortsByPodUID(endpoints *api.Endpoints) PortsByPodUID {
	m := PortsByPodUID{}
	for _, ss := range endpoints.Subsets {
		for _, port := range ss.Ports {
			for _, addr := range ss.Addresses {
				containerPort := port.Port
				hostPort := port.Port

				// use endpoint annotations to recover the container port in a Mesos setup
				// compare contrib/mesos/pkg/service/endpoints_controller.syncService
				key := fmt.Sprintf("k8s.mesosphere.io/containerPort_%s_%s_%d", port.Protocol, addr.IP, hostPort)
				mesosContainerPortString := endpoints.Annotations[key]
				if mesosContainerPortString != "" {
					mesosContainerPort, err := strconv.Atoi(mesosContainerPortString)
					if err != nil {
						continue
					}
					containerPort = int32(mesosContainerPort)
					framework.Logf("Mapped mesos host port %d to container port %d via annotation %s=%s", hostPort, containerPort, key, mesosContainerPortString)
				}

				// framework.Logf("Found pod %v, host port %d and container port %d", addr.TargetRef.UID, hostPort, containerPort)
				if _, ok := m[addr.TargetRef.UID]; !ok {
					m[addr.TargetRef.UID] = make([]int, 0)
				}
				m[addr.TargetRef.UID] = append(m[addr.TargetRef.UID], int(containerPort))
			}
		}
	}
	return m
}

type PortsByPodName map[string][]int
type PortsByPodUID map[types.UID][]int

func translatePodNameToUIDOrFail(c *client.Client, ns string, expectedEndpoints PortsByPodName) PortsByPodUID {
	portsByUID := make(PortsByPodUID)

	for name, portList := range expectedEndpoints {
		pod, err := c.Pods(ns).Get(name)
		if err != nil {
			framework.Failf("failed to get pod %s, that's pretty weird. validation failed: %s", name, err)
		}
		portsByUID[pod.ObjectMeta.UID] = portList
	}
	// framework.Logf("successfully translated pod names to UIDs: %v -> %v on namespace %s", expectedEndpoints, portsByUID, ns)
	return portsByUID
}

func validatePortsOrFail(endpoints PortsByPodUID, expectedEndpoints PortsByPodUID) {
	if len(endpoints) != len(expectedEndpoints) {
		// should not happen because we check this condition before
		framework.Failf("invalid number of endpoints got %v, expected %v", endpoints, expectedEndpoints)
	}
	for podUID := range expectedEndpoints {
		if _, ok := endpoints[podUID]; !ok {
			framework.Failf("endpoint %v not found", podUID)
		}
		if len(endpoints[podUID]) != len(expectedEndpoints[podUID]) {
			framework.Failf("invalid list of ports for uid %v. Got %v, expected %v", podUID, endpoints[podUID], expectedEndpoints[podUID])
		}
		sort.Ints(endpoints[podUID])
		sort.Ints(expectedEndpoints[podUID])
		for index := range endpoints[podUID] {
			if endpoints[podUID][index] != expectedEndpoints[podUID][index] {
				framework.Failf("invalid list of ports for uid %v. Got %v, expected %v", podUID, endpoints[podUID], expectedEndpoints[podUID])
			}
		}
	}
}

func validateEndpointsOrFail(c *client.Client, namespace, serviceName string, expectedEndpoints PortsByPodName) {
	By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to expose endpoints %v", framework.ServiceStartTimeout, serviceName, namespace, expectedEndpoints))
	i := 1
	for start := time.Now(); time.Since(start) < framework.ServiceStartTimeout; time.Sleep(1 * time.Second) {
		endpoints, err := c.Endpoints(namespace).Get(serviceName)
		if err != nil {
			framework.Logf("Get endpoints failed (%v elapsed, ignoring for 5s): %v", time.Since(start), err)
			continue
		}
		// framework.Logf("Found endpoints %v", endpoints)

		portsByPodUID := getContainerPortsByPodUID(endpoints)
		// framework.Logf("Found port by pod UID %v", portsByPodUID)

		expectedPortsByPodUID := translatePodNameToUIDOrFail(c, namespace, expectedEndpoints)
		if len(portsByPodUID) == len(expectedEndpoints) {
			validatePortsOrFail(portsByPodUID, expectedPortsByPodUID)
			framework.Logf("successfully validated that service %s in namespace %s exposes endpoints %v (%v elapsed)",
				serviceName, namespace, expectedEndpoints, time.Since(start))
			return
		}

		if i%5 == 0 {
			framework.Logf("Unexpected endpoints: found %v, expected %v (%v elapsed, will retry)", portsByPodUID, expectedEndpoints, time.Since(start))
		}
		i++
	}

	if pods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{}); err == nil {
		for _, pod := range pods.Items {
			framework.Logf("Pod %s\t%s\t%s\t%s", pod.Namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
		}
	} else {
		framework.Logf("Can't list pod debug info: %v", err)
	}
	framework.Failf("Timed out waiting for service %s in namespace %s to expose endpoints %v (%v elapsed)", serviceName, namespace, expectedEndpoints, framework.ServiceStartTimeout)
}

// createExecPodOrFail creates a simple busybox pod in a sleep loop used as a
// vessel for kubectl exec commands.
func createExecPodOrFail(c *client.Client, ns, name string) {
	framework.Logf("Creating new exec pod")
	immediate := int64(0)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PodSpec{
			TerminationGracePeriodSeconds: &immediate,
			Containers: []api.Container{
				{
					Name:    "exec",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", "while true; do sleep 5; done"},
				},
			},
		},
	}
	_, err := c.Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	err = wait.PollImmediate(framework.Poll, 5*time.Minute, func() (bool, error) {
		retrievedPod, err := c.Pods(pod.Namespace).Get(pod.Name)
		if err != nil {
			return false, nil
		}
		return retrievedPod.Status.Phase == api.PodRunning, nil
	})
	Expect(err).NotTo(HaveOccurred())
}

func createPodOrFail(c *client.Client, ns, name string, labels map[string]string, containerPorts []api.ContainerPort) {
	By(fmt.Sprintf("creating pod %s in namespace %s", name, ns))
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "test",
					Image: "gcr.io/google_containers/pause-amd64:3.0",
					Ports: containerPorts,
					// Add a dummy environment variable to work around a docker issue.
					// https://github.com/docker/docker/issues/14203
					Env: []api.EnvVar{{Name: "FOO", Value: " "}},
				},
			},
		},
	}
	_, err := c.Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
}

func deletePodOrFail(c *client.Client, ns, name string) {
	By(fmt.Sprintf("deleting pod %s in namespace %s", name, ns))
	err := c.Pods(ns).Delete(name, nil)
	Expect(err).NotTo(HaveOccurred())
}

func collectAddresses(nodes *api.NodeList, addressType api.NodeAddressType) []string {
	ips := []string{}
	for i := range nodes.Items {
		item := &nodes.Items[i]
		for j := range item.Status.Addresses {
			nodeAddress := &item.Status.Addresses[j]
			if nodeAddress.Type == addressType {
				ips = append(ips, nodeAddress.Address)
			}
		}
	}
	return ips
}

func getNodePublicIps(c *client.Client) ([]string, error) {
	nodes := framework.ListSchedulableNodesOrDie(c)

	ips := collectAddresses(nodes, api.NodeExternalIP)
	if len(ips) == 0 {
		ips = collectAddresses(nodes, api.NodeLegacyHostIP)
	}
	return ips, nil
}

func pickNodeIP(c *client.Client) string {
	publicIps, err := getNodePublicIps(c)
	Expect(err).NotTo(HaveOccurred())
	if len(publicIps) == 0 {
		framework.Failf("got unexpected number (%d) of public IPs", len(publicIps))
	}
	ip := publicIps[0]
	return ip
}

func testReachableHTTP(ip string, port int, request string, expect string) (bool, error) {
	url := fmt.Sprintf("http://%s:%d%s", ip, port, request)
	if ip == "" {
		framework.Failf("Got empty IP for reachability check (%s)", url)
		return false, nil
	}
	if port == 0 {
		framework.Failf("Got port==0 for reachability check (%s)", url)
		return false, nil
	}

	framework.Logf("Testing HTTP reachability of %v", url)

	resp, err := httpGetNoConnectionPool(url)
	if err != nil {
		framework.Logf("Got error testing for reachability of %s: %v", url, err)
		return false, nil
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		framework.Logf("Got error reading response from %s: %v", url, err)
		return false, nil
	}
	if resp.StatusCode != 200 {
		return false, fmt.Errorf("received non-success return status %q trying to access %s; got body: %s", resp.Status, url, string(body))
	}
	if !strings.Contains(string(body), expect) {
		return false, fmt.Errorf("received response body without expected substring %q: %s", expect, string(body))
	}
	framework.Logf("Successfully reached %v", url)
	return true, nil
}

func testNotReachableHTTP(ip string, port int) (bool, error) {
	url := fmt.Sprintf("http://%s:%d", ip, port)
	if ip == "" {
		framework.Failf("Got empty IP for non-reachability check (%s)", url)
		return false, nil
	}
	if port == 0 {
		framework.Failf("Got port==0 for non-reachability check (%s)", url)
		return false, nil
	}

	framework.Logf("Testing HTTP non-reachability of %v", url)

	resp, err := httpGetNoConnectionPool(url)
	if err != nil {
		framework.Logf("Confirmed that %s is not reachable", url)
		return true, nil
	}
	resp.Body.Close()
	return false, nil
}

func testReachableUDP(ip string, port int, request string, expect string) (bool, error) {
	uri := fmt.Sprintf("udp://%s:%d", ip, port)
	if ip == "" {
		framework.Failf("Got empty IP for reachability check (%s)", uri)
		return false, nil
	}
	if port == 0 {
		framework.Failf("Got port==0 for reachability check (%s)", uri)
		return false, nil
	}

	framework.Logf("Testing UDP reachability of %v", uri)

	con, err := net.Dial("udp", ip+":"+strconv.Itoa(port))
	if err != nil {
		return false, fmt.Errorf("Failed to dial %s:%d: %v", ip, port, err)
	}

	_, err = con.Write([]byte(fmt.Sprintf("%s\n", request)))
	if err != nil {
		return false, fmt.Errorf("Failed to send request: %v", err)
	}

	var buf []byte = make([]byte, len(expect)+1)

	err = con.SetDeadline(time.Now().Add(3 * time.Second))
	if err != nil {
		return false, fmt.Errorf("Failed to set deadline: %v", err)
	}

	_, err = con.Read(buf)
	if err != nil {
		return false, nil
	}

	if !strings.Contains(string(buf), expect) {
		return false, fmt.Errorf("Failed to retrieve %q, got %q", expect, string(buf))
	}

	framework.Logf("Successfully reached %v", uri)
	return true, nil
}

func testNotReachableUDP(ip string, port int, request string) (bool, error) {
	uri := fmt.Sprintf("udp://%s:%d", ip, port)
	if ip == "" {
		framework.Failf("Got empty IP for reachability check (%s)", uri)
		return false, nil
	}
	if port == 0 {
		framework.Failf("Got port==0 for reachability check (%s)", uri)
		return false, nil
	}

	framework.Logf("Testing UDP non-reachability of %v", uri)

	con, err := net.Dial("udp", ip+":"+strconv.Itoa(port))
	if err != nil {
		framework.Logf("Confirmed that %s is not reachable", uri)
		return true, nil
	}

	_, err = con.Write([]byte(fmt.Sprintf("%s\n", request)))
	if err != nil {
		framework.Logf("Confirmed that %s is not reachable", uri)
		return true, nil
	}

	var buf []byte = make([]byte, 1)

	err = con.SetDeadline(time.Now().Add(3 * time.Second))
	if err != nil {
		return false, fmt.Errorf("Failed to set deadline: %v", err)
	}

	_, err = con.Read(buf)
	if err != nil {
		framework.Logf("Confirmed that %s is not reachable", uri)
		return true, nil
	}

	return false, nil
}

// Creates a replication controller that serves its hostname and a service on top of it.
func startServeHostnameService(c *client.Client, ns, name string, port, replicas int) ([]string, string, error) {
	podNames := make([]string, replicas)

	By("creating service " + name + " in namespace " + ns)
	_, err := c.Services(ns).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Port:       int32(port),
				TargetPort: intstr.FromInt(9376),
				Protocol:   "TCP",
			}},
			Selector: map[string]string{
				"name": name,
			},
		},
	})
	if err != nil {
		return podNames, "", err
	}

	var createdPods []*api.Pod
	maxContainerFailures := 0
	config := framework.RCConfig{
		Client:               c,
		Image:                "gcr.io/google_containers/serve_hostname:v1.4",
		Name:                 name,
		Namespace:            ns,
		PollInterval:         3 * time.Second,
		Timeout:              framework.PodReadyBeforeTimeout,
		Replicas:             replicas,
		CreatedPods:          &createdPods,
		MaxContainerFailures: &maxContainerFailures,
	}
	err = framework.RunRC(config)
	if err != nil {
		return podNames, "", err
	}

	if len(createdPods) != replicas {
		return podNames, "", fmt.Errorf("Incorrect number of running pods: %v", len(createdPods))
	}

	for i := range createdPods {
		podNames[i] = createdPods[i].ObjectMeta.Name
	}
	sort.StringSlice(podNames).Sort()

	service, err := c.Services(ns).Get(name)
	if err != nil {
		return podNames, "", err
	}
	if service.Spec.ClusterIP == "" {
		return podNames, "", fmt.Errorf("Service IP is blank for %v", name)
	}
	serviceIP := service.Spec.ClusterIP
	return podNames, serviceIP, nil
}

func stopServeHostnameService(c *client.Client, ns, name string) error {
	if err := framework.DeleteRC(c, ns, name); err != nil {
		return err
	}
	if err := c.Services(ns).Delete(name); err != nil {
		return err
	}
	return nil
}

// verifyServeHostnameServiceUp wgets the given serviceIP:servicePort from the
// given host and from within a pod. The host is expected to be an SSH-able node
// in the cluster. Each pod in the service is expected to echo its name. These
// names are compared with the given expectedPods list after a sort | uniq.
func verifyServeHostnameServiceUp(c *client.Client, ns, host string, expectedPods []string, serviceIP string, servicePort int) error {
	execPodName := "execpod"
	createExecPodOrFail(c, ns, execPodName)
	defer func() {
		deletePodOrFail(c, ns, execPodName)
	}()

	// Loop a bunch of times - the proxy is randomized, so we want a good
	// chance of hitting each backend at least once.
	buildCommand := func(wget string) string {
		return fmt.Sprintf("for i in $(seq 1 %d); do %s http://%s:%d 2>&1 || true; echo; done",
			50*len(expectedPods), wget, serviceIP, servicePort)
	}
	commands := []func() string{
		// verify service from node
		func() string {
			cmd := "set -e; " + buildCommand("wget -q --timeout=0.2 --tries=1 -O -")
			framework.Logf("Executing cmd %q on host %v", cmd, host)
			result, err := framework.SSH(cmd, host, framework.TestContext.Provider)
			if err != nil || result.Code != 0 {
				framework.LogSSHResult(result)
				framework.Logf("error while SSH-ing to node: %v", err)
			}
			return result.Stdout
		},
		// verify service from pod
		func() string {
			cmd := buildCommand("wget -q -T 1 -O -")
			framework.Logf("Executing cmd %q in pod %v/%v", cmd, ns, execPodName)
			// TODO: Use exec-over-http via the netexec pod instead of kubectl exec.
			output, err := framework.RunHostCmd(ns, execPodName, cmd)
			if err != nil {
				framework.Logf("error while kubectl execing %q in pod %v/%v: %v\nOutput: %v", cmd, ns, execPodName, err, output)
			}
			return output
		},
	}

	expectedEndpoints := sets.NewString(expectedPods...)
	By(fmt.Sprintf("verifying service has %d reachable backends", len(expectedPods)))
	for _, cmdFunc := range commands {
		passed := false
		gotEndpoints := sets.NewString()

		// Retry cmdFunc for a while
		for start := time.Now(); time.Since(start) < kubeProxyLagTimeout; time.Sleep(5 * time.Second) {
			for _, endpoint := range strings.Split(cmdFunc(), "\n") {
				trimmedEp := strings.TrimSpace(endpoint)
				if trimmedEp != "" {
					gotEndpoints.Insert(trimmedEp)
				}
			}
			// TODO: simply checking that the retrieved endpoints is a superset
			// of the expected allows us to ignore intermitten network flakes that
			// result in output like "wget timed out", but these should be rare
			// and we need a better way to track how often it occurs.
			if gotEndpoints.IsSuperset(expectedEndpoints) {
				if !gotEndpoints.Equal(expectedEndpoints) {
					framework.Logf("Ignoring unexpected output wgetting endpoints of service %s: %v", serviceIP, gotEndpoints.Difference(expectedEndpoints))
				}
				passed = true
				break
			}
			framework.Logf("Unable to reach the following endpoints of service %s: %v", serviceIP, expectedEndpoints.Difference(gotEndpoints))
		}
		if !passed {
			// Sort the lists so they're easier to visually diff.
			exp := expectedEndpoints.List()
			got := gotEndpoints.List()
			sort.StringSlice(exp).Sort()
			sort.StringSlice(got).Sort()
			return fmt.Errorf("service verification failed for: %s\nexpected %v\nreceived %v", serviceIP, exp, got)
		}
	}
	return nil
}

func verifyServeHostnameServiceDown(c *client.Client, host string, serviceIP string, servicePort int) error {
	command := fmt.Sprintf(
		"curl -s --connect-timeout 2 http://%s:%d && exit 99", serviceIP, servicePort)

	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := framework.SSH(command, host, framework.TestContext.Provider)
		if err != nil {
			framework.LogSSHResult(result)
			framework.Logf("error while SSH-ing to node: %v", err)
		}
		if result.Code != 99 {
			return nil
		}
		framework.Logf("service still alive - still waiting")
	}
	return fmt.Errorf("waiting for service to be down timed out")
}

// Does an HTTP GET, but does not reuse TCP connections
// This masks problems where the iptables rule has changed, but we don't see it
// This is intended for relatively quick requests (status checks), so we set a short (5 seconds) timeout
func httpGetNoConnectionPool(url string) (*http.Response, error) {
	tr := utilnet.SetTransportDefaults(&http.Transport{
		DisableKeepAlives: true,
	})
	client := &http.Client{
		Transport: tr,
		Timeout:   5 * time.Second,
	}

	return client.Get(url)
}

// A test jig to help testing.
type ServiceTestJig struct {
	ID     string
	Name   string
	Client *client.Client
	Labels map[string]string
}

// NewServiceTestJig allocates and inits a new ServiceTestJig.
func NewServiceTestJig(client *client.Client, name string) *ServiceTestJig {
	j := &ServiceTestJig{}
	j.Client = client
	j.Name = name
	j.ID = j.Name + "-" + string(util.NewUUID())
	j.Labels = map[string]string{"testid": j.ID}

	return j
}

// newServiceTemplate returns the default api.Service template for this jig, but
// does not actually create the Service.  The default Service has the same name
// as the jig and exposes port 80.
func (j *ServiceTestJig) newServiceTemplate(namespace string, proto api.Protocol) *api.Service {
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: api.ServiceSpec{
			Selector: j.Labels,
			Ports: []api.ServicePort{
				{
					Protocol: proto,
					Port:     80,
				},
			},
		},
	}
	return service
}

// CreateTCPServiceOrFail creates a new TCP Service based on the jig's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *ServiceTestJig) CreateTCPServiceOrFail(namespace string, tweak func(svc *api.Service)) *api.Service {
	svc := j.newServiceTemplate(namespace, api.ProtocolTCP)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.Services(namespace).Create(svc)
	if err != nil {
		framework.Failf("Failed to create TCP Service %q: %v", svc.Name, err)
	}
	return result
}

// CreateUDPServiceOrFail creates a new UDP Service based on the jig's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *ServiceTestJig) CreateUDPServiceOrFail(namespace string, tweak func(svc *api.Service)) *api.Service {
	svc := j.newServiceTemplate(namespace, api.ProtocolUDP)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.Services(namespace).Create(svc)
	if err != nil {
		framework.Failf("Failed to create UDP Service %q: %v", svc.Name, err)
	}
	return result
}

func (j *ServiceTestJig) SanityCheckService(svc *api.Service, svcType api.ServiceType) {
	if svc.Spec.Type != svcType {
		framework.Failf("unexpected Spec.Type (%s) for service, expected %s", svc.Spec.Type, svcType)
	}
	expectNodePorts := false
	if svcType != api.ServiceTypeClusterIP {
		expectNodePorts = true
	}
	for i, port := range svc.Spec.Ports {
		hasNodePort := (port.NodePort != 0)
		if hasNodePort != expectNodePorts {
			framework.Failf("unexpected Spec.Ports[%d].NodePort (%d) for service", i, port.NodePort)
		}
		if hasNodePort {
			if !ServiceNodePortRange.Contains(int(port.NodePort)) {
				framework.Failf("out-of-range nodePort (%d) for service", port.NodePort)
			}
		}
	}
	expectIngress := false
	if svcType == api.ServiceTypeLoadBalancer {
		expectIngress = true
	}
	hasIngress := len(svc.Status.LoadBalancer.Ingress) != 0
	if hasIngress != expectIngress {
		framework.Failf("unexpected number of Status.LoadBalancer.Ingress (%d) for service", len(svc.Status.LoadBalancer.Ingress))
	}
	if hasIngress {
		for i, ing := range svc.Status.LoadBalancer.Ingress {
			if ing.IP == "" && ing.Hostname == "" {
				framework.Failf("unexpected Status.LoadBalancer.Ingress[%d] for service: %#v", i, ing)
			}
		}
	}
}

// UpdateService fetches a service, calls the update function on it, and
// then attempts to send the updated service. It tries up to 3 times in the
// face of timeouts and conflicts.
func (j *ServiceTestJig) UpdateService(namespace, name string, update func(*api.Service)) (*api.Service, error) {
	for i := 0; i < 3; i++ {
		service, err := j.Client.Services(namespace).Get(name)
		if err != nil {
			return nil, fmt.Errorf("Failed to get Service %q: %v", name, err)
		}

		update(service)
		service, err = j.Client.Services(namespace).Update(service)
		if err == nil {
			return service, nil
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			return nil, fmt.Errorf("Failed to update Service %q: %v", name, err)
		}
	}
	return nil, fmt.Errorf("Too many retries updating Service %q", name)
}

// UpdateServiceOrFail fetches a service, calls the update function on it, and
// then attempts to send the updated service. It tries up to 3 times in the
// face of timeouts and conflicts.
func (j *ServiceTestJig) UpdateServiceOrFail(namespace, name string, update func(*api.Service)) *api.Service {
	svc, err := j.UpdateService(namespace, name, update)
	if err != nil {
		framework.Failf(err.Error())
	}
	return svc
}

func (j *ServiceTestJig) ChangeServiceNodePortOrFail(namespace, name string, initial int) *api.Service {
	var err error
	var service *api.Service
	for i := 1; i < ServiceNodePortRange.Size; i++ {
		offs1 := initial - ServiceNodePortRange.Base
		offs2 := (offs1 + i) % ServiceNodePortRange.Size
		newPort := ServiceNodePortRange.Base + offs2
		service, err = j.UpdateService(namespace, name, func(s *api.Service) {
			s.Spec.Ports[0].NodePort = int32(newPort)
		})
		if err != nil && strings.Contains(err.Error(), "provided port is already allocated") {
			framework.Logf("tried nodePort %d, but it is in use, will try another", newPort)
			continue
		}
		// Otherwise err was nil or err was a real error
		break
	}
	if err != nil {
		framework.Failf("Could not change the nodePort: %v", err)
	}
	return service
}

func (j *ServiceTestJig) WaitForLoadBalancerOrFail(namespace, name string) *api.Service {
	var service *api.Service
	framework.Logf("Waiting up to %v for service %q to have a LoadBalancer", loadBalancerCreateTimeout, name)
	pollFunc := func() (bool, error) {
		svc, err := j.Client.Services(namespace).Get(name)
		if err != nil {
			return false, err
		}
		if len(svc.Status.LoadBalancer.Ingress) > 0 {
			service = svc
			return true, nil
		}
		return false, nil
	}
	if err := wait.PollImmediate(framework.Poll, loadBalancerCreateTimeout, pollFunc); err != nil {
		framework.Failf("Timeout waiting for service %q to have a load balancer", name)
	}
	return service
}

func (j *ServiceTestJig) WaitForLoadBalancerDestroyOrFail(namespace, name string, ip string, port int) *api.Service {
	// TODO: once support ticket 21807001 is resolved, reduce this timeout back to something reasonable
	defer func() {
		if err := framework.EnsureLoadBalancerResourcesDeleted(ip, strconv.Itoa(port)); err != nil {
			framework.Logf("Failed to delete cloud resources for service: %s %d (%v)", ip, port, err)
		}
	}()

	var service *api.Service
	framework.Logf("Waiting up to %v for service %q to have no LoadBalancer", loadBalancerCreateTimeout, name)
	pollFunc := func() (bool, error) {
		svc, err := j.Client.Services(namespace).Get(name)
		if err != nil {
			return false, err
		}
		if len(svc.Status.LoadBalancer.Ingress) == 0 {
			service = svc
			return true, nil
		}
		return false, nil
	}
	if err := wait.PollImmediate(framework.Poll, loadBalancerCreateTimeout, pollFunc); err != nil {
		framework.Failf("Timeout waiting for service %q to have no load balancer", name)
	}
	return service
}

func (j *ServiceTestJig) TestReachableHTTP(host string, port int, timeout time.Duration) {
	if err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) { return testReachableHTTP(host, port, "/echo?msg=hello", "hello") }); err != nil {
		framework.Failf("Could not reach HTTP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

func (j *ServiceTestJig) TestNotReachableHTTP(host string, port int, timeout time.Duration) {
	if err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) { return testNotReachableHTTP(host, port) }); err != nil {
		framework.Failf("Could still reach HTTP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

func (j *ServiceTestJig) TestReachableUDP(host string, port int, timeout time.Duration) {
	if err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) { return testReachableUDP(host, port, "echo hello", "hello") }); err != nil {
		framework.Failf("Could not reach UDP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

func (j *ServiceTestJig) TestNotReachableUDP(host string, port int, timeout time.Duration) {
	if err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) { return testNotReachableUDP(host, port, "echo hello") }); err != nil {
		framework.Failf("Could still reach UDP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

func getIngressPoint(ing *api.LoadBalancerIngress) string {
	host := ing.IP
	if host == "" {
		host = ing.Hostname
	}
	return host
}

// newRCTemplate returns the default api.ReplicationController object for
// this jig, but does not actually create the RC.  The default RC has the same
// name as the jig and runs the "netexec" container.
func (j *ServiceTestJig) newRCTemplate(namespace string) *api.ReplicationController {
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: j.Labels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: j.Labels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "netexec",
							Image: "gcr.io/google_containers/netexec:1.4",
							Args:  []string{"--http-port=80", "--udp-port=80"},
							ReadinessProbe: &api.Probe{
								PeriodSeconds: 3,
								Handler: api.Handler{
									HTTPGet: &api.HTTPGetAction{
										Port: intstr.FromInt(80),
										Path: "/hostName",
									},
								},
							},
						},
					},
					TerminationGracePeriodSeconds: new(int64),
				},
			},
		},
	}
	return rc
}

// RunOrFail creates a ReplicationController and Pod(s) and waits for the
// Pod(s) to be running. Callers can provide a function to tweak the RC object
// before it is created.
func (j *ServiceTestJig) RunOrFail(namespace string, tweak func(rc *api.ReplicationController)) *api.ReplicationController {
	rc := j.newRCTemplate(namespace)
	if tweak != nil {
		tweak(rc)
	}
	result, err := j.Client.ReplicationControllers(namespace).Create(rc)
	if err != nil {
		framework.Failf("Failed to created RC %q: %v", rc.Name, err)
	}
	pods, err := j.waitForPodsCreated(namespace, int(rc.Spec.Replicas))
	if err != nil {
		framework.Failf("Failed to create pods: %v", err)
	}
	if err := j.waitForPodsReady(namespace, pods); err != nil {
		framework.Failf("Failed waiting for pods to be running: %v", err)
	}
	return result
}

func (j *ServiceTestJig) waitForPodsCreated(namespace string, replicas int) ([]string, error) {
	timeout := 2 * time.Minute
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(j.Labels))
	framework.Logf("Waiting up to %v for %d pods to be created", timeout, replicas)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		options := api.ListOptions{LabelSelector: label}
		pods, err := j.Client.Pods(namespace).List(options)
		if err != nil {
			return nil, err
		}

		found := []string{}
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp != nil {
				continue
			}
			found = append(found, pod.Name)
		}
		if len(found) == replicas {
			framework.Logf("Found all %d pods", replicas)
			return found, nil
		}
		framework.Logf("Found %d/%d pods - will retry", len(found), replicas)
	}
	return nil, fmt.Errorf("Timeout waiting for %d pods to be created", replicas)
}

func (j *ServiceTestJig) waitForPodsReady(namespace string, pods []string) error {
	timeout := 2 * time.Minute
	if !framework.CheckPodsRunningReady(j.Client, namespace, pods, timeout) {
		return fmt.Errorf("Timeout waiting for %d pods to be ready", len(pods))
	}
	return nil
}

// Simple helper class to avoid too much boilerplate in tests
type ServiceTestFixture struct {
	ServiceName string
	Namespace   string
	Client      *client.Client

	TestId string
	Labels map[string]string

	rcs      map[string]bool
	services map[string]bool
	name     string
	image    string
}

func NewServerTest(client *client.Client, namespace string, serviceName string) *ServiceTestFixture {
	t := &ServiceTestFixture{}
	t.Client = client
	t.Namespace = namespace
	t.ServiceName = serviceName
	t.TestId = t.ServiceName + "-" + string(util.NewUUID())
	t.Labels = map[string]string{
		"testid": t.TestId,
	}

	t.rcs = make(map[string]bool)
	t.services = make(map[string]bool)

	t.name = "webserver"
	t.image = "gcr.io/google_containers/test-webserver:e2e"

	return t
}

// Build default config for a service (which can then be changed)
func (t *ServiceTestFixture) BuildServiceSpec() *api.Service {
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      t.ServiceName,
			Namespace: t.Namespace,
		},
		Spec: api.ServiceSpec{
			Selector: t.Labels,
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
	return service
}

// CreateWebserverRC creates rc-backed pods with the well-known webserver
// configuration and records it for cleanup.
func (t *ServiceTestFixture) CreateWebserverRC(replicas int32) *api.ReplicationController {
	rcSpec := rcByNamePort(t.name, replicas, t.image, 80, api.ProtocolTCP, t.Labels)
	rcAct, err := t.createRC(rcSpec)
	if err != nil {
		framework.Failf("Failed to create rc %s: %v", rcSpec.Name, err)
	}
	if err := framework.VerifyPods(t.Client, t.Namespace, t.name, false, replicas); err != nil {
		framework.Failf("Failed to create %d pods with name %s: %v", replicas, t.name, err)
	}
	return rcAct
}

// createRC creates a replication controller and records it for cleanup.
func (t *ServiceTestFixture) createRC(rc *api.ReplicationController) (*api.ReplicationController, error) {
	rc, err := t.Client.ReplicationControllers(t.Namespace).Create(rc)
	if err == nil {
		t.rcs[rc.Name] = true
	}
	return rc, err
}

// Create a service, and record it for cleanup
func (t *ServiceTestFixture) CreateService(service *api.Service) (*api.Service, error) {
	result, err := t.Client.Services(t.Namespace).Create(service)
	if err == nil {
		t.services[service.Name] = true
	}
	return result, err
}

// Delete a service, and remove it from the cleanup list
func (t *ServiceTestFixture) DeleteService(serviceName string) error {
	err := t.Client.Services(t.Namespace).Delete(serviceName)
	if err == nil {
		delete(t.services, serviceName)
	}
	return err
}

func (t *ServiceTestFixture) Cleanup() []error {
	var errs []error
	for rcName := range t.rcs {
		By("stopping RC " + rcName + " in namespace " + t.Namespace)
		// First, resize the RC to 0.
		old, err := t.Client.ReplicationControllers(t.Namespace).Get(rcName)
		if err != nil {
			errs = append(errs, err)
		}
		old.Spec.Replicas = 0
		if _, err := t.Client.ReplicationControllers(t.Namespace).Update(old); err != nil {
			errs = append(errs, err)
		}
		// TODO(mikedanese): Wait.

		// Then, delete the RC altogether.
		if err := t.Client.ReplicationControllers(t.Namespace).Delete(rcName); err != nil {
			errs = append(errs, err)
		}
	}

	for serviceName := range t.services {
		By("deleting service " + serviceName + " in namespace " + t.Namespace)
		err := t.Client.Services(t.Namespace).Delete(serviceName)
		if err != nil {
			errs = append(errs, err)
		}
	}

	return errs
}
