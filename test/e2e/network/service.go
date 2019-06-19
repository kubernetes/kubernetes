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

package network

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpoints "k8s.io/kubernetes/test/e2e/framework/endpoints"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	imageutils "k8s.io/kubernetes/test/utils/image"
	gcecloud "k8s.io/legacy-cloud-providers/gce"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	defaultServeHostnameServicePort = 80
	defaultServeHostnameServiceName = "svc-hostname"
)

var (
	defaultServeHostnameService = v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultServeHostnameServiceName,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       int32(defaultServeHostnameServicePort),
				TargetPort: intstr.FromInt(9376),
				Protocol:   v1.ProtocolTCP,
			}},
			Selector: map[string]string{
				"name": defaultServeHostnameServiceName,
			},
		},
	}
)

func getServeHostnameService(name string) *v1.Service {
	svc := defaultServeHostnameService.DeepCopy()
	svc.ObjectMeta.Name = name
	svc.Spec.Selector["name"] = name
	return svc
}

var _ = SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface
	serviceLBNames := []string{}

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentGinkgoTestDescription().Failed {
			framework.DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			e2elog.Logf("cleaning load balancer resource for %s", lb)
			framework.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	// TODO: We get coverage of TCP/UDP and multi-port services through the DNS test. We should have a simpler test for multi-port TCP here.

	/*
		Release : v1.9
		Testname: Kubernetes Service
		Description: By default when a kubernetes cluster is running there MUST be a ‘kubernetes’ service running in the cluster.
	*/
	framework.ConformanceIt("should provide secure master service ", func() {
		_, err := cs.CoreV1().Services(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the service object for the service named kubernetes")
	})

	/*
		Release : v1.9
		Testname: Service, endpoints
		Description: Create a service with a endpoint without any Pods, the service MUST run and show empty endpoints. Add a pod to the service and the service MUST validate to show all the endpoints for the ports exposed by the Pod. Add another Pod then the list of all Ports exposed by both the Pods MUST be valid and have corresponding service endpoint. Once the second Pod is deleted then set of endpoint MUST be validated to show only ports from the first container that are exposed. Once both pods are deleted the endpoints from the service MUST be empty.
	*/
	framework.ConformanceIt("should serve a basic endpoint from pods ", func() {
		serviceName := "endpoint-test2"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		defer func() {
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		ports := []v1.ServicePort{{
			Port:       80,
			TargetPort: intstr.FromInt(80),
		}}
		_, err := jig.CreateServiceWithServicePort(labels, ns, ports)

		framework.ExpectNoError(err, "failed to create service with ServicePorts in namespace: %s", ns)

		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(name, nil)
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		name1 := "pod1"
		name2 := "pod2"

		e2epod.CreatePodOrFail(cs, ns, name1, labels, []v1.ContainerPort{{ContainerPort: 80}})
		names[name1] = true
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{name1: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.CreatePodOrFail(cs, ns, name2, labels, []v1.ContainerPort{{ContainerPort: 80}})
		names[name2] = true
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{name1: {80}, name2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, name1)
		delete(names, name1)
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{name2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, name2)
		delete(names, name2)
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)
	})

	/*
		Release : v1.9
		Testname: Service, endpoints with multiple ports
		Description: Create a service with two ports but no Pods are added to the service yet.  The service MUST run and show empty set of endpoints. Add a Pod to the first port, service MUST list one endpoint for the Pod on that port. Add another Pod to the second port, service MUST list both the endpoints. Delete the first Pod and the service MUST list only the endpoint to the second Pod. Delete the second Pod and the service must now have empty set of endpoints.
	*/
	framework.ConformanceIt("should serve multiport endpoints from pods ", func() {
		// repacking functionality is intentionally not tested here - it's better to test it in an integration test.
		serviceName := "multi-endpoint-test"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		defer func() {
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()

		labels := map[string]string{"foo": "bar"}

		svc1port := "svc1"
		svc2port := "svc2"

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		ports := []v1.ServicePort{
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
		}
		_, err := jig.CreateServiceWithServicePort(labels, ns, ports)
		framework.ExpectNoError(err, "failed to create service with ServicePorts in namespace: %s", ns)
		port1 := 100
		port2 := 101
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(name, nil)
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		containerPorts1 := []v1.ContainerPort{
			{
				Name:          svc1port,
				ContainerPort: int32(port1),
			},
		}
		containerPorts2 := []v1.ContainerPort{
			{
				Name:          svc2port,
				ContainerPort: int32(port2),
			},
		}

		podname1 := "pod1"
		podname2 := "pod2"

		e2epod.CreatePodOrFail(cs, ns, podname1, labels, containerPorts1)
		names[podname1] = true
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{podname1: {port1}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.CreatePodOrFail(cs, ns, podname2, labels, containerPorts2)
		names[podname2] = true
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{podname1: {port1}, podname2: {port2}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, podname1)
		delete(names, podname1)
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{podname2: {port2}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, podname2)
		delete(names, podname2)
		err = e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)
	})

	ginkgo.It("should preserve source pod IP for traffic thru service cluster IP", func() {

		// This behavior is not supported if Kube-proxy is in "userspace" mode.
		// So we check the kube-proxy mode and skip this test if that's the case.
		if proxyMode, err := framework.ProxyMode(f); err == nil {
			if proxyMode == "userspace" {
				framework.Skipf("The test doesn't work with kube-proxy in userspace mode")
			}
		} else {
			e2elog.Logf("Couldn't detect KubeProxy mode - test failure may be expected: %v", err)
		}

		serviceName := "sourceip-test"
		ns := f.Namespace.Name

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		jig := framework.NewServiceTestJig(cs, serviceName)
		servicePort := 8080
		tcpService := jig.CreateTCPServiceWithPort(ns, nil, int32(servicePort))
		jig.SanityCheckService(tcpService, v1.ServiceTypeClusterIP)
		defer func() {
			e2elog.Logf("Cleaning up the sourceip test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		serviceIP := tcpService.Spec.ClusterIP
		e2elog.Logf("sourceip-test cluster ip: %s", serviceIP)

		ginkgo.By("Picking multiple nodes")
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)

		if len(nodes.Items) == 1 {
			framework.Skipf("The test requires two Ready nodes on %s, but found just one.", framework.TestContext.Provider)
		}

		node1 := nodes.Items[0]
		node2 := nodes.Items[1]

		ginkgo.By("Creating a webserver pod be part of the TCP service which echoes back source ip")
		serverPodName := "echoserver-sourceip"
		jig.LaunchEchoserverPodOnNode(f, node1.Name, serverPodName)
		defer func() {
			e2elog.Logf("Cleaning up the echo server pod")
			err := cs.CoreV1().Pods(ns).Delete(serverPodName, nil)
			framework.ExpectNoError(err, "failed to delete pod: %s on node: %s", serverPodName, node1.Name)
		}()

		// Waiting for service to expose endpoint.
		err := e2eendpoints.ValidateEndpointsPorts(cs, ns, serviceName, e2eendpoints.PortsByPodName{serverPodName: {servicePort}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		ginkgo.By("Retrieve sourceip from a pod on the same node")
		sourceIP1, execPodIP1 := execSourceipTest(f, cs, ns, node1.Name, serviceIP, servicePort)
		ginkgo.By("Verifying the preserved source ip")
		gomega.Expect(sourceIP1).To(gomega.Equal(execPodIP1))

		ginkgo.By("Retrieve sourceip from a pod on a different node")
		sourceIP2, execPodIP2 := execSourceipTest(f, cs, ns, node2.Name, serviceIP, servicePort)
		ginkgo.By("Verifying the preserved source ip")
		gomega.Expect(sourceIP2).To(gomega.Equal(execPodIP2))
	})

	ginkgo.It("should be able to up and down services", func() {
		// TODO: use the ServiceTestJig here
		// this test uses e2essh.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		// this test does not work if the Node does not support SSH Key
		framework.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort

		svc1 := "up-down-1"
		svc2 := "up-down-2"
		svc3 := "up-down-3"

		ginkgo.By("creating " + svc1 + " in namespace " + ns)
		podNames1, svc1IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)
		ginkgo.By("creating " + svc2 + " in namespace " + ns)
		podNames2, svc2IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ginkgo.By("verifying service " + svc1 + " is up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		ginkgo.By("verifying service " + svc2 + " is up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		// Stop service 1 and make sure it is gone.
		ginkgo.By("stopping service " + svc1)
		framework.ExpectNoError(framework.StopServeHostnameService(f.ClientSet, ns, svc1))

		ginkgo.By("verifying service " + svc1 + " is not up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceDown(cs, host, svc1IP, servicePort))
		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		// Start another service and verify both are up.
		ginkgo.By("creating service " + svc3 + " in namespace " + ns)
		podNames3, svc3IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc3), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc3, ns)

		if svc2IP == svc3IP {
			framework.Failf("service IPs conflict: %v", svc2IP)
		}

		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		ginkgo.By("verifying service " + svc3 + " is up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames3, svc3IP, servicePort))
	})

	ginkgo.It("should work after restarting kube-proxy [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		framework.SkipUnlessProviderIs("gce", "gke")

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort

		svc1 := "restart-proxy-1"
		svc2 := "restart-proxy-2"

		defer func() {
			framework.ExpectNoError(framework.StopServeHostnameService(f.ClientSet, ns, svc1))
		}()
		podNames1, svc1IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)

		defer func() {
			framework.ExpectNoError(framework.StopServeHostnameService(f.ClientSet, ns, svc2))
		}()
		podNames2, svc2IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		ginkgo.By(fmt.Sprintf("Restarting kube-proxy on %v", host))
		if err := framework.RestartKubeProxy(host); err != nil {
			framework.Failf("error restarting kube-proxy: %v", err)
		}
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		ginkgo.By("Removing iptable rules")
		result, err := e2essh.SSH(`
			sudo iptables -t nat -F KUBE-SERVICES || true;
			sudo iptables -t nat -F KUBE-PORTALS-HOST || true;
			sudo iptables -t nat -F KUBE-PORTALS-CONTAINER || true`, host, framework.TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
			framework.Failf("couldn't remove iptable rules: %v", err)
		}
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))
	})

	ginkgo.It("should work after restarting apiserver [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		framework.SkipUnlessProviderIs("gce", "gke")

		ns := f.Namespace.Name
		numPods, servicePort := 3, 80

		svc1 := "restart-apiserver-1"
		svc2 := "restart-apiserver-2"

		defer func() {
			framework.ExpectNoError(framework.StopServeHostnameService(f.ClientSet, ns, svc1))
		}()
		podNames1, svc1IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		// Restart apiserver
		ginkgo.By("Restarting apiserver")
		if err := framework.RestartApiserver(cs); err != nil {
			framework.Failf("error restarting apiserver: %v", err)
		}
		ginkgo.By("Waiting for apiserver to come up by polling /healthz")
		if err := framework.WaitForApiserverUp(cs); err != nil {
			framework.Failf("error while waiting for apiserver up: %v", err)
		}
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		// Create a new service and check if it's not reusing IP.
		defer func() {
			framework.ExpectNoError(framework.StopServeHostnameService(f.ClientSet, ns, svc2))
		}()
		podNames2, svc2IP, err := framework.StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))
	})

	// TODO: Run this test against the userspace proxy and nodes
	// configured with a default deny firewall to validate that the
	// proxy whitelists NodePort traffic.
	ginkgo.It("should be able to create a functioning NodePort service", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name

		jig := framework.NewServiceTestJig(cs, serviceName)
		nodeIP := framework.PickNodeIP(jig.Client) // for later

		ginkgo.By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		service := jig.CreateTCPServiceOrFail(ns, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		jig.SanityCheckService(service, v1.ServiceTypeNodePort)
		nodePort := int(service.Spec.Ports[0].NodePort)

		ginkgo.By("creating pod to be part of service " + serviceName)
		jig.RunOrFail(ns, nil)

		ginkgo.By("hitting the pod through the service's NodePort")
		jig.TestReachableHTTP(nodeIP, nodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("verifying the node port is locked")
		hostExec := e2epod.LaunchHostExecPod(f.ClientSet, f.Namespace.Name, "hostexec")
		// Even if the node-ip:node-port check above passed, this hostexec pod
		// might fall on a node with a laggy kube-proxy.
		cmd := fmt.Sprintf(`for i in $(seq 1 300); do if ss -ant46 'sport = :%d' | grep ^LISTEN; then exit 0; fi; sleep 1; done; exit 1`, nodePort)
		stdout, err := framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
		if err != nil {
			framework.Failf("expected node port %d to be in use, stdout: %v. err: %v", nodePort, stdout, err)
		}
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should be able to change the type and ports of a service [Slow] [DisabledForLargeClusters]", func() {
		// requires cloud load-balancer support
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerSupportsUDP := !framework.ProviderIs("aws")

		loadBalancerLagTimeout := framework.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = framework.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := framework.LoadBalancerCreateTimeoutDefault
		if nodes := framework.GetReadySchedulableNodesOrDie(cs); len(nodes.Items) > framework.LargeClusterMinNodesNumber {
			loadBalancerCreateTimeout = framework.LoadBalancerCreateTimeoutLarge
		}

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		e2elog.Logf("namespace for TCP test: %s", ns1)

		ginkgo.By("creating a second namespace")
		namespacePtr, err := f.CreateNamespace("services", nil)
		framework.ExpectNoError(err, "failed to create namespace")
		ns2 := namespacePtr.Name // LB2 in ns2 on UDP
		e2elog.Logf("namespace for UDP test: %s", ns2)

		jig := framework.NewServiceTestJig(cs, serviceName)
		nodeIP := framework.PickNodeIP(jig.Client) // for later

		// Test TCP and UDP Services.  Services with the same name in different
		// namespaces should get different node ports and load balancers.

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpService := jig.CreateTCPServiceOrFail(ns1, nil)
		jig.SanityCheckService(tcpService, v1.ServiceTypeClusterIP)

		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in namespace " + ns2)
		udpService := jig.CreateUDPServiceOrFail(ns2, nil)
		jig.SanityCheckService(udpService, v1.ServiceTypeClusterIP)

		ginkgo.By("verifying that TCP and UDP use the same port")
		if tcpService.Spec.Ports[0].Port != udpService.Spec.Ports[0].Port {
			framework.Failf("expected to use the same port for TCP and UDP")
		}
		svcPort := int(tcpService.Spec.Ports[0].Port)
		e2elog.Logf("service port (TCP and UDP): %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		jig.RunOrFail(ns1, nil)

		ginkgo.By("creating a pod to be part of the UDP service " + serviceName)
		jig.RunOrFail(ns2, nil)

		// Change the services to NodePort.

		ginkgo.By("changing the TCP service to type=NodePort")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		jig.SanityCheckService(tcpService, v1.ServiceTypeNodePort)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		e2elog.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("changing the UDP service to type=NodePort")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		jig.SanityCheckService(udpService, v1.ServiceTypeNodePort)
		udpNodePort := int(udpService.Spec.Ports[0].NodePort)
		e2elog.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		// Change the services to LoadBalancer.

		// Here we test that LoadBalancers can receive static IP addresses.  This isn't
		// necessary, but is an additional feature this monolithic test checks.
		requestedIP := ""
		staticIPName := ""
		if framework.ProviderIs("gce", "gke") {
			ginkgo.By("creating a static load balancer IP")
			staticIPName = fmt.Sprintf("e2e-external-lb-test-%s", framework.RunID)
			gceCloud, err := gce.GetGCECloud()
			framework.ExpectNoError(err, "failed to get GCE cloud provider")

			err = gceCloud.ReserveRegionAddress(&compute.Address{Name: staticIPName}, gceCloud.Region())
			defer func() {
				if staticIPName != "" {
					// Release GCE static IP - this is not kube-managed and will not be automatically released.
					if err := gceCloud.DeleteRegionAddress(staticIPName, gceCloud.Region()); err != nil {
						e2elog.Logf("failed to release static IP %s: %v", staticIPName, err)
					}
				}
			}()
			framework.ExpectNoError(err, "failed to create region address: %s", staticIPName)
			reservedAddr, err := gceCloud.GetRegionAddress(staticIPName, gceCloud.Region())
			framework.ExpectNoError(err, "failed to get region address: %s", staticIPName)

			requestedIP = reservedAddr.Address
			e2elog.Logf("Allocated static load balancer IP: %s", requestedIP)
		}

		ginkgo.By("changing the TCP service to type=LoadBalancer")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *v1.Service) {
			s.Spec.LoadBalancerIP = requestedIP // will be "" if not applicable
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})

		if loadBalancerSupportsUDP {
			ginkgo.By("changing the UDP service to type=LoadBalancer")
			udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *v1.Service) {
				s.Spec.Type = v1.ServiceTypeLoadBalancer
			})
		}
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(tcpService))
		if loadBalancerSupportsUDP {
			serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(udpService))
		}

		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService = jig.WaitForLoadBalancerOrFail(ns1, tcpService.Name, loadBalancerCreateTimeout)
		jig.SanityCheckService(tcpService, v1.ServiceTypeLoadBalancer)
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", tcpNodePort, tcpService.Spec.Ports[0].NodePort)
		}
		if requestedIP != "" && framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != requestedIP {
			framework.Failf("unexpected TCP Status.LoadBalancer.Ingress (expected %s, got %s)", requestedIP, framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		tcpIngressIP := framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		e2elog.Logf("TCP load balancer: %s", tcpIngressIP)

		if framework.ProviderIs("gce", "gke") {
			// Do this as early as possible, which overrides the `defer` above.
			// This is mostly out of fear of leaking the IP in a timeout case
			// (as of this writing we're not 100% sure where the leaks are
			// coming from, so this is first-aid rather than surgery).
			ginkgo.By("demoting the static IP to ephemeral")
			if staticIPName != "" {
				gceCloud, err := gce.GetGCECloud()
				framework.ExpectNoError(err, "failed to get GCE cloud provider")
				// Deleting it after it is attached "demotes" it to an
				// ephemeral IP, which can be auto-released.
				if err := gceCloud.DeleteRegionAddress(staticIPName, gceCloud.Region()); err != nil {
					framework.Failf("failed to release static IP %s: %v", staticIPName, err)
				}
				staticIPName = ""
			}
		}

		var udpIngressIP string
		if loadBalancerSupportsUDP {
			ginkgo.By("waiting for the UDP service to have a load balancer")
			// 2nd one should be faster since they ran in parallel.
			udpService = jig.WaitForLoadBalancerOrFail(ns2, udpService.Name, loadBalancerCreateTimeout)
			jig.SanityCheckService(udpService, v1.ServiceTypeLoadBalancer)
			if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
				framework.Failf("UDP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", udpNodePort, udpService.Spec.Ports[0].NodePort)
			}
			udpIngressIP = framework.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
			e2elog.Logf("UDP load balancer: %s", udpIngressIP)

			ginkgo.By("verifying that TCP and UDP use different load balancers")
			if tcpIngressIP == udpIngressIP {
				framework.Failf("Load balancers are not different: %s", framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
			}
		}

		ginkgo.By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' node ports.

		ginkgo.By("changing the TCP service's NodePort")
		tcpService = jig.ChangeServiceNodePortOrFail(ns1, tcpService.Name, tcpNodePort)
		jig.SanityCheckService(tcpService, v1.ServiceTypeLoadBalancer)
		tcpNodePortOld := tcpNodePort
		tcpNodePort = int(tcpService.Spec.Ports[0].NodePort)
		if tcpNodePort == tcpNodePortOld {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) did not change", tcpNodePort)
		}
		if framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		e2elog.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("changing the UDP service's NodePort")
		udpService = jig.ChangeServiceNodePortOrFail(ns2, udpService.Name, udpNodePort)
		if loadBalancerSupportsUDP {
			jig.SanityCheckService(udpService, v1.ServiceTypeLoadBalancer)
		} else {
			jig.SanityCheckService(udpService, v1.ServiceTypeNodePort)
		}
		udpNodePortOld := udpNodePort
		udpNodePort = int(udpService.Spec.Ports[0].NodePort)
		if udpNodePort == udpNodePortOld {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) did not change", udpNodePort)
		}
		if loadBalancerSupportsUDP && framework.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, framework.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}
		e2elog.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the TCP service's new NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's new NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("checking the old TCP NodePort is closed")
		jig.TestNotReachableHTTP(nodeIP, tcpNodePortOld, framework.KubeProxyLagTimeout)

		ginkgo.By("checking the old UDP NodePort is closed")
		jig.TestNotReachableUDP(nodeIP, udpNodePortOld, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' main ports.

		ginkgo.By("changing the TCP service's port")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		jig.SanityCheckService(tcpService, v1.ServiceTypeLoadBalancer)
		svcPortOld := svcPort
		svcPort = int(tcpService.Spec.Ports[0].Port)
		if svcPort == svcPortOld {
			framework.Failf("TCP Spec.Ports[0].Port (%d) did not change", svcPort)
		}
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) changed", tcpService.Spec.Ports[0].NodePort)
		}
		if framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, framework.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}

		ginkgo.By("changing the UDP service's port")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		if loadBalancerSupportsUDP {
			jig.SanityCheckService(udpService, v1.ServiceTypeLoadBalancer)
		} else {
			jig.SanityCheckService(udpService, v1.ServiceTypeNodePort)
		}
		if int(udpService.Spec.Ports[0].Port) != svcPort {
			framework.Failf("UDP Spec.Ports[0].Port (%d) did not change", udpService.Spec.Ports[0].Port)
		}
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) changed", udpService.Spec.Ports[0].NodePort)
		}
		if loadBalancerSupportsUDP && framework.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, framework.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}

		e2elog.Logf("service port (TCP and UDP): %d", svcPort)

		ginkgo.By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		ginkgo.By("Scaling the pods to 0")
		jig.Scale(ns1, 0)
		jig.Scale(ns2, 0)

		ginkgo.By("looking for ICMP REJECT on the TCP service's NodePort")
		jig.TestRejectedHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the UDP service's NodePort")
		jig.TestRejectedUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the TCP service's LoadBalancer")
		jig.TestRejectedHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("looking for ICMP REJECT on the UDP service's LoadBalancer")
			jig.TestRejectedUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		ginkgo.By("Scaling the pods to 1")
		jig.Scale(ns1, 1)
		jig.Scale(ns2, 1)

		ginkgo.By("hitting the TCP service's NodePort")
		jig.TestReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		jig.TestReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			jig.TestReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		// Change the services back to ClusterIP.

		ginkgo.By("changing TCP service back to type=ClusterIP")
		tcpService = jig.UpdateServiceOrFail(ns1, tcpService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		// Wait for the load balancer to be destroyed asynchronously
		tcpService = jig.WaitForLoadBalancerDestroyOrFail(ns1, tcpService.Name, tcpIngressIP, svcPort, loadBalancerCreateTimeout)
		jig.SanityCheckService(tcpService, v1.ServiceTypeClusterIP)

		ginkgo.By("changing UDP service back to type=ClusterIP")
		udpService = jig.UpdateServiceOrFail(ns2, udpService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		if loadBalancerSupportsUDP {
			// Wait for the load balancer to be destroyed asynchronously
			udpService = jig.WaitForLoadBalancerDestroyOrFail(ns2, udpService.Name, udpIngressIP, svcPort, loadBalancerCreateTimeout)
			jig.SanityCheckService(udpService, v1.ServiceTypeClusterIP)
		}

		ginkgo.By("checking the TCP NodePort is closed")
		jig.TestNotReachableHTTP(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("checking the UDP NodePort is closed")
		jig.TestNotReachableUDP(nodeIP, udpNodePort, framework.KubeProxyLagTimeout)

		ginkgo.By("checking the TCP LoadBalancer is closed")
		jig.TestNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("checking the UDP LoadBalancer is closed")
			jig.TestNotReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}
	})

	ginkgo.It("should be able to update NodePorts with two same port numbers but different protocols", func() {
		serviceName := "nodeport-update-service"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		tcpService := jig.CreateTCPServiceOrFail(ns, nil)
		defer func() {
			e2elog.Logf("Cleaning up the updating NodePorts test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		jig.SanityCheckService(tcpService, v1.ServiceTypeClusterIP)
		svcPort := int(tcpService.Spec.Ports[0].Port)
		e2elog.Logf("service port TCP: %d", svcPort)

		// Change the services to NodePort and add a UDP port.

		ginkgo.By("changing the TCP service to type=NodePort and add a UDP port")
		newService := jig.UpdateServiceOrFail(ns, tcpService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
			s.Spec.Ports = []v1.ServicePort{
				{
					Name:     "tcp-port",
					Port:     80,
					Protocol: v1.ProtocolTCP,
				},
				{
					Name:     "udp-port",
					Port:     80,
					Protocol: v1.ProtocolUDP,
				},
			}
		})
		jig.SanityCheckService(newService, v1.ServiceTypeNodePort)
		if len(newService.Spec.Ports) != 2 {
			framework.Failf("new service should have two Ports")
		}
		for _, port := range newService.Spec.Ports {
			if port.NodePort == 0 {
				framework.Failf("new service failed to allocate NodePort for Port %s", port.Name)
			}

			e2elog.Logf("new service allocates NodePort %d for Port %s", port.NodePort, port.Name)
		}
	})

	ginkgo.It("should be able to change the type from ExternalName to ClusterIP", func() {
		serviceName := "externalname-service"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ExternalName in namespace " + ns)
		externalNameService := jig.CreateExternalNameServiceOrFail(ns, nil)
		defer func() {
			e2elog.Logf("Cleaning up the ExternalName to ClusterIP test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()
		jig.SanityCheckService(externalNameService, v1.ServiceTypeExternalName)
		ginkgo.By("changing the ExternalName service to type=ClusterIP")
		clusterIPService := jig.UpdateServiceOrFail(ns, externalNameService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.ExternalName = ""
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
			}
		})
		jig.SanityCheckService(clusterIPService, v1.ServiceTypeClusterIP)
	})

	ginkgo.It("should be able to change the type from ExternalName to NodePort", func() {
		serviceName := "externalname-service"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ExternalName in namespace " + ns)
		externalNameService := jig.CreateExternalNameServiceOrFail(ns, nil)
		defer func() {
			e2elog.Logf("Cleaning up the ExternalName to NodePort test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()
		jig.SanityCheckService(externalNameService, v1.ServiceTypeExternalName)
		ginkgo.By("changing the ExternalName service to type=NodePort")
		nodePortService := jig.UpdateServiceOrFail(ns, externalNameService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
			s.Spec.ExternalName = ""
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
			}
		})
		jig.SanityCheckService(nodePortService, v1.ServiceTypeNodePort)
	})

	ginkgo.It("should be able to change the type from ClusterIP to ExternalName", func() {
		serviceName := "clusterip-service"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ClusterIP in namespace " + ns)
		clusterIPService := jig.CreateTCPServiceOrFail(ns, nil)
		defer func() {
			e2elog.Logf("Cleaning up the ClusterIP to ExternalName test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()
		jig.SanityCheckService(clusterIPService, v1.ServiceTypeClusterIP)
		ginkgo.By("changing the ClusterIP service to type=ExternalName")
		externalNameService := jig.UpdateServiceOrFail(ns, clusterIPService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeExternalName
			s.Spec.ExternalName = "foo.example.com"
			s.Spec.ClusterIP = ""
		})
		jig.SanityCheckService(externalNameService, v1.ServiceTypeExternalName)
	})

	ginkgo.It("should be able to change the type from NodePort to ExternalName", func() {
		serviceName := "nodeport-service"
		ns := f.Namespace.Name
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=NodePort in namespace " + ns)
		nodePortService := jig.CreateTCPServiceOrFail(ns, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		defer func() {
			e2elog.Logf("Cleaning up the NodePort to ExternalName test service")
			err := cs.CoreV1().Services(ns).Delete(serviceName, nil)
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()
		jig.SanityCheckService(nodePortService, v1.ServiceTypeNodePort)
		ginkgo.By("changing the NodePort service to type=ExternalName")
		externalNameService := jig.UpdateServiceOrFail(ns, nodePortService.Name, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeExternalName
			s.Spec.ExternalName = "foo.example.com"
			s.Spec.ClusterIP = ""
			s.Spec.Ports[0].NodePort = 0
		})
		jig.SanityCheckService(externalNameService, v1.ServiceTypeExternalName)
	})

	ginkgo.It("should use same NodePort with same port but different protocols", func() {
		serviceName := "nodeports"
		ns := f.Namespace.Name

		t := framework.NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + serviceName + " with same NodePort but different protocols in namespace " + ns)
		service := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      t.ServiceName,
				Namespace: t.Namespace,
			},
			Spec: v1.ServiceSpec{
				Selector: t.Labels,
				Type:     v1.ServiceTypeNodePort,
				Ports: []v1.ServicePort{
					{
						Name:     "tcp-port",
						Port:     53,
						Protocol: v1.ProtocolTCP,
					},
					{
						Name:     "udp-port",
						Port:     53,
						Protocol: v1.ProtocolUDP,
					},
				},
			},
		}
		result, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		if len(result.Spec.Ports) != 2 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", result)
		}
		if result.Spec.Ports[0].NodePort != result.Spec.Ports[1].NodePort {
			framework.Failf("should use same NodePort for new service: %v", result)
		}
	})

	ginkgo.It("should prevent NodePort collisions", func() {
		// TODO: use the ServiceTestJig here
		baseName := "nodeport-collision-"
		serviceName1 := baseName + "1"
		serviceName2 := baseName + "2"
		ns := f.Namespace.Name

		t := framework.NewServerTest(cs, ns, serviceName1)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + serviceName1 + " with type NodePort in namespace " + ns)
		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		result, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName1, ns)

		if result.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", result)
		}
		if len(result.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", result)
		}
		port := result.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].NodePort for new service: %v", result)
		}

		ginkgo.By("creating service " + serviceName2 + " with conflicting NodePort")
		service2 := t.BuildServiceSpec()
		service2.Name = serviceName2
		service2.Spec.Type = v1.ServiceTypeNodePort
		service2.Spec.Ports[0].NodePort = port.NodePort
		result2, err := t.CreateService(service2)
		if err == nil {
			framework.Failf("Created service with conflicting NodePort: %v", result2)
		}
		expectedErr := fmt.Sprintf("%d.*port is already allocated", port.NodePort)
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))

		ginkgo.By("deleting service " + serviceName1 + " to release NodePort")
		err = t.DeleteService(serviceName1)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName1, ns)

		ginkgo.By("creating service " + serviceName2 + " with no-longer-conflicting NodePort")
		_, err = t.CreateService(service2)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName1, ns)
	})

	ginkgo.It("should check NodePort out-of-range", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-range-test"
		ns := f.Namespace.Name

		t := framework.NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort

		ginkgo.By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		if service.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !framework.ServiceNodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}

		outOfRangeNodePort := 0
		rand.Seed(time.Now().UnixNano())
		for {
			outOfRangeNodePort = 1 + rand.Intn(65535)
			if !framework.ServiceNodePortRange.Contains(outOfRangeNodePort) {
				break
			}
		}
		ginkgo.By(fmt.Sprintf("changing service "+serviceName+" to out-of-range NodePort %d", outOfRangeNodePort))
		result, err := framework.UpdateService(cs, ns, serviceName, func(s *v1.Service) {
			s.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		})
		if err == nil {
			framework.Failf("failed to prevent update of service with out-of-range NodePort: %v", result)
		}
		expectedErr := fmt.Sprintf("%d.*port is not in the valid range", outOfRangeNodePort)
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))

		ginkgo.By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)

		ginkgo.By(fmt.Sprintf("creating service "+serviceName+" with out-of-range NodePort %d", outOfRangeNodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		service, err = t.CreateService(service)
		if err == nil {
			framework.Failf("failed to prevent create of service with out-of-range NodePort (%d): %v", outOfRangeNodePort, service)
		}
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))
	})

	ginkgo.It("should release NodePorts on delete", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-reuse"
		ns := f.Namespace.Name

		t := framework.NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort

		ginkgo.By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		if service.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !framework.ServiceNodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}
		nodePort := port.NodePort

		ginkgo.By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)

		hostExec := e2epod.LaunchHostExecPod(f.ClientSet, f.Namespace.Name, "hostexec")
		cmd := fmt.Sprintf(`! ss -ant46 'sport = :%d' | tail -n +2 | grep LISTEN`, nodePort)
		var stdout string
		if pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				e2elog.Logf("expected node port (%d) to not be in use, stdout: %v", nodePort, stdout)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected node port (%d) to not be in use in %v, stdout: %v", nodePort, framework.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By(fmt.Sprintf("creating service "+serviceName+" with same NodePort %d", nodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = nodePort
		service, err = t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)
	})

	ginkgo.It("should create endpoints for unready pods", func() {
		serviceName := "tolerate-unready"
		ns := f.Namespace.Name

		t := framework.NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		t.Name = "slow-terminating-unready-pod"
		t.Image = imageutils.GetE2EImage(imageutils.Netexec)
		port := 80
		terminateSeconds := int64(600)

		service := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:        t.ServiceName,
				Namespace:   t.Namespace,
				Annotations: map[string]string{endpoint.TolerateUnreadyEndpointsAnnotation: "true"},
			},
			Spec: v1.ServiceSpec{
				Selector: t.Labels,
				Ports: []v1.ServicePort{{
					Name:       "http",
					Port:       int32(port),
					TargetPort: intstr.FromInt(port),
				}},
			},
		}
		rcSpec := framework.RcByNameContainer(t.Name, 1, t.Image, t.Labels, v1.Container{
			Args:  []string{fmt.Sprintf("--http-port=%d", port)},
			Name:  t.Name,
			Image: t.Image,
			Ports: []v1.ContainerPort{{ContainerPort: int32(port), Protocol: v1.ProtocolTCP}},
			ReadinessProbe: &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"/bin/false"},
					},
				},
			},
			Lifecycle: &v1.Lifecycle{
				PreStop: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"/bin/sleep", fmt.Sprintf("%d", terminateSeconds)},
					},
				},
			},
		}, nil)
		rcSpec.Spec.Template.Spec.TerminationGracePeriodSeconds = &terminateSeconds

		ginkgo.By(fmt.Sprintf("creating RC %v with selectors %v", rcSpec.Name, rcSpec.Spec.Selector))
		_, err := t.CreateRC(rcSpec)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("creating Service %v with selectors %v", service.Name, service.Spec.Selector))
		_, err = t.CreateService(service)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying pods for RC " + t.Name)
		framework.ExpectNoError(e2epod.VerifyPods(t.Client, t.Namespace, t.Name, false, 1))

		svcName := fmt.Sprintf("%v.%v.svc.%v", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		ginkgo.By("Waiting for endpoints of Service with DNS name " + svcName)

		execPodName := e2epod.CreateExecPodOrFail(f.ClientSet, f.Namespace.Name, "execpod-", nil)
		cmd := fmt.Sprintf("wget -qO- http://%s:%d/", svcName, port)
		var stdout string
		if pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				e2elog.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, framework.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Scaling down replication controller to zero")
		framework.ScaleRC(f.ClientSet, f.ScalesGetter, t.Namespace, rcSpec.Name, 0, false)

		ginkgo.By("Update service to not tolerate unready services")
		_, err = framework.UpdateService(f.ClientSet, t.Namespace, t.ServiceName, func(s *v1.Service) {
			s.ObjectMeta.Annotations[endpoint.TolerateUnreadyEndpointsAnnotation] = "false"
		})
		framework.ExpectNoError(err)

		ginkgo.By("Check if pod is unreachable")
		cmd = fmt.Sprintf("wget -qO- -T 2 http://%s:%d/; test \"$?\" -eq \"1\"", svcName, port)
		if pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				e2elog.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, framework.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Update service to tolerate unready services again")
		_, err = framework.UpdateService(f.ClientSet, t.Namespace, t.ServiceName, func(s *v1.Service) {
			s.ObjectMeta.Annotations[endpoint.TolerateUnreadyEndpointsAnnotation] = "true"
		})
		framework.ExpectNoError(err)

		ginkgo.By("Check if terminating pod is available through service")
		cmd = fmt.Sprintf("wget -qO- http://%s:%d/", svcName, port)
		if pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				e2elog.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, framework.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Remove pods immediately")
		label := labels.SelectorFromSet(labels.Set(t.Labels))
		options := metav1.ListOptions{LabelSelector: label.String()}
		podClient := t.Client.CoreV1().Pods(f.Namespace.Name)
		pods, err := podClient.List(options)
		if err != nil {
			e2elog.Logf("warning: error retrieving pods: %s", err)
		} else {
			for _, pod := range pods.Items {
				var gracePeriodSeconds int64 = 0
				err := podClient.Delete(pod.Name, &metav1.DeleteOptions{GracePeriodSeconds: &gracePeriodSeconds})
				if err != nil {
					e2elog.Logf("warning: error force deleting pod '%s': %s", pod.Name, err)
				}
			}
		}
	})

	ginkgo.It("should only allow access from service loadbalancer source ranges [Slow]", func() {
		// this feature currently supported only on GCE/GKE/AWS
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := framework.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = framework.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := framework.LoadBalancerCreateTimeoutDefault
		if nodes := framework.GetReadySchedulableNodesOrDie(cs); len(nodes.Items) > framework.LargeClusterMinNodesNumber {
			loadBalancerCreateTimeout = framework.LoadBalancerCreateTimeoutLarge
		}

		namespace := f.Namespace.Name
		serviceName := "lb-sourcerange"
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("Prepare allow source ips")
		// prepare the exec pods
		// acceptPod are allowed to access the loadbalancer
		acceptPodName := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-accept", nil)
		dropPodName := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-drop", nil)

		acceptPod, err := cs.CoreV1().Pods(namespace).Get(acceptPodName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch pod: %s in namespace: %s", acceptPodName, namespace)
		dropPod, err := cs.CoreV1().Pods(namespace).Get(dropPodName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch pod: %s in namespace: %s", dropPodName, namespace)

		ginkgo.By("creating a pod to be part of the service " + serviceName)
		// This container is an nginx container listening on port 80
		// See kubernetes/contrib/ingress/echoheaders/nginx.conf for content of response
		jig.RunOrFail(namespace, nil)
		// Create loadbalancer service with source range from node[0] and podAccept
		svc := jig.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.LoadBalancerSourceRanges = []string{acceptPod.Status.PodIP + "/32"}
		})

		// Clean up loadbalancer service
		defer func() {
			jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			err = cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		svc = jig.WaitForLoadBalancerOrFail(namespace, serviceName, loadBalancerCreateTimeout)
		jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)

		// timeout when we haven't just created the load balancer
		normalReachabilityTimeout := 2 * time.Minute

		ginkgo.By("check reachability from different sources")
		svcIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		// Wait longer as this is our first request after creation.  We can't check using a separate method,
		// because the LB should only be reachable from the "accept" pod
		framework.CheckReachabilityFromPod(true, loadBalancerLagTimeout, namespace, acceptPodName, svcIP)
		framework.CheckReachabilityFromPod(false, normalReachabilityTimeout, namespace, dropPodName, svcIP)

		ginkgo.By("Update service LoadBalancerSourceRange and check reachability")
		jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			// only allow access from dropPod
			svc.Spec.LoadBalancerSourceRanges = []string{dropPod.Status.PodIP + "/32"}
		})
		framework.CheckReachabilityFromPod(false, normalReachabilityTimeout, namespace, acceptPodName, svcIP)
		framework.CheckReachabilityFromPod(true, normalReachabilityTimeout, namespace, dropPodName, svcIP)

		ginkgo.By("Delete LoadBalancerSourceRange field and check reachability")
		jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.LoadBalancerSourceRanges = nil
		})
		framework.CheckReachabilityFromPod(true, normalReachabilityTimeout, namespace, acceptPodName, svcIP)
		framework.CheckReachabilityFromPod(true, normalReachabilityTimeout, namespace, dropPodName, svcIP)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should be able to create an internal type load balancer [Slow] [DisabledForLargeClusters]", func() {
		framework.SkipUnlessProviderIs("azure", "gke", "gce")

		createTimeout := framework.LoadBalancerCreateTimeoutDefault
		if nodes := framework.GetReadySchedulableNodesOrDie(cs); len(nodes.Items) > framework.LargeClusterMinNodesNumber {
			createTimeout = framework.LoadBalancerCreateTimeoutLarge
		}

		pollInterval := framework.Poll * 10

		namespace := f.Namespace.Name
		serviceName := "lb-internal"
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("creating pod to be part of service " + serviceName)
		jig.RunOrFail(namespace, nil)

		enableILB, disableILB := framework.EnableAndDisableInternalLB()

		isInternalEndpoint := func(lbIngress *v1.LoadBalancerIngress) bool {
			ingressEndpoint := framework.GetIngressPoint(lbIngress)
			// Needs update for providers using hostname as endpoint.
			return strings.HasPrefix(ingressEndpoint, "10.")
		}

		ginkgo.By("creating a service with type LoadBalancer and cloud specific Internal-LB annotation enabled")
		svc := jig.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			enableILB(svc)
		})
		svc = jig.WaitForLoadBalancerOrFail(namespace, serviceName, createTimeout)
		jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
		lbIngress := &svc.Status.LoadBalancer.Ingress[0]
		svcPort := int(svc.Spec.Ports[0].Port)
		// should have an internal IP.
		gomega.Expect(isInternalEndpoint(lbIngress)).To(gomega.BeTrue())

		// ILBs are not accessible from the test orchestrator, so it's necessary to use
		//  a pod to test the service.
		ginkgo.By("hitting the internal load balancer from pod")
		e2elog.Logf("creating pod with host network")
		hostExec := e2epod.LaunchHostExecPod(f.ClientSet, f.Namespace.Name, "ilb-host-exec")

		e2elog.Logf("Waiting up to %v for service %q's internal LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP := framework.GetIngressPoint(lbIngress)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			cmd := fmt.Sprintf(`curl -m 5 'http://%v:%v/echo?msg=hello'`, tcpIngressIP, svcPort)
			stdout, err := framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				e2elog.Logf("error curling; stdout: %v. err: %v", stdout, err)
				return false, nil
			}

			if !strings.Contains(stdout, "hello") {
				e2elog.Logf("Expected output to contain 'hello', got %q; retrying...", stdout)
				return false, nil
			}

			e2elog.Logf("Successful curl; stdout: %v", stdout)
			return true, nil
		}); pollErr != nil {
			framework.Failf("ginkgo.Failed to hit ILB IP, err: %v", pollErr)
		}

		ginkgo.By("switching to external type LoadBalancer")
		svc = jig.UpdateServiceOrFail(namespace, serviceName, func(svc *v1.Service) {
			disableILB(svc)
		})
		e2elog.Logf("Waiting up to %v for service %q to have an external LoadBalancer", createTimeout, serviceName)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			svc, err := jig.Client.CoreV1().Services(namespace).Get(serviceName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			lbIngress = &svc.Status.LoadBalancer.Ingress[0]
			return !isInternalEndpoint(lbIngress), nil
		}); pollErr != nil {
			framework.Failf("Loadbalancer IP not changed to external.")
		}
		// should have an external IP.
		jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
		gomega.Expect(isInternalEndpoint(lbIngress)).To(gomega.BeFalse())

		ginkgo.By("hitting the external load balancer")
		e2elog.Logf("Waiting up to %v for service %q's external LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP = framework.GetIngressPoint(lbIngress)
		jig.TestReachableHTTP(tcpIngressIP, svcPort, framework.LoadBalancerLagTimeoutDefault)

		// GCE cannot test a specific IP because the test may not own it. This cloud specific condition
		// will be removed when GCP supports similar functionality.
		if framework.ProviderIs("azure") {
			ginkgo.By("switching back to interal type LoadBalancer, with static IP specified.")
			internalStaticIP := "10.240.11.11"
			svc = jig.UpdateServiceOrFail(namespace, serviceName, func(svc *v1.Service) {
				svc.Spec.LoadBalancerIP = internalStaticIP
				enableILB(svc)
			})
			e2elog.Logf("Waiting up to %v for service %q to have an internal LoadBalancer", createTimeout, serviceName)
			if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
				svc, err := jig.Client.CoreV1().Services(namespace).Get(serviceName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				lbIngress = &svc.Status.LoadBalancer.Ingress[0]
				return isInternalEndpoint(lbIngress), nil
			}); pollErr != nil {
				framework.Failf("Loadbalancer IP not changed to internal.")
			}
			// should have the given static internal IP.
			jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
			gomega.Expect(framework.GetIngressPoint(lbIngress)).To(gomega.Equal(internalStaticIP))
		}

		ginkgo.By("switching to ClusterIP type to destroy loadbalancer")
		jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, createTimeout)
	})

	// This test creates a load balancer, make sure its health check interval
	// equals to gceHcCheckIntervalSeconds. Then the interval is manipulated
	// to be something else, see if the interval will be reconciled.
	ginkgo.It("should reconcile LB health check interval [Slow][Serial]", func() {
		const gceHcCheckIntervalSeconds = int64(8)
		// This test is for clusters on GCE.
		// (It restarts kube-controller-manager, which we don't support on GKE)
		framework.SkipUnlessProviderIs("gce")
		clusterID, err := gce.GetClusterID(cs)
		if err != nil {
			framework.Failf("framework.GetClusterID(cs) = _, %v; want nil", err)
		}
		gceCloud, err := gce.GetGCECloud()
		if err != nil {
			framework.Failf("framework.GetGCECloud() = _, %v; want nil", err)
		}

		namespace := f.Namespace.Name
		serviceName := "lb-hc-int"
		jig := framework.NewServiceTestJig(cs, serviceName)

		ginkgo.By("create load balancer service")
		// Create loadbalancer service with source range from node[0] and podAccept
		svc := jig.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})

		// Clean up loadbalancer service
		defer func() {
			jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
			})
			err = cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		svc = jig.WaitForLoadBalancerOrFail(namespace, serviceName, framework.LoadBalancerCreateTimeoutDefault)

		hcName := gcecloud.MakeNodesHealthCheckName(clusterID)
		hc, err := gceCloud.GetHTTPHealthCheck(hcName)
		if err != nil {
			framework.Failf("gceCloud.GetHttpHealthCheck(%q) = _, %v; want nil", hcName, err)
		}
		gomega.Expect(hc.CheckIntervalSec).To(gomega.Equal(gceHcCheckIntervalSeconds))

		ginkgo.By("modify the health check interval")
		hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1
		if err = gceCloud.UpdateHTTPHealthCheck(hc); err != nil {
			framework.Failf("gcecloud.UpdateHttpHealthCheck(%#v) = %v; want nil", hc, err)
		}

		ginkgo.By("restart kube-controller-manager")
		if err := framework.RestartControllerManager(); err != nil {
			framework.Failf("framework.RestartControllerManager() = %v; want nil", err)
		}
		if err := framework.WaitForControllerManagerUp(); err != nil {
			framework.Failf("framework.WaitForControllerManagerUp() = %v; want nil", err)
		}

		ginkgo.By("health check should be reconciled")
		pollInterval := framework.Poll * 10
		if pollErr := wait.PollImmediate(pollInterval, framework.LoadBalancerCreateTimeoutDefault, func() (bool, error) {
			hc, err := gceCloud.GetHTTPHealthCheck(hcName)
			if err != nil {
				e2elog.Logf("ginkgo.Failed to get HttpHealthCheck(%q): %v", hcName, err)
				return false, err
			}
			e2elog.Logf("hc.CheckIntervalSec = %v", hc.CheckIntervalSec)
			return hc.CheckIntervalSec == gceHcCheckIntervalSeconds, nil
		}); pollErr != nil {
			framework.Failf("Health check %q does not reconcile its check interval to %d.", hcName, gceHcCheckIntervalSeconds)
		}
	})

	ginkgo.It("should have session affinity work for service with type clusterIP", func() {
		svc := getServeHostnameService("affinity-clusterip")
		svc.Spec.Type = v1.ServiceTypeClusterIP
		execAffinityTestForNonLBService(f, cs, svc)
	})

	ginkgo.It("should be able to switch session affinity for service with type clusterIP", func() {
		svc := getServeHostnameService("affinity-clusterip-transition")
		svc.Spec.Type = v1.ServiceTypeClusterIP
		execAffinityTestForNonLBServiceWithTransition(f, cs, svc)
	})

	ginkgo.It("should have session affinity work for NodePort service", func() {
		svc := getServeHostnameService("affinity-nodeport")
		svc.Spec.Type = v1.ServiceTypeNodePort
		execAffinityTestForNonLBService(f, cs, svc)
	})

	ginkgo.It("should be able to switch session affinity for NodePort service", func() {
		svc := getServeHostnameService("affinity-nodeport-transition")
		svc.Spec.Type = v1.ServiceTypeNodePort
		execAffinityTestForNonLBServiceWithTransition(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		framework.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		framework.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		framework.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		framework.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	ginkgo.It("should implement service.kubernetes.io/service-proxy-name", func() {
		// this test uses e2essh.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		// this test does not work if the Node does not support SSH Key
		framework.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort
		serviceProxyNameLabels := map[string]string{"service.kubernetes.io/service-proxy-name": "foo-bar"}

		// We will create 2 services to test creating services in both states and also dynamic updates
		// svcDisabled: Created with the label, will always be disabled. We create this early and
		//              test again late to make sure it never becomes available.
		// svcToggled: Created without the label then the label is toggled verifying reachability at each step.

		ginkgo.By("creating service-disabled in namespace " + ns)
		svcDisabled := getServeHostnameService("service-proxy-disabled")
		svcDisabled.ObjectMeta.Labels = serviceProxyNameLabels
		_, svcDisabledIP, err := framework.StartServeHostnameService(cs, svcDisabled, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svcDisabledIP, ns)

		ginkgo.By("creating service in namespace " + ns)
		svcToggled := getServeHostnameService("service-proxy-toggled")
		podToggledNames, svcToggledIP, err := framework.StartServeHostnameService(cs, svcToggled, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svcToggledIP, ns)

		jig := framework.NewServiceTestJig(cs, svcToggled.ObjectMeta.Name)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is not up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceDown(cs, host, svcDisabledIP, servicePort))

		ginkgo.By("adding service-proxy-name label")
		jig.UpdateServiceOrFail(ns, svcToggled.ObjectMeta.Name, func(svc *v1.Service) {
			svc.ObjectMeta.Labels = serviceProxyNameLabels
		})

		ginkgo.By("verifying service is not up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceDown(cs, host, svcToggledIP, servicePort))

		ginkgo.By("removing service-proxy-name annotation")
		jig.UpdateServiceOrFail(ns, svcToggled.ObjectMeta.Name, func(svc *v1.Service) {
			svc.ObjectMeta.Labels = nil
		})

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceUp(cs, ns, host, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is still not up")
		framework.ExpectNoError(framework.VerifyServeHostnameServiceDown(cs, host, svcDisabledIP, servicePort))
	})

	ginkgo.It("should be rejected when no endpoints exist", func() {
		namespace := f.Namespace.Name
		serviceName := "no-pods"
		jig := framework.NewServiceTestJig(cs, serviceName)
		nodes := jig.GetNodes(framework.MaxNodesForEndpointsTests)
		labels := map[string]string{
			"nopods": "nopods",
		}
		port := 80
		ports := []v1.ServicePort{{
			Port:       int32(port),
			TargetPort: intstr.FromInt(80),
		}}

		ginkgo.By("creating a service with no endpoints")
		_, err := jig.CreateServiceWithServicePort(labels, namespace, ports)
		if err != nil {
			framework.Failf("ginkgo.Failed to create service: %v", err)
		}

		nodeName := nodes.Items[0].Name
		podName := "execpod-noendpoints"

		ginkgo.By(fmt.Sprintf("creating %v on node %v", podName, nodeName))
		execPodName := e2epod.CreateExecPodOrFail(f.ClientSet, namespace, podName, func(pod *v1.Pod) {
			pod.Spec.NodeName = nodeName
		})
		execPod, err := f.ClientSet.CoreV1().Pods(namespace).Get(execPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		serviceAddress := net.JoinHostPort(serviceName, strconv.Itoa(port))
		e2elog.Logf("waiting up to %v wget %v", framework.KubeProxyEndpointLagTimeout, serviceAddress)
		cmd := fmt.Sprintf(`wget -T 3 -qO- %v`, serviceAddress)

		ginkgo.By(fmt.Sprintf("hitting service %v from pod %v on node %v", serviceAddress, podName, nodeName))
		expectedErr := "connection refused"
		if pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyEndpointLagTimeout, func() (bool, error) {
			_, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)

			if err != nil {
				if strings.Contains(strings.ToLower(err.Error()), expectedErr) {
					e2elog.Logf("error contained '%s', as expected: %s", expectedErr, err.Error())
					return true, nil
				}
				e2elog.Logf("error didn't contain '%s', keep trying: %s", expectedErr, err.Error())
				return false, nil
			}
			return true, errors.New("expected wget call to fail")
		}); pollErr != nil {
			framework.ExpectNoError(pollErr)
		}
	})

	// This test verifies if service load balancer cleanup finalizer can be removed
	// when feature gate isn't enabled on the cluster.
	// This ensures downgrading from higher version cluster will not break LoadBalancer
	// type service.
	ginkgo.It("should remove load balancer cleanup finalizer when service is deleted [Slow]", func() {
		jig := framework.NewServiceTestJig(cs, "lb-remove-finalizer")

		ginkgo.By("Create load balancer service")
		svc := jig.CreateTCPServiceOrFail(f.Namespace.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})

		defer func() {
			waitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		ginkgo.By("Wait for load balancer to serve traffic")
		svc = jig.WaitForLoadBalancerOrFail(svc.Namespace, svc.Name, framework.GetServiceLoadBalancerCreationTimeout(cs))

		ginkgo.By("Manually add load balancer cleanup finalizer to service")
		svc.Finalizers = append(svc.Finalizers, "service.kubernetes.io/load-balancer-cleanup")
		if _, err := cs.CoreV1().Services(svc.Namespace).Update(svc); err != nil {
			framework.Failf("Failed to add finalizer to service %s/%s: %v", svc.Namespace, svc.Name, err)
		}
	})

	// This test verifies if service load balancer cleanup finalizer is properly
	// handled during service lifecycle.
	// 1. Create service with type=LoadBalancer. Finalizer should be added.
	// 2. Update service to type=ClusterIP. Finalizer should be removed.
	// 3. Update service to type=LoadBalancer. Finalizer should be added.
	// 4. Delete service with type=LoadBalancer. Finalizer should be removed.
	ginkgo.It("should handle load balancer cleanup finalizer for service [Slow] [Feature:ServiceFinalizer]", func() {
		jig := framework.NewServiceTestJig(cs, "lb-finalizer")

		ginkgo.By("Create load balancer service")
		svc := jig.CreateTCPServiceOrFail(f.Namespace.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})

		defer func() {
			waitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		ginkgo.By("Wait for load balancer to serve traffic")
		svc = jig.WaitForLoadBalancerOrFail(svc.Namespace, svc.Name, framework.GetServiceLoadBalancerCreationTimeout(cs))

		ginkgo.By("Check if finalizer presents on service with type=LoadBalancer")
		waitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)

		ginkgo.By("Check if finalizer is removed on service after changed to type=ClusterIP")
		jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, framework.GetServiceLoadBalancerCreationTimeout(cs))
		waitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, false)

		ginkgo.By("Check if finalizer is added back to service after changed to type=LoadBalancer")
		jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeLoadBalancer, framework.GetServiceLoadBalancerCreationTimeout(cs))
		waitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)
	})
})

func waitForServiceDeletedWithFinalizer(cs clientset.Interface, namespace, name string) {
	ginkgo.By("Delete service with finalizer")
	if err := cs.CoreV1().Services(namespace).Delete(name, nil); err != nil {
		framework.Failf("Failed to delete service %s/%s", namespace, name)
	}

	ginkgo.By("Wait for service to disappear")
	if pollErr := wait.PollImmediate(framework.LoadBalancerPollInterval, framework.GetServiceLoadBalancerCreationTimeout(cs), func() (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				e2elog.Logf("Service %s/%s is gone.", namespace, name)
				return true, nil
			}
			return false, err
		}
		e2elog.Logf("Service %s/%s still exists with finalizers: %v", namespace, name, svc.Finalizers)
		return false, nil
	}); pollErr != nil {
		framework.Failf("Failed to wait for service to disappear: %v", pollErr)
	}
}

func waitForServiceUpdatedWithFinalizer(cs clientset.Interface, namespace, name string, hasFinalizer bool) {
	ginkgo.By(fmt.Sprintf("Wait for service to hasFinalizer=%t", hasFinalizer))
	if pollErr := wait.PollImmediate(framework.LoadBalancerPollInterval, framework.GetServiceLoadBalancerCreationTimeout(cs), func() (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		foundFinalizer := false
		for _, finalizer := range svc.Finalizers {
			if finalizer == "service.kubernetes.io/load-balancer-cleanup" {
				foundFinalizer = true
			}
		}
		if foundFinalizer != hasFinalizer {
			e2elog.Logf("Service %s/%s hasFinalizer=%t, want %t", namespace, name, foundFinalizer, hasFinalizer)
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		framework.Failf("Failed to wait for service to hasFinalizer=%t: %v", hasFinalizer, pollErr)
	}
}

// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
var _ = SIGDescribe("ESIPP [Slow] [DisabledForLargeClusters]", func() {
	f := framework.NewDefaultFramework("esipp")
	loadBalancerCreateTimeout := framework.LoadBalancerCreateTimeoutDefault

	var cs clientset.Interface
	serviceLBNames := []string{}

	ginkgo.BeforeEach(func() {
		// requires cloud load-balancer support - this feature currently supported only on GCE/GKE
		framework.SkipUnlessProviderIs("gce", "gke")

		cs = f.ClientSet
		if nodes := framework.GetReadySchedulableNodesOrDie(cs); len(nodes.Items) > framework.LargeClusterMinNodesNumber {
			loadBalancerCreateTimeout = framework.LoadBalancerCreateTimeoutLarge
		}
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentGinkgoTestDescription().Failed {
			framework.DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			e2elog.Logf("cleaning load balancer resource for %s", lb)
			framework.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	ginkgo.It("should work for type=LoadBalancer", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-lb"
		jig := framework.NewServiceTestJig(cs, serviceName)

		svc := jig.CreateOnlyLocalLoadBalancerService(namespace, serviceName, loadBalancerCreateTimeout, true, nil)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}
		defer func() {
			jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)

			// Make sure we didn't leak the health check node port.
			threshold := 2
			for _, ips := range jig.GetEndpointNodes(svc) {
				err := jig.TestHTTPHealthCheckNodePort(ips[0], healthCheckNodePort, "/healthz", framework.KubeProxyEndpointLagTimeout, false, threshold)
				framework.ExpectNoError(err)
			}
			err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		ingressIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])

		ginkgo.By("reading clientIP using the TCP service's service port via its external VIP")
		content := jig.GetHTTPContent(ingressIP, svcTCPPort, framework.KubeProxyLagTimeout, "/clientip")
		clientIP := content.String()
		e2elog.Logf("ClientIP detected by target pod using VIP:SvcPort is %s", clientIP)

		ginkgo.By("checking if Source IP is preserved")
		if strings.HasPrefix(clientIP, "10.") {
			framework.Failf("Source IP was NOT preserved")
		}
	})

	ginkgo.It("should work for type=NodePort", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodeport"
		jig := framework.NewServiceTestJig(cs, serviceName)

		svc := jig.CreateOnlyLocalNodePortService(namespace, serviceName, true)
		defer func() {
			err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		tcpNodePort := int(svc.Spec.Ports[0].NodePort)
		endpointsNodeMap := jig.GetEndpointNodes(svc)
		path := "/clientip"

		for nodeName, nodeIPs := range endpointsNodeMap {
			nodeIP := nodeIPs[0]
			ginkgo.By(fmt.Sprintf("reading clientIP using the TCP service's NodePort, on node %v: %v%v%v", nodeName, nodeIP, tcpNodePort, path))
			content := jig.GetHTTPContent(nodeIP, tcpNodePort, framework.KubeProxyLagTimeout, path)
			clientIP := content.String()
			e2elog.Logf("ClientIP detected by target pod using NodePort is %s", clientIP)
			if strings.HasPrefix(clientIP, "10.") {
				framework.Failf("Source IP was NOT preserved")
			}
		}
	})

	ginkgo.It("should only target nodes with endpoints", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodes"
		jig := framework.NewServiceTestJig(cs, serviceName)
		nodes := jig.GetNodes(framework.MaxNodesForEndpointsTests)

		svc := jig.CreateOnlyLocalLoadBalancerService(namespace, serviceName, loadBalancerCreateTimeout, false,
			func(svc *v1.Service) {
				// Change service port to avoid collision with opened hostPorts
				// in other tests that run in parallel.
				if len(svc.Spec.Ports) != 0 {
					svc.Spec.Ports[0].TargetPort = intstr.FromInt(int(svc.Spec.Ports[0].Port))
					svc.Spec.Ports[0].Port = 8081
				}

			})
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeExternalIP)

		ingressIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcTCPPort := int(svc.Spec.Ports[0].Port)

		threshold := 2
		path := "/healthz"
		for i := 0; i < len(nodes.Items); i++ {
			endpointNodeName := nodes.Items[i].Name

			ginkgo.By("creating a pod to be part of the service " + serviceName + " on node " + endpointNodeName)
			jig.RunOrFail(namespace, func(rc *v1.ReplicationController) {
				rc.Name = serviceName
				if endpointNodeName != "" {
					rc.Spec.Template.Spec.NodeName = endpointNodeName
				}
			})

			ginkgo.By(fmt.Sprintf("waiting for service endpoint on node %v", endpointNodeName))
			jig.WaitForEndpointOnNode(namespace, serviceName, endpointNodeName)

			// HealthCheck should pass only on the node where num(endpoints) > 0
			// All other nodes should fail the healthcheck on the service healthCheckNodePort
			for n, publicIP := range ips {
				// Make sure the loadbalancer picked up the health check change.
				// Confirm traffic can reach backend through LB before checking healthcheck nodeport.
				jig.TestReachableHTTP(ingressIP, svcTCPPort, framework.KubeProxyLagTimeout)
				expectedSuccess := nodes.Items[n].Name == endpointNodeName
				port := strconv.Itoa(healthCheckNodePort)
				ipPort := net.JoinHostPort(publicIP, port)
				e2elog.Logf("Health checking %s, http://%s%s, expectedSuccess %v", nodes.Items[n].Name, ipPort, path, expectedSuccess)
				err := jig.TestHTTPHealthCheckNodePort(publicIP, healthCheckNodePort, path, framework.KubeProxyEndpointLagTimeout, expectedSuccess, threshold)
				framework.ExpectNoError(err)
			}
			framework.ExpectNoError(framework.DeleteRCAndWaitForGC(f.ClientSet, namespace, serviceName))
		}
	})

	ginkgo.It("should work from pods", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-pods"
		jig := framework.NewServiceTestJig(cs, serviceName)
		nodes := jig.GetNodes(framework.MaxNodesForEndpointsTests)

		svc := jig.CreateOnlyLocalLoadBalancerService(namespace, serviceName, loadBalancerCreateTimeout, true, nil)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		ingressIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		port := strconv.Itoa(int(svc.Spec.Ports[0].Port))
		ipPort := net.JoinHostPort(ingressIP, port)
		path := fmt.Sprintf("%s/clientip", ipPort)
		nodeName := nodes.Items[0].Name
		podName := "execpod-sourceip"

		ginkgo.By(fmt.Sprintf("Creating %v on node %v", podName, nodeName))
		execPodName := e2epod.CreateExecPodOrFail(f.ClientSet, namespace, podName, func(pod *v1.Pod) {
			pod.Spec.NodeName = nodeName
		})
		defer func() {
			err := cs.CoreV1().Pods(namespace).Delete(execPodName, nil)
			framework.ExpectNoError(err, "failed to delete pod: %s", execPodName)
		}()
		execPod, err := f.ClientSet.CoreV1().Pods(namespace).Get(execPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		e2elog.Logf("Waiting up to %v wget %v", framework.KubeProxyLagTimeout, path)
		cmd := fmt.Sprintf(`wget -T 30 -qO- %v`, path)

		var srcIP string
		ginkgo.By(fmt.Sprintf("Hitting external lb %v from pod %v on node %v", ingressIP, podName, nodeName))
		if pollErr := wait.PollImmediate(framework.Poll, framework.LoadBalancerCreateTimeoutDefault, func() (bool, error) {
			stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				e2elog.Logf("got err: %v, retry until timeout", err)
				return false, nil
			}
			srcIP = strings.TrimSpace(strings.Split(stdout, ":")[0])
			return srcIP == execPod.Status.PodIP, nil
		}); pollErr != nil {
			framework.Failf("Source IP not preserved from %v, expected '%v' got '%v'", podName, execPod.Status.PodIP, srcIP)
		}
	})

	ginkgo.It("should handle updates to ExternalTrafficPolicy field", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-update"
		jig := framework.NewServiceTestJig(cs, serviceName)

		nodes := jig.GetNodes(framework.MaxNodesForEndpointsTests)
		if len(nodes.Items) < 2 {
			framework.Failf("Need at least 2 nodes to verify source ip from a node without endpoint")
		}

		svc := jig.CreateOnlyLocalLoadBalancerService(namespace, serviceName, loadBalancerCreateTimeout, true, nil)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			jig.ChangeServiceType(svc.Namespace, svc.Name, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
		}()

		// save the health check node port because it disappears when ESIPP is turned off.
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)

		ginkgo.By("turning ESIPP off")
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		})
		if svc.Spec.HealthCheckNodePort > 0 {
			framework.Failf("Service HealthCheck NodePort still present")
		}

		endpointNodeMap := jig.GetEndpointNodes(svc)
		noEndpointNodeMap := map[string][]string{}
		for _, n := range nodes.Items {
			if _, ok := endpointNodeMap[n.Name]; ok {
				continue
			}
			noEndpointNodeMap[n.Name] = e2enode.GetAddresses(&n, v1.NodeExternalIP)
		}

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		svcNodePort := int(svc.Spec.Ports[0].NodePort)
		ingressIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		path := "/clientip"

		ginkgo.By(fmt.Sprintf("endpoints present on nodes %v, absent on nodes %v", endpointNodeMap, noEndpointNodeMap))
		for nodeName, nodeIPs := range noEndpointNodeMap {
			ginkgo.By(fmt.Sprintf("Checking %v (%v:%v%v) proxies to endpoints on another node", nodeName, nodeIPs[0], svcNodePort, path))
			jig.GetHTTPContent(nodeIPs[0], svcNodePort, framework.KubeProxyLagTimeout, path)
		}

		for nodeName, nodeIPs := range endpointNodeMap {
			ginkgo.By(fmt.Sprintf("checking kube-proxy health check fails on node with endpoint (%s), public IP %s", nodeName, nodeIPs[0]))
			var body bytes.Buffer
			pollfn := func() (bool, error) {
				result := framework.PokeHTTP(nodeIPs[0], healthCheckNodePort, "/healthz", nil)
				if result.Code == 0 {
					return true, nil
				}
				body.Reset()
				body.Write(result.Body)
				return false, nil
			}
			if pollErr := wait.PollImmediate(framework.Poll, framework.ServiceTestTimeout, pollfn); pollErr != nil {
				framework.Failf("Kube-proxy still exposing health check on node %v:%v, after ESIPP was turned off. body %s",
					nodeName, healthCheckNodePort, body.String())
			}
		}

		// Poll till kube-proxy re-adds the MASQUERADE rule on the node.
		ginkgo.By(fmt.Sprintf("checking source ip is NOT preserved through loadbalancer %v", ingressIP))
		var clientIP string
		pollErr := wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			content := jig.GetHTTPContent(ingressIP, svcTCPPort, framework.KubeProxyLagTimeout, "/clientip")
			clientIP = content.String()
			if strings.HasPrefix(clientIP, "10.") {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP WAS preserved even after ESIPP turned off. Got %v, expected a ten-dot cluster ip.", clientIP)
		}

		// TODO: We need to attempt to create another service with the previously
		// allocated healthcheck nodePort. If the health check nodePort has been
		// freed, the new service creation will succeed, upon which we cleanup.
		// If the health check nodePort has NOT been freed, the new service
		// creation will fail.

		ginkgo.By("setting ExternalTraffic field back to OnlyLocal")
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			// Request the same healthCheckNodePort as before, to test the user-requested allocation path
			svc.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		})
		pollErr = wait.PollImmediate(framework.Poll, framework.KubeProxyLagTimeout, func() (bool, error) {
			content := jig.GetHTTPContent(ingressIP, svcTCPPort, framework.KubeProxyLagTimeout, path)
			clientIP = content.String()
			ginkgo.By(fmt.Sprintf("Endpoint %v:%v%v returned client ip %v", ingressIP, svcTCPPort, path, clientIP))
			if !strings.HasPrefix(clientIP, "10.") {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP (%v) is not the client IP even after ESIPP turned on, expected a public IP.", clientIP)
		}
	})
})

func execSourceipTest(f *framework.Framework, c clientset.Interface, ns, nodeName, serviceIP string, servicePort int) (string, string) {
	e2elog.Logf("Creating an exec pod on node %v", nodeName)
	execPodName := e2epod.CreateExecPodOrFail(f.ClientSet, ns, fmt.Sprintf("execpod-sourceip-%s", nodeName), func(pod *v1.Pod) {
		pod.Spec.NodeName = nodeName
	})
	defer func() {
		e2elog.Logf("Cleaning up the exec pod")
		err := c.CoreV1().Pods(ns).Delete(execPodName, nil)
		framework.ExpectNoError(err, "failed to delete pod: %s", execPodName)
	}()
	execPod, err := f.ClientSet.CoreV1().Pods(ns).Get(execPodName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	var stdout string
	serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	timeout := 2 * time.Minute
	e2elog.Logf("Waiting up to %v wget %s", timeout, serviceIPPort)
	cmd := fmt.Sprintf(`wget -T 30 -qO- %s | grep client_address`, serviceIPPort)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		stdout, err = framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
		if err != nil {
			e2elog.Logf("got err: %v, retry until timeout", err)
			continue
		}
		// Need to check output because wget -q might omit the error.
		if strings.TrimSpace(stdout) == "" {
			e2elog.Logf("got empty stdout, retry until timeout")
			continue
		}
		break
	}

	framework.ExpectNoError(err)

	// The stdout return from RunHostCmd seems to come with "\n", so TrimSpace is needed.
	// Desired stdout in this format: client_address=x.x.x.x
	outputs := strings.Split(strings.TrimSpace(stdout), "=")
	if len(outputs) != 2 {
		// ginkgo.Fail the test if output format is unexpected.
		framework.Failf("exec pod returned unexpected stdout format: [%v]\n", stdout)
	}
	return execPod.Status.PodIP, outputs[1]
}

func execAffinityTestForNonLBServiceWithTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForNonLBServiceWithOptionalTransition(f, cs, svc, true)
}

func execAffinityTestForNonLBService(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForNonLBServiceWithOptionalTransition(f, cs, svc, false)
}

// execAffinityTestForNonLBServiceWithOptionalTransition is a helper function that wrap the logic of
// affinity test for non-load-balancer services. Session afinity will be
// enabled when the service is created. If parameter isTransitionTest is true,
// session affinity will be switched off/on and test if the service converges
// to a stable affinity state.
func execAffinityTestForNonLBServiceWithOptionalTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service, isTransitionTest bool) {
	ns := f.Namespace.Name
	numPods, servicePort, serviceName := 3, defaultServeHostnameServicePort, svc.ObjectMeta.Name
	ginkgo.By("creating service in namespace " + ns)
	serviceType := svc.Spec.Type
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	_, _, err := framework.StartServeHostnameService(cs, svc, ns, numPods)
	framework.ExpectNoError(err, "failed to create replication controller with service in the namespace: %s", ns)
	defer func() {
		framework.StopServeHostnameService(cs, ns, serviceName)
	}()
	jig := framework.NewServiceTestJig(cs, serviceName)
	svc, err = jig.Client.CoreV1().Services(ns).Get(serviceName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to fetch service: %s in namespace: %s", serviceName, ns)
	var svcIP string
	if serviceType == v1.ServiceTypeNodePort {
		nodes := framework.GetReadySchedulableNodesOrDie(cs)
		addrs := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)
		gomega.Expect(len(addrs)).To(gomega.BeNumerically(">", 0), "ginkgo.Failed to get Node internal IP")
		svcIP = addrs[0]
		servicePort = int(svc.Spec.Ports[0].NodePort)
	} else {
		svcIP = svc.Spec.ClusterIP
	}

	execPodName := e2epod.CreateExecPodOrFail(cs, ns, "execpod-affinity", nil)
	defer func() {
		e2elog.Logf("Cleaning up the exec pod")
		err := cs.CoreV1().Pods(ns).Delete(execPodName, nil)
		framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", execPodName, ns)
	}()
	execPod, err := cs.CoreV1().Pods(ns).Get(execPodName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to fetch pod: %s in namespace: %s", execPodName, ns)

	if !isTransitionTest {
		gomega.Expect(framework.CheckAffinity(jig, execPod, svcIP, servicePort, true)).To(gomega.BeTrue())
	}
	if isTransitionTest {
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		})
		gomega.Expect(framework.CheckAffinity(jig, execPod, svcIP, servicePort, false)).To(gomega.BeTrue())
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
		})
		gomega.Expect(framework.CheckAffinity(jig, execPod, svcIP, servicePort, true)).To(gomega.BeTrue())
	}
}

func execAffinityTestForLBServiceWithTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForLBServiceWithOptionalTransition(f, cs, svc, true)
}

func execAffinityTestForLBService(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForLBServiceWithOptionalTransition(f, cs, svc, false)
}

// execAffinityTestForLBServiceWithOptionalTransition is a helper function that wrap the logic of
// affinity test for load balancer services, similar to
// execAffinityTestForNonLBServiceWithOptionalTransition.
func execAffinityTestForLBServiceWithOptionalTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service, isTransitionTest bool) {
	numPods, ns, serviceName := 3, f.Namespace.Name, svc.ObjectMeta.Name

	ginkgo.By("creating service in namespace " + ns)
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	_, _, err := framework.StartServeHostnameService(cs, svc, ns, numPods)
	framework.ExpectNoError(err, "failed to create replication controller with service in the namespace: %s", ns)
	jig := framework.NewServiceTestJig(cs, serviceName)
	ginkgo.By("waiting for loadbalancer for service " + ns + "/" + serviceName)
	svc = jig.WaitForLoadBalancerOrFail(ns, serviceName, framework.LoadBalancerCreateTimeoutDefault)
	jig.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
	defer func() {
		podNodePairs, err := framework.PodNodePairs(cs, ns)
		e2elog.Logf("[pod,node] pairs: %+v; err: %v", podNodePairs, err)
		framework.StopServeHostnameService(cs, ns, serviceName)
		lb := cloudprovider.DefaultLoadBalancerName(svc)
		e2elog.Logf("cleaning load balancer resource for %s", lb)
		framework.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
	}()
	ingressIP := framework.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
	port := int(svc.Spec.Ports[0].Port)

	if !isTransitionTest {
		gomega.Expect(framework.CheckAffinity(jig, nil, ingressIP, port, true)).To(gomega.BeTrue())
	}
	if isTransitionTest {
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		})
		gomega.Expect(framework.CheckAffinity(jig, nil, ingressIP, port, false)).To(gomega.BeTrue())
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
		})
		gomega.Expect(framework.CheckAffinity(jig, nil, ingressIP, port, true)).To(gomega.BeTrue())
	}
}
