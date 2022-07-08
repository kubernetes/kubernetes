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
	"context"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2ekubesystem "k8s.io/kubernetes/test/e2e/framework/kubesystem"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	gcecloud "k8s.io/legacy-cloud-providers/gce"
	admissionapi "k8s.io/pod-security-admission/api"
	utilpointer "k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("LoadBalancers", func() {
	f := framework.NewDefaultFramework("loadbalancers")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	serviceLBNames := []string{}
	var subnetPrefix []string
	var err error

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		subnetPrefix, err = e2enode.GetSubnetPrefix(cs)
		framework.ExpectNoError(err)
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentSpecReport().Failed() {
			DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning load balancer resource for %s", lb)
			e2eservice.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	ginkgo.It("should be able to change the type and ports of a TCP service [Slow]", func() {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		nodeIP, err := e2enode.PickIP(cs) // for later
		framework.ExpectNoError(err)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpJig := e2eservice.NewTestJig(cs, ns1, serviceName)
		tcpService, err := tcpJig.CreateTCPService(nil)
		framework.ExpectNoError(err)

		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		_, err = tcpJig.Run(nil)
		framework.ExpectNoError(err)

		// Change the services to NodePort.

		ginkgo.By("changing the TCP service to type=NodePort")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

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
						framework.Logf("failed to release static IP %s: %v", staticIPName, err)
					}
				}
			}()
			framework.ExpectNoError(err, "failed to create region address: %s", staticIPName)
			reservedAddr, err := gceCloud.GetRegionAddress(staticIPName, gceCloud.Region())
			framework.ExpectNoError(err, "failed to get region address: %s", staticIPName)

			requestedIP = reservedAddr.Address
			framework.Logf("Allocated static load balancer IP: %s", requestedIP)
		}

		ginkgo.By("changing the TCP service to type=LoadBalancer")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.LoadBalancerIP = requestedIP // will be "" if not applicable
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(tcpService))
		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService, err = tcpJig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", tcpNodePort, tcpService.Spec.Ports[0].NodePort)
		}
		if requestedIP != "" && e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != requestedIP {
			framework.Failf("unexpected TCP Status.LoadBalancer.Ingress (expected %s, got %s)", requestedIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

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

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("changing the TCP service's NodePort")
		tcpService, err = tcpJig.ChangeServiceNodePort(tcpNodePort)
		framework.ExpectNoError(err)
		tcpNodePortOld := tcpNodePort
		tcpNodePort = int(tcpService.Spec.Ports[0].NodePort)
		if tcpNodePort == tcpNodePortOld {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) did not change", tcpNodePort)
		}
		if e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("hitting the TCP service's new NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the old TCP NodePort is closed")
		testNotReachableHTTP(nodeIP, tcpNodePortOld, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' main ports.

		ginkgo.By("changing the TCP service's port")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		framework.ExpectNoError(err)
		svcPortOld := svcPort
		svcPort = int(tcpService.Spec.Ports[0].Port)
		if svcPort == svcPortOld {
			framework.Failf("TCP Spec.Ports[0].Port (%d) did not change", svcPort)
		}
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) changed", tcpService.Spec.Ports[0].NodePort)
		}
		if e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}

		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 0")
		err = tcpJig.Scale(0)
		framework.ExpectNoError(err)

		ginkgo.By("looking for ICMP REJECT on the TCP service's NodePort")
		testRejectedHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the TCP service's LoadBalancer")
		testRejectedHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 1")
		err = tcpJig.Scale(1)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		// Change the services back to ClusterIP.

		ginkgo.By("changing TCP service back to type=ClusterIP")
		tcpReadback, err := tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)
		if tcpReadback.Spec.Ports[0].NodePort != 0 {
			framework.Fail("TCP Spec.Ports[0].NodePort was not cleared")
		}
		// Wait for the load balancer to be destroyed asynchronously
		_, err = tcpJig.WaitForLoadBalancerDestroy(tcpIngressIP, svcPort, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the TCP NodePort is closed")
		testNotReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the TCP LoadBalancer is closed")
		testNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	ginkgo.It("should be able to change the type and ports of a UDP service [Slow]", func() {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns2 := f.Namespace.Name // LB1 in ns2 on TCP
		framework.Logf("namespace for TCP test: %s", ns2)

		nodeIP, err := e2enode.PickIP(cs) // for later
		framework.ExpectNoError(err)

		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in namespace " + ns2)
		udpJig := e2eservice.NewTestJig(cs, ns2, serviceName)
		udpService, err := udpJig.CreateUDPService(nil)
		framework.ExpectNoError(err)

		svcPort := int(udpService.Spec.Ports[0].Port)
		framework.Logf("service port UDP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the UDP service " + serviceName)
		_, err = udpJig.Run(nil)
		framework.ExpectNoError(err)

		// Change the services to NodePort.

		ginkgo.By("changing the UDP service to type=NodePort")
		udpService, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		udpNodePort := int(udpService.Spec.Ports[0].NodePort)
		framework.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		// Change the services to LoadBalancer.

		// Here we test that LoadBalancers can receive static IP addresses.  This isn't
		// necessary, but is an additional feature this monolithic test checks.
		requestedIP := ""
		staticIPName := ""
		ginkgo.By("creating a static load balancer IP")
		staticIPName = fmt.Sprintf("e2e-external-lb-test-%s", framework.RunID)
		gceCloud, err := gce.GetGCECloud()
		framework.ExpectNoError(err, "failed to get GCE cloud provider")

		err = gceCloud.ReserveRegionAddress(&compute.Address{Name: staticIPName}, gceCloud.Region())
		defer func() {
			if staticIPName != "" {
				// Release GCE static IP - this is not kube-managed and will not be automatically released.
				if err := gceCloud.DeleteRegionAddress(staticIPName, gceCloud.Region()); err != nil {
					framework.Logf("failed to release static IP %s: %v", staticIPName, err)
				}
			}
		}()
		framework.ExpectNoError(err, "failed to create region address: %s", staticIPName)
		reservedAddr, err := gceCloud.GetRegionAddress(staticIPName, gceCloud.Region())
		framework.ExpectNoError(err, "failed to get region address: %s", staticIPName)

		requestedIP = reservedAddr.Address
		framework.Logf("Allocated static load balancer IP: %s", requestedIP)

		ginkgo.By("changing the UDP service to type=LoadBalancer")
		udpService, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(udpService))

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

		var udpIngressIP string
		ginkgo.By("waiting for the UDP service to have a load balancer")
		// 2nd one should be faster since they ran in parallel.
		udpService, err = udpJig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", udpNodePort, udpService.Spec.Ports[0].NodePort)
		}
		udpIngressIP = e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("UDP load balancer: %s", udpIngressIP)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("changing the UDP service's NodePort")
		udpService, err = udpJig.ChangeServiceNodePort(udpNodePort)
		framework.ExpectNoError(err)
		udpNodePortOld := udpNodePort
		udpNodePort = int(udpService.Spec.Ports[0].NodePort)
		if udpNodePort == udpNodePortOld {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) did not change", udpNodePort)
		}
		if e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the UDP service's new NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the old UDP NodePort is closed")
		testNotReachableUDP(nodeIP, udpNodePortOld, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' main ports.

		ginkgo.By("changing the UDP service's port")
		udpService, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		framework.ExpectNoError(err)
		svcPortOld := svcPort
		svcPort = int(udpService.Spec.Ports[0].Port)
		if svcPort == svcPortOld {
			framework.Failf("UDP Spec.Ports[0].Port (%d) did not change", svcPort)
		}
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) changed", udpService.Spec.Ports[0].NodePort)
		}
		if e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}

		framework.Logf("service port UDP: %d", svcPort)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 0")
		err = udpJig.Scale(0)
		framework.ExpectNoError(err)

		ginkgo.By("looking for ICMP REJECT on the UDP service's NodePort")
		testRejectedUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the UDP service's LoadBalancer")
		testRejectedUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 1")
		err = udpJig.Scale(1)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		// Change the services back to ClusterIP.

		ginkgo.By("changing UDP service back to type=ClusterIP")
		udpReadback, err := udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)
		if udpReadback.Spec.Ports[0].NodePort != 0 {
			framework.Fail("UDP Spec.Ports[0].NodePort was not cleared")
		}
		// Wait for the load balancer to be destroyed asynchronously
		_, err = udpJig.WaitForLoadBalancerDestroy(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the UDP NodePort is closed")
		testNotReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the UDP LoadBalancer is closed")
		testNotReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	ginkgo.It("should only allow access from service loadbalancer source ranges [Slow]", func() {
		// this feature currently supported only on GCE/GKE/AWS
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		namespace := f.Namespace.Name
		serviceName := "lb-sourcerange"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("Prepare allow source ips")
		// prepare the exec pods
		// acceptPod are allowed to access the loadbalancer
		acceptPod := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-accept", nil)
		dropPod := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-drop", nil)

		ginkgo.By("creating a pod to be part of the service " + serviceName)
		// This container is an nginx container listening on port 80
		// See kubernetes/contrib/ingress/echoheaders/nginx.conf for content of response
		_, err := jig.Run(nil)
		framework.ExpectNoError(err)
		// Make sure acceptPod is running. There are certain chances that pod might be teminated due to unexpected reasons.
		acceptPod, err = cs.CoreV1().Pods(namespace).Get(context.TODO(), acceptPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", acceptPod.Name)
		framework.ExpectEqual(acceptPod.Status.Phase, v1.PodRunning)
		framework.ExpectNotEqual(acceptPod.Status.PodIP, "")

		// Create loadbalancer service with source range from node[0] and podAccept
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.LoadBalancerSourceRanges = []string{acceptPod.Status.PodIP + "/32"}
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("check reachability from different sources")
		svcIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		// We should wait until service changes are actually propagated in the cloud-provider,
		// as this may take significant amount of time, especially in large clusters.
		// However, the information whether it was already programmed isn't achievable.
		// So we're resolving it by using loadBalancerCreateTimeout that takes cluster size into account.
		checkReachabilityFromPod(true, loadBalancerCreateTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(false, loadBalancerCreateTimeout, namespace, dropPod.Name, svcIP)

		// Make sure dropPod is running. There are certain chances that the pod might be teminated due to unexpected reasons.
		dropPod, err = cs.CoreV1().Pods(namespace).Get(context.TODO(), dropPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", dropPod.Name)
		framework.ExpectEqual(acceptPod.Status.Phase, v1.PodRunning)
		framework.ExpectNotEqual(acceptPod.Status.PodIP, "")

		ginkgo.By("Update service LoadBalancerSourceRange and check reachability")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			// only allow access from dropPod
			svc.Spec.LoadBalancerSourceRanges = []string{dropPod.Status.PodIP + "/32"}
		})
		framework.ExpectNoError(err)

		// We should wait until service changes are actually propagates, as this may take
		// significant amount of time, especially in large clusters.
		// However, the information whether it was already programmed isn't achievable.
		// So we're resolving it by using loadBalancerCreateTimeout that takes cluster size into account.
		checkReachabilityFromPod(false, loadBalancerCreateTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(true, loadBalancerCreateTimeout, namespace, dropPod.Name, svcIP)

		ginkgo.By("Delete LoadBalancerSourceRange field and check reachability")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.LoadBalancerSourceRanges = nil
		})
		framework.ExpectNoError(err)
		// We should wait until service changes are actually propagates, as this may take
		// significant amount of time, especially in large clusters.
		// However, the information whether it was already programmed isn't achievable.
		// So we're resolving it by using loadBalancerCreateTimeout that takes cluster size into account.
		checkReachabilityFromPod(true, loadBalancerCreateTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(true, loadBalancerCreateTimeout, namespace, dropPod.Name, svcIP)
	})

	ginkgo.It("should be able to create an internal type load balancer [Slow]", func() {
		e2eskipper.SkipUnlessProviderIs("azure", "gke", "gce")

		createTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)
		pollInterval := framework.Poll * 10

		namespace := f.Namespace.Name
		serviceName := "lb-internal"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("creating pod to be part of service " + serviceName)
		_, err := jig.Run(nil)
		framework.ExpectNoError(err)

		enableILB, disableILB := enableAndDisableInternalLB()

		isInternalEndpoint := func(lbIngress *v1.LoadBalancerIngress) bool {
			ingressEndpoint := e2eservice.GetIngressPoint(lbIngress)
			// Needs update for providers using hostname as endpoint.
			return strings.HasPrefix(ingressEndpoint, subnetPrefix[0]+".")
		}

		ginkgo.By("creating a service with type LoadBalancer and cloud specific Internal-LB annotation enabled")
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			enableILB(svc)
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(createTimeout)
		framework.ExpectNoError(err)
		lbIngress := &svc.Status.LoadBalancer.Ingress[0]
		svcPort := int(svc.Spec.Ports[0].Port)
		// should have an internal IP.
		if !isInternalEndpoint(lbIngress) {
			framework.Failf("lbIngress %v doesn't have an internal IP", lbIngress)
		}

		// ILBs are not accessible from the test orchestrator, so it's necessary to use
		//  a pod to test the service.
		ginkgo.By("hitting the internal load balancer from pod")
		framework.Logf("creating pod with host network")
		hostExec := launchHostExecPod(f.ClientSet, f.Namespace.Name, "ilb-host-exec")

		framework.Logf("Waiting up to %v for service %q's internal LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP := e2eservice.GetIngressPoint(lbIngress)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			cmd := fmt.Sprintf(`curl -m 5 'http://%v:%v/echo?msg=hello'`, tcpIngressIP, svcPort)
			stdout, err := framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				framework.Logf("error curling; stdout: %v. err: %v", stdout, err)
				return false, nil
			}

			if !strings.Contains(stdout, "hello") {
				framework.Logf("Expected output to contain 'hello', got %q; retrying...", stdout)
				return false, nil
			}

			framework.Logf("Successful curl; stdout: %v", stdout)
			return true, nil
		}); pollErr != nil {
			framework.Failf("ginkgo.Failed to hit ILB IP, err: %v", pollErr)
		}

		ginkgo.By("switching to external type LoadBalancer")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			disableILB(svc)
		})
		framework.ExpectNoError(err)
		framework.Logf("Waiting up to %v for service %q to have an external LoadBalancer", createTimeout, serviceName)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			svc, err := cs.CoreV1().Services(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			lbIngress = &svc.Status.LoadBalancer.Ingress[0]
			return !isInternalEndpoint(lbIngress), nil
		}); pollErr != nil {
			framework.Failf("Loadbalancer IP not changed to external.")
		}
		// should have an external IP.
		gomega.Expect(isInternalEndpoint(lbIngress)).To(gomega.BeFalse())

		ginkgo.By("hitting the external load balancer")
		framework.Logf("Waiting up to %v for service %q's external LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP = e2eservice.GetIngressPoint(lbIngress)
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, e2eservice.LoadBalancerLagTimeoutDefault)

		// GCE cannot test a specific IP because the test may not own it. This cloud specific condition
		// will be removed when GCP supports similar functionality.
		if framework.ProviderIs("azure") {
			ginkgo.By("switching back to interal type LoadBalancer, with static IP specified.")
			// For a cluster created with CAPZ, node-subnet may not be "10.240.0.0/16", e.g. "10.1.0.0/16".
			internalStaticIP := fmt.Sprintf("%s.%s.11.11", subnetPrefix[0], subnetPrefix[1])

			svc, err = jig.UpdateService(func(svc *v1.Service) {
				svc.Spec.LoadBalancerIP = internalStaticIP
				enableILB(svc)
			})
			framework.ExpectNoError(err)
			framework.Logf("Waiting up to %v for service %q to have an internal LoadBalancer", createTimeout, serviceName)
			if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
				svc, err := cs.CoreV1().Services(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				lbIngress = &svc.Status.LoadBalancer.Ingress[0]
				return isInternalEndpoint(lbIngress), nil
			}); pollErr != nil {
				framework.Failf("Loadbalancer IP not changed to internal.")
			}
			// should have the given static internal IP.
			framework.ExpectEqual(e2eservice.GetIngressPoint(lbIngress), internalStaticIP)
		}
	})

	// This test creates a load balancer, make sure its health check interval
	// equals to gceHcCheckIntervalSeconds. Then the interval is manipulated
	// to be something else, see if the interval will be reconciled.
	ginkgo.It("should reconcile LB health check interval [Slow][Serial][Disruptive]", func() {
		const gceHcCheckIntervalSeconds = int64(8)
		// This test is for clusters on GCE.
		// (It restarts kube-controller-manager, which we don't support on GKE)
		e2eskipper.SkipUnlessProviderIs("gce")
		e2eskipper.SkipUnlessSSHKeyPresent()

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
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("create load balancer service")
		// Create loadbalancer service with source range from node[0] and podAccept
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)

		hcName := gcecloud.MakeNodesHealthCheckName(clusterID)
		hc, err := gceCloud.GetHTTPHealthCheck(hcName)
		if err != nil {
			framework.Failf("gceCloud.GetHttpHealthCheck(%q) = _, %v; want nil", hcName, err)
		}
		framework.ExpectEqual(hc.CheckIntervalSec, gceHcCheckIntervalSeconds)

		ginkgo.By("modify the health check interval")
		hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1
		if err = gceCloud.UpdateHTTPHealthCheck(hc); err != nil {
			framework.Failf("gcecloud.UpdateHttpHealthCheck(%#v) = %v; want nil", hc, err)
		}

		ginkgo.By("restart kube-controller-manager")
		if err := e2ekubesystem.RestartControllerManager(); err != nil {
			framework.Failf("e2ekubesystem.RestartControllerManager() = %v; want nil", err)
		}
		if err := e2ekubesystem.WaitForControllerManagerUp(); err != nil {
			framework.Failf("e2ekubesystem.WaitForControllerManagerUp() = %v; want nil", err)
		}

		ginkgo.By("health check should be reconciled")
		pollInterval := framework.Poll * 10
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
		if pollErr := wait.PollImmediate(pollInterval, loadBalancerPropagationTimeout, func() (bool, error) {
			hc, err := gceCloud.GetHTTPHealthCheck(hcName)
			if err != nil {
				framework.Logf("ginkgo.Failed to get HttpHealthCheck(%q): %v", hcName, err)
				return false, err
			}
			framework.Logf("hc.CheckIntervalSec = %v", hc.CheckIntervalSec)
			return hc.CheckIntervalSec == gceHcCheckIntervalSeconds, nil
		}); pollErr != nil {
			framework.Failf("Health check %q does not reconcile its check interval to %d.", hcName, gceHcCheckIntervalSeconds)
		}
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	// This test verifies if service load balancer cleanup finalizer is properly
	// handled during service lifecycle.
	// 1. Create service with type=LoadBalancer. Finalizer should be added.
	// 2. Update service to type=ClusterIP. Finalizer should be removed.
	// 3. Update service to type=LoadBalancer. Finalizer should be added.
	// 4. Delete service with type=LoadBalancer. Finalizer should be removed.
	ginkgo.It("should handle load balancer cleanup finalizer for service [Slow]", func() {
		jig := e2eservice.NewTestJig(cs, f.Namespace.Name, "lb-finalizer")

		ginkgo.By("Create load balancer service")
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Check that service can be deleted with finalizer")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		ginkgo.By("Wait for load balancer to serve traffic")
		svc, err = jig.WaitForLoadBalancer(e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)

		ginkgo.By("Check if finalizer presents on service with type=LoadBalancer")
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)

		ginkgo.By("Check if finalizer is removed on service after changed to type=ClusterIP")
		err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, false)

		ginkgo.By("Check if finalizer is added back to service after changed to type=LoadBalancer")
		err = jig.ChangeServiceType(v1.ServiceTypeLoadBalancer, e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)
	})

	ginkgo.It("should be able to create LoadBalancer Service without NodePort and change it [Slow]", func() {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "reallocate-nodeport-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		nodeIP, err := e2enode.PickIP(cs) // for later
		framework.ExpectNoError(err)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpJig := e2eservice.NewTestJig(cs, ns1, serviceName)
		tcpService, err := tcpJig.CreateTCPService(nil)
		framework.ExpectNoError(err)

		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		_, err = tcpJig.Run(nil)
		framework.ExpectNoError(err)

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
						framework.Logf("failed to release static IP %s: %v", staticIPName, err)
					}
				}
			}()
			framework.ExpectNoError(err, "failed to create region address: %s", staticIPName)
			reservedAddr, err := gceCloud.GetRegionAddress(staticIPName, gceCloud.Region())
			framework.ExpectNoError(err, "failed to get region address: %s", staticIPName)

			requestedIP = reservedAddr.Address
			framework.Logf("Allocated static load balancer IP: %s", requestedIP)
		}

		ginkgo.By("changing the TCP service to type=LoadBalancer")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.LoadBalancerIP = requestedIP // will be "" if not applicable
			s.Spec.Type = v1.ServiceTypeLoadBalancer
			s.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(false)
		})
		framework.ExpectNoError(err)

		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(tcpService))
		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService, err = tcpJig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(tcpService.Spec.Ports[0].NodePort) != 0 {
			framework.Failf("TCP Spec.Ports[0].NodePort allocated %d when not expected", tcpService.Spec.Ports[0].NodePort)
		}
		if requestedIP != "" && e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != requestedIP {
			framework.Failf("unexpected TCP Status.LoadBalancer.Ingress (expected %s, got %s)", requestedIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

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

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("adding a TCP service's NodePort")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(true)
		})
		framework.ExpectNoError(err)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		if tcpNodePort == 0 {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) not allocated", tcpNodePort)
		}
		if e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("hitting the TCP service's new NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)
	})
})

var _ = common.SIGDescribe("LoadBalancers ESIPP [Slow]", func() {
	f := framework.NewDefaultFramework("esipp")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	var loadBalancerCreateTimeout time.Duration

	var cs clientset.Interface
	serviceLBNames := []string{}
	var subnetPrefix []string
	var err error

	ginkgo.BeforeEach(func() {
		// requires cloud load-balancer support - this feature currently supported only on GCE/GKE
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		cs = f.ClientSet
		loadBalancerCreateTimeout = e2eservice.GetServiceLoadBalancerCreationTimeout(cs)
		subnetPrefix, err = e2enode.GetSubnetPrefix(cs)
		framework.ExpectNoError(err)
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentSpecReport().Failed() {
			DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning load balancer resource for %s", lb)
			e2eservice.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	ginkgo.It("should work for type=LoadBalancer", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-lb"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)

			// Make sure we didn't leak the health check node port.
			const threshold = 2
			nodes, err := getEndpointNodesWithInternalIP(jig)
			framework.ExpectNoError(err)
			config := e2enetwork.NewNetworkingTestConfig(f)
			for _, internalIP := range nodes {
				err := testHTTPHealthCheckNodePortFromTestContainer(
					config,
					internalIP,
					healthCheckNodePort,
					e2eservice.KubeProxyLagTimeout,
					false,
					threshold)
				framework.ExpectNoError(err)
			}
			err = cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])

		ginkgo.By("reading clientIP using the TCP service's service port via its external VIP")
		clientIP, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, "/clientip")
		framework.ExpectNoError(err)
		framework.Logf("ClientIP detected by target pod using VIP:SvcPort is %s", clientIP)

		ginkgo.By("checking if Source IP is preserved")
		if strings.HasPrefix(clientIP, subnetPrefix[0]+".") {
			framework.Failf("Source IP was NOT preserved")
		}
	})

	ginkgo.It("should work for type=NodePort", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodeport"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalNodePortService(true)
		framework.ExpectNoError(err)
		defer func() {
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		tcpNodePort := int(svc.Spec.Ports[0].NodePort)

		endpointsNodeMap, err := getEndpointNodesWithInternalIP(jig)
		framework.ExpectNoError(err)

		dialCmd := "clientip"
		config := e2enetwork.NewNetworkingTestConfig(f)

		for nodeName, nodeIP := range endpointsNodeMap {
			ginkgo.By(fmt.Sprintf("reading clientIP using the TCP service's NodePort, on node %v: %v:%v/%v", nodeName, nodeIP, tcpNodePort, dialCmd))
			clientIP, err := GetHTTPContentFromTestContainer(config, nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout, dialCmd)
			framework.ExpectNoError(err)
			framework.Logf("ClientIP detected by target pod using NodePort is %s, the ip of test container is %s", clientIP, config.TestContainerPod.Status.PodIP)
			// the clientIP returned by agnhost contains port
			if !strings.HasPrefix(clientIP, config.TestContainerPod.Status.PodIP) {
				framework.Failf("Source IP was NOT preserved")
			}
		}
	})

	ginkgo.It("should only target nodes with endpoints", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodes"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, false,
			func(svc *v1.Service) {
				// Change service port to avoid collision with opened hostPorts
				// in other tests that run in parallel.
				if len(svc.Spec.Ports) != 0 {
					svc.Spec.Ports[0].TargetPort = intstr.FromInt(int(svc.Spec.Ports[0].Port))
					svc.Spec.Ports[0].Port = 8081
				}

			})
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcTCPPort := int(svc.Spec.Ports[0].Port)

		const threshold = 2
		config := e2enetwork.NewNetworkingTestConfig(f)
		for i := 0; i < len(nodes.Items); i++ {
			endpointNodeName := nodes.Items[i].Name

			ginkgo.By("creating a pod to be part of the service " + serviceName + " on node " + endpointNodeName)
			_, err = jig.Run(func(rc *v1.ReplicationController) {
				rc.Name = serviceName
				if endpointNodeName != "" {
					rc.Spec.Template.Spec.NodeName = endpointNodeName
				}
			})
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("waiting for service endpoint on node %v", endpointNodeName))
			err = jig.WaitForEndpointOnNode(endpointNodeName)
			framework.ExpectNoError(err)

			// HealthCheck should pass only on the node where num(endpoints) > 0
			// All other nodes should fail the healthcheck on the service healthCheckNodePort
			for n, internalIP := range ips {
				// Make sure the loadbalancer picked up the health check change.
				// Confirm traffic can reach backend through LB before checking healthcheck nodeport.
				e2eservice.TestReachableHTTP(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout)
				expectedSuccess := nodes.Items[n].Name == endpointNodeName
				port := strconv.Itoa(healthCheckNodePort)
				ipPort := net.JoinHostPort(internalIP, port)
				framework.Logf("Health checking %s, http://%s/healthz, expectedSuccess %v", nodes.Items[n].Name, ipPort, expectedSuccess)
				err := testHTTPHealthCheckNodePortFromTestContainer(
					config,
					internalIP,
					healthCheckNodePort,
					e2eservice.KubeProxyEndpointLagTimeout,
					expectedSuccess,
					threshold)
				framework.ExpectNoError(err)
			}
			framework.ExpectNoError(e2erc.DeleteRCAndWaitForGC(f.ClientSet, namespace, serviceName))
		}
	})

	ginkgo.It("should work from pods", func() {
		var err error
		namespace := f.Namespace.Name
		serviceName := "external-local-pods"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		port := strconv.Itoa(int(svc.Spec.Ports[0].Port))
		ipPort := net.JoinHostPort(ingressIP, port)
		path := fmt.Sprintf("%s/clientip", ipPort)

		ginkgo.By("Creating pause pod deployment to make sure, pausePods are in desired state")
		deployment := createPausePodDeployment(cs, "pause-pod-deployment", namespace, 1)
		framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(cs, deployment), "Failed to complete pause pod deployment")

		defer func() {
			framework.Logf("Deleting deployment")
			err = cs.AppsV1().Deployments(namespace).Delete(context.TODO(), deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete deployment %s", deployment.Name)
		}()

		deployment, err = cs.AppsV1().Deployments(namespace).Get(context.TODO(), deployment.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error in retrieving pause pod deployment")
		labelSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
		framework.ExpectNoError(err, "Error in setting LabelSelector as selector from deployment")

		pausePods, err := cs.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector.String()})
		framework.ExpectNoError(err, "Error in listing pods associated with pause pod deployments")

		pausePod := pausePods.Items[0]
		framework.Logf("Waiting up to %v curl %v", e2eservice.KubeProxyLagTimeout, path)
		cmd := fmt.Sprintf(`curl -q -s --connect-timeout 30 %v`, path)

		var srcIP string
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
		ginkgo.By(fmt.Sprintf("Hitting external lb %v from pod %v on node %v", ingressIP, pausePod.Name, pausePod.Spec.NodeName))
		if pollErr := wait.PollImmediate(framework.Poll, loadBalancerPropagationTimeout, func() (bool, error) {
			stdout, err := framework.RunHostCmd(pausePod.Namespace, pausePod.Name, cmd)
			if err != nil {
				framework.Logf("got err: %v, retry until timeout", err)
				return false, nil
			}
			srcIP = strings.TrimSpace(strings.Split(stdout, ":")[0])
			return srcIP == pausePod.Status.PodIP, nil
		}); pollErr != nil {
			framework.Failf("Source IP not preserved from %v, expected '%v' got '%v'", pausePod.Name, pausePod.Status.PodIP, srcIP)
		}
	})

	ginkgo.It("should handle updates to ExternalTrafficPolicy field", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-update"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			framework.Failf("Need at least 2 nodes to verify source ip from a node without endpoint")
		}

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		// save the health check node port because it disappears when ESIPP is turned off.
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)

		ginkgo.By("turning ESIPP off")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		})
		framework.ExpectNoError(err)
		if svc.Spec.HealthCheckNodePort > 0 {
			framework.Failf("Service HealthCheck NodePort still present")
		}

		epNodes, err := jig.ListNodesWithEndpoint()
		framework.ExpectNoError(err)
		// map from name of nodes with endpoint to internal ip
		// it is assumed that there is only a single node with the endpoint
		endpointNodeMap := make(map[string]string)
		// map from name of nodes without endpoint to internal ip
		noEndpointNodeMap := make(map[string]string)
		for _, node := range epNodes {
			ips := e2enode.GetAddresses(&node, v1.NodeInternalIP)
			if len(ips) < 1 {
				framework.Failf("No internal ip found for node %s", node.Name)
			}
			endpointNodeMap[node.Name] = ips[0]
		}
		for _, n := range nodes.Items {
			ips := e2enode.GetAddresses(&n, v1.NodeInternalIP)
			if len(ips) < 1 {
				framework.Failf("No internal ip found for node %s", n.Name)
			}
			if _, ok := endpointNodeMap[n.Name]; !ok {
				noEndpointNodeMap[n.Name] = ips[0]
			}
		}
		framework.ExpectNotEqual(len(endpointNodeMap), 0)
		framework.ExpectNotEqual(len(noEndpointNodeMap), 0)

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		svcNodePort := int(svc.Spec.Ports[0].NodePort)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		path := "/clientip"
		dialCmd := "clientip"

		config := e2enetwork.NewNetworkingTestConfig(f)

		ginkgo.By(fmt.Sprintf("endpoints present on nodes %v, absent on nodes %v", endpointNodeMap, noEndpointNodeMap))
		for nodeName, nodeIP := range noEndpointNodeMap {
			ginkgo.By(fmt.Sprintf("Checking %v (%v:%v/%v) proxies to endpoints on another node", nodeName, nodeIP[0], svcNodePort, dialCmd))
			_, err := GetHTTPContentFromTestContainer(config, nodeIP, svcNodePort, e2eservice.KubeProxyLagTimeout, dialCmd)
			framework.ExpectNoError(err, "Could not reach HTTP service through %v:%v/%v after %v", nodeIP, svcNodePort, dialCmd, e2eservice.KubeProxyLagTimeout)
		}

		for nodeName, nodeIP := range endpointNodeMap {
			ginkgo.By(fmt.Sprintf("checking kube-proxy health check fails on node with endpoint (%s), public IP %s", nodeName, nodeIP))
			var body string
			pollFn := func() (bool, error) {
				// we expect connection failure here, but not other errors
				resp, err := config.GetResponseFromTestContainer(
					"http",
					"healthz",
					nodeIP,
					healthCheckNodePort)
				if err != nil {
					return false, nil
				}
				if len(resp.Errors) > 0 {
					return true, nil
				}
				if len(resp.Responses) > 0 {
					body = resp.Responses[0]
				}
				return false, nil
			}
			if pollErr := wait.PollImmediate(framework.Poll, e2eservice.TestTimeout, pollFn); pollErr != nil {
				framework.Failf("Kube-proxy still exposing health check on node %v:%v, after ESIPP was turned off. body %s",
					nodeName, healthCheckNodePort, body)
			}
		}

		// Poll till kube-proxy re-adds the MASQUERADE rule on the node.
		ginkgo.By(fmt.Sprintf("checking source ip is NOT preserved through loadbalancer %v", ingressIP))
		var clientIP string
		pollErr := wait.PollImmediate(framework.Poll, 3*e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			clientIP, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, path)
			if err != nil {
				return false, nil
			}
			if strings.HasPrefix(clientIP, subnetPrefix[0]+".") {
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
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			// Request the same healthCheckNodePort as before, to test the user-requested allocation path
			svc.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		})
		framework.ExpectNoError(err)
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
		pollErr = wait.PollImmediate(framework.PollShortTimeout, loadBalancerPropagationTimeout, func() (bool, error) {
			clientIP, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, path)
			if err != nil {
				return false, nil
			}
			ginkgo.By(fmt.Sprintf("Endpoint %v:%v%v returned client ip %v", ingressIP, svcTCPPort, path, clientIP))
			if !strings.HasPrefix(clientIP, subnetPrefix[0]+".") {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP (%v) is not the client IP even after ESIPP turned on, expected a public IP.", clientIP)
		}
	})
})
