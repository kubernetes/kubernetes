//go:build !providerless
// +build !providerless

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
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	e2eapps "k8s.io/kubernetes/test/e2e/apps"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// getInternalIP returns node internal IP
func getInternalIP(node *v1.Node) (string, error) {
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP && address.Address != "" {
			return address.Address, nil
		}
	}
	return "", fmt.Errorf("couldn't get the internal IP of host %s with addresses %v", node.Name, node.Status.Addresses)
}

// getSubnetPrefix returns a network prefix based on one of the workers
// InternalIP adding a /16 or /64 mask depending on the IP family of the node.
// IMPORTANT: These assumes a flat network assigned to the nodes, that is common
// on cloud providers.
func getSubnetPrefix(ctx context.Context, c clientset.Interface) (*net.IPNet, error) {
	node, err := getReadySchedulableWorkerNode(ctx, c)
	if err != nil {
		return nil, fmt.Errorf("error getting a ready schedulable worker Node, err: %w", err)
	}
	internalIP, err := getInternalIP(node)
	if err != nil {
		return nil, fmt.Errorf("error getting Node internal IP, err: %w", err)
	}
	ip := netutils.ParseIPSloppy(internalIP)
	if ip == nil {
		return nil, fmt.Errorf("invalid IP address format: %s", internalIP)
	}

	// if IPv6 return a net.IPNet with IP = ip and mask /64
	ciderMask := net.CIDRMask(64, 128)
	// if IPv4 return a net.IPNet with IP = ip and mask /16
	if netutils.IsIPv4(ip) {
		ciderMask = net.CIDRMask(16, 32)
	}
	return &net.IPNet{IP: ip.Mask(ciderMask), Mask: ciderMask}, nil
}

// getReadySchedulableWorkerNode gets a single worker node which is available for
// running pods on. If there are no such available nodes it will return an error.
func getReadySchedulableWorkerNode(ctx context.Context, c clientset.Interface) (*v1.Node, error) {
	nodes, err := e2enode.GetReadySchedulableNodes(ctx, c)
	if err != nil {
		return nil, err
	}
	for i := range nodes.Items {
		node := nodes.Items[i]
		_, isMaster := node.Labels["node-role.kubernetes.io/master"]
		_, isControlPlane := node.Labels["node-role.kubernetes.io/control-plane"]
		if !isMaster && !isControlPlane {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("there are currently no ready, schedulable worker nodes in the cluster")
}

var _ = common.SIGDescribe("LoadBalancers", func() {
	f := framework.NewDefaultFramework("loadbalancers")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = f.ClientSet
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		if ginkgo.CurrentSpecReport().Failed() {
			DescribeSvc(f.Namespace.Name)
		}
	})

	f.It("should be able to change the type and ports of a TCP service", f.WithSlow(), func(ctx context.Context) {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpJig := e2eservice.NewTestJig(cs, ns1, serviceName)
		tcpService, err := tcpJig.CreateTCPService(ctx, nil)
		framework.ExpectNoError(err)

		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		_, err = tcpJig.Run(ctx, nil)
		framework.ExpectNoError(err)

		execPod := e2epod.CreateExecPodOrFail(ctx, cs, ns1, "execpod", nil)
		err = tcpJig.CheckServiceReachability(ctx, tcpService, execPod)
		framework.ExpectNoError(err)

		// Change the services to NodePort.

		ginkgo.By("changing the TCP service to type=NodePort")
		tcpService, err = tcpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		framework.Logf("TCP node port: %d", tcpNodePort)

		err = tcpJig.CheckServiceReachability(ctx, tcpService, execPod)
		framework.ExpectNoError(err)

		// Change the services to LoadBalancer.
		ginkgo.By("changing the TCP service to type=LoadBalancer")
		_, err = tcpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService, err = tcpJig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", tcpNodePort, tcpService.Spec.Ports[0].NodePort)
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

		err = tcpJig.CheckServiceReachability(ctx, tcpService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("changing the TCP service's NodePort")
		tcpService, err = tcpJig.ChangeServiceNodePort(ctx, tcpNodePort)
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

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' main ports.

		ginkgo.By("changing the TCP service's port")
		tcpService, err = tcpJig.UpdateService(ctx, func(s *v1.Service) {
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

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		ginkgo.By("Scaling the pods to 0")
		err = tcpJig.Scale(ctx, 0)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the TCP service's LoadBalancer with no backends, no answer expected")
		testNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		ginkgo.By("Scaling the pods to 1")
		err = tcpJig.Scale(ctx, 1)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services back to ClusterIP.

		ginkgo.By("changing TCP service back to type=ClusterIP")
		tcpReadback, err := tcpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)
		if tcpReadback.Spec.Ports[0].NodePort != 0 {
			framework.Fail("TCP Spec.Ports[0].NodePort was not cleared")
		}
		// Wait for the load balancer to be destroyed asynchronously
		_, err = tcpJig.WaitForLoadBalancerDestroy(ctx, tcpIngressIP, svcPort, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the TCP LoadBalancer is closed")
		testNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	f.It("should be able to change the type and ports of a UDP service", f.WithSlow(), func(ctx context.Context) {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns2 := f.Namespace.Name // LB1 in ns2 on TCP
		framework.Logf("namespace for TCP test: %s", ns2)

		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in namespace " + ns2)
		udpJig := e2eservice.NewTestJig(cs, ns2, serviceName)
		udpService, err := udpJig.CreateUDPService(ctx, nil)
		framework.ExpectNoError(err)

		svcPort := int(udpService.Spec.Ports[0].Port)
		framework.Logf("service port UDP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the UDP service " + serviceName)
		_, err = udpJig.Run(ctx, nil)
		framework.ExpectNoError(err)

		execPod := e2epod.CreateExecPodOrFail(ctx, cs, ns2, "execpod", nil)
		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		// Change the services to NodePort.

		ginkgo.By("changing the UDP service to type=NodePort")
		udpService, err = udpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		udpNodePort := int(udpService.Spec.Ports[0].NodePort)
		framework.Logf("UDP node port: %d", udpNodePort)

		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		// Change the services to LoadBalancer.
		ginkgo.By("changing the UDP service to type=LoadBalancer")
		_, err = udpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		var udpIngressIP string
		ginkgo.By("waiting for the UDP service to have a load balancer")
		// 2nd one should be faster since they ran in parallel.
		udpService, err = udpJig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", udpNodePort, udpService.Spec.Ports[0].NodePort)
		}
		udpIngressIP = e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("UDP load balancer: %s", udpIngressIP)

		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("changing the UDP service's NodePort")
		udpService, err = udpJig.ChangeServiceNodePort(ctx, udpNodePort)
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

		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' main ports.

		ginkgo.By("changing the UDP service's port")
		udpService, err = udpJig.UpdateService(ctx, func(s *v1.Service) {
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
		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 0")
		err = udpJig.Scale(ctx, 0)
		framework.ExpectNoError(err)

		ginkgo.By("looking for ICMP REJECT on the UDP service's LoadBalancer")
		testRejectedUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		ginkgo.By("Scaling the pods to 1")
		err = udpJig.Scale(ctx, 1)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's NodePort")
		err = udpJig.CheckServiceReachability(ctx, udpService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the UDP service's LoadBalancer")
		testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)

		// Change the services back to ClusterIP.

		ginkgo.By("changing UDP service back to type=ClusterIP")
		udpReadback, err := udpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)
		if udpReadback.Spec.Ports[0].NodePort != 0 {
			framework.Fail("UDP Spec.Ports[0].NodePort was not cleared")
		}
		// Wait for the load balancer to be destroyed asynchronously
		_, err = udpJig.WaitForLoadBalancerDestroy(ctx, udpIngressIP, svcPort, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the UDP LoadBalancer is closed")
		testNotReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	f.It("should only allow access from service loadbalancer source ranges", f.WithSlow(), func(ctx context.Context) {
		// this feature currently supported only on GCE/GKE/AWS/AZURE
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws", "azure")

		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		namespace := f.Namespace.Name
		serviceName := "lb-sourcerange"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("Prepare allow source ips")
		// prepare the exec pods
		// acceptPod are allowed to access the loadbalancer
		acceptPod := e2epod.CreateExecPodOrFail(ctx, cs, namespace, "execpod-accept", nil)
		dropPod := e2epod.CreateExecPodOrFail(ctx, cs, namespace, "execpod-drop", nil)

		ginkgo.By("creating a pod to be part of the service " + serviceName)
		// This container is an nginx container listening on port 80
		// See kubernetes/contrib/ingress/echoheaders/nginx.conf for content of response
		_, err := jig.Run(ctx, nil)
		framework.ExpectNoError(err)
		// Make sure acceptPod is running. There are certain chances that pod might be terminated due to unexpected reasons.
		acceptPod, err = cs.CoreV1().Pods(namespace).Get(ctx, acceptPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", acceptPod.Name)
		gomega.Expect(acceptPod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(acceptPod.Status.PodIP).ToNot(gomega.BeEmpty())

		// Create loadbalancer service with source range from node[0] and podAccept
		svc, err := jig.CreateTCPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.LoadBalancerSourceRanges = []string{acceptPod.Status.PodIP + "/32"}
		})
		framework.ExpectNoError(err)

		ginkgo.DeferCleanup(func(ctx context.Context) {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(ctx, cs, svc.Namespace, svc.Name)
		})

		svc, err = jig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("check reachability from different sources")
		svcIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		// We should wait until service changes are actually propagated in the cloud-provider,
		// as this may take significant amount of time, especially in large clusters.
		// However, the information whether it was already programmed isn't achievable.
		// So we're resolving it by using loadBalancerCreateTimeout that takes cluster size into account.
		checkReachabilityFromPod(true, loadBalancerCreateTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(false, loadBalancerCreateTimeout, namespace, dropPod.Name, svcIP)

		// Make sure dropPod is running. There are certain chances that the pod might be terminated due to unexpected reasons.
		dropPod, err = cs.CoreV1().Pods(namespace).Get(ctx, dropPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", dropPod.Name)
		gomega.Expect(acceptPod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(acceptPod.Status.PodIP).ToNot(gomega.BeEmpty())

		ginkgo.By("Update service LoadBalancerSourceRange and check reachability")
		_, err = jig.UpdateService(ctx, func(svc *v1.Service) {
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
		_, err = jig.UpdateService(ctx, func(svc *v1.Service) {
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

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should have session affinity work for LoadBalancer service with ESIPP on", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		execAffinityTestForLBService(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should be able to switch session affinity for LoadBalancer service with ESIPP on", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		execAffinityTestForLBServiceWithTransition(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should have session affinity work for LoadBalancer service with ESIPP off", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
		execAffinityTestForLBService(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should be able to switch session affinity for LoadBalancer service with ESIPP off", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
		execAffinityTestForLBServiceWithTransition(ctx, f, cs, svc)
	})

	// This test verifies if service load balancer cleanup finalizer is properly
	// handled during service lifecycle.
	// 1. Create service with type=LoadBalancer. Finalizer should be added.
	// 2. Update service to type=ClusterIP. Finalizer should be removed.
	// 3. Update service to type=LoadBalancer. Finalizer should be added.
	// 4. Delete service with type=LoadBalancer. Finalizer should be removed.
	f.It("should handle load balancer cleanup finalizer for service", f.WithSlow(), func(ctx context.Context) {
		jig := e2eservice.NewTestJig(cs, f.Namespace.Name, "lb-finalizer")

		ginkgo.By("Create load balancer service")
		svc, err := jig.CreateTCPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		ginkgo.DeferCleanup(func(ctx context.Context) {
			ginkgo.By("Check that service can be deleted with finalizer")
			e2eservice.WaitForServiceDeletedWithFinalizer(ctx, cs, svc.Namespace, svc.Name)
		})

		ginkgo.By("Wait for load balancer to serve traffic")
		svc, err = jig.WaitForLoadBalancer(ctx, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs))
		framework.ExpectNoError(err)

		ginkgo.By("Check if finalizer presents on service with type=LoadBalancer")
		e2eservice.WaitForServiceUpdatedWithFinalizer(ctx, cs, svc.Namespace, svc.Name, true)

		ginkgo.By("Check if finalizer is removed on service after changed to type=ClusterIP")
		err = jig.ChangeServiceType(ctx, v1.ServiceTypeClusterIP, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(ctx, cs, svc.Namespace, svc.Name, false)

		ginkgo.By("Check if finalizer is added back to service after changed to type=LoadBalancer")
		err = jig.ChangeServiceType(ctx, v1.ServiceTypeLoadBalancer, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(ctx, cs, svc.Namespace, svc.Name, true)
	})

	f.It("should be able to create LoadBalancer Service without NodePort and change it", f.WithSlow(), func(ctx context.Context) {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "reallocate-nodeport-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpJig := e2eservice.NewTestJig(cs, ns1, serviceName)
		tcpService, err := tcpJig.CreateTCPService(ctx, nil)
		framework.ExpectNoError(err)

		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		_, err = tcpJig.Run(ctx, nil)
		framework.ExpectNoError(err)

		// Change the services to LoadBalancer.
		ginkgo.By("changing the TCP service to type=LoadBalancer")
		_, err = tcpJig.UpdateService(ctx, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeLoadBalancer
			s.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(false)
		})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService, err = tcpJig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(tcpService.Spec.Ports[0].NodePort) != 0 {
			framework.Failf("TCP Spec.Ports[0].NodePort allocated %d when not expected", tcpService.Spec.Ports[0].NodePort)
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("adding a TCP service's NodePort")
		tcpService, err = tcpJig.UpdateService(ctx, func(s *v1.Service) {
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

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on different nodes", func(ctx context.Context) {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "azure")
		ns := f.Namespace.Name
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// Create a LoadBalancer service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=LoadBalancer in " + ns)
		_, err = udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		var udpIngressIP string
		ginkgo.By("waiting for the UDP service to have a load balancer")
		udpService, err := udpJig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		udpIngressIP = e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("UDP load balancer: %s", udpIngressIP)

		// keep hitting the loadbalancer to check it fails over to the second pod
		ginkgo.By("hitting the UDP service's LoadBalancer with same source port")
		stopCh := make(chan struct{})
		defer close(stopCh)
		var mu sync.Mutex
		hostnames := sets.NewString()
		go func() {
			defer ginkgo.GinkgoRecover()
			port := int(udpService.Spec.Ports[0].Port)
			laddr, err := net.ResolveUDPAddr("udp", ":54321")
			if err != nil {
				framework.Failf("Failed to resolve local address: %v", err)
			}
			raddr := net.UDPAddr{IP: netutils.ParseIPSloppy(udpIngressIP), Port: port}

			for {
				select {
				case <-stopCh:
					if len(hostnames) != 2 {
						framework.Failf("Failed to hit the 2 UDP LoadBalancer backends successfully, got %v", hostnames.List())
					}
					return
				default:
					time.Sleep(1 * time.Second)
				}

				conn, err := net.DialUDP("udp", laddr, &raddr)
				if err != nil {
					framework.Logf("Failed to connect to: %s %d", udpIngressIP, port)
					continue
				}
				conn.SetDeadline(time.Now().Add(3 * time.Second))
				framework.Logf("Connected successfully to: %s", raddr.String())
				conn.Write([]byte("hostname\n"))
				buff := make([]byte, 1024)
				n, _, err := conn.ReadFrom(buff)
				if err == nil {
					mu.Lock()
					hostnames.Insert(string(buff[:n]))
					mu.Unlock()
					framework.Logf("Connected successfully to hostname: %s", string(buff[:n]))
				}
				conn.Close()
			}
		}()

		// Add a backend pod to the service in one node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		serverPod1.Spec.Hostname = "hostname1"
		nodeSelection := e2epod.NodeSelection{Name: nodes.Items[0].Name}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(f).CreateSync(ctx, serverPod1)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node " + nodes.Items[0].Name)
		if err := wait.PollImmediate(1*time.Second, loadBalancerLagTimeout, func() (bool, error) {
			mu.Lock()
			defer mu.Unlock()
			return hostnames.Has(serverPod1.Spec.Hostname), nil
		}); err != nil {
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		serverPod2.Spec.Hostname = "hostname2"
		nodeSelection = e2epod.NodeSelection{Name: nodes.Items[1].Name}
		e2epod.SetNodeSelection(&serverPod2.Spec, nodeSelection)
		e2epod.NewPodClient(f).CreateSync(ctx, serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(f).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend2: {80}})

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node " + nodes.Items[1].Name)
		if err := wait.PollImmediate(1*time.Second, loadBalancerLagTimeout, func() (bool, error) {
			mu.Lock()
			defer mu.Unlock()
			return hostnames.Has(serverPod2.Spec.Hostname), nil
		}); err != nil {
			framework.Failf("Failed to connect to backend 2")
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on the same nodes", func(ctx context.Context) {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "azure")
		ns := f.Namespace.Name
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 1)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 1 {
			e2eskipper.Skipf(
				"Test requires >= 1 Ready nodes, but there are only %d nodes",
				len(nodes.Items))
		}

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// Create a LoadBalancer service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=LoadBalancer in " + ns)
		_, err = udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		var udpIngressIP string
		ginkgo.By("waiting for the UDP service to have a load balancer")
		udpService, err := udpJig.WaitForLoadBalancer(ctx, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		udpIngressIP = e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("UDP load balancer: %s", udpIngressIP)

		// keep hitting the loadbalancer to check it fails over to the second pod
		ginkgo.By("hitting the UDP service's LoadBalancer with same source port")
		stopCh := make(chan struct{})
		defer close(stopCh)
		var mu sync.Mutex
		hostnames := sets.NewString()
		go func() {
			defer ginkgo.GinkgoRecover()
			port := int(udpService.Spec.Ports[0].Port)
			laddr, err := net.ResolveUDPAddr("udp", ":54322")
			if err != nil {
				framework.Failf("Failed to resolve local address: %v", err)
			}
			raddr := net.UDPAddr{IP: netutils.ParseIPSloppy(udpIngressIP), Port: port}

			for {
				select {
				case <-stopCh:
					if len(hostnames) != 2 {
						framework.Failf("Failed to hit the 2 UDP LoadBalancer backends successfully, got %v", hostnames.List())
					}
					return
				default:
					time.Sleep(1 * time.Second)
				}

				conn, err := net.DialUDP("udp", laddr, &raddr)
				if err != nil {
					framework.Logf("Failed to connect to: %s %d", udpIngressIP, port)
					continue
				}
				conn.SetDeadline(time.Now().Add(3 * time.Second))
				framework.Logf("Connected successfully to: %s", raddr.String())
				conn.Write([]byte("hostname\n"))
				buff := make([]byte, 1024)
				n, _, err := conn.ReadFrom(buff)
				if err == nil {
					mu.Lock()
					hostnames.Insert(string(buff[:n]))
					mu.Unlock()
					framework.Logf("Connected successfully to hostname: %s", string(buff[:n]))
				}
				conn.Close()
			}
		}()

		// Add a backend pod to the service in one node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		serverPod1.Spec.Hostname = "hostname1"
		nodeSelection := e2epod.NodeSelection{Name: nodes.Items[0].Name}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(f).CreateSync(ctx, serverPod1)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node " + nodes.Items[0].Name)
		if err := wait.PollImmediate(1*time.Second, loadBalancerLagTimeout, func() (bool, error) {
			mu.Lock()
			defer mu.Unlock()
			return hostnames.Has(serverPod1.Spec.Hostname), nil
		}); err != nil {
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod on the same node
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		serverPod2.Spec.Hostname = "hostname2"
		// use the same node as previous pod
		e2epod.SetNodeSelection(&serverPod2.Spec, nodeSelection)
		e2epod.NewPodClient(f).CreateSync(ctx, serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(f).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend2: {80}})

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node " + nodes.Items[0].Name)
		if err := wait.PollImmediate(1*time.Second, loadBalancerLagTimeout, func() (bool, error) {
			mu.Lock()
			defer mu.Unlock()
			return hostnames.Has(serverPod2.Spec.Hostname), nil
		}); err != nil {
			framework.Failf("Failed to connect to backend 2")
		}
	})

	f.It("should not have connectivity disruption during rolling update with externalTrafficPolicy=Cluster", f.WithSlow(), func(ctx context.Context) {
		// We start with a low but reasonable threshold to analyze the results.
		// The goal is to achieve 99% minimum success rate.
		// TODO: We should do incremental steps toward the goal.
		minSuccessRate := 0.95

		testRollingUpdateLBConnectivityDisruption(ctx, f, v1.ServiceExternalTrafficPolicyTypeCluster, minSuccessRate)
	})

	f.It("should not have connectivity disruption during rolling update with externalTrafficPolicy=Local", f.WithSlow(), func(ctx context.Context) {
		// We start with a low but reasonable threshold to analyze the results.
		// The goal is to achieve 99% minimum success rate.
		// TODO: We should do incremental steps toward the goal.
		minSuccessRate := 0.95

		testRollingUpdateLBConnectivityDisruption(ctx, f, v1.ServiceExternalTrafficPolicyTypeLocal, minSuccessRate)
	})
})

var _ = common.SIGDescribe("LoadBalancers ESIPP", framework.WithSlow(), func() {
	f := framework.NewDefaultFramework("esipp")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var loadBalancerCreateTimeout time.Duration

	var cs clientset.Interface
	var subnetPrefix *net.IPNet
	var err error

	ginkgo.BeforeEach(func(ctx context.Context) {
		// requires cloud load-balancer support - this feature currently supported only on GCE/GKE
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		cs = f.ClientSet
		loadBalancerCreateTimeout = e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)
		subnetPrefix, err = getSubnetPrefix(ctx, cs)
		framework.ExpectNoError(err)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		if ginkgo.CurrentSpecReport().Failed() {
			DescribeSvc(f.Namespace.Name)
		}
	})

	ginkgo.It("should work for type=LoadBalancer", func(ctx context.Context) {
		namespace := f.Namespace.Name
		serviceName := "external-local-lb"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err = jig.ChangeServiceType(ctx, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)

			// Make sure we didn't leak the health check node port.
			const threshold = 2
			nodes, err := getEndpointNodesWithInternalIP(ctx, jig)
			framework.ExpectNoError(err)
			config := e2enetwork.NewNetworkingTestConfig(ctx, f)
			for _, internalIP := range nodes {
				err := testHTTPHealthCheckNodePortFromTestContainer(ctx,
					config,
					internalIP,
					healthCheckNodePort,
					e2eservice.KubeProxyLagTimeout,
					false,
					threshold)
				framework.ExpectNoError(err)
			}
			err = cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])

		ginkgo.By("reading clientIP using the TCP service's service port via its external VIP")
		clientIPPort, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, "/clientip")
		framework.ExpectNoError(err)
		framework.Logf("ClientIP detected by target pod using VIP:SvcPort is %s", clientIPPort)

		ginkgo.By("checking if Source IP is preserved")
		// The clientIPPort returned from GetHTTPContent is in this format: x.x.x.x:port or [xx:xx:xx::x]:port
		host, _, err := net.SplitHostPort(clientIPPort)
		if err != nil {
			framework.Failf("SplitHostPort returned unexpected error: %q", clientIPPort)
		}
		ip := netutils.ParseIPSloppy(host)
		if ip == nil {
			framework.Failf("Invalid client IP address format: %q", host)
		}
		if subnetPrefix.Contains(ip) {
			framework.Failf("Source IP was NOT preserved")
		}
	})

	ginkgo.It("should work for type=NodePort", func(ctx context.Context) {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodeport"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalNodePortService(ctx, true)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		tcpNodePort := int(svc.Spec.Ports[0].NodePort)

		endpointsNodeMap, err := getEndpointNodesWithInternalIP(ctx, jig)
		framework.ExpectNoError(err)

		dialCmd := "clientip"
		config := e2enetwork.NewNetworkingTestConfig(ctx, f)

		for nodeName, nodeIP := range endpointsNodeMap {
			ginkgo.By(fmt.Sprintf("reading clientIP using the TCP service's NodePort, on node %v: %v:%v/%v", nodeName, nodeIP, tcpNodePort, dialCmd))
			clientIP, err := GetHTTPContentFromTestContainer(ctx, config, nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout, dialCmd)
			framework.ExpectNoError(err)
			framework.Logf("ClientIP detected by target pod using NodePort is %s, the ip of test container is %s", clientIP, config.TestContainerPod.Status.PodIP)
			// the clientIP returned by agnhost contains port
			if !strings.HasPrefix(clientIP, config.TestContainerPod.Status.PodIP) {
				framework.Failf("Source IP was NOT preserved")
			}
		}
	})

	ginkgo.It("should only target nodes with endpoints", func(ctx context.Context) {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodes"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, loadBalancerCreateTimeout, false,
			func(svc *v1.Service) {
				// Change service port to avoid collision with opened hostPorts
				// in other tests that run in parallel.
				if len(svc.Spec.Ports) != 0 {
					svc.Spec.Ports[0].TargetPort = intstr.FromInt32(svc.Spec.Ports[0].Port)
					svc.Spec.Ports[0].Port = 8081
				}

			})
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err = jig.ChangeServiceType(ctx, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcTCPPort := int(svc.Spec.Ports[0].Port)

		const threshold = 2
		config := e2enetwork.NewNetworkingTestConfig(ctx, f)
		for i := 0; i < len(nodes.Items); i++ {
			endpointNodeName := nodes.Items[i].Name

			ginkgo.By("creating a pod to be part of the service " + serviceName + " on node " + endpointNodeName)
			_, err = jig.Run(ctx, func(rc *v1.ReplicationController) {
				rc.Name = serviceName
				if endpointNodeName != "" {
					rc.Spec.Template.Spec.NodeName = endpointNodeName
				}
			})
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("waiting for service endpoint on node %v", endpointNodeName))
			err = jig.WaitForEndpointOnNode(ctx, endpointNodeName)
			framework.ExpectNoError(err)

			// HealthCheck should pass only on the node where num(endpoints) > 0
			// All other nodes should fail the healthcheck on the service healthCheckNodePort
			for n, internalIP := range ips {
				// Make sure the loadbalancer picked up the health check change.
				// Confirm traffic can reach backend through LB before checking healthcheck nodeport.
				e2eservice.TestReachableHTTP(ctx, ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout)
				expectedSuccess := nodes.Items[n].Name == endpointNodeName
				port := strconv.Itoa(healthCheckNodePort)
				ipPort := net.JoinHostPort(internalIP, port)
				framework.Logf("Health checking %s, http://%s/healthz, expectedSuccess %v", nodes.Items[n].Name, ipPort, expectedSuccess)
				err := testHTTPHealthCheckNodePortFromTestContainer(ctx,
					config,
					internalIP,
					healthCheckNodePort,
					e2eservice.KubeProxyEndpointLagTimeout,
					expectedSuccess,
					threshold)
				framework.ExpectNoError(err)
			}
			framework.ExpectNoError(e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, namespace, serviceName))
		}
	})

	ginkgo.It("should work from pods", func(ctx context.Context) {
		var err error
		namespace := f.Namespace.Name
		serviceName := "external-local-pods"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err = jig.ChangeServiceType(ctx, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		port := strconv.Itoa(int(svc.Spec.Ports[0].Port))
		ipPort := net.JoinHostPort(ingressIP, port)
		path := fmt.Sprintf("%s/clientip", ipPort)

		ginkgo.By("Creating pause pod deployment to make sure, pausePods are in desired state")
		deployment := createPausePodDeployment(ctx, cs, "pause-pod-deployment", namespace, 1)
		framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(cs, deployment), "Failed to complete pause pod deployment")

		ginkgo.DeferCleanup(func(ctx context.Context) {
			framework.Logf("Deleting deployment")
			err = cs.AppsV1().Deployments(namespace).Delete(ctx, deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete deployment %s", deployment.Name)
		})

		deployment, err = cs.AppsV1().Deployments(namespace).Get(ctx, deployment.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error in retrieving pause pod deployment")
		labelSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
		framework.ExpectNoError(err, "Error in setting LabelSelector as selector from deployment")

		pausePods, err := cs.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector.String()})
		framework.ExpectNoError(err, "Error in listing pods associated with pause pod deployments")

		pausePod := pausePods.Items[0]
		framework.Logf("Waiting up to %v curl %v", e2eservice.KubeProxyLagTimeout, path)
		cmd := fmt.Sprintf(`curl -q -s --connect-timeout 30 %v`, path)

		var srcIP string
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, cs)
		ginkgo.By(fmt.Sprintf("Hitting external lb %v from pod %v on node %v", ingressIP, pausePod.Name, pausePod.Spec.NodeName))
		if pollErr := wait.PollImmediate(framework.Poll, loadBalancerPropagationTimeout, func() (bool, error) {
			stdout, err := e2eoutput.RunHostCmd(pausePod.Namespace, pausePod.Name, cmd)
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

	ginkgo.It("should handle updates to ExternalTrafficPolicy field", func(ctx context.Context) {
		namespace := f.Namespace.Name
		serviceName := "external-local-update"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			framework.Failf("Need at least 2 nodes to verify source ip from a node without endpoint")
		}

		svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err = jig.ChangeServiceType(ctx, v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		// save the health check node port because it disappears when ESIPP is turned off.
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)

		ginkgo.By("turning ESIPP off")
		svc, err = jig.UpdateService(ctx, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
		})
		framework.ExpectNoError(err)
		if svc.Spec.HealthCheckNodePort > 0 {
			framework.Failf("Service HealthCheck NodePort still present")
		}

		epNodes, err := jig.ListNodesWithEndpoint(ctx)
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
		gomega.Expect(endpointNodeMap).ToNot(gomega.BeEmpty())
		gomega.Expect(noEndpointNodeMap).ToNot(gomega.BeEmpty())

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		svcNodePort := int(svc.Spec.Ports[0].NodePort)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		path := "/clientip"
		dialCmd := "clientip"

		config := e2enetwork.NewNetworkingTestConfig(ctx, f)

		ginkgo.By(fmt.Sprintf("endpoints present on nodes %v, absent on nodes %v", endpointNodeMap, noEndpointNodeMap))
		for nodeName, nodeIP := range noEndpointNodeMap {
			ginkgo.By(fmt.Sprintf("Checking %v (%v:%v/%v) proxies to endpoints on another node", nodeName, nodeIP[0], svcNodePort, dialCmd))
			_, err := GetHTTPContentFromTestContainer(ctx, config, nodeIP, svcNodePort, e2eservice.KubeProxyLagTimeout, dialCmd)
			framework.ExpectNoError(err, "Could not reach HTTP service through %v:%v/%v after %v", nodeIP, svcNodePort, dialCmd, e2eservice.KubeProxyLagTimeout)
		}

		for nodeName, nodeIP := range endpointNodeMap {
			ginkgo.By(fmt.Sprintf("checking kube-proxy health check fails on node with endpoint (%s), public IP %s", nodeName, nodeIP))
			var body string
			pollFn := func() (bool, error) {
				// we expect connection failure here, but not other errors
				resp, err := config.GetResponseFromTestContainer(ctx,
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
			clientIPPort, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, path)
			if err != nil {
				return false, nil
			}
			// The clientIPPort returned from GetHTTPContent is in this format: x.x.x.x:port or [xx:xx:xx::x]:port
			host, _, err := net.SplitHostPort(clientIPPort)
			if err != nil {
				framework.Logf("SplitHostPort returned unexpected error: %q", clientIPPort)
				return false, nil
			}
			ip := netutils.ParseIPSloppy(host)
			if ip == nil {
				framework.Logf("Invalid client IP address format: %q", host)
				return false, nil
			}
			if subnetPrefix.Contains(ip) {
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
		svc, err = jig.UpdateService(ctx, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			// Request the same healthCheckNodePort as before, to test the user-requested allocation path
			svc.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		})
		framework.ExpectNoError(err)
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, cs)
		pollErr = wait.PollImmediate(framework.PollShortTimeout, loadBalancerPropagationTimeout, func() (bool, error) {
			clientIPPort, err := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, path)
			if err != nil {
				return false, nil
			}
			ginkgo.By(fmt.Sprintf("Endpoint %v:%v%v returned client ip %v", ingressIP, svcTCPPort, path, clientIPPort))
			// The clientIPPort returned from GetHTTPContent is in this format: x.x.x.x:port or [xx:xx:xx::x]:port
			host, _, err := net.SplitHostPort(clientIPPort)
			if err != nil {
				framework.Logf("SplitHostPort returned unexpected error: %q", clientIPPort)
				return false, nil
			}
			ip := netutils.ParseIPSloppy(host)
			if ip == nil {
				framework.Logf("Invalid client IP address format: %q", host)
				return false, nil
			}
			if !subnetPrefix.Contains(ip) {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP (%v) is not the client IP even after ESIPP turned on, expected a public IP.", clientIP)
		}
	})
})

func testRollingUpdateLBConnectivityDisruption(ctx context.Context, f *framework.Framework, externalTrafficPolicy v1.ServiceExternalTrafficPolicyType, minSuccessRate float64) {
	cs := f.ClientSet
	ns := f.Namespace.Name
	name := "test-lb-rolling-update"
	labels := map[string]string{"name": name}
	gracePeriod := int64(60)
	maxUnavailable := intstr.FromString("10%")
	ds := e2edaemonset.NewDaemonSet(name, e2eapps.AgnhostImage, labels, nil, nil,
		[]v1.ContainerPort{
			{ContainerPort: 80},
		},
		"netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod),
	)
	ds.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{
		Type: appsv1.RollingUpdateDaemonSetStrategyType,
		RollingUpdate: &appsv1.RollingUpdateDaemonSet{
			MaxUnavailable: &maxUnavailable,
		},
	}
	ds.Spec.Template.Labels = labels
	ds.Spec.Template.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(gracePeriod)

	nodeNames := e2edaemonset.SchedulableNodes(ctx, cs, ds)
	e2eskipper.SkipUnlessAtLeast(len(nodeNames), 2, "load-balancer rolling update test requires at least 2 schedulable nodes for the DaemonSet")
	if len(nodeNames) > 25 {
		e2eskipper.Skipf("load-balancer rolling update test skipped for large environments with more than 25 nodes")
	}

	ginkgo.By(fmt.Sprintf("Creating DaemonSet %q", name))
	ds, err := cs.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Checking that daemon pods launch on every schedulable node of the cluster")
	creationTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)
	err = wait.PollUntilContextTimeout(ctx, framework.Poll, creationTimeout, true, e2edaemonset.CheckDaemonPodOnNodes(f, ds, nodeNames))
	framework.ExpectNoError(err, "error waiting for daemon pods to start")
	err = e2edaemonset.CheckDaemonStatus(ctx, f, name)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating a service %s with type=LoadBalancer externalTrafficPolicy=%s in namespace %s", name, externalTrafficPolicy, ns))
	jig := e2eservice.NewTestJig(cs, ns, name)
	jig.Labels = labels
	service, err := jig.CreateLoadBalancerService(ctx, creationTimeout, func(svc *v1.Service) {
		svc.Spec.ExternalTrafficPolicy = externalTrafficPolicy
	})
	framework.ExpectNoError(err)

	lbNameOrAddress := e2eservice.GetIngressPoint(&service.Status.LoadBalancer.Ingress[0])
	svcPort := int(service.Spec.Ports[0].Port)

	ginkgo.By("Hitting the DaemonSet's pods through the service's load balancer")
	timeout := e2eservice.LoadBalancerLagTimeoutDefault
	if framework.ProviderIs("aws") {
		timeout = e2eservice.LoadBalancerLagTimeoutAWS
	}
	e2eservice.TestReachableHTTP(ctx, lbNameOrAddress, svcPort, timeout)

	ginkgo.By("Starting a goroutine to continuously hit the DaemonSet's pods through the service's load balancer")
	var totalRequests uint64 = 0
	var networkErrors uint64 = 0
	var httpErrors uint64 = 0
	done := make(chan struct{})
	defer close(done)
	go func() {
		defer ginkgo.GinkgoRecover()

		wait.Until(func() {
			atomic.AddUint64(&totalRequests, 1)
			client := &http.Client{
				Transport: utilnet.SetTransportDefaults(&http.Transport{
					DisableKeepAlives: true,
				}),
				Timeout: 5 * time.Second,
			}
			ipPort := net.JoinHostPort(lbNameOrAddress, strconv.Itoa(svcPort))
			msg := "hello"
			url := fmt.Sprintf("http://%s/echo?msg=%s", ipPort, msg)
			resp, err := client.Get(url)
			if err != nil {
				framework.Logf("Got error testing for reachability of %s: %v", url, err)
				atomic.AddUint64(&networkErrors, 1)
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				framework.Logf("Got bad status code: %d", resp.StatusCode)
				atomic.AddUint64(&httpErrors, 1)
				return
			}
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				framework.Logf("Got error reading HTTP body: %v", err)
				atomic.AddUint64(&httpErrors, 1)
				return
			}
			if string(body) != msg {
				framework.Logf("The response body does not contain expected string %s", string(body))
				atomic.AddUint64(&httpErrors, 1)
				return
			}
		}, time.Duration(0), done)
	}()

	ginkgo.By("Triggering DaemonSet rolling update several times")
	var previousTotalRequests uint64 = 0
	var previousNetworkErrors uint64 = 0
	var previousHttpErrors uint64 = 0
	for i := 1; i <= 5; i++ {
		framework.Logf("Update daemon pods environment: [{\"name\":\"VERSION\",\"value\":\"%d\"}]", i)
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","env":[{"name":"VERSION","value":"%d"}]}]}}}}`, ds.Spec.Template.Spec.Containers[0].Name, i)
		ds, err = cs.AppsV1().DaemonSets(ns).Patch(context.TODO(), name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err)

		framework.Logf("Check that daemon pods are available on every node of the cluster with the updated environment.")
		err = wait.PollImmediate(framework.Poll, creationTimeout, func() (bool, error) {
			podList, err := cs.CoreV1().Pods(ds.Namespace).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				return false, err
			}
			pods := podList.Items

			readyPods := 0
			for _, pod := range pods {
				if !metav1.IsControlledBy(&pod, ds) {
					continue
				}
				if pod.DeletionTimestamp != nil {
					continue
				}
				podVersion := ""
				for _, env := range pod.Spec.Containers[0].Env {
					if env.Name == "VERSION" {
						podVersion = env.Value
						break
					}
				}
				if podVersion != fmt.Sprintf("%d", i) {
					continue
				}
				podReady := podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now())
				if !podReady {
					continue
				}
				readyPods += 1
			}
			framework.Logf("Number of running nodes: %d, number of updated ready pods: %d in daemonset %s", len(nodeNames), readyPods, ds.Name)
			return readyPods == len(nodeNames), nil
		})
		framework.ExpectNoError(err, "error waiting for daemon pods to be ready")

		// assert that the HTTP requests success rate is above the acceptable threshold after this rolling update
		currentTotalRequests := atomic.LoadUint64(&totalRequests)
		currentNetworkErrors := atomic.LoadUint64(&networkErrors)
		currentHttpErrors := atomic.LoadUint64(&httpErrors)

		partialTotalRequests := currentTotalRequests - previousTotalRequests
		partialNetworkErrors := currentNetworkErrors - previousNetworkErrors
		partialHttpErrors := currentHttpErrors - previousHttpErrors
		partialSuccessRate := (float64(partialTotalRequests) - float64(partialNetworkErrors+partialHttpErrors)) / float64(partialTotalRequests)

		framework.Logf("Load Balancer total HTTP requests: %d", partialTotalRequests)
		framework.Logf("Network errors: %d", partialNetworkErrors)
		framework.Logf("HTTP errors: %d", partialHttpErrors)
		framework.Logf("Success rate: %.2f%%", partialSuccessRate*100)
		if partialSuccessRate < minSuccessRate {
			framework.Failf("Encountered too many errors when doing HTTP requests to the load balancer address. Success rate is %.2f%%, and the minimum allowed threshold is %.2f%%.", partialSuccessRate*100, minSuccessRate*100)
		}

		previousTotalRequests = currentTotalRequests
		previousNetworkErrors = currentNetworkErrors
		previousHttpErrors = currentHttpErrors
	}

	// assert that the load balancer address is still reachable after the rolling updates are finished
	e2eservice.TestReachableHTTP(ctx, lbNameOrAddress, svcPort, timeout)
}
