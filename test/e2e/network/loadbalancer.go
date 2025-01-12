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
	imageutils "k8s.io/kubernetes/test/utils/image"
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
	"k8s.io/kubernetes/test/e2e/feature"
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
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
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

func waitForSvcStatus(ctx context.Context, cs clientset.Interface, ns string, serviceName string, timeout time.Duration, statusFn func(service *v1.Service) (bool, error)) error {
	return wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, func(ctx context.Context) (bool, error) {
		svc, err := cs.CoreV1().Services(ns).Get(ctx, serviceName, metav1.GetOptions{})

		if err != nil {
			framework.Logf("Retrying .... error trying to get Service %s: %v", serviceName, err)

			return false, err
		}

		return statusFn(svc)
	})
}

func changeServiceNodePort(ctx context.Context, cs clientset.Interface, ns string, serviceName string, initial int) (*v1.Service, error) {
	var err error
	var svc *v1.Service
	for i := 1; i < e2eservice.NodePortRange.Size; i++ {
		offs1 := initial - e2eservice.NodePortRange.Base
		offs2 := (offs1 + i) % e2eservice.NodePortRange.Size
		newPort := e2eservice.NodePortRange.Base + offs2

		svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
			s.Spec.Ports[0].NodePort = int32(newPort)
		})
		if err != nil && strings.Contains(err.Error(), "provided port is already allocated") {
			framework.Logf("tried nodePort %d, but it is in use, will try another", newPort)
			continue
		}
		// Otherwise err was nil or err was a real error
		break
	}
	return svc, err
}

var _ = common.SIGDescribe("LoadBalancers", feature.LoadBalancer, func() {
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

	protocols := []struct {
		name               string
		protocol           v1.Protocol
		lbLagTimeout       time.Duration
		testReachableFn    func(ctx context.Context, host string, port int, timeout time.Duration)
		testNotReachableFn func(ctx context.Context, host string, port int, timeout time.Duration)
	}{
		{"TCP", v1.ProtocolTCP, e2eservice.LoadBalancerLagTimeoutAWS, e2eservice.TestReachableHTTP, testNotReachableHTTP},
		//{"UDP", v1.ProtocolUDP, e2eservice.LoadBalancerLagTimeoutDefault, testReachableUDP, testNotReachableUDP},
	}

	for _, protocolTest := range protocols {
		f.It(fmt.Sprintf("[%s] should be able to change the type and ports of the service", protocolTest.name),
			f.WithSlow(),
			func(ctx context.Context) {
				// FIXME: need a better platform-independent timeout
				loadBalancerLagTimeout := protocolTest.lbLagTimeout
				loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

				// This test is more monolithic than we'd like because LB turnup can be
				// very slow, so we lumped all the tests into one LB lifecycle.

				serviceName := fmt.Sprintf("mutability-test-%s", strings.ToLower(protocolTest.name))
				serviceSelector := map[string]string{"testid": serviceName}
				ns := f.Namespace.Name // LB1 in ns on TCP
				framework.Logf("[%s] namespace for test: %s", protocolTest.name, ns)

				// Setup test service
				ginkgo.By(fmt.Sprintf("[%s] creating service %s with type=ClusterIP in namespace %s",
					protocolTest.name,
					serviceName,
					ns))
				svcSpec := e2eservice.CreateServiceSpec(serviceName, "", false, serviceSelector)
				svcSpec.Spec.Ports[0].Protocol = protocolTest.protocol

				svc, err := cs.CoreV1().Services(ns).Create(ctx, svcSpec, metav1.CreateOptions{})
				framework.ExpectNoError(err)

				svcPort := int(svc.Spec.Ports[0].Port)
				framework.Logf("[%s] service port: %d", protocolTest.name, svcPort)

				// Setup test deployment
				ginkgo.By(fmt.Sprintf("[%s] creating a deployment to be part of the service %s",
					protocolTest.name,
					serviceName))

				deploymentSpec := e2edeployment.NewDeployment(serviceName,
					1,
					serviceSelector,
					serviceName,
					imageutils.GetE2EImage(imageutils.Agnhost),
					appsv1.RollingUpdateDeploymentStrategyType)
				deploymentSpec.Spec.Template.Spec.Containers[0].Args = []string{"netexec", "--http-port=80", "--udp-port=80"}
				deploymentSpec.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
					PeriodSeconds: 3,
					ProbeHandler: v1.ProbeHandler{
						HTTPGet: &v1.HTTPGetAction{
							Port: svc.Spec.Ports[0].TargetPort,
							Path: "/hostName",
						},
					},
				}

				deployment, err := cs.AppsV1().Deployments(ns).Create(ctx, deploymentSpec, metav1.CreateOptions{})
				framework.ExpectNoError(err)

				// Create test bastion pod
				execPod := e2epod.CreateExecPodOrFail(ctx, cs, ns, "execpod", nil)

				// Start running tests

				// Check ClusterIP service is reachable
				err = e2eservice.TestAllEndpointsReachable(ctx, cs, svc, execPod)
				framework.ExpectNoError(err)

				// Change the service to NodePort.
				ginkgo.By(fmt.Sprintf("[%s] changing the service to type=NodePort", protocolTest.name))
				svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
					s.Spec.Type = v1.ServiceTypeNodePort
				})
				framework.ExpectNoError(err)

				tcpNodePort := int(svc.Spec.Ports[0].NodePort)
				framework.Logf("[%s] node port: %d", protocolTest.name, tcpNodePort)

				// Check NodePort service is reachable
				err = e2eservice.TestAllEndpointsReachable(ctx, cs, svc, execPod)
				framework.ExpectNoError(err)

				// Change the service to LoadBalancer.
				ginkgo.By(fmt.Sprintf("[%s] changing the service to type=LoadBalancer", protocolTest.name))
				svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
					s.Spec.Type = v1.ServiceTypeLoadBalancer
				})
				framework.ExpectNoError(err)

				// Wait for the load balancer to be created asynchronously
				ginkgo.By(fmt.Sprintf("[%s] waiting for the service to have a load balancer", protocolTest.name))
				err = waitForSvcStatus(ctx, cs, ns, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
					return len(service.Status.LoadBalancer.Ingress) > 0, nil
				})
				framework.ExpectNoError(err)

				// If the NodePort has changed, fail
				if int(svc.Spec.Ports[0].NodePort) != tcpNodePort {
					framework.Failf("[%s] Spec.Ports[0].NodePort changed (%d -> %d) when not expected",
						protocolTest.name,
						tcpNodePort,
						svc.Spec.Ports[0].NodePort)
				}

				svc, err = cs.CoreV1().Services(ns).Get(ctx, serviceName, metav1.GetOptions{})
				framework.ExpectNoError(err)

				ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
				framework.Logf("[%s] load balancer: %s", protocolTest.name, ingressIP)

				// Check LoadBalancer service is reachable
				err = e2eservice.TestAllEndpointsReachable(ctx, cs, svc, execPod)
				framework.ExpectNoError(err)

				// Additionally, check that the LoadBalancer is reachable
				ginkgo.By(fmt.Sprintf("[%s] hitting the service's LoadBalancer", protocolTest.name))
				protocolTest.testReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)

				// Change the service node ports.
				ginkgo.By(fmt.Sprintf("[%s] changing the service's NodePort", protocolTest.name))
				svc, err = changeServiceNodePort(ctx, cs, ns, serviceName, tcpNodePort)
				framework.ExpectNoError(err)

				// If the NodePort has not changed, fail
				tcpNodePortOld := tcpNodePort
				tcpNodePort = int(svc.Spec.Ports[0].NodePort)
				if tcpNodePort == tcpNodePortOld {
					framework.Failf("[%s] Spec.Ports[0].NodePort (%d) did not change", protocolTest.name, tcpNodePort)
				}

				// If the LoadBalancer Ingress has changed, fail
				if e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]) != ingressIP {
					framework.Failf("[%s] Status.LoadBalancer.Ingress changed (%s -> %s) when not expected",
						protocolTest.name,
						ingressIP,
						e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]))
				}
				framework.Logf("[%s] node port: %d", protocolTest.name, tcpNodePort)

				// Check that the LoadBalancer is still reachable
				ginkgo.By(fmt.Sprintf("[%s] hitting the service's LoadBalancer", protocolTest.name))
				protocolTest.testReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)

				// Change the service main ports.
				ginkgo.By(fmt.Sprintf("[%s] changing the service's port", protocolTest.name))
				svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
					s.Spec.Ports[0].Port++
				})
				framework.ExpectNoError(err)

				// If the port has not changed, fail
				// If the NodePort has changed, fail
				// If the LoadBalancer Ingress has changed, fail
				svcPortOld := svcPort
				svcPort = int(svc.Spec.Ports[0].Port)
				if svcPort == svcPortOld {
					framework.Failf("[%s] Spec.Ports[0].Port (%d) did not change", protocolTest.name, svcPort)
				}
				if int(svc.Spec.Ports[0].NodePort) != tcpNodePort {
					framework.Failf("[%s] Spec.Ports[0].NodePort (%d) changed",
						protocolTest.name,
						svc.Spec.Ports[0].NodePort)
				}
				if e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]) != ingressIP {
					framework.Failf("[%s] Status.LoadBalancer.Ingress changed (%s -> %s) when not expected",
						protocolTest.name,
						ingressIP,
						e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]))
				}

				framework.Logf("[%s] service port: %d", protocolTest.name, svcPort)

				// Check that the LoadBalancer is still reachable
				ginkgo.By(fmt.Sprintf("[%s] hitting the service's LoadBalancer", protocolTest.name))
				protocolTest.testReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)

				// Scale the deployment to 0 replicas
				ginkgo.By("Scaling the pods to 0")
				deployment, err = e2edeployment.UpdateDeploymentWithRetries(cs,
					ns,
					deployment.Name,
					func(d *appsv1.Deployment) {
						newReplicas := int32(0)
						d.Spec.Replicas = &newReplicas
					})
				framework.ExpectNoError(err)

				err = e2edeployment.WaitForDeploymentComplete(cs, deployment)
				framework.ExpectNoError(err)

				// Check that the LoadBalancer is not reachable anymore
				ginkgo.By(fmt.Sprintf("[%s] hitting the service's LoadBalancer with no backends, no answer expected",
					protocolTest.name))
				protocolTest.testNotReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)

				// Scale the deployment to 1 replica
				ginkgo.By("Scaling the pods to 1")
				deployment, err = e2edeployment.UpdateDeploymentWithRetries(cs,
					ns,
					deployment.Name,
					func(d *appsv1.Deployment) {
						newReplicas := int32(1)
						d.Spec.Replicas = &newReplicas
					})
				framework.ExpectNoError(err)

				err = e2edeployment.WaitForDeploymentComplete(cs, deployment)
				framework.ExpectNoError(err)

				// Check that the LoadBalancer is reachable again
				ginkgo.By(fmt.Sprintf("[%s] hitting the service's LoadBalancer", protocolTest.name))
				protocolTest.testReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)

				// Change the services back to ClusterIP.
				ginkgo.By(fmt.Sprintf("[%s] changing service back to type=ClusterIP", protocolTest.name))
				svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
					s.Spec.Type = v1.ServiceTypeClusterIP
				})
				framework.ExpectNoError(err)

				// Check that the service doesn't have a NodePort anymore
				if svc.Spec.Ports[0].NodePort != 0 {
					framework.Failf("[%s] Spec.Ports[0].NodePort was not cleared", protocolTest.name)
				}

				// Wait for the LoadBalancer to be gone
				err = waitForSvcStatus(ctx, cs, ns, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
					return len(service.Status.LoadBalancer.Ingress) == 0, nil
				})

				framework.ExpectNoError(err)

				// Check that the LoadBalancer is not reachable anymore
				ginkgo.By(fmt.Sprintf("[%s] checking the LoadBalancer is closed", protocolTest.name))
				protocolTest.testNotReachableFn(ctx, ingressIP, svcPort, loadBalancerLagTimeout)
			})
	}

	f.It("should only allow access from service loadbalancer source ranges", f.WithSlow(), func(ctx context.Context) {
		// FIXME: need a better platform-independent timeout
		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutAWS
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		namespace := f.Namespace.Name
		serviceName := "lb-sourcerange"
		serviceSelector := map[string]string{"testid": serviceName}

		ginkgo.By("creating a LoadBalancer Service (with no LoadBalancerSourceRanges)")
		svcSpec := e2eservice.CreateServiceSpec(serviceName, "", false, serviceSelector)

		svc, err := cs.CoreV1().Services(namespace).Create(ctx, svcSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating LoadBalancer")

		ginkgo.DeferCleanup(func(ctx context.Context) {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(ctx, cs, svc.Namespace, svc.Name)
		})

		// Provisioning the LB may take some time, so create pods while we wait.
		ginkgo.By("creating a pod to be part of the service " + serviceName)
		deploymentSpec := e2edeployment.NewDeployment(serviceName,
			1,
			serviceSelector,
			serviceName,
			imageutils.GetE2EImage(imageutils.Agnhost),
			appsv1.RollingUpdateDeploymentStrategyType)
		deploymentSpec.Spec.Template.Spec.Containers[0].Args = []string{"netexec", "--http-port=80", "--udp-port=80"}
		deploymentSpec.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
			PeriodSeconds: 3,
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Port: svc.Spec.Ports[0].TargetPort,
					Path: "/hostName",
				},
			},
		}

		_, err = cs.AppsV1().Deployments(namespace).Create(ctx, deploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating service backend")

		ginkgo.By("creating client pods on two different nodes")
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err, "getting list of nodes")

		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %d nodes",
				len(nodes.Items))
		}

		acceptPod := e2epod.CreateExecPodOrFail(ctx, cs, namespace, "execpod-accept",
			func(pod *v1.Pod) {
				pod.Spec.NodeName = nodes.Items[0].Name
			},
		)
		acceptPod, err = cs.CoreV1().Pods(namespace).Get(ctx, acceptPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting IP of acceptPod")

		dropPod := e2epod.CreateExecPodOrFail(ctx, cs, namespace, "execpod-drop",
			func(pod *v1.Pod) {
				pod.Spec.NodeName = nodes.Items[1].Name
			},
		)
		dropPod, err = cs.CoreV1().Pods(namespace).Get(ctx, dropPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting IP of dropPod")

		ginkgo.By("waiting for the LoadBalancer to be provisioned")
		err = waitForSvcStatus(ctx, cs, namespace, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
			return len(service.Status.LoadBalancer.Ingress) > 0, nil
		})
		framework.ExpectNoError(err, "waiting for LoadBalancer to be provisioned")

		ingress := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcPort := int(svc.Spec.Ports[0].Port)
		framework.Logf("Load balancer is at %s, port %d", ingress, svcPort)

		ginkgo.By("checking reachability from outside the cluster when LoadBalancerSourceRanges is unset")
		e2eservice.TestReachableHTTP(ctx, ingress, svcPort, loadBalancerLagTimeout)

		ginkgo.By("checking reachability from pods when LoadBalancerSourceRanges is unset")
		// There are different propagation delay for the APIs for different nodes, so it tries
		// a few times, despite previously it was confirmed that the Service was reachable.
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, acceptPod.Name, ingress)
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, dropPod.Name, ingress)

		// Create source ranges that allow acceptPod but not dropPod or
		// cluster-external sources. We assume that the LBSR rules will either see
		// the traffic unmasqueraded, or else see it masqueraded to the pod's node
		// IP. (Since acceptPod and dropPod are on different nodes, this should
		// still be unambiguous.)
		sourceRanges := []string{
			ipToSourceRange(acceptPod.Status.PodIP),
			ipToSourceRange(acceptPod.Status.HostIP),
		}
		ginkgo.By(fmt.Sprintf("setting LoadBalancerSourceRanges to %v", sourceRanges))
		svc, err = e2eservice.UpdateService(ctx, cs, namespace, serviceName, func(s *v1.Service) {
			s.Spec.LoadBalancerSourceRanges = sourceRanges
		})
		framework.ExpectNoError(err)

		ginkgo.By("checking reachability from outside the cluster when LoadBalancerSourceRanges blocks it")
		testNotReachableHTTP(ctx, ingress, svcPort, loadBalancerLagTimeout)

		ginkgo.By("checking reachability from pods when LoadBalancerSourceRanges only allows acceptPod")
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, acceptPod.Name, ingress)
		checkReachabilityFromPod(ctx, false, e2eservice.KubeProxyEndpointLagTimeout, namespace, dropPod.Name, ingress)

		// "Allow all". (We should be able to set this dual-stack but maybe
		// some IPv4-only cloud providers won't handle that.)
		if netutils.IsIPv4String(acceptPod.Status.HostIP) {
			sourceRanges = []string{"0.0.0.0/1", "128.0.0.0/1"}
		} else {
			sourceRanges = []string{"::/1", "8000::/1"}
		}

		ginkgo.By(fmt.Sprintf("setting LoadBalancerSourceRanges to %v", sourceRanges))
		svc, err = e2eservice.UpdateService(ctx, cs, namespace, serviceName, func(s *v1.Service) {
			s.Spec.LoadBalancerSourceRanges = sourceRanges
		})
		framework.ExpectNoError(err, "updating LoadBalancerSourceRanges")

		ginkgo.By("checking reachability from outside the cluster when LoadBalancerSourceRanges allows everything")
		e2eservice.TestReachableHTTP(ctx, ingress, svcPort, loadBalancerLagTimeout)

		ginkgo.By("checking reachability from pods when LoadBalancerSourceRanges allows everything")
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, acceptPod.Name, ingress)
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, dropPod.Name, ingress)

		// "Deny all, essentially"
		if netutils.IsIPv4String(acceptPod.Status.HostIP) {
			sourceRanges = []string{"255.0.0.0/32"}
		} else {
			sourceRanges = []string{"ffff::/128"}
		}

		ginkgo.By(fmt.Sprintf("setting LoadBalancerSourceRanges to %v", sourceRanges))
		svc, err = e2eservice.UpdateService(ctx, cs, namespace, serviceName, func(s *v1.Service) {
			s.Spec.LoadBalancerSourceRanges = sourceRanges
		})
		framework.ExpectNoError(err, "updating LoadBalancerSourceRanges")

		ginkgo.By("checking reachability from outside the cluster when LoadBalancerSourceRanges blocks everything")
		testNotReachableHTTP(ctx, ingress, svcPort, loadBalancerLagTimeout)

		ginkgo.By("checking reachability from pods when LoadBalancerSourceRanges blocks everything")
		checkReachabilityFromPod(ctx, false, e2eservice.KubeProxyEndpointLagTimeout, namespace, acceptPod.Name, ingress)
		checkReachabilityFromPod(ctx, false, e2eservice.KubeProxyEndpointLagTimeout, namespace, dropPod.Name, ingress)

		ginkgo.By("clearing LoadBalancerSourceRanges")
		svc, err = e2eservice.UpdateService(ctx, cs, namespace, serviceName, func(s *v1.Service) {
			s.Spec.LoadBalancerSourceRanges = nil
		})
		framework.ExpectNoError(err, "updating LoadBalancerSourceRanges")

		ginkgo.By("checking reachability from outside the cluster after LoadBalancerSourceRanges is cleared")
		e2eservice.TestReachableHTTP(ctx, ingress, svcPort, loadBalancerLagTimeout)

		ginkgo.By("checking reachability from pods after LoadBalancerSourceRanges is cleared")
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, acceptPod.Name, ingress)
		checkReachabilityFromPod(ctx, true, e2eservice.KubeProxyEndpointLagTimeout, namespace, dropPod.Name, ingress)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should have session affinity work for LoadBalancer service with Local traffic policy", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// FIXME: some cloud providers do not support k8s-compatible affinity

		svc := getServeHostnameService("affinity-lb-esipp")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		execAffinityTestForLBService(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should be able to switch session affinity for LoadBalancer service with Local traffic policy", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// FIXME: some cloud providers do not support k8s-compatible affinity

		svc := getServeHostnameService("affinity-lb-esipp-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		execAffinityTestForLBServiceWithTransition(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should have session affinity work for LoadBalancer service with Cluster traffic policy", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// FIXME: some cloud providers do not support k8s-compatible affinity

		svc := getServeHostnameService("affinity-lb")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
		execAffinityTestForLBService(ctx, f, cs, svc)
	})

	// [LinuxOnly]: Windows does not support session affinity.
	f.It("should be able to switch session affinity for LoadBalancer service with Cluster traffic policy", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
		// FIXME: some cloud providers do not support k8s-compatible affinity

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
		// FIXME: need a better platform-independent timeout
		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutAWS
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "reallocate-nodeport-test"
		ns := f.Namespace.Name // LB1 in ns on TCP
		serviceSelector := map[string]string{"testid": serviceName}

		framework.Logf("namespace for TCP test: %s", ns)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		svcSpec := e2eservice.CreateServiceSpec(serviceName, "", false, serviceSelector)

		svc, err := cs.CoreV1().Services(ns).Create(ctx, svcSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		svcPort := int(svc.Spec.Ports[0].Port)
		framework.Logf("service port TCP: %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		deploymentSpec := e2edeployment.NewDeployment(serviceName,
			1,
			serviceSelector,
			serviceName,
			imageutils.GetE2EImage(imageutils.Agnhost),
			appsv1.RollingUpdateDeploymentStrategyType)
		deploymentSpec.Spec.Template.Spec.Containers[0].Args = []string{"netexec", "--http-port=80", "--udp-port=80"}
		deploymentSpec.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
			PeriodSeconds: 3,
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Port: svc.Spec.Ports[0].TargetPort,
					Path: "/hostName",
				},
			},
		}

		_, err = cs.AppsV1().Deployments(ns).Create(ctx, deploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// Change the services to LoadBalancer.
		ginkgo.By("changing the TCP service to type=LoadBalancer")
		svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeLoadBalancer
			s.Spec.AllocateLoadBalancerNodePorts = ptr.To(false)
		})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		err = waitForSvcStatus(ctx, cs, ns, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
			return len(service.Status.LoadBalancer.Ingress) > 0, nil
		})
		framework.ExpectNoError(err)

		if int(svc.Spec.Ports[0].NodePort) != 0 {
			framework.Failf("TCP Spec.Ports[0].NodePort allocated %d when not expected", svc.Spec.Ports[0].NodePort)
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)

		// Change the services' node ports.

		ginkgo.By("adding a TCP service's NodePort")
		svc, err = e2eservice.UpdateService(ctx, cs, ns, serviceName, func(s *v1.Service) {
			s.Spec.AllocateLoadBalancerNodePorts = ptr.To(true)
		})
		framework.ExpectNoError(err)

		tcpNodePort := int(svc.Spec.Ports[0].NodePort)

		if tcpNodePort == 0 {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) not allocated", tcpNodePort)
		}
		if e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0]))
		}

		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(ctx, tcpIngressIP, svcPort, loadBalancerLagTimeout)
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on different nodes", func(ctx context.Context) {
		// FIXME: some cloud providers do not support UDP LoadBalancers

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
				_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
				framework.Logf("Connected successfully to: %s", raddr.String())
				_, _ = conn.Write([]byte("hostname\n"))
				buff := make([]byte, 1024)
				n, _, err := conn.ReadFrom(buff)
				if err == nil {
					mu.Lock()
					hostnames.Insert(string(buff[:n]))
					mu.Unlock()
					framework.Logf("Connected successfully to hostname: %s", string(buff[:n]))
				}
				_ = conn.Close()
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
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, loadBalancerLagTimeout, true, func(ctx context.Context) (bool, error) {
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
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, loadBalancerLagTimeout, true, func(ctx context.Context) (bool, error) {
			mu.Lock()
			defer mu.Unlock()
			return hostnames.Has(serverPod2.Spec.Hostname), nil
		}); err != nil {
			framework.Failf("Failed to connect to backend 2")
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on the same nodes", func(ctx context.Context) {
		// FIXME: some cloud providers do not support UDP LoadBalancers

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
				_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
				framework.Logf("Connected successfully to: %s", raddr.String())
				_, _ = conn.Write([]byte("hostname\n"))
				buff := make([]byte, 1024)
				n, _, err := conn.ReadFrom(buff)
				if err == nil {
					mu.Lock()
					hostnames.Insert(string(buff[:n]))
					mu.Unlock()
					framework.Logf("Connected successfully to hostname: %s", string(buff[:n]))
				}
				_ = conn.Close()
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
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, loadBalancerLagTimeout, true, func(ctx context.Context) (bool, error) {
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
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, loadBalancerLagTimeout, true, func(ctx context.Context) (bool, error) {
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

var _ = common.SIGDescribe("LoadBalancers ExternalTrafficPolicy: Local", feature.LoadBalancer, framework.WithSlow(), func() {
	f := framework.NewDefaultFramework("esipp")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var loadBalancerCreateTimeout time.Duration

	var cs clientset.Interface
	var subnetPrefix *net.IPNet
	var err error

	ginkgo.BeforeEach(func(ctx context.Context) {
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

		// FIXME: figure out the actual expected semantics for
		// "ExternalTrafficPolicy: Local" + "IPMode: Proxy".
		// https://issues.k8s.io/123714
		ingress := &svc.Status.LoadBalancer.Ingress[0]
		if ingress.IP == "" || (ingress.IPMode != nil && *ingress.IPMode == v1.LoadBalancerIPModeProxy) {
			e2eskipper.Skipf("LoadBalancer uses 'Proxy' IPMode")
		}

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

	ginkgo.It("should only target nodes with endpoints", func(ctx context.Context) {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodes"
		serviceSelector := map[string]string{"testid": serviceName}

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)

		svcSpec := e2eservice.CreateServiceSpec(serviceName, "", false, serviceSelector)
		svcSpec.Spec.Type = v1.ServiceTypeLoadBalancer
		svcSpec.Spec.SessionAffinity = v1.ServiceAffinityNone
		svcSpec.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal

		// Change service port to avoid collision with opened hostPorts
		// in other tests that run in parallel.
		svcSpec.Spec.Ports[0].TargetPort = intstr.FromInt32(svcSpec.Spec.Ports[0].Port)
		svcSpec.Spec.Ports[0].Port = 8081

		svc, err := cs.CoreV1().Services(namespace).Create(ctx, svcSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = waitForSvcStatus(ctx, cs, namespace, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
			return len(service.Status.LoadBalancer.Ingress) > 0, nil
		})
		framework.ExpectNoError(err)

		ginkgo.DeferCleanup(func(ctx context.Context) {
			svc, err = e2eservice.UpdateService(ctx, cs, namespace, serviceName, func(s *v1.Service) {
				s.Spec.Type = v1.ServiceTypeClusterIP
			})
			framework.ExpectNoError(err)

			err = waitForSvcStatus(ctx, cs, namespace, serviceName, loadBalancerCreateTimeout, func(service *v1.Service) (bool, error) {
				return len(service.Status.LoadBalancer.Ingress) == 0, nil
			})
			framework.ExpectNoError(err)

			err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})

		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		svc, err = cs.CoreV1().Services(namespace).Get(ctx, serviceName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcTCPPort := int(svc.Spec.Ports[0].Port)

		const threshold = 2
		config := e2enetwork.NewNetworkingTestConfig(ctx, f)
		for i := 0; i < len(nodes.Items); i++ {
			endpointNodeName := nodes.Items[i].Name

			ginkgo.By("creating a pod to be part of the service " + serviceName + " on node " + endpointNodeName)
			deploymentSpec := e2edeployment.NewDeployment(serviceName,
				1,
				serviceSelector,
				serviceName,
				imageutils.GetE2EImage(imageutils.Agnhost),
				appsv1.RollingUpdateDeploymentStrategyType)
			deploymentSpec.Spec.Template.Spec.Containers[0].Args = []string{"netexec", "--http-port=80", "--udp-port=80"}
			deploymentSpec.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
				PeriodSeconds: 3,
				ProbeHandler: v1.ProbeHandler{
					HTTPGet: &v1.HTTPGetAction{
						Port: svc.Spec.Ports[0].TargetPort,
						Path: "/hostName",
					},
				},
			}
			deploymentSpec.Spec.Template.Spec.NodeName = endpointNodeName

			_, err := cs.AppsV1().Deployments(namespace).Create(ctx, deploymentSpec, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("waiting for service endpoint on node %v", endpointNodeName))
			err = wait.PollUntilContextTimeout(ctx, framework.Poll, e2eservice.KubeProxyLagTimeout, true, func(ctx context.Context) (bool, error) {
				endpoints, err := cs.CoreV1().Endpoints(namespace).Get(ctx, serviceName, metav1.GetOptions{})

				if err != nil {
					framework.Logf("Get endpoints for service %s/%s failed (%s)", namespace, serviceName, err)
					return false, nil
				}

				if len(endpoints.Subsets) == 0 {
					framework.Logf("Expect endpoints with subsets, got none.")
					return false, nil
				}

				// TODO: Handle multiple endpoints
				if len(endpoints.Subsets[0].Addresses) == 0 {
					framework.Logf("Expected Ready endpoints - found none")
					return false, nil
				}

				epHostName := *endpoints.Subsets[0].Addresses[0].NodeName
				framework.Logf("Pod for service %s/%s is on node %s", namespace, serviceName, epHostName)
				if epHostName != endpointNodeName {
					framework.Logf("Found endpoint on wrong node, expected %v, got %v", endpointNodeName, epHostName)
					return false, nil
				}

				return true, nil
			})
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

	ginkgo.It("should target all nodes with endpoints", func(ctx context.Context) {
		// FIXME: need a better platform-independent timeout
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs)

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) == 1 {
			e2eskipper.Skipf("Test requires multiple schedulable nodes")
		}

		namespace := f.Namespace.Name
		serviceName := "external-local-update"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("creating the service")
		svc, err := jig.CreateOnlyLocalLoadBalancerService(ctx, loadBalancerCreateTimeout, false, nil)
		framework.ExpectNoError(err, "creating the service")
		ingress := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcPort := int(svc.Spec.Ports[0].Port)
		framework.Logf("ingress is %s:%d", ingress, svcPort)

		ginkgo.By("creating endpoints on multiple nodes")
		_, err = jig.Run(ctx, func(rc *v1.ReplicationController) {
			rc.Spec.Replicas = ptr.To[int32](2)
			rc.Spec.Template.Spec.Affinity = &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: jig.Labels},
							TopologyKey:   "kubernetes.io/hostname",
						},
					},
				},
			}
		})
		framework.ExpectNoError(err, "creating the endpoints")

		ginkgo.By("ensuring that the LoadBalancer targets all endpoints")
		// We're not testing affinity here, but we can use checkAffinity(false) to
		// test that affinity *isn't* enabled, which is to say, that connecting to
		// ingress:svcPort multiple times eventually reaches at least 2 different
		// endpoints.
		if !checkAffinity(ctx, cs, nil, ingress, svcPort, false) {
			framework.Failf("Load balancer connections only reached one of the two endpoints")
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

		// FIXME: figure out the actual expected semantics for
		// "ExternalTrafficPolicy: Local" + "IPMode: Proxy".
		// https://issues.k8s.io/123714
		ingress := &svc.Status.LoadBalancer.Ingress[0]
		if ingress.IP == "" || (ingress.IPMode != nil && *ingress.IPMode == v1.LoadBalancerIPModeProxy) {
			e2eskipper.Skipf("LoadBalancer uses 'Proxy' IPMode")
		}

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
		if pollErr := wait.PollUntilContextTimeout(ctx, framework.Poll, loadBalancerPropagationTimeout, true, func(ctx context.Context) (bool, error) {
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
})

func ipToSourceRange(ip string) string {
	if netutils.IsIPv6String(ip) {
		return ip + "/128"
	}
	return ip + "/32"
}

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
	ds.Spec.Template.Spec.TerminationGracePeriodSeconds = ptr.To(gracePeriod)

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
	// FIXME: need a better platform-independent timeout
	timeout := e2eservice.LoadBalancerLagTimeoutAWS
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
			defer func() { _ = resp.Body.Close() }()
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
	var previousHTTPErrors uint64 = 0
	for i := 1; i <= 5; i++ {
		framework.Logf("Update daemon pods environment: [{\"name\":\"VERSION\",\"value\":\"%d\"}]", i)
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","env":[{"name":"VERSION","value":"%d"}]}]}}}}`, ds.Spec.Template.Spec.Containers[0].Name, i)
		ds, err = cs.AppsV1().DaemonSets(ns).Patch(context.TODO(), name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err)

		framework.Logf("Check that daemon pods are available on every node of the cluster with the updated environment.")
		err = wait.PollUntilContextTimeout(ctx, framework.Poll, creationTimeout, true, func(ctx context.Context) (bool, error) {
			podList, err := cs.CoreV1().Pods(ds.Namespace).List(ctx, metav1.ListOptions{})
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
		currentHTTPErrors := atomic.LoadUint64(&httpErrors)

		partialTotalRequests := currentTotalRequests - previousTotalRequests
		partialNetworkErrors := currentNetworkErrors - previousNetworkErrors
		partialHTTPErrors := currentHTTPErrors - previousHTTPErrors
		partialSuccessRate := (float64(partialTotalRequests) - float64(partialNetworkErrors+partialHTTPErrors)) / float64(partialTotalRequests)

		framework.Logf("Load Balancer total HTTP requests: %d", partialTotalRequests)
		framework.Logf("Network errors: %d", partialNetworkErrors)
		framework.Logf("HTTP errors: %d", partialHTTPErrors)
		framework.Logf("Success rate: %.2f%%", partialSuccessRate*100)
		if partialSuccessRate < minSuccessRate {
			framework.Failf("Encountered too many errors when doing HTTP requests to the load balancer address. Success rate is %.2f%%, and the minimum allowed threshold is %.2f%%.", partialSuccessRate*100, minSuccessRate*100)
		}

		previousTotalRequests = currentTotalRequests
		previousNetworkErrors = currentNetworkErrors
		previousHTTPErrors = currentHTTPErrors
	}

	// assert that the load balancer address is still reachable after the rolling updates are finished
	e2eservice.TestReachableHTTP(ctx, lbNameOrAddress, svcPort, timeout)
}
