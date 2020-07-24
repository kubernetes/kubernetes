/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	imageutils "k8s.io/kubernetes/test/utils/image"
	netutils "k8s.io/utils/net"
)

// Tests for ipv6 dual stack feature
var _ = SIGDescribe("[Feature:IPv6DualStackAlphaFeature] [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("dualstack")

	var cs clientset.Interface
	var podClient *framework.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = f.PodClient()
	})

	ginkgo.It("should have ipv4 and ipv6 internal node ip", func() {
		// TODO (aramase) can switch to new function to get all nodes
		nodeList, err := e2enode.GetReadySchedulableNodes(cs)
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			// get all internal ips for node
			internalIPs := e2enode.GetAddresses(&node, v1.NodeInternalIP)

			framework.ExpectEqual(len(internalIPs), 2)
			// assert 2 ips belong to different families
			framework.ExpectEqual(isIPv4(internalIPs[0]) != isIPv4(internalIPs[1]), true)
		}
	})

	ginkgo.It("should have ipv4 and ipv6 node podCIDRs", func() {
		// TODO (aramase) can switch to new function to get all nodes
		nodeList, err := e2enode.GetReadySchedulableNodes(cs)
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			framework.ExpectEqual(len(node.Spec.PodCIDRs), 2)
			// assert podCIDR is same as podCIDRs[0]
			framework.ExpectEqual(node.Spec.PodCIDR, node.Spec.PodCIDRs[0])
			// assert one is ipv4 and other is ipv6
			framework.ExpectEqual(isIPv4CIDR(node.Spec.PodCIDRs[0]) != isIPv4CIDR(node.Spec.PodCIDRs[1]), true)
		}
	})

	ginkgo.It("should create pod, add ipv6 and ipv4 ip to pod ips", func() {
		podName := "pod-dualstack-ips"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "dualstack-pod-ips"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "dualstack-pod-ips",
						Image: imageutils.GetE2EImage(imageutils.BusyBox),
					},
				},
			},
		}

		ginkgo.By("submitting the pod to kubernetes")
		podClient.CreateSync(pod)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name))

		p, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get pod %q", pod.Name)

		gomega.Expect(p.Status.PodIP).ShouldNot(gomega.BeEquivalentTo(""))
		gomega.Expect(p.Status.PodIPs).ShouldNot(gomega.BeNil())

		// validate there are 2 ips in podIPs
		framework.ExpectEqual(len(p.Status.PodIPs), 2)
		// validate first ip in PodIPs is same as PodIP
		framework.ExpectEqual(p.Status.PodIP, p.Status.PodIPs[0].IP)
		// assert 2 pod ips belong to different families
		framework.ExpectEqual(isIPv4(p.Status.PodIPs[0].IP) != isIPv4(p.Status.PodIPs[1].IP), true)

		ginkgo.By("deleting the pod")
		err = podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err, "failed to delete pod")
	})

	// takes close to 140s to complete, so doesn't need to be marked [SLOW]
	// this test is tagged with phase2 so we can skip this until phase 2 is completed and merged
	// TODO (aramase) remove phase 2 tag once phase 2 of dual stack is merged
	ginkgo.It("should be able to reach pod on ipv4 and ipv6 ip [Feature:IPv6DualStackAlphaFeature:Phase2]", func() {
		serverDeploymentName := "dualstack-server"
		clientDeploymentName := "dualstack-client"

		// get all schedulable nodes to determine the number of replicas for pods
		// this is to ensure connectivity from all nodes on cluster
		// FIXME: tests may be run in large clusters. This test is O(n^2) in the
		// number of nodes used. It should use GetBoundedReadySchedulableNodes().
		nodeList, err := e2enode.GetReadySchedulableNodes(cs)
		framework.ExpectNoError(err)

		replicas := int32(len(nodeList.Items))

		serverDeploymentSpec := e2edeployment.NewDeployment(serverDeploymentName,
			replicas,
			map[string]string{"test": "dual-stack-server"},
			"dualstack-test-server",
			imageutils.GetE2EImage(imageutils.Agnhost),
			appsv1.RollingUpdateDeploymentStrategyType)
		serverDeploymentSpec.Spec.Template.Spec.Containers[0].Args = []string{"test-webserver"}

		// to ensure all the pods land on different nodes and we can thereby
		// validate connectivity across all nodes.
		serverDeploymentSpec.Spec.Template.Spec.Affinity = &v1.Affinity{
			PodAntiAffinity: &v1.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "test",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"dualstack-test-server"},
								},
							},
						},
						TopologyKey: "kubernetes.io/hostname",
					},
				},
			},
		}

		clientDeploymentSpec := e2edeployment.NewDeployment(clientDeploymentName,
			replicas,
			map[string]string{"test": "dual-stack-client"},
			"dualstack-test-client",
			imageutils.GetE2EImage(imageutils.Agnhost),
			appsv1.RollingUpdateDeploymentStrategyType)

		clientDeploymentSpec.Spec.Template.Spec.Containers[0].Command = []string{"sleep", "3600"}
		clientDeploymentSpec.Spec.Template.Spec.Affinity = &v1.Affinity{
			PodAntiAffinity: &v1.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "test",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"dualstack-test-client"},
								},
							},
						},
						TopologyKey: "kubernetes.io/hostname",
					},
				},
			},
		}

		serverDeployment, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(context.TODO(), serverDeploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		clientDeployment, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(context.TODO(), clientDeploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2edeployment.WaitForDeploymentComplete(cs, serverDeployment)
		framework.ExpectNoError(err)
		err = e2edeployment.WaitForDeploymentComplete(cs, clientDeployment)
		framework.ExpectNoError(err)

		serverPods, err := e2edeployment.GetPodsForDeployment(cs, serverDeployment)
		framework.ExpectNoError(err)

		clientPods, err := e2edeployment.GetPodsForDeployment(cs, clientDeployment)
		framework.ExpectNoError(err)

		assertNetworkConnectivity(f, *serverPods, *clientPods, "dualstack-test-client", "80")
	})

	ginkgo.It("should create service with cluster ip from primary service range [Feature:IPv6DualStackAlphaFeature:Phase2]", func() {
		serviceName := "defaultclusterip"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		defaultIPFamily := v1.IPv4Protocol
		if framework.TestContext.ClusterIsIPv6() {
			defaultIPFamily = v1.IPv6Protocol
		}

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily not set")
		service := createService(t.ServiceName, t.Namespace, t.Labels, nil)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to default ip family
		validateServiceAndClusterIPFamily(svc, defaultIPFamily)

		// ensure endpoint belong to same ipfamily as service
		if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
			endpoint, err := cs.CoreV1().Endpoints(svc.Namespace).Get(context.TODO(), svc.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			validateEndpointsBelongToIPFamily(svc, endpoint, defaultIPFamily)
			return true, nil
		}); err != nil {
			framework.Failf("Get endpoints for service %s/%s failed (%s)", svc.Namespace, svc.Name, err)
		}
	})

	ginkgo.It("should create service with ipv4 cluster ip [Feature:IPv6DualStackAlphaFeature:Phase2]", func() {
		serviceName := "ipv4clusterip"
		ns := f.Namespace.Name
		ipv4 := v1.IPv4Protocol

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv4" + ns)
		service := createService(t.ServiceName, t.Namespace, t.Labels, &ipv4)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv4 and cluster ip belong to IPv4 family
		validateServiceAndClusterIPFamily(svc, ipv4)

		// ensure endpoints belong to same ipfamily as service
		if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
			endpoint, err := cs.CoreV1().Endpoints(svc.Namespace).Get(context.TODO(), svc.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			validateEndpointsBelongToIPFamily(svc, endpoint, ipv4)
			return true, nil
		}); err != nil {
			framework.Failf("Get endpoints for service %s/%s failed (%s)", svc.Namespace, svc.Name, err)
		}
	})

	ginkgo.It("should create service with ipv6 cluster ip [Feature:IPv6DualStackAlphaFeature:Phase2]", func() {
		serviceName := "ipv6clusterip"
		ns := f.Namespace.Name
		ipv6 := v1.IPv6Protocol

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv6" + ns)
		service := createService(t.ServiceName, t.Namespace, t.Labels, &ipv6)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv6 and cluster ip belongs to IPv6 family
		validateServiceAndClusterIPFamily(svc, ipv6)

		// ensure endpoints belong to same ipfamily as service
		if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
			endpoint, err := cs.CoreV1().Endpoints(svc.Namespace).Get(context.TODO(), svc.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			validateEndpointsBelongToIPFamily(svc, endpoint, ipv6)
			return true, nil
		}); err != nil {
			framework.Failf("Get endpoints for service %s/%s failed (%s)", svc.Namespace, svc.Name, err)
		}
	})
})

func validateNumOfServicePorts(svc *v1.Service, expectedNumOfPorts int) {
	if len(svc.Spec.Ports) != expectedNumOfPorts {
		framework.Failf("got unexpected len(Spec.Ports) for service: %v", svc)
	}
}

func validateServiceAndClusterIPFamily(svc *v1.Service, expectedIPFamily v1.IPFamily) {
	if svc.Spec.IPFamily == nil {
		framework.Failf("service ip family nil for service %s/%s", svc.Namespace, svc.Name)
	}
	if *svc.Spec.IPFamily != expectedIPFamily {
		framework.Failf("ip family mismatch for service: %s/%s, expected: %s, actual: %s", svc.Namespace, svc.Name, expectedIPFamily, *svc.Spec.IPFamily)
	}

	isIPv6ClusterIP := netutils.IsIPv6String(svc.Spec.ClusterIP)
	if (expectedIPFamily == v1.IPv4Protocol && isIPv6ClusterIP) || (expectedIPFamily == v1.IPv6Protocol && !isIPv6ClusterIP) {
		framework.Failf("got unexpected service ip %s, should belong to %s ip family", svc.Spec.ClusterIP, expectedIPFamily)
	}
}

func validateEndpointsBelongToIPFamily(svc *v1.Service, endpoint *v1.Endpoints, expectedIPFamily v1.IPFamily) {
	if len(endpoint.Subsets) == 0 {
		framework.Failf("Endpoint has no subsets, cannot determine service ip family matches endpoints ip family for service %s/%s", svc.Namespace, svc.Name)
	}
	for _, ss := range endpoint.Subsets {
		for _, e := range ss.Addresses {
			if (expectedIPFamily == v1.IPv6Protocol && isIPv4(e.IP)) || (expectedIPFamily == v1.IPv4Protocol && netutils.IsIPv6String(e.IP)) {
				framework.Failf("service endpoint %s doesn't belong to %s ip family", e.IP, expectedIPFamily)
			}
		}
	}
}

func assertNetworkConnectivity(f *framework.Framework, serverPods v1.PodList, clientPods v1.PodList, containerName, port string) {
	// curl from each client pod to all server pods to assert connectivity
	duration := "10s"
	pollInterval := "1s"
	timeout := 10

	var serverIPs []string
	for _, pod := range serverPods.Items {
		if pod.Status.PodIPs == nil || len(pod.Status.PodIPs) != 2 {
			framework.Failf("PodIPs list not expected value, got %v", pod.Status.PodIPs)
		}
		if isIPv4(pod.Status.PodIPs[0].IP) == isIPv4(pod.Status.PodIPs[1].IP) {
			framework.Failf("PodIPs should belong to different families, got %v", pod.Status.PodIPs)
		}
		serverIPs = append(serverIPs, pod.Status.PodIPs[0].IP, pod.Status.PodIPs[1].IP)
	}

	for _, clientPod := range clientPods.Items {
		for _, ip := range serverIPs {
			gomega.Consistently(func() error {
				ginkgo.By(fmt.Sprintf("checking connectivity from pod %s to serverIP: %s, port: %s", clientPod.Name, ip, port))
				cmd := checkNetworkConnectivity(ip, port, timeout)
				_, _, err := f.ExecCommandInContainerWithFullOutput(clientPod.Name, containerName, cmd...)
				return err
			}, duration, pollInterval).ShouldNot(gomega.HaveOccurred())
		}
	}
}

func checkNetworkConnectivity(ip, port string, timeout int) []string {
	curl := fmt.Sprintf("curl -g --connect-timeout %v http://%s", timeout, net.JoinHostPort(ip, port))
	cmd := []string{"/bin/sh", "-c", curl}
	return cmd
}

// isIPv4 checks if the provided ip belongs to ipv4 family.
// If ip belongs to ipv4 family, return true else it returns false
// TODO (aramase) move this to https://github.com/kubernetes/utils/blob/master/net/net.go
func isIPv4(ip string) bool {
	return net.ParseIP(ip).To4() != nil
}

// isIPv4CIDR checks if the provided cidr block belongs to ipv4 family.
// If cidr belongs to ipv4 family, return true else it returns false
// TODO (aramase) move this to https://github.com/kubernetes/utils/blob/master/net/net.go
func isIPv4CIDR(cidr string) bool {
	ip, _, err := net.ParseCIDR(cidr)
	framework.ExpectNoError(err)
	return isIPv4(ip.String())
}

// createService returns a service spec with defined arguments
func createService(name, ns string, labels map[string]string, ipFamily *v1.IPFamily) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
			Type:     v1.ServiceTypeNodePort,
			IPFamily: ipFamily,
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
}
