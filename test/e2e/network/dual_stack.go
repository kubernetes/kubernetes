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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
)

// Tests for ipv4-ipv6 dual-stack feature
var _ = common.SIGDescribe(feature.IPv6DualStack, func() {
	f := framework.NewDefaultFramework("dualstack")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should have ipv4 and ipv6 internal node ip", func(ctx context.Context) {
		// TODO (aramase) can switch to new function to get all nodes
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, cs)
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			// get all internal ips for node
			internalIPs := e2enode.GetAddresses(&node, v1.NodeInternalIP)

			gomega.Expect(internalIPs).To(gomega.HaveLen(2))
			// assert 2 ips belong to different families
			if netutils.IsIPv4String(internalIPs[0]) == netutils.IsIPv4String(internalIPs[1]) {
				framework.Failf("both internalIPs %s and %s belong to the same families", internalIPs[0], internalIPs[1])
			}
		}
	})

	ginkgo.It("should create pod, add ipv6 and ipv4 ip to pod ips", func(ctx context.Context) {
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
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		}

		ginkgo.By("submitting the pod to kubernetes")
		p := podClient.CreateSync(ctx, pod)

		gomega.Expect(p.Status.PodIP).ShouldNot(gomega.BeEquivalentTo(""))
		gomega.Expect(p.Status.PodIPs).ShouldNot(gomega.BeNil())

		// validate there are 2 ips in podIPs
		gomega.Expect(p.Status.PodIPs).To(gomega.HaveLen(2))
		// validate first ip in PodIPs is same as PodIP
		gomega.Expect(p.Status.PodIP).To(gomega.Equal(p.Status.PodIPs[0].IP))
		// assert 2 pod ips belong to different families
		if netutils.IsIPv4String(p.Status.PodIPs[0].IP) == netutils.IsIPv4String(p.Status.PodIPs[1].IP) {
			framework.Failf("both internalIPs %s and %s belong to the same families", p.Status.PodIPs[0].IP, p.Status.PodIPs[1].IP)
		}

		ginkgo.By("deleting the pod")
		err := podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err, "failed to delete pod")
	})

	f.It("should create pod, add ipv6 and ipv4 ip to host ips", func(ctx context.Context) {
		podName := "pod-dualstack-ips"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "dualstack-host-ips"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "dualstack-host-ips",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		}

		ginkgo.By("submitting the pod to kubernetes")
		p := podClient.CreateSync(ctx, pod)

		gomega.Expect(p.Status.HostIP).ShouldNot(gomega.BeEquivalentTo(""))
		gomega.Expect(p.Status.HostIPs).ShouldNot(gomega.BeNil())

		// validate there are 2 ips in hostIPs
		gomega.Expect(p.Status.HostIPs).To(gomega.HaveLen(2))
		// validate first ip in hostIPs is same as HostIP
		gomega.Expect(p.Status.HostIP).To(gomega.Equal(p.Status.HostIPs[0].IP))
		// assert 2 host ips belong to different families
		if netutils.IsIPv4String(p.Status.HostIPs[0].IP) == netutils.IsIPv4String(p.Status.HostIPs[1].IP) {
			framework.Failf("both internalIPs %s and %s belong to the same families", p.Status.HostIPs[0], p.Status.HostIPs[1])
		}

		ginkgo.By("deleting the pod")
		err := podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err, "failed to delete pod")
	})

	// takes close to 140s to complete, so doesn't need to be marked [SLOW]
	ginkgo.It("should be able to reach pod on ipv4 and ipv6 ip", func(ctx context.Context) {
		serverDeploymentName := "dualstack-server"
		clientDeploymentName := "dualstack-client"

		// get all schedulable nodes to determine the number of replicas for pods
		// this is to ensure connectivity from all nodes on cluster
		nodeList, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 3)
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

		serverDeployment, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(ctx, serverDeploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		clientDeployment, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(ctx, clientDeploymentSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2edeployment.WaitForDeploymentComplete(cs, serverDeployment)
		framework.ExpectNoError(err)
		err = e2edeployment.WaitForDeploymentComplete(cs, clientDeployment)
		framework.ExpectNoError(err)

		serverPods, err := e2edeployment.GetPodsForDeployment(ctx, cs, serverDeployment)
		framework.ExpectNoError(err)

		clientPods, err := e2edeployment.GetPodsForDeployment(ctx, cs, clientDeployment)
		framework.ExpectNoError(err)

		assertNetworkConnectivity(ctx, f, *serverPods, *clientPods, "dualstack-test-client", "80")
	})

	ginkgo.It("should create a single stack service with cluster ip from primary service range", func(ctx context.Context) {
		serviceName := "defaultclusterip"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamilies not set nil policy")
		service := createService(t.ServiceName, t.Namespace, t.Labels, nil, nil)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		expectedPolicy := v1.IPFamilyPolicySingleStack
		expectedFamilies := []v1.IPFamily{v1.IPv4Protocol}
		if framework.TestContext.ClusterIsIPv6() {
			expectedFamilies = []v1.IPFamily{v1.IPv6Protocol}
		}

		// check the spec has been set to default ip family
		validateServiceAndClusterIPFamily(svc, expectedFamilies, &expectedPolicy)

		// ensure endpoint belong to same ipfamily as service
		validateEndpointSlices(ctx, f, svc, expectedFamilies)
	})

	ginkgo.It("should create service with ipv4 cluster ip", func(ctx context.Context) {
		serviceName := "ipv4clusterip"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv4" + ns)

		expectedPolicy := v1.IPFamilyPolicySingleStack
		expectedFamilies := []v1.IPFamily{v1.IPv4Protocol}

		service := createService(t.ServiceName, t.Namespace, t.Labels, nil, expectedFamilies)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv4 and cluster ip belong to IPv4 family
		validateServiceAndClusterIPFamily(svc, expectedFamilies, &expectedPolicy)

		// ensure endpoints belong to same ipfamily as service
		validateEndpointSlices(ctx, f, svc, expectedFamilies)
	})

	ginkgo.It("should create service with ipv6 cluster ip", func(ctx context.Context) {
		serviceName := "ipv6clusterip"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv6" + ns)
		expectedPolicy := v1.IPFamilyPolicySingleStack
		expectedFamilies := []v1.IPFamily{v1.IPv6Protocol}

		service := createService(t.ServiceName, t.Namespace, t.Labels, nil, expectedFamilies)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv6 and cluster ip belongs to IPv6 family
		validateServiceAndClusterIPFamily(svc, expectedFamilies, &expectedPolicy)

		// ensure endpoints belong to same ipfamily as service
		validateEndpointSlices(ctx, f, svc, expectedFamilies)
	})

	ginkgo.It("should create service with ipv4,v6 cluster ip", func(ctx context.Context) {
		serviceName := "ipv4ipv6clusterip"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv4, IPv6" + ns)

		expectedPolicy := v1.IPFamilyPolicyRequireDualStack
		expectedFamilies := []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol}

		service := createService(t.ServiceName, t.Namespace, t.Labels, &expectedPolicy, expectedFamilies)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv4 and cluster ip belong to IPv4 family
		validateServiceAndClusterIPFamily(svc, expectedFamilies, &expectedPolicy)

		// ensure endpoints belong to same ipfamily as service
		validateEndpointSlices(ctx, f, svc, expectedFamilies)
	})

	ginkgo.It("should create service with ipv6,v4 cluster ip", func(ctx context.Context) {
		serviceName := "ipv6ipv4clusterip"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			if errs := t.Cleanup(); len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + ns + "/" + serviceName + " with Service.Spec.IPFamily IPv4, IPv6" + ns)

		expectedPolicy := v1.IPFamilyPolicyRequireDualStack
		expectedFamilies := []v1.IPFamily{v1.IPv6Protocol, v1.IPv4Protocol}

		service := createService(t.ServiceName, t.Namespace, t.Labels, &expectedPolicy, expectedFamilies)

		jig.Labels = t.Labels
		err := jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		svc, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		validateNumOfServicePorts(svc, 2)

		// check the spec has been set to IPv4 and cluster ip belong to IPv4 family
		validateServiceAndClusterIPFamily(svc, expectedFamilies, &expectedPolicy)

		// ensure endpoints belong to same ipfamily as service
		validateEndpointSlices(ctx, f, svc, expectedFamilies)
	})

	// Service Granular Checks as in k8s.io/kubernetes/test/e2e/network/networking.go
	// but using the secondary IP, so we run the same tests for each ClusterIP family
	ginkgo.Describe("Granular Checks: Services Secondary IP Family [LinuxOnly]", func() {

		ginkgo.It("should function for pod-Service: http", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err := config.DialFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.SecondaryNodeIP, config.NodeHTTPPort))
			err = config.DialFromTestContainer(ctx, "http", config.SecondaryNodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should function for pod-Service: udp", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			err := config.DialFromTestContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.SecondaryNodeIP, config.NodeUDPPort))
			err = config.DialFromTestContainer(ctx, "udp", config.SecondaryNodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		f.It("should function for pod-Service: sctp", feature.SCTPConnectivity, func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack, e2enetwork.EnableSCTP)
			ginkgo.By(fmt.Sprintf("dialing(sctp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterSCTPPort))
			err := config.DialFromTestContainer(ctx, "sctp", config.SecondaryClusterIP, e2enetwork.ClusterSCTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By(fmt.Sprintf("dialing(sctp) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.SecondaryNodeIP, config.NodeSCTPPort))
			err = config.DialFromTestContainer(ctx, "sctp", config.SecondaryNodeIP, config.NodeSCTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should function for node-Service: http", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack, e2enetwork.UseHostNetwork)
			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (config.clusterIP)", config.SecondaryNodeIP, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err := config.DialFromNode(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.SecondaryNodeIP, config.SecondaryNodeIP, config.NodeHTTPPort))
			err = config.DialFromNode(ctx, "http", config.SecondaryNodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should function for node-Service: udp", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack, e2enetwork.UseHostNetwork)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (config.clusterIP)", config.SecondaryNodeIP, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			err := config.DialFromNode(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.SecondaryNodeIP, config.SecondaryNodeIP, config.NodeUDPPort))
			err = config.DialFromNode(ctx, "udp", config.SecondaryNodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should function for endpoint-Service: http", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err := config.DialFromEndpointContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			ginkgo.By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.SecondaryNodeIP, config.NodeHTTPPort))
			err = config.DialFromEndpointContainer(ctx, "http", config.SecondaryNodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should function for endpoint-Service: udp", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			err := config.DialFromEndpointContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.SecondaryNodeIP, config.NodeUDPPort))
			err = config.DialFromEndpointContainer(ctx, "udp", config.SecondaryNodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should update endpoints: http", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err := config.DialFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			config.DeleteNetProxyPod(ctx)

			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err = config.DialFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		ginkgo.It("should update endpoints: udp", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			err := config.DialFromTestContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			config.DeleteNetProxyPod(ctx)

			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			err = config.DialFromTestContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})

		// [LinuxOnly]: Windows does not support session affinity.
		ginkgo.It("should function for client IP based session affinity: http [LinuxOnly]", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIPs[1], e2enetwork.ClusterHTTPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer(ctx, "http", config.SessionAffinityService.Spec.ClusterIPs[1], e2enetwork.ClusterHTTPPort, e2enetwork.SessionAffinityChecks)
			if err != nil {
				framework.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				framework.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				framework.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})

		// [LinuxOnly]: Windows does not support session affinity.
		ginkgo.It("should function for client IP based session affinity: udp [LinuxOnly]", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIPs[1], e2enetwork.ClusterUDPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer(ctx, "udp", config.SessionAffinityService.Spec.ClusterIPs[1], e2enetwork.ClusterUDPPort, e2enetwork.SessionAffinityChecks)
			if err != nil {
				framework.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				framework.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				framework.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})

		ginkgo.It("should be able to handle large requests: http", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			message := strings.Repeat("42", 1000)
			config.DialEchoFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, message)
		})

		ginkgo.It("should be able to handle large requests: udp", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			message := "n" + strings.Repeat("o", 1999)
			config.DialEchoFromTestContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, message)
		})

		// if the endpoints pods use hostNetwork, several tests can't run in parallel
		// because the pods will try to acquire the same port in the host.
		// We run the test in serial, to avoid port conflicts.
		ginkgo.It("should function for service endpoints using hostNetwork", func(ctx context.Context) {
			config := e2enetwork.NewNetworkingTestConfig(ctx, f, e2enetwork.EnableDualStack, e2enetwork.UseHostNetwork, e2enetwork.EndpointsUseHostNetwork)

			ginkgo.By("pod-Service(hostNetwork): http")

			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err := config.DialFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.SecondaryNodeIP, config.NodeHTTPPort))
			err = config.DialFromTestContainer(ctx, "http", config.SecondaryNodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By("node-Service(hostNetwork): http")

			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (config.clusterIP)", config.SecondaryNodeIP, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			err = config.DialFromNode(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.SecondaryNodeIP, config.SecondaryNodeIP, config.NodeHTTPPort))
			err = config.DialFromNode(ctx, "http", config.SecondaryNodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By("node-Service(hostNetwork): udp")

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (config.clusterIP)", config.SecondaryNodeIP, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))

			err = config.DialFromNode(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				time.Sleep(10 * time.Hour)
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.SecondaryNodeIP, config.SecondaryNodeIP, config.NodeUDPPort))
			err = config.DialFromNode(ctx, "udp", config.SecondaryNodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By("handle large requests: http(hostNetwork)")

			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.SecondaryClusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort))
			message := strings.Repeat("42", 1000)
			err = config.DialEchoFromTestContainer(ctx, "http", config.SecondaryClusterIP, e2enetwork.ClusterHTTPPort, config.MaxTries, 0, message)
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}

			ginkgo.By("handle large requests: udp(hostNetwork)")

			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.SecondaryClusterIP)", config.TestContainerPod.Name, config.SecondaryClusterIP, e2enetwork.ClusterUDPPort))
			message = "n" + strings.Repeat("o", 1999)
			err = config.DialEchoFromTestContainer(ctx, "udp", config.SecondaryClusterIP, e2enetwork.ClusterUDPPort, config.MaxTries, 0, message)
			if err != nil {
				framework.Failf("failed dialing endpoint, %v", err)
			}
		})
	})
})

func validateNumOfServicePorts(svc *v1.Service, expectedNumOfPorts int) {
	if len(svc.Spec.Ports) != expectedNumOfPorts {
		framework.Failf("got unexpected len(Spec.Ports) for service: %v", svc)
	}
}

func validateServiceAndClusterIPFamily(svc *v1.Service, expectedIPFamilies []v1.IPFamily, expectedPolicy *v1.IPFamilyPolicy) {
	if len(svc.Spec.IPFamilies) != len(expectedIPFamilies) {
		framework.Failf("service ip family nil for service %s/%s", svc.Namespace, svc.Name)
	}

	for idx, family := range expectedIPFamilies {
		if svc.Spec.IPFamilies[idx] != family {
			framework.Failf("service %s/%s expected family %v at index[%v] got %v", svc.Namespace, svc.Name, family, idx, svc.Spec.IPFamilies[idx])
		}
	}

	// validate ip assigned is from the family
	if len(svc.Spec.ClusterIPs) != len(svc.Spec.IPFamilies) {
		framework.Failf("service %s/%s assigned ips [%+v] does not match families [%+v]", svc.Namespace, svc.Name, svc.Spec.ClusterIPs, svc.Spec.IPFamilies)
	}

	for idx, family := range svc.Spec.IPFamilies {
		if (family == v1.IPv6Protocol) != netutils.IsIPv6String(svc.Spec.ClusterIPs[idx]) {
			framework.Failf("service %s/%s assigned ips at [%v]:%v does not match family:%v", svc.Namespace, svc.Name, idx, svc.Spec.ClusterIPs[idx], family)
		}
	}
	// validate policy
	if expectedPolicy == nil && svc.Spec.IPFamilyPolicy != nil {
		framework.Failf("service %s/%s expected nil for IPFamilyPolicy", svc.Namespace, svc.Name)
	}
	if expectedPolicy != nil && svc.Spec.IPFamilyPolicy == nil {
		framework.Failf("service %s/%s expected value %v for IPFamilyPolicy", svc.Namespace, svc.Name, expectedPolicy)
	}

	if expectedPolicy != nil && *(svc.Spec.IPFamilyPolicy) != *(expectedPolicy) {
		framework.Failf("service %s/%s expected value %v for IPFamilyPolicy", svc.Namespace, svc.Name, expectedPolicy)
	}
}

func validateEndpointSlices(ctx context.Context, f *framework.Framework, svc *v1.Service, expectedIPFamilies []v1.IPFamily) {
	var slices []discoveryv1.EndpointSlice
	err := e2eendpointslice.WaitForEndpointSlices(ctx, f.ClientSet, svc.Namespace, svc.Name, 500*time.Millisecond, 10*time.Second, func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (bool, error) {
		if len(endpointSlices) < len(expectedIPFamilies) {
			return false, nil
		}
		slices = endpointSlices
		return true, nil
	})
	framework.ExpectNoError(err, "could not validate EndpointSlices for service %s/%s", svc.Namespace, svc.Name)

	var wantIPv4, wantIPv6 bool
	for _, family := range expectedIPFamilies {
		if family == v1.IPv4Protocol {
			wantIPv4 = true
		} else if family == v1.IPv6Protocol {
			wantIPv6 = true
		}
	}

	for _, slice := range slices {
		ip := "(none)"
		if len(slice.Endpoints) > 0 && len(slice.Endpoints[0].Addresses) > 0 {
			ip = slice.Endpoints[0].Addresses[0]
		}
		if slice.AddressType == discoveryv1.AddressTypeIPv4 {
			if !wantIPv4 {
				framework.Failf("did not want IPv4 slice but got slice %s with IP %s", slice.Name, ip)
			}
			wantIPv4 = false
		} else if slice.AddressType == discoveryv1.AddressTypeIPv6 {
			if !wantIPv6 {
				framework.Failf("did not want IPv6 slice but got slice %s with IP %s", slice.Name, ip)
			}
			wantIPv6 = false
		}
	}
	if wantIPv4 {
		framework.Failf("wanted an IPv4 slice but did not get one")
	}
	if wantIPv6 {
		framework.Failf("wanted an IPv6 slice but did not get one")
	}
}

func assertNetworkConnectivity(ctx context.Context, f *framework.Framework, serverPods v1.PodList, clientPods v1.PodList, containerName, port string) {
	// curl from each client pod to all server pods to assert connectivity
	duration := "10s"
	pollInterval := "1s"
	timeout := 10

	var serverIPs []string
	for _, pod := range serverPods.Items {
		if pod.Status.PodIPs == nil || len(pod.Status.PodIPs) != 2 {
			framework.Failf("PodIPs list not expected value, got %v", pod.Status.PodIPs)
		}
		if netutils.IsIPv4String(pod.Status.PodIPs[0].IP) == netutils.IsIPv4String(pod.Status.PodIPs[1].IP) {
			framework.Failf("PodIPs should belong to different families, got %v", pod.Status.PodIPs)
		}
		serverIPs = append(serverIPs, pod.Status.PodIPs[0].IP, pod.Status.PodIPs[1].IP)
	}

	for _, clientPod := range clientPods.Items {
		for _, ip := range serverIPs {
			gomega.Consistently(ctx, func() error {
				ginkgo.By(fmt.Sprintf("checking connectivity from pod %s to serverIP: %s, port: %s", clientPod.Name, ip, port))
				cmd := checkNetworkConnectivity(ip, port, timeout)
				_, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, clientPod.Name, containerName, cmd...)
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

// createService returns a service spec with defined arguments
func createService(name, ns string, labels map[string]string, ipFamilyPolicy *v1.IPFamilyPolicy, ipFamilies []v1.IPFamily) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.ServiceSpec{
			Selector:       labels,
			Type:           v1.ServiceTypeNodePort,
			IPFamilyPolicy: ipFamilyPolicy,
			IPFamilies:     ipFamilies,
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
