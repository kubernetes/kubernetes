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
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This should match whatever the default/configured range is
var ServiceNodePortRange = util.PortRange{Base: 30000, Size: 2767}

var _ = Describe("Services", func() {
	var c *client.Client
	// Use these in tests.  They're unique for each test to prevent name collisions.
	var namespaces [2]string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		By("Building a namespace api objects")
		for i := range namespaces {
			namespacePtr, err := createTestingNS(fmt.Sprintf("service-%d", i), c)
			Expect(err).NotTo(HaveOccurred())
			namespaces[i] = namespacePtr.Name
		}
	})

	AfterEach(func() {
		for _, ns := range namespaces {
			By(fmt.Sprintf("Destroying namespace %v", ns))
			if err := c.Namespaces().Delete(ns); err != nil {
				Failf("Couldn't delete namespace %s: %s", ns, err)
			}
		}
	})
	// TODO: We get coverage of TCP/UDP and multi-port services through the DNS test. We should have a simpler test for multi-port TCP here.
	It("should provide RW and RO services", func() {
		svc := api.ServiceList{}
		err := c.Get().
			AbsPath("/api/v1beta3/proxy/namespaces/default/services/kubernetes-ro/api/v1beta3/services").
			Do().
			Into(&svc)
		if err != nil {
			Failf("unexpected error listing services using ro service: %v", err)
		}
		var foundRW, foundRO bool
		for i := range svc.Items {
			if svc.Items[i].Name == "kubernetes" {
				foundRW = true
			}
			if svc.Items[i].Name == "kubernetes-ro" {
				foundRO = true
			}
		}
		Expect(foundRW).To(Equal(true))
		Expect(foundRO).To(Equal(true))
	})

	It("should serve a basic endpoint from pods", func(done Done) {
		serviceName := "endpoint-test2"
		ns := namespaces[0]
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}

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
					TargetPort: util.NewIntOrStringFromInt(80),
				}},
			},
		}
		_, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{})

		var names []string
		defer func() {
			for _, name := range names {
				err := c.Pods(ns).Delete(name, nil)
				Expect(err).NotTo(HaveOccurred())
			}
		}()

		name1 := "test1"
		addEndpointPodOrFail(c, ns, name1, labels, []api.ContainerPort{{ContainerPort: 80}})
		names = append(names, name1)

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{name1: {80}})

		name2 := "test2"
		addEndpointPodOrFail(c, ns, name2, labels, []api.ContainerPort{{ContainerPort: 80}})
		names = append(names, name2)

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{name1: {80}, name2: {80}})

		err = c.Pods(ns).Delete(name1, nil)
		Expect(err).NotTo(HaveOccurred())
		names = []string{name2}

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{name2: {80}})

		err = c.Pods(ns).Delete(name2, nil)
		Expect(err).NotTo(HaveOccurred())
		names = []string{}

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{})

		// We deferred Gingko pieces that may Fail, we aren't done.
		defer func() {
			close(done)
		}()
	}, 240.0)

	It("should serve multiport endpoints from pods", func(done Done) {
		// repacking functionality is intentionally not tested here - it's better to test it in an integration test.
		serviceName := "multi-endpoint-test"
		ns := namespaces[0]

		defer func() {
			err := c.Services(ns).Delete(serviceName)
			Expect(err).NotTo(HaveOccurred())
		}()

		labels := map[string]string{"foo": "bar"}

		svc1port := "svc1"
		svc2port := "svc2"

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
						TargetPort: util.NewIntOrStringFromString(svc1port),
					},
					{
						Name:       "portname2",
						Port:       81,
						TargetPort: util.NewIntOrStringFromString(svc2port),
					},
				},
			},
		}
		_, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())
		port1 := 100
		port2 := 101
		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{})

		var names []string
		defer func() {
			for _, name := range names {
				err := c.Pods(ns).Delete(name, nil)
				Expect(err).NotTo(HaveOccurred())
			}
		}()

		containerPorts1 := []api.ContainerPort{
			{
				Name:          svc1port,
				ContainerPort: port1,
			},
		}
		containerPorts2 := []api.ContainerPort{
			{
				Name:          svc2port,
				ContainerPort: port2,
			},
		}

		podname1 := "podname1"
		addEndpointPodOrFail(c, ns, podname1, labels, containerPorts1)
		names = append(names, podname1)
		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{podname1: {port1}})

		podname2 := "podname2"
		addEndpointPodOrFail(c, ns, podname2, labels, containerPorts2)
		names = append(names, podname2)
		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{podname1: {port1}, podname2: {port2}})

		podname3 := "podname3"
		addEndpointPodOrFail(c, ns, podname3, labels, append(containerPorts1, containerPorts2...))
		names = append(names, podname3)
		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{podname1: {port1}, podname2: {port2}, podname3: {port1, port2}})

		err = c.Pods(ns).Delete(podname1, nil)
		Expect(err).NotTo(HaveOccurred())
		names = []string{podname2, podname3}

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{podname2: {port2}, podname3: {port1, port2}})

		err = c.Pods(ns).Delete(podname2, nil)
		Expect(err).NotTo(HaveOccurred())
		names = []string{podname3}

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{podname3: {port1, port2}})

		err = c.Pods(ns).Delete(podname3, nil)
		Expect(err).NotTo(HaveOccurred())
		names = []string{}

		validateEndpointsOrFail(c, ns, serviceName, map[string][]int{})

		// We deferred Gingko pieces that may Fail, we aren't done.
		defer func() {
			close(done)
		}()
	}, 240.0)

	It("should be able to create a functioning external load balancer", func() {
		if !providerIs("gce", "gke", "aws") {
			By(fmt.Sprintf("Skipping service external load balancer test; uses ServiceTypeLoadBalancer, a (gce|gke|aws) feature"))
			return
		}

		serviceName := "external-lb-test"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeLoadBalancer

		By("creating service " + serviceName + " with external load balancer in namespace " + ns)
		result, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		// Wait for the load balancer to be created asynchronously, which is
		// currently indicated by ingress point(s) being added to the status.
		result, err = waitForLoadBalancerIngress(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())
		if len(result.Status.LoadBalancer.Ingress) != 1 {
			Failf("got unexpected number (%v) of ingress points for externally load balanced service: %v", result.Status.LoadBalancer.Ingress, result)
		}
		ingress := result.Status.LoadBalancer.Ingress[0]
		if len(result.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for LoadBalancer service: %v", result)
		}
		port := result.Spec.Ports[0]
		if port.NodePort == 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for LoadBalancer service: %v", result)
		}
		if !ServiceNodePortRange.Contains(port.NodePort) {
			Failf("got unexpected (out-of-range) port for LoadBalancer service: %v", result)
		}

		By("creating pod to be part of service " + serviceName)
		t.CreateWebserverPod()

		By("hitting the pod through the service's NodePort")
		testReachable(pickMinionIP(c), port.NodePort)

		By("hitting the pod through the service's external load balancer")
		testLoadBalancerReachable(ingress, 80)
	})

	It("should be able to create a functioning NodePort service", func() {
		serviceName := "nodeportservice-test"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		result, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())
		defer func(ns, serviceName string) { // clean up when we're done
			By("deleting service " + serviceName + " in namespace " + ns)
			err := c.Services(ns).Delete(serviceName)
			Expect(err).NotTo(HaveOccurred())
		}(ns, serviceName)

		if len(result.Spec.Ports) != 1 {
			Failf("got unexpected number (%d) of Ports for NodePort service: %v", len(result.Spec.Ports), result)
		}

		nodePort := result.Spec.Ports[0].NodePort
		if nodePort == 0 {
			Failf("got unexpected nodePort (%d) on Ports[0] for NodePort service: %v", nodePort, result)
		}
		if !ServiceNodePortRange.Contains(nodePort) {
			Failf("got unexpected (out-of-range) port for NodePort service: %v", result)
		}

		By("creating pod to be part of service " + serviceName)
		t.CreateWebserverPod()

		By("hitting the pod through the service's NodePort")
		ip := pickMinionIP(c)
		testReachable(ip, nodePort)
	})

	It("should be able to change the type and nodeport settings of a service", func() {
		serviceName := "mutability-service-test"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()

		By("creating service " + serviceName + " with type unspecified in namespace " + t.Namespace)
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeClusterIP {
			Failf("got unexpected Spec.Type for default service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for default service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort != 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for default service: %v", service)
		}
		if len(service.Status.LoadBalancer.Ingress) != 0 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for default service: %v", service)
		}

		By("creating pod to be part of service " + t.ServiceName)
		t.CreateWebserverPod()

		By("changing service " + serviceName + " to type=NodePort")
		service.Spec.Type = api.ServiceTypeNodePort
		service, err = c.Services(ns).Update(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			Failf("got unexpected Spec.Type for NodePort service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for NodePort service: %v", service)
		}
		port = service.Spec.Ports[0]
		if port.NodePort == 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for NodePort service: %v", service)
		}
		if !ServiceNodePortRange.Contains(port.NodePort) {
			Failf("got unexpected (out-of-range) port for NodePort service: %v", service)
		}

		if len(service.Status.LoadBalancer.Ingress) != 0 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for NodePort service: %v", service)
		}
		By("hitting the pod through the service's NodePort")
		ip := pickMinionIP(c)
		nodePort1 := port.NodePort // Save for later!
		testReachable(ip, nodePort1)

		By("changing service " + serviceName + " to type=LoadBalancer")
		service.Spec.Type = api.ServiceTypeLoadBalancer
		service, err = c.Services(ns).Update(service)
		Expect(err).NotTo(HaveOccurred())

		// Wait for the load balancer to be created asynchronously
		service, err = waitForLoadBalancerIngress(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeLoadBalancer {
			Failf("got unexpected Spec.Type for LoadBalancer service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for LoadBalancer service: %v", service)
		}
		port = service.Spec.Ports[0]
		if port.NodePort != nodePort1 {
			Failf("got unexpected Spec.Ports[0].nodePort for LoadBalancer service: %v", service)
		}
		if len(service.Status.LoadBalancer.Ingress) != 1 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for LoadBalancer service: %v", service)
		}
		ingress1 := service.Status.LoadBalancer.Ingress[0]
		if ingress1.IP == "" && ingress1.Hostname == "" {
			Failf("got unexpected Status.LoadBalancer.Ingresss[0] for LoadBalancer service: %v", service)
		}
		By("hitting the pod through the service's NodePort")
		ip = pickMinionIP(c)
		testReachable(ip, nodePort1)
		By("hitting the pod through the service's LoadBalancer")
		testLoadBalancerReachable(ingress1, 80)

		By("changing service " + serviceName + " update NodePort")
		nodePort2 := nodePort1 - 1
		if !ServiceNodePortRange.Contains(nodePort2) {
			//Check for (unlikely) assignment at bottom of range
			nodePort2 = nodePort1 + 1
		}
		service.Spec.Ports[0].NodePort = nodePort2
		service, err = c.Services(ns).Update(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeLoadBalancer {
			Failf("got unexpected Spec.Type for updated-NodePort service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for updated-NodePort service: %v", service)
		}
		port = service.Spec.Ports[0]
		if port.NodePort != nodePort2 {
			Failf("got unexpected Spec.Ports[0].nodePort for NodePort service: %v", service)
		}
		if len(service.Status.LoadBalancer.Ingress) != 1 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for NodePort service: %v", service)
		}
		ingress2 := service.Status.LoadBalancer.Ingress[0]
		// TODO: This is a problem on AWS; we can't just always be changing the LB
		Expect(ingress1).To(Equal(ingress2))

		By("hitting the pod through the service's updated NodePort")
		testReachable(ip, nodePort2)
		By("hitting the pod through the service's LoadBalancer")
		testLoadBalancerReachable(ingress2, 80)
		By("checking the old NodePort is closed")
		testNotReachable(ip, nodePort1)

		By("changing service " + serviceName + " back to type=ClusterIP")
		service.Spec.Type = api.ServiceTypeClusterIP
		service, err = c.Services(ns).Update(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeClusterIP {
			Failf("got unexpected Spec.Type for back-to-ClusterIP service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for back-to-ClusterIP service: %v", service)
		}
		port = service.Spec.Ports[0]
		if port.NodePort != 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for back-to-ClusterIP service: %v", service)
		}

		// Wait for the load balancer to be destroyed asynchronously
		service, err = waitForLoadBalancerDestroy(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())

		if len(service.Status.LoadBalancer.Ingress) != 0 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for back-to-ClusterIP service: %v", service)
		}
		By("checking the NodePort (original) is closed")
		ip = pickMinionIP(c)
		testNotReachable(ip, nodePort1)
		By("checking the NodePort (updated) is closed")
		ip = pickMinionIP(c)
		testNotReachable(ip, nodePort2)
		By("checking the LoadBalancer is closed")
		testLoadBalancerNotReachable(ingress2, 80)
	})

	It("should release the load balancer when Type goes from LoadBalancer -> NodePort", func() {
		serviceName := "service-release-lb"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeLoadBalancer

		By("creating service " + serviceName + " with type LoadBalancer")
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		By("creating pod to be part of service " + t.ServiceName)
		t.CreateWebserverPod()

		if service.Spec.Type != api.ServiceTypeLoadBalancer {
			Failf("got unexpected Spec.Type for LoadBalancer service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for LoadBalancer service: %v", service)
		}
		nodePort := service.Spec.Ports[0].NodePort
		if nodePort == 0 {
			Failf("got unexpected Spec.Ports[0].NodePort for LoadBalancer service: %v", service)
		}

		// Wait for the load balancer to be created asynchronously
		service, err = waitForLoadBalancerIngress(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())

		if len(service.Status.LoadBalancer.Ingress) != 1 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for LoadBalancer service: %v", service)
		}
		ingress := service.Status.LoadBalancer.Ingress[0]
		if ingress.IP == "" && ingress.Hostname == "" {
			Failf("got unexpected Status.LoadBalancer.Ingresss[0] for LoadBalancer service: %v", service)
		}

		By("hitting the pod through the service's NodePort")
		ip := pickMinionIP(c)
		testReachable(ip, nodePort)
		By("hitting the pod through the service's LoadBalancer")
		testLoadBalancerReachable(ingress, 80)

		By("changing service " + serviceName + " to type=NodePort")
		service.Spec.Type = api.ServiceTypeNodePort
		service, err = c.Services(ns).Update(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			Failf("got unexpected Spec.Type for NodePort service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for NodePort service: %v", service)
		}
		if service.Spec.Ports[0].NodePort != nodePort {
			Failf("got unexpected Spec.Ports[0].NodePort for NodePort service: %v", service)
		}

		// Wait for the load balancer to be created asynchronously
		service, err = waitForLoadBalancerDestroy(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())

		if len(service.Status.LoadBalancer.Ingress) != 0 {
			Failf("got unexpected len(Status.LoadBalancer.Ingresss) for NodePort service: %v", service)
		}

		By("hitting the pod through the service's NodePort")
		testReachable(ip, nodePort)
		By("checking the LoadBalancer is closed")
		testLoadBalancerNotReachable(ingress, 80)
	})

	It("should prevent NodePort collisions", func() {
		serviceName := "nodeport-collision"
		serviceName2 := serviceName + "2"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		result, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if result.Spec.Type != api.ServiceTypeNodePort {
			Failf("got unexpected Spec.Type for new service: %v", result)
		}
		if len(result.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for new service: %v", result)
		}
		port := result.Spec.Ports[0]
		if port.NodePort == 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", result)
		}

		By("creating service " + serviceName + " with conflicting NodePort")

		service2 := t.BuildServiceSpec()
		service2.Name = serviceName2
		service2.Spec.Type = api.ServiceTypeNodePort
		service2.Spec.Ports[0].NodePort = port.NodePort

		By("creating service " + serviceName2 + " with conflicting NodePort")
		result2, err := t.CreateService(service2)
		if err == nil {
			Failf("Created service with conflicting NodePort: %v", result2)
		}
		expectedErr := fmt.Sprintf("Service \"%s\" is invalid: spec.ports[0].nodePort: invalid value '%d': provided port is already allocated", serviceName2, port.NodePort)
		Expect(fmt.Sprintf("%v", err)).To(Equal(expectedErr))

		By("deleting original service " + serviceName + " with type NodePort in namespace " + ns)
		err = t.DeleteService(serviceName)
		Expect(err).NotTo(HaveOccurred())

		By("creating service " + serviceName2 + " with no-longer-conflicting NodePort")
		_, err = t.CreateService(service2)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should check NodePort out-of-range", func() {
		serviceName := "nodeport-range-test"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !ServiceNodePortRange.Contains(port.NodePort) {
			Failf("got unexpected (out-of-range) port for new service: %v", service)
		}

		outOfRangeNodePort := 0
		for {
			outOfRangeNodePort = 1 + rand.Intn(65535)
			if !ServiceNodePortRange.Contains(outOfRangeNodePort) {
				break
			}
		}
		By(fmt.Sprintf("changing service "+serviceName+" to out-of-range NodePort %d", outOfRangeNodePort))
		service.Spec.Ports[0].NodePort = outOfRangeNodePort
		result, err := t.Client.Services(t.Namespace).Update(service)
		if err == nil {
			Failf("failed to prevent update of service with out-of-range NodePort: %v", result)
		}
		expectedErr := fmt.Sprintf("Service \"%s\" is invalid: spec.ports[0].nodePort: invalid value '%d': provided port is not in the valid range", serviceName, outOfRangeNodePort)
		Expect(fmt.Sprintf("%v", err)).To(Equal(expectedErr))

		By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("creating service "+serviceName+" with out-of-range NodePort %d", outOfRangeNodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = outOfRangeNodePort
		service, err = t.CreateService(service)
		if err == nil {
			Failf("failed to prevent create of service with out-of-range NodePort (%d): %v", outOfRangeNodePort, service)
		}
		Expect(fmt.Sprintf("%v", err)).To(Equal(expectedErr))
	})

	It("should release NodePorts on delete", func() {
		serviceName := "nodeport-reuse"
		ns := namespaces[0]

		t := NewWebserverTest(c, ns, serviceName)
		defer func() {
			defer GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort

		By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())

		if service.Spec.Type != api.ServiceTypeNodePort {
			Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !ServiceNodePortRange.Contains(port.NodePort) {
			Failf("got unexpected (out-of-range) port for new service: %v", service)
		}
		port1 := port.NodePort

		By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("creating service "+serviceName+" with same NodePort %d", port1))
		service = t.BuildServiceSpec()
		service.Spec.Type = api.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = port1
		service, err = t.CreateService(service)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should correctly serve identically named services in different namespaces on different external IP addresses", func() {
		if !providerIs("gce", "gke", "aws") {
			By(fmt.Sprintf("Skipping service namespace collision test; uses ServiceTypeLoadBalancer, a (gce|gke|aws) feature"))
			return
		}

		serviceNames := []string{"s0"} // Could add more here, but then it takes longer.
		labels := map[string]string{
			"key0": "value0",
			"key1": "value1",
		}
		service := &api.Service{
			ObjectMeta: api.ObjectMeta{},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{{
					Port:       80,
					TargetPort: util.NewIntOrStringFromInt(80),
				}},
				Type: api.ServiceTypeLoadBalancer,
			},
		}

		ingressPoints := []string{}
		for _, namespace := range namespaces {
			for _, serviceName := range serviceNames {
				service.ObjectMeta.Name = serviceName
				service.ObjectMeta.Namespace = namespace
				By("creating service " + serviceName + " in namespace " + namespace)
				_, err := c.Services(namespace).Create(service)
				Expect(err).NotTo(HaveOccurred())
				defer func(namespace, serviceName string) { // clean up when we're done
					By("deleting service " + serviceName + " in namespace " + namespace)
					err := c.Services(namespace).Delete(serviceName)
					Expect(err).NotTo(HaveOccurred())
				}(namespace, serviceName)
			}
		}
		for _, namespace := range namespaces {
			for _, serviceName := range serviceNames {
				result, err := waitForLoadBalancerIngress(c, serviceName, namespace)
				Expect(err).NotTo(HaveOccurred())
				for i := range result.Status.LoadBalancer.Ingress {
					ingress := result.Status.LoadBalancer.Ingress[i].IP
					if ingress == "" {
						ingress = result.Status.LoadBalancer.Ingress[i].Hostname
					}
					ingressPoints = append(ingressPoints, ingress) // Save 'em to check uniqueness
				}
			}
		}
		validateUniqueOrFail(ingressPoints)
	})
})

func waitForLoadBalancerIngress(c *client.Client, serviceName, namespace string) (*api.Service, error) {
	const timeout = 4 * time.Minute
	var service *api.Service
	By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to have a LoadBalancer ingress point", timeout, serviceName, namespace))
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		service, err := c.Services(namespace).Get(serviceName)
		if err != nil {
			Logf("Get service failed, ignoring for 5s: %v", err)
			continue
		}
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			return service, nil
		}
		Logf("Waiting for service %s in namespace %s to have a LoadBalancer ingress point (%v)", serviceName, namespace, time.Since(start))
	}
	return service, fmt.Errorf("service %s in namespace %s doesn't have a LoadBalancer ingress point after %.2f seconds", serviceName, namespace, timeout.Seconds())
}

func waitForLoadBalancerDestroy(c *client.Client, serviceName, namespace string) (*api.Service, error) {
	const timeout = 4 * time.Minute
	var service *api.Service
	By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to have no LoadBalancer ingress points", timeout, serviceName, namespace))
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		service, err := c.Services(namespace).Get(serviceName)
		if err != nil {
			Logf("Get service failed, ignoring for 5s: %v", err)
			continue
		}
		if len(service.Status.LoadBalancer.Ingress) == 0 {
			return service, nil
		}
		Logf("Waiting for service %s in namespace %s to have no LoadBalancer ingress points (%v)", serviceName, namespace, time.Since(start))
	}
	return service, fmt.Errorf("service %s in namespace %s still has LoadBalancer ingress points after %.2f seconds", serviceName, namespace, timeout.Seconds())
}

func validateUniqueOrFail(s []string) {
	By(fmt.Sprintf("validating unique: %v", s))
	sort.Strings(s)
	var prev string
	for i, elem := range s {
		if i > 0 && elem == prev {
			Fail("duplicate found: " + elem)
		}
		prev = elem
	}
}

func getPortsByIp(subsets []api.EndpointSubset) map[string][]int {
	m := make(map[string][]int)
	for _, ss := range subsets {
		for _, port := range ss.Ports {
			for _, addr := range ss.Addresses {
				Logf("Found IP %v and port %v", addr.IP, port.Port)
				if _, ok := m[addr.IP]; !ok {
					m[addr.IP] = make([]int, 0)
				}
				m[addr.IP] = append(m[addr.IP], port.Port)
			}
		}
	}
	return m
}

func translatePodNameToIpOrFail(c *client.Client, ns string, expectedEndpoints map[string][]int) map[string][]int {
	portsByIp := make(map[string][]int)

	for name, portList := range expectedEndpoints {
		pod, err := c.Pods(ns).Get(name)
		if err != nil {
			Failf("failed to get pod %s, that's pretty weird. validation failed: %s", name, err)
		}
		portsByIp[pod.Status.PodIP] = portList
		By(fmt.Sprintf(""))
	}
	By(fmt.Sprintf("successfully translated pod names to ips: %v -> %v on namespace %s", expectedEndpoints, portsByIp, ns))
	return portsByIp
}

func validatePortsOrFail(endpoints map[string][]int, expectedEndpoints map[string][]int) {
	if len(endpoints) != len(expectedEndpoints) {
		// should not happen because we check this condition before
		Failf("invalid number of endpoints got %v, expected %v", endpoints, expectedEndpoints)
	}
	for ip := range expectedEndpoints {
		if _, ok := endpoints[ip]; !ok {
			Failf("endpoint %v not found", ip)
		}
		if len(endpoints[ip]) != len(expectedEndpoints[ip]) {
			Failf("invalid list of ports for ip %v. Got %v, expected %v", ip, endpoints[ip], expectedEndpoints[ip])
		}
		sort.Ints(endpoints[ip])
		sort.Ints(expectedEndpoints[ip])
		for index := range endpoints[ip] {
			if endpoints[ip][index] != expectedEndpoints[ip][index] {
				Failf("invalid list of ports for ip %v. Got %v, expected %v", ip, endpoints[ip], expectedEndpoints[ip])
			}
		}
	}
}

func validateEndpointsOrFail(c *client.Client, ns, serviceName string, expectedEndpoints map[string][]int) {
	By(fmt.Sprintf("Validating endpoints %v with on service %s/%s", expectedEndpoints, ns, serviceName))
	for {
		endpoints, err := c.Endpoints(ns).Get(serviceName)
		if err == nil {
			By(fmt.Sprintf("Found endpoints %v", endpoints))

			portsByIp := getPortsByIp(endpoints.Subsets)

			By(fmt.Sprintf("Found ports by ip %v", portsByIp))
			if len(portsByIp) == len(expectedEndpoints) {
				expectedPortsByIp := translatePodNameToIpOrFail(c, ns, expectedEndpoints)
				validatePortsOrFail(portsByIp, expectedPortsByIp)
				break
			} else {
				By(fmt.Sprintf("Unexpected number of endpoints: found %v, expected %v (ignoring for 1 second)", portsByIp, expectedEndpoints))
			}
		} else {
			By(fmt.Sprintf("Failed to get endpoints: %v (ignoring for 1 second)", err))
		}
		time.Sleep(time.Second)
	}
	By(fmt.Sprintf("successfully validated endpoints %v with on service %s/%s", expectedEndpoints, ns, serviceName))
}

func addEndpointPodOrFail(c *client.Client, ns, name string, labels map[string]string, containerPorts []api.ContainerPort) {
	By(fmt.Sprintf("Adding pod %v in namespace %v", name, ns))
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "test",
					Image: "gcr.io/google_containers/pause",
					Ports: containerPorts,
				},
			},
		},
	}
	_, err := c.Pods(ns).Create(pod)
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

func getMinionPublicIps(c *client.Client) ([]string, error) {
	nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return nil, err
	}

	ips := collectAddresses(nodes, api.NodeExternalIP)
	if len(ips) == 0 {
		ips = collectAddresses(nodes, api.NodeLegacyHostIP)
	}
	return ips, nil
}

func pickMinionIP(c *client.Client) string {
	publicIps, err := getMinionPublicIps(c)
	Expect(err).NotTo(HaveOccurred())
	if len(publicIps) == 0 {
		Failf("got unexpected number (%d) of public IPs", len(publicIps))
	}
	ip := publicIps[0]
	return ip
}

func testLoadBalancerReachable(ingress api.LoadBalancerIngress, port int) {
	ip := ingress.IP
	if ip == "" {
		ip = ingress.Hostname
	}

	testReachable(ip, port)
}

func testLoadBalancerNotReachable(ingress api.LoadBalancerIngress, port int) {
	ip := ingress.IP
	if ip == "" {
		ip = ingress.Hostname
	}

	testNotReachable(ip, port)
}

func testReachable(ip string, port int) {
	var err error
	var resp *http.Response

	url := fmt.Sprintf("http://%s:%d", ip, port)
	if ip == "" {
		Failf("got empty IP for reachability check", url)
	}
	if port == 0 {
		Failf("got port==0 for reachability check", url)
	}

	By(fmt.Sprintf("Checking reachability of %s", url))
	for t := time.Now(); time.Since(t) < podStartTimeout; time.Sleep(5 * time.Second) {
		resp, err = httpGetNoConnectionPool(url)
		if err == nil {
			break
		}
		By(fmt.Sprintf("Got error waiting for reachability of %s: %v", url, err))
	}
	Expect(err).NotTo(HaveOccurred())
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	if resp.StatusCode != 200 {
		Failf("received non-success return status %q trying to access %s; got body: %s", resp.Status, url, string(body))
	}
	if !strings.Contains(string(body), "test-webserver") {
		Failf("received response body without expected substring 'test-webserver': %s", string(body))
	}
}

func testNotReachable(ip string, port int) {
	var err error
	var resp *http.Response
	var body []byte

	url := fmt.Sprintf("http://%s:%d", ip, port)
	if ip == "" {
		Failf("got empty IP for non-reachability check", url)
	}
	if port == 0 {
		Failf("got port==0 for non-reachability check", url)
	}

	for t := time.Now(); time.Since(t) < podStartTimeout; time.Sleep(5 * time.Second) {
		resp, err = httpGetNoConnectionPool(url)
		if err != nil {
			break
		}
		body, err = ioutil.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		resp.Body.Close()
		By(fmt.Sprintf("Got success waiting for non-reachability of %s: %v", url, resp.Status))
	}
	if err == nil {
		Failf("able to reach service %s when should no longer have been reachable: %q body=%s", url, resp.Status, string(body))
	}
	// TODO: Check type of error
	By(fmt.Sprintf("Found (expected) error during not-reachability test %v", err))
}

// Does an HTTP GET, but does not reuse TCP connections
// This masks problems where the iptables rule has changed, but we don't see it
func httpGetNoConnectionPool(url string) (*http.Response, error) {
	tr := &http.Transport{
		DisableKeepAlives: true,
	}
	client := &http.Client{
		Transport: tr,
	}

	return client.Get(url)
}

// Simple helper class to avoid too much boilerplate in tests
type WebserverTest struct {
	ServiceName string
	Namespace   string
	Client      *client.Client

	TestId string
	Labels map[string]string

	pods     map[string]bool
	services map[string]bool

	// Used for generating e.g. unique pod names
	sequence int32
}

func NewWebserverTest(client *client.Client, namespace string, serviceName string) *WebserverTest {
	t := &WebserverTest{}
	t.Client = client
	t.Namespace = namespace
	t.ServiceName = serviceName
	t.TestId = t.ServiceName + "-" + string(util.NewUUID())
	t.Labels = map[string]string{
		"testid": t.TestId,
	}

	t.pods = make(map[string]bool)
	t.services = make(map[string]bool)

	return t
}

func (t *WebserverTest) SequenceNext() int {
	n := atomic.AddInt32(&t.sequence, 1)
	return int(n)
}

// Build default config for a service (which can then be changed)
func (t *WebserverTest) BuildServiceSpec() *api.Service {
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: t.ServiceName,
		},
		Spec: api.ServiceSpec{
			Selector: t.Labels,
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: util.NewIntOrStringFromInt(80),
			}},
		},
	}
	return service
}

// Create a pod with the well-known webserver configuration, and record it for cleanup
func (t *WebserverTest) CreateWebserverPod() {
	name := t.ServiceName + "-" + strconv.Itoa(t.SequenceNext())
	pod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: t.Labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "webserver",
					Image: "gcr.io/google_containers/test-webserver",
				},
			},
		},
	}
	_, err := t.CreatePod(pod)
	if err != nil {
		Failf("Failed to create pod %s: %v", pod.Name, err)
	}
	expectNoError(waitForPodRunningInNamespace(t.Client, pod.Name, t.Namespace))
}

// Create a pod, and record it for cleanup
func (t *WebserverTest) CreatePod(pod *api.Pod) (*api.Pod, error) {
	podClient := t.Client.Pods(t.Namespace)
	result, err := podClient.Create(pod)
	if err == nil {
		t.pods[pod.Name] = true
	}
	return result, err
}

// Create a service, and record it for cleanup
func (t *WebserverTest) CreateService(service *api.Service) (*api.Service, error) {
	result, err := t.Client.Services(t.Namespace).Create(service)
	if err == nil {
		t.services[service.Name] = true
	}
	return result, err
}

// Delete a service, and remove it from the cleanup list
func (t *WebserverTest) DeleteService(serviceName string) error {
	err := t.Client.Services(t.Namespace).Delete(serviceName)
	if err == nil {
		delete(t.services, serviceName)
	}
	return err
}

func (t *WebserverTest) Cleanup() []error {
	var errs []error

	for podName := range t.pods {
		podClient := t.Client.Pods(t.Namespace)
		By("deleting pod " + podName + " in namespace " + t.Namespace)
		err := podClient.Delete(podName, nil)
		if err != nil {
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
