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
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

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
		if !providerIs("gce", "gke") {
			By(fmt.Sprintf("Skipping service external load balancer test; uses createExternalLoadBalancer, a (gce|gke) feature"))
			return
		}

		serviceName := "external-lb-test"
		ns := namespaces[0]
		labels := map[string]string{
			"key0": "value0",
		}
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
				CreateExternalLoadBalancer: true,
			},
		}

		By("creating service " + serviceName + " with external load balancer in namespace " + ns)
		result, err := c.Services(ns).Create(service)
		Expect(err).NotTo(HaveOccurred())
		defer func(ns, serviceName string) { // clean up when we're done
			By("deleting service " + serviceName + " in namespace " + ns)
			err := c.Services(ns).Delete(serviceName)
			Expect(err).NotTo(HaveOccurred())
		}(ns, serviceName)

		// Wait for the load balancer to be created asynchronously, which is
		// currently indicated by a public IP address being added to the spec.
		result, err = waitForPublicIPs(c, serviceName, ns)
		Expect(err).NotTo(HaveOccurred())
		if len(result.Status.LoadBalancer.Ingress) != 1 {
			Failf("got unexpected number (%v) of ingress points for externally load balanced service: %v", result.Status.LoadBalancer.Ingress, result)
		}
		ingress := result.Status.LoadBalancer.Ingress[0]
		ip := ingress.IP
		if ip == "" {
			ip = ingress.Hostname
		}
		port := result.Spec.Ports[0].Port

		pod := &api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: latest.Version,
			},
			ObjectMeta: api.ObjectMeta{
				Name:   "elb-test-" + string(util.NewUUID()),
				Labels: labels,
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

		By("creating pod to be part of service " + serviceName)
		podClient := c.Pods(ns)
		defer func() {
			By("deleting pod " + pod.Name)
			defer GinkgoRecover()
			podClient.Delete(pod.Name, nil)
		}()
		if _, err := podClient.Create(pod); err != nil {
			Failf("Failed to create pod %s: %v", pod.Name, err)
		}
		expectNoError(waitForPodRunningInNamespace(c, pod.Name, ns))

		By("hitting the pod through the service's external load balancer")
		var resp *http.Response
		for t := time.Now(); time.Since(t) < podStartTimeout; time.Sleep(5 * time.Second) {
			resp, err = http.Get(fmt.Sprintf("http://%s:%d", ip, port))
			if err == nil {
				break
			}
		}
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		body, err := ioutil.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		if resp.StatusCode != 200 {
			Failf("received non-success return status %q trying to access pod through load balancer; got body: %s", resp.Status, string(body))
		}
		if !strings.Contains(string(body), "test-webserver") {
			Failf("received response body without expected substring 'test-webserver': %s", string(body))
		}
	})

	It("should correctly serve identically named services in different namespaces on different external IP addresses", func() {
		if !providerIs("gce", "gke") {
			By(fmt.Sprintf("Skipping service namespace collision test; uses createExternalLoadBalancer, a (gce|gke) feature"))
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
				CreateExternalLoadBalancer: true,
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
				result, err := waitForPublicIPs(c, serviceName, namespace)
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

func waitForPublicIPs(c *client.Client, serviceName, namespace string) (*api.Service, error) {
	const timeout = 4 * time.Minute
	var service *api.Service
	By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to have a public IP", timeout, serviceName, namespace))
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		service, err := c.Services(namespace).Get(serviceName)
		if err != nil {
			Logf("Get service failed, ignoring for 5s: %v", err)
			continue
		}
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			return service, nil
		}
		Logf("Waiting for service %s in namespace %s to have an ingress point (%v)", serviceName, namespace, time.Since(start))
	}
	return service, fmt.Errorf("service %s in namespace %s doesn't have an ingress point after %.2f seconds", serviceName, namespace, timeout.Seconds())
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
