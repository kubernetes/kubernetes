/*
Copyright 2014 Google Inc. All rights reserved.

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
	"net"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func validateIPs(c *client.Client, ns, expectedPort string, expectedEndpoints []string, endpoints *api.Endpoints) bool {
	ips := util.StringSet{}
	for _, spec := range endpoints.Endpoints {
		host, port, err := net.SplitHostPort(spec)
		if err != nil {
			glog.Warningf("invalid endpoint spec: %s", spec)
			return false
		}
		if port != expectedPort {
			glog.Warningf("invalid port, expected %s, got %s", expectedPort, port)
			return false
		}
		ips.Insert(host)
	}

	for _, name := range expectedEndpoints {
		pod, err := c.Pods(ns).Get(name)
		if err != nil {
			glog.Warningf("failed to get pod %s, that's pretty weird. validation failed.", name)
			return false
		}
		if !ips.Has(pod.Status.PodIP) {
			glog.Warningf("ip validation failed, expected: %v, saw: %v", ips, pod.Status.PodIP)
			return false
		}
	}
	return true
}

func validateEndpoints(c *client.Client, ns, serviceName, expectedPort string, expectedEndpoints []string) bool {
	done := make(chan bool)
	running := true
	go func() {
		ok := true
		for running {
			endpoints, err := c.Endpoints(ns).Get(serviceName)
			if err == nil {
				if len(endpoints.Endpoints) != len(expectedEndpoints) {
					glog.Warningf("Unexpected endpoints: %#v, expected %v", endpoints, expectedEndpoints)
				} else if validateIPs(c, ns, expectedPort, expectedEndpoints, endpoints) {
					break
				}
			} else {
				glog.Warningf("Failed to get endpoints: %v", err)
			}
			time.Sleep(time.Second)
		}
		done <- ok
	}()

	select {
	case result := <-done:
		return result
	case <-time.After(60 * time.Second):
		glog.Errorf("Timed out waiting for endpoints.")
		running = false
		return false
	}
}

func addPod(c *client.Client, ns, name string, labels map[string]string) error {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "test",
					Image: "kubernetes/pause",
					Ports: []api.Port{{ContainerPort: 80}},
				},
			},
		},
	}
	_, err := c.Pods(ns).Create(pod)
	return err
}

func TestEndpoints(c *client.Client) bool {
	serviceName := "endpoint-test"
	ns := api.NamespaceDefault
	labels := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}

	defer func() {
		err := c.Services(ns).Delete(serviceName)
		if err != nil {
			glog.Errorf("Failed to delete service: %v (%v)", serviceName, err)
		}
	}()

	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: serviceName,
		},
		Spec: api.ServiceSpec{
			Port:          80,
			Selector:      labels,
			ContainerPort: util.NewIntOrStringFromInt(80),
		},
	}
	if _, err := c.Services(ns).Create(service); err != nil {
		glog.Errorf("Failed to create endpoint test service: %v", err)
		return false
	}
	expectedPort := "80"

	if !validateEndpoints(c, ns, serviceName, expectedPort, []string{}) {
		return false
	}

	name1 := "test1"
	if err := addPod(c, ns, name1, labels); err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	names := []string{name1}
	defer func() {
		for _, name := range names {
			err := c.Pods(ns).Delete(name)
			if err != nil {
				glog.Errorf("Failed to delete pod: %v (%v)", name, err)
			}
		}
	}()

	if !validateEndpoints(c, ns, serviceName, expectedPort, names) {
		return false
	}

	name2 := "test2"
	if err := addPod(c, ns, name2, labels); err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	names = append(names, name2)

	if !validateEndpoints(c, ns, serviceName, expectedPort, names) {
		return false
	}

	if err := c.Pods(ns).Delete(name1); err != nil {
		glog.Errorf("Failed to delete pod: %s", name1)
		return false
	}
	names = []string{name2}

	if !validateEndpoints(c, ns, serviceName, expectedPort, names) {
		return false
	}

	return true
}

var _ = Describe("TestEndpoints", func() {
	It("should pass", func() {
		c, err := loadClient()
		Expect(err).NotTo(HaveOccurred())
		Expect(TestEndpoints(c)).To(BeTrue())
	})
})
