/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/metrics"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("MetricsGrabber", func() {
	framework := NewDefaultFramework("metrics-grabber")
	var c *client.Client
	var grabber *metrics.MetricsGrabber
	BeforeEach(func() {
		var err error
		c = framework.Client
		expectNoError(err)
		grabber, err = metrics.NewMetricsGrabber(c, true, true, true, true)
		expectNoError(err)
	})

	It("should grab all metrics from API server.", func() {
		By("Connecting to /metrics endpoint")
		response, err := grabber.GrabFromApiServer()
		expectNoError(err)
		Expect(response).NotTo(BeEmpty())
	})

	It("should grab all metrics from a Kubelet.", func() {
		By("Proxying to Node through the API server")
		nodes := ListSchedulableNodesOrDie(c)
		Expect(nodes.Items).NotTo(BeEmpty())
		response, err := grabber.GrabFromKubelet(nodes.Items[0].Name)
		expectNoError(err)
		Expect(response).NotTo(BeEmpty())
	})

	It("should grab all metrics from a Scheduler.", func() {
		By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.Nodes().List(api.ListOptions{})
		expectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			Logf("Master is node registered. Skipping testing Scheduler metrics.")
			return
		}
		response, err := grabber.GrabFromScheduler()
		expectNoError(err)
		Expect(response).NotTo(BeEmpty())
	})

	It("should grab all metrics from a ControllerManager.", func() {
		By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.Nodes().List(api.ListOptions{})
		expectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			Logf("Master is node registered. Skipping testing ControllerManager metrics.")
			return
		}
		response, err := grabber.GrabFromControllerManager()
		expectNoError(err)
		Expect(response).NotTo(BeEmpty())
	})
})
