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
	"fmt"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Networking", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	// Create a unique namespace for this test.
	ns := "nettest-" + randomSuffix()

	It("should function for pods", func() {
		if testContext.provider == "vagrant" {
			By("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
			return
		}

		// Obtain a list of nodes so we can place one webserver container on each node.
		nodes, err := c.Nodes().List()
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}
		peers := len(nodes.Items)
		if peers == 0 {
			Failf("Failed to find any nodes")
		}

		// Test basic external connectivity.
		resp, err := http.Get("http://google.com/")
		if err != nil {
			Failf("unable to talk to the external internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			Failf("unexpected error code. expected 200, got: %v (%v)", resp.StatusCode, resp)
		}

		name := "nettest"

		By(fmt.Sprintf("Creating service with name %s in namespace %s", name, ns))
		svc, err := c.Services(ns).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": name,
				},
			},
			Spec: api.ServiceSpec{
				Port:       8080,
				TargetPort: util.NewIntOrStringFromInt(8080),
				Selector: map[string]string{
					"name": name,
				},
			},
		})
		if err != nil {
			Failf("unable to create test service %s: %v", svc.Name, err)
		}
		// Clean up service
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the service")
			if err = c.Services(ns).Delete(svc.Name); err != nil {
				Failf("unable to delete svc %v: %v", svc.Name, err)
			}
		}()

		By("Creating a webserver pod on each node")
		podNames := []string{}
		for i, node := range nodes.Items {
			podName := fmt.Sprintf("%s-%d", name, i)
			podNames = append(podNames, podName)
			Logf("Creating pod %s on node %s", podName, node.Name)
			_, err := c.Pods(ns).Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
					Labels: map[string]string{
						"name": name,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "webserver",
							Image: "kubernetes/nettest:1.1",
							Command: []string{
								"-service=" + name,
								fmt.Sprintf("-peers=%d", peers),
								"-namespace=" + ns},
							Ports: []api.ContainerPort{{ContainerPort: 8080}},
						},
					},
					Host:          node.Name,
					RestartPolicy: api.RestartPolicyNever,
				},
			})
			Expect(err).NotTo(HaveOccurred())
		}
		// Clean up the pods
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the webserver pods")
			for _, podName := range podNames {
				if err = c.Pods(ns).Delete(podName); err != nil {
					Logf("Failed to delete pod %s: %v", podName, err)
				}
			}
		}()

		By("Wait for the webserver pods to be ready")
		for _, podName := range podNames {
			err = waitForPodRunningInNamespace(c, podName, ns)
			Expect(err).NotTo(HaveOccurred())
		}

		By("Waiting for connectivity to be verified")
		const maxAttempts = 60
		passed := false
		var body []byte
		for i := 0; i < maxAttempts && !passed; i++ {
			time.Sleep(2 * time.Second)
			body, err = c.Get().
				Namespace(ns).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("status").
				Do().Raw()
			if err != nil {
				Logf("Attempt %v/%v: service/pod still starting. (error: '%v')", i, maxAttempts, err)
				continue
			}
			switch string(body) {
			case "pass":
				Logf("Passed on attempt %v. Cleaning up.", i)
				passed = true
				break
			case "running":
				Logf("Attempt %v/%v: test still running", i, maxAttempts)
				break
			case "fail":
				if body, err = c.Get().
					Namespace(ns).Prefix("proxy").
					Resource("services").
					Name(svc.Name).Suffix("read").
					Do().Raw(); err != nil {
					Failf("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
				} else {
					Failf("Failed on attempt %v. Cleaning up. Details:\n%s", i, string(body))
				}
				break
			}
		}

		if !passed {
			if body, err = c.Get().
				Namespace(ns).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("read").
				Do().Raw(); err != nil {
				Failf("Timed out. Cleaning up. Error reading details: %v", err)
			} else {
				Failf("Timed out. Cleaning up. Details:\n%s", string(body))
			}
		}
		Expect(string(body)).To(Equal("pass"))
	})

	It("should provide unchanging URLs", func() {
		tests := []struct {
			path string
		}{
			{path: "/validate"},
			{path: "/healthz"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := c.RESTClient.Get().
				Namespace(ns).
				AbsPath(test.path).
				Do().Raw()
			if err != nil {
				Failf("Failed: %v\nBody: %s", err, string(data))
			}
		}
	})
})
