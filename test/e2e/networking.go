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
	//"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	//"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var c *client.Client = nil

func LaunchNetTestPodPerNode(nodes *api.NodeList, name string, c *client.Client, ns string) []string {
	podNames := []string{}

	totalPods := len(nodes.Items)

	Expect(totalPods).NotTo(Equal(0))

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
						Image: "gcr.io/google_containers/nettest:1.1",
						Args: []string{
							"-service=" + name,
							//peers >= totalPods should be asserted by the container.
							//the nettest container finds peers by looking up list of svc endpoints.
							fmt.Sprintf("-peers=%d", totalPods),
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
	return podNames
}

var _ = Describe("Networking", func() {

	//This namespace is modified throughout the course of the test.
	var namespace *api.Namespace
	var svcname = "nettest"
	var c *client.Client = nil

	BeforeEach(func() {
		//Assert basic external connectivity.
		//Since this is not really a test of kubernetes in any way, we
		//leave it as a pre-test assertion, rather than a Ginko test.
		By("Executing a successful http request from the external internet")
		resp, err := http.Get("http://google.com")
		if err != nil {
			Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}

		By("Creating a kubernetes client")
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		By("Building a namespace api object")
		namespace, err = createTestingNS("nettest", c)
		Expect(err).NotTo(HaveOccurred())

	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", namespace.Name))
		if err := c.Namespaces().Delete(namespace.Name); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	// First test because it has no dependencies on variables created later on.
	It("should provide unchanging, static URL paths for kubernetes api services.", func() {
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
				Namespace(namespace.Name).
				AbsPath(test.path).
				DoRaw()
			if err != nil {
				Failf("Failed: %v\nBody: %s", err, string(data))
			}
		}
	})

	//Now we can proceed with the test.
	It("should function for intra-pod communication", func() {

		if testContext.Provider == "vagrant" {
			By("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
			return
		}

		By(fmt.Sprintf("Creating a service named [%s] in namespace %s", svcname, namespace.Name))
		svc, err := c.Services(namespace.Name).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: svcname,
				Labels: map[string]string{
					"name": svcname,
				},
			},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol:   "TCP",
					Port:       8080,
					TargetPort: util.NewIntOrStringFromInt(8080),
				}},
				Selector: map[string]string{
					"name": svcname,
				},
			},
		})
		if err != nil {
			Failf("unable to create test service named [%s] %v", svc.Name, err)
		}

		// Clean up service
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the service")
			if err = c.Services(namespace.Name).Delete(svc.Name); err != nil {
				Failf("unable to delete svc %v: %v", svc.Name, err)
			}
		}()

		By("Creating a webserver (pending) pod on each node")

		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}

		podNames := LaunchNetTestPodPerNode(nodes, svcname, c, namespace.Name)

		// Clean up the pods
		defer func() {
			defer GinkgoRecover()
			By("Cleaning up the webserver pods")
			for _, podName := range podNames {
				if err = c.Pods(namespace.Name).Delete(podName, nil); err != nil {
					Logf("Failed to delete pod %s: %v", podName, err)
				}
			}
		}()

		By("Waiting for the webserver pods to transition to Running state")
		for _, podName := range podNames {
			err = waitForPodRunningInNamespace(c, podName, namespace.Name)
			Expect(err).NotTo(HaveOccurred())
		}

		By("Waiting for connectivity to be verified")
		const maxAttempts = 60
		passed := false

		//once response OK, evaluate response body for pass/fail.
		var body []byte

		for i := 0; i < maxAttempts && !passed; i++ {
			time.Sleep(2 * time.Second)
			Logf("About to make a proxy status call")
			start := time.Now()
			body, err = c.Get().
				Namespace(namespace.Name).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("status").
				DoRaw()
			Logf("Proxy status call returned in %v", time.Since(start))
			if err != nil {
				Logf("Attempt %v/%v: service/pod still starting. (error: '%v')", i, maxAttempts, err)
				continue
			}
			//Finally, we pass/fail the test based on if the container's response body, as to wether or not it was able to find peers.
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
					Namespace(namespace.Name).Prefix("proxy").
					Resource("services").
					Name(svc.Name).Suffix("read").
					DoRaw(); err != nil {
					Failf("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
				} else {
					Failf("Failed on attempt %v. Cleaning up. Details:\n%s", i, string(body))
				}
				break
			}
		}

		if !passed {
			if body, err = c.Get().
				Namespace(namespace.Name).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("read").
				DoRaw(); err != nil {
				Failf("Timed out. Cleaning up. Error reading details: %v", err)
			} else {
				Failf("Timed out. Cleaning up. Details:\n%s", string(body))
			}
		}
		Expect(string(body)).To(Equal("pass"))
	})

})
