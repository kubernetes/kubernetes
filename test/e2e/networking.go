/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Networking", func() {
	f := framework.NewDefaultFramework("nettest")

	var svcname = "nettest"

	BeforeEach(func() {
		//Assert basic external connectivity.
		//Since this is not really a test of kubernetes in any way, we
		//leave it as a pre-test assertion, rather than a Ginko test.
		By("Executing a successful http request from the external internet")
		resp, err := http.Get("http://google.com")
		if err != nil {
			framework.Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			framework.Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}
	})

	It("should provide Internet connection for containers [Conformance]", func() {
		By("Running container which tries to wget google.com")
		framework.ExpectNoError(framework.CheckConnectivityToHost(f, "", "wget-test", "google.com", 30))
	})

	// First test because it has no dependencies on variables created later on.
	It("should provide unchanging, static URL paths for kubernetes api services [Conformance]", func() {
		tests := []struct {
			path string
		}{
			{path: "/healthz"},
			{path: "/api"},
			{path: "/apis"},
			{path: "/logs"},
			{path: "/metrics"},
			{path: "/swaggerapi"},
			{path: "/version"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := f.Client.RESTClient.Get().
				AbsPath(test.path).
				DoRaw()
			if err != nil {
				framework.Failf("Failed: %v\nBody: %s", err, string(data))
			}
		}
	})

	//Now we can proceed with the test.
	It("should function for intra-pod communication [Conformance]", func() {

		By(fmt.Sprintf("Creating a service named %q in namespace %q", svcname, f.Namespace.Name))
		svc, err := f.Client.Services(f.Namespace.Name).Create(&api.Service{
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
					TargetPort: intstr.FromInt(8080),
				}},
				Selector: map[string]string{
					"name": svcname,
				},
			},
		})
		if err != nil {
			framework.Failf("unable to create test service named [%s] %v", svc.Name, err)
		}

		// Clean up service
		defer func() {
			By("Cleaning up the service")
			if err = f.Client.Services(f.Namespace.Name).Delete(svc.Name); err != nil {
				framework.Failf("unable to delete svc %v: %v", svc.Name, err)
			}
		}()

		By("Creating a webserver (pending) pod on each node")

		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(f.Client))
		nodes := framework.GetReadySchedulableNodesOrDie(f.Client)
		// This test is super expensive in terms of network usage - large services
		// result in huge "Endpoint" objects and all underlying pods read them
		// periodically. Moreover, all KubeProxies watch all of them.
		// Thus we limit the maximum number of pods under a service.
		//
		// TODO: Remove this limitation once services, endpoints and data flows
		// between nodes and master are better optimized.
		maxNodeCount := 250
		if len(nodes.Items) > maxNodeCount {
			nodes.Items = nodes.Items[:maxNodeCount]
		}

		if len(nodes.Items) == 1 {
			// in general, the test requires two nodes. But for local development, often a one node cluster
			// is created, for simplicity and speed. (see issue #10012). We permit one-node test
			// only in some cases
			if !framework.ProviderIs("local") {
				framework.Failf(fmt.Sprintf("The test requires two Ready nodes on %s, but found just one.", framework.TestContext.Provider))
			}
			framework.Logf("Only one ready node is detected. The test has limited scope in such setting. " +
				"Rerun it with at least two nodes to get complete coverage.")
		}

		podNames := LaunchNetTestPodPerNode(f, nodes, svcname)

		// Clean up the pods
		defer func() {
			By("Cleaning up the webserver pods")
			for _, podName := range podNames {
				if err = f.Client.Pods(f.Namespace.Name).Delete(podName, nil); err != nil {
					framework.Logf("Failed to delete pod %s: %v", podName, err)
				}
			}
		}()

		By("Waiting for the webserver pods to transition to Running state")
		for _, podName := range podNames {
			err = f.WaitForPodRunning(podName)
			Expect(err).NotTo(HaveOccurred())
		}

		By("Waiting for connectivity to be verified")
		passed := false

		//once response OK, evaluate response body for pass/fail.
		var body []byte
		getDetails := func() ([]byte, error) {
			proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
			if errProxy != nil {
				return nil, errProxy
			}
			return proxyRequest.Namespace(f.Namespace.Name).
				Name(svc.Name).
				Suffix("read").
				DoRaw()
		}

		getStatus := func() ([]byte, error) {
			proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
			if errProxy != nil {
				return nil, errProxy
			}
			return proxyRequest.Namespace(f.Namespace.Name).
				Name(svc.Name).
				Suffix("status").
				DoRaw()
		}

		// nettest containers will wait for all service endpoints to come up for 2 minutes
		// apply a 3 minutes observation period here to avoid this test to time out before the nettest starts to contact peers
		timeout := time.Now().Add(3 * time.Minute)
		for i := 0; !passed && timeout.After(time.Now()); i++ {
			time.Sleep(2 * time.Second)
			framework.Logf("About to make a proxy status call")
			start := time.Now()
			body, err = getStatus()
			framework.Logf("Proxy status call returned in %v", time.Since(start))
			if err != nil {
				framework.Logf("Attempt %v: service/pod still starting. (error: '%v')", i, err)
				continue
			}
			// Finally, we pass/fail the test based on if the container's response body, as to whether or not it was able to find peers.
			switch {
			case string(body) == "pass":
				framework.Logf("Passed on attempt %v. Cleaning up.", i)
				passed = true
			case string(body) == "running":
				framework.Logf("Attempt %v: test still running", i)
			case string(body) == "fail":
				if body, err = getDetails(); err != nil {
					framework.Failf("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
				} else {
					framework.Failf("Failed on attempt %v. Cleaning up. Details:\n%s", i, string(body))
				}
			case strings.Contains(string(body), "no endpoints available"):
				framework.Logf("Attempt %v: waiting on service/endpoints", i)
			default:
				framework.Logf("Unexpected response:\n%s", body)
			}
		}

		if !passed {
			if body, err = getDetails(); err != nil {
				framework.Failf("Timed out. Cleaning up. Error reading details: %v", err)
			} else {
				framework.Failf("Timed out. Cleaning up. Details:\n%s", string(body))
			}
		}
		Expect(string(body)).To(Equal("pass"))
	})

	framework.KubeDescribe("Granular Checks", func() {

		connectivityTimeout := 10

		It("should function for pod communication on a single node", func() {

			By("Picking a node")
			nodes := framework.GetReadySchedulableNodesOrDie(f.Client)
			node := nodes.Items[0]

			By("Creating a webserver pod")
			podName := "same-node-webserver"
			defer f.Client.Pods(f.Namespace.Name).Delete(podName, nil)
			ip := framework.LaunchWebserverPod(f, podName, node.Name)

			By("Checking that the webserver is accessible from a pod on the same node")
			framework.ExpectNoError(framework.CheckConnectivityToHost(f, node.Name, "same-node-wget", ip, connectivityTimeout))
		})

		It("should function for pod communication between nodes", func() {

			podClient := f.Client.Pods(f.Namespace.Name)

			By("Picking multiple nodes")
			nodes := framework.GetReadySchedulableNodesOrDie(f.Client)

			if len(nodes.Items) == 1 {
				framework.Skipf("The test requires two Ready nodes on %s, but found just one.", framework.TestContext.Provider)
			}

			node1 := nodes.Items[0]
			node2 := nodes.Items[1]

			By("Creating a webserver pod")
			podName := "different-node-webserver"
			defer podClient.Delete(podName, nil)
			ip := framework.LaunchWebserverPod(f, podName, node1.Name)

			By("Checking that the webserver is accessible from a pod on a different node")
			framework.ExpectNoError(framework.CheckConnectivityToHost(f, node2.Name, "different-node-wget", ip, connectivityTimeout))
		})
	})
})

func LaunchNetTestPodPerNode(f *framework.Framework, nodes *api.NodeList, name string) []string {
	podNames := []string{}

	totalPods := len(nodes.Items)

	Expect(totalPods).NotTo(Equal(0))

	for _, node := range nodes.Items {
		pod, err := f.Client.Pods(f.Namespace.Name).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				GenerateName: name + "-",
				Labels: map[string]string{
					"name": name,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "webserver",
						Image: "gcr.io/google_containers/nettest:1.9",
						Args: []string{
							"-service=" + name,
							//peers >= totalPods should be asserted by the container.
							//the nettest container finds peers by looking up list of svc endpoints.
							fmt.Sprintf("-peers=%d", totalPods),
							"-namespace=" + f.Namespace.Name},
						Ports: []api.ContainerPort{{ContainerPort: 8080}},
					},
				},
				NodeName:      node.Name,
				RestartPolicy: api.RestartPolicyNever,
			},
		})
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Created pod %s on node %s", pod.ObjectMeta.Name, node.Name)
		podNames = append(podNames, pod.ObjectMeta.Name)
	}
	return podNames
}
