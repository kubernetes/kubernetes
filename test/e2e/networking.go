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
	"net/http"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Networking", func() {
	f := NewFramework("nettest")

	var svcname = "nettest"

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
	})

	It("should provide Internet connection for containers [Conformance]", func() {
		By("Running container which tries to wget google.com")
		podName := "wget-test"
		contName := "wget-test-container"
		pod := &api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    contName,
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"wget", "-s", "google.com"},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}
		_, err := f.Client.Pods(f.Namespace.Name).Create(pod)
		expectNoError(err)
		defer f.Client.Pods(f.Namespace.Name).Delete(podName, nil)

		By("Verify that the pod succeed")
		expectNoError(waitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name))
	})

	// First test because it has no dependencies on variables created later on.
	It("should provide unchanging, static URL paths for kubernetes api services [Conformance]", func() {
		tests := []struct {
			path string
		}{
			{path: "/validate"},
			{path: "/healthz"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := f.Client.RESTClient.Get().
				Namespace(f.Namespace.Name).
				AbsPath(test.path).
				DoRaw()
			if err != nil {
				Failf("Failed: %v\nBody: %s", err, string(data))
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
			By("Cleaning up the service")
			if err = f.Client.Services(f.Namespace.Name).Delete(svc.Name); err != nil {
				Failf("unable to delete svc %v: %v", svc.Name, err)
			}
		}()

		By("Creating a webserver (pending) pod on each node")

		nodes, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}
		// previous tests may have cause failures of some nodes. Let's skip
		// 'Not Ready' nodes, just in case (there is no need to fail the test).
		filterNodes(nodes, func(node api.Node) bool {
			return isNodeReadySetAsExpected(&node, true)
		})

		if len(nodes.Items) == 0 {
			Failf("No Ready nodes found.")
		}
		if len(nodes.Items) == 1 {
			// in general, the test requires two nodes. But for local development, often a one node cluster
			// is created, for simplicity and speed. (see issue #10012). We permit one-node test
			// only in some cases
			if !providerIs("local") {
				Failf(fmt.Sprintf("The test requires two Ready nodes on %s, but found just one.", testContext.Provider))
			}
			Logf("Only one ready node is detected. The test has limited scope in such setting. " +
				"Rerun it with at least two nodes to get complete coverage.")
		}

		podNames := LaunchNetTestPodPerNode(f, nodes, svcname, "1.6")

		// Clean up the pods
		defer func() {
			By("Cleaning up the webserver pods")
			for _, podName := range podNames {
				if err = f.Client.Pods(f.Namespace.Name).Delete(podName, nil); err != nil {
					Logf("Failed to delete pod %s: %v", podName, err)
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
			return f.Client.Get().
				Namespace(f.Namespace.Name).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("read").
				DoRaw()
		}

		getStatus := func() ([]byte, error) {
			return f.Client.Get().
				Namespace(f.Namespace.Name).
				Prefix("proxy").
				Resource("services").
				Name(svc.Name).
				Suffix("status").
				DoRaw()
		}

		timeout := time.Now().Add(2 * time.Minute)
		for i := 0; !passed && timeout.After(time.Now()); i++ {
			time.Sleep(2 * time.Second)
			Logf("About to make a proxy status call")
			start := time.Now()
			body, err = getStatus()
			Logf("Proxy status call returned in %v", time.Since(start))
			if err != nil {
				Logf("Attempt %v: service/pod still starting. (error: '%v')", i, err)
				continue
			}
			// Finally, we pass/fail the test based on if the container's response body, as to whether or not it was able to find peers.
			switch {
			case string(body) == "pass":
				Logf("Passed on attempt %v. Cleaning up.", i)
				passed = true
			case string(body) == "running":
				Logf("Attempt %v: test still running", i)
			case string(body) == "fail":
				if body, err = getDetails(); err != nil {
					Failf("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
				} else {
					Failf("Failed on attempt %v. Cleaning up. Details:\n%s", i, string(body))
				}
			case strings.Contains(string(body), "no endpoints available"):
				Logf("Attempt %v: waiting on service/endpoints", i)
			default:
				Logf("Unexpected response:\n%s", body)
			}
		}

		if !passed {
			if body, err = getDetails(); err != nil {
				Failf("Timed out. Cleaning up. Error reading details: %v", err)
			} else {
				Failf("Timed out. Cleaning up. Details:\n%s", string(body))
			}
		}
		Expect(string(body)).To(Equal("pass"))
	})

})

func LaunchNetTestPodPerNode(f *Framework, nodes *api.NodeList, name, version string) []string {
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
						Image: "gcr.io/google_containers/nettest:" + version,
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
		Logf("Created pod %s on node %s", pod.ObjectMeta.Name, node.Name)
		podNames = append(podNames, pod.ObjectMeta.Name)
	}
	return podNames
}
