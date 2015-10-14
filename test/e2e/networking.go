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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
	"net/http"
	"strconv"
	"strings"
	"time"
)

//versions ~ 1.3 (original RO test), 1.6 uses newer services/tokens,...
const nettestVersion = "1.6"

var _ = Describe("Networking", func() {
	f := NewFramework("nettest")
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
	It("should provide unchanging, static URL paths for kubernetes api services [Conformance].", func() {
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

	//Each tuple defined in this struct array represents
	//a number of services, and a timeout.  So for example,
	//{1, 300} defines a test where 1 service is created and
	//we give it 300 seconds to timeout.  This is how we test
	//services in parallel... we can create a tuple like {5,400}
	//to confirm that services over 5 ports all pass the networking test.
	serviceSoakTests := []struct {
		service        int
		timeoutSeconds time.Duration
	}{
		//These are very liberal, once this test is running regularly,
		//We will DECREASE the timeout value.
		//Replace this with --scale constants eventually https://github.com/kubernetes/kubernetes/issues/10479.
		{1, time.Duration(100 * time.Second)},
		{3, time.Duration(200 * time.Second)}, //Test that parallel nettests running on different ports complete.
	}

	for _, svcSoak := range serviceSoakTests {
		//copy to local to avoid range overwriting
		timeoutSeconds := svcSoak.timeoutSeconds
		serviceNum := svcSoak.service
		It(fmt.Sprintf("should function for intrapod communication between all hosts in %v parallel services [Conformance]", serviceNum),
			func() {
				Logf("running service test with timeout = %v for %v", timeout, serviceNum)
				runNetTest(timeoutSeconds, f, makePorts(serviceNum), nettestVersion)
			})
	}
})

//pollPeerStatus will either fail, pass, or continue polling.
//When completed, it will write the service name to the channel provided, thus
//facilitating parallel service testing.
func pollPeerStatus(serviceDoneChannel chan string, f *Framework, svc *api.Service, pollTimeoutSeconds time.Duration) {

	Logf("Begin polling " + svc.Name)
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

	passed := false

	expectNoError(wait.Poll(2*time.Second, pollTimeoutSeconds, func() (bool, error) {
		body, err := getStatus()
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}

		// Finally, we pass/fail the test based on if the container's response body, as to whether or not it was able to find peers.
		switch {
		case string(body) == "pass":
			passed = true
			return true, nil
		case string(body) == "running":
		case string(body) == "fail":
			if body, err = getDetails(); err != nil {
				Failf("Failed on attempt. Error reading details: %v", err)
				return false, err
			} else {
				Failf("Failed on attempt. Details:\n%s", string(body))
				return false, nil
			}
		case strings.Contains(string(body), "no endpoints available"):
			Logf("Attempt: waiting on service/endpoints")
		default:
			Logf("Unexpected response:\n%s", body)
		}
		return false, nil
	}))

	if !passed {
		if body, err := getDetails(); err != nil {
			Failf("Timed out.  Major error : Couldn't read service details: %v", err)
		} else {
			Failf("Timed out. Service details :\n%s", string(body))
		}
	}
	serviceDoneChannel <- svc.Name
}

//runNetTest Creates a single pod on each host which serves
//on a unique port in the cluster.  It then binds a service to
//that port, so that there are "n" nodes to balance traffic to -
//finally, each node reaches out to ping every other node in
//the cluster on the given port.
//The more ports given, the more services will be spun up,
//i.e. one service per port.
//To test basic pod networking, send a single port.
//To soak test the services, we can send a range (i.e. 8000-9000).
func runNetTest(timeoutSeconds time.Duration, f *Framework, ports []int, nettestVersion string) {
	nodes, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
	expectNoError(err, "Failed to list nodes")

	// previous tests may have caused failures of some nodes. Let's skip
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
			Failf("The test requires two Ready nodes on %s, but found just one.", testContext.Provider)
		}
		Logf("Only one ready node is detected. The test has limited scope in such setting. " +
			"Rerun it with at least two nodes to get complete coverage.")
	}

	portCompleteChannel := make(chan string, len(ports))

	for p := range ports {
		go func(nodes *api.NodeList, port int) {
			var svcname = fmt.Sprintf("nettest-%v", port)

			defer GinkgoRecover()

			By(fmt.Sprintf("creating a service named %q in namespace %q", svcname, f.Namespace.Name))
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
						Port:       port,
						TargetPort: util.NewIntOrStringFromInt(port),
					}},
					Selector: map[string]string{
						"name": svcname,
					},
				},
			})

			if err != nil {
				Failf("unable to create test service named [%s] %v on port %v", svc.Name, err, port)
			} else {
				Logf("Created service successfully [%s]", svc.Name)
			}

			// Clean up service
			defer func() {
				By("Cleaning up the service")
				if err := f.Client.Services(f.Namespace.Name).Delete(svc.Name); err != nil {
					Failf("unable to delete svc %v: %v", svc.Name, err)
				}
			}()

			Logf("launching pod per node.....")
			podNames := launchNetTestPodPerNode(port, nettestVersion, f, nodes, svcname)
			// Clean up the pods
			defer func() {
				By("Cleaning up the webserver pods")
				for _, podName := range podNames {
					if err = f.Client.Pods(f.Namespace.Name).Delete(podName, nil); err != nil {
						Logf("Failed to delete pod %s: %v", podName, err)
					}
				}
			}()

			Logf("Launched test pods for %v", port)
			By("waiting for all of the webserver pods to transition to Running + reaching the Passing state.")

			for _, podName := range podNames {
				err = f.WaitForPodRunning(podName)
				Expect(err).NotTo(HaveOccurred())
				By(fmt.Sprintf("waiting for connectivity to be verified [ port =  %v ] ", port))
				//once response OK, evaluate response body for pass/fail.
				pollPeerStatus(portCompleteChannel, f, svc, timeoutSeconds)
			}

			Logf("Finished test pods for %v", port)
		}(nodes, ports[p])
	}
	//now wait for the all X nettests to complete...
	for pReturned := range ports {
		Logf("Waiting on ports to report back.  So far %v have been discovered...", pReturned)
		Logf("... Another service has successfully been discovered: %v ( %v ) ", pReturned, <-portCompleteChannel)
	}
	Logf("Completed test on %v port/services", len(ports))
}

//makePorts makes a bunch of ports from 8080->8080+n
func makePorts(n int) []int {
	m := make([]int, n)
	for i := 0; i < n; i++ {
		m[i] = 8080 + i
	}
	return m
}

//launchNetTestPodPerNode launches nettest pods, and returns their names.
func launchNetTestPodPerNode(port int, version string, f *Framework, nodes *api.NodeList, name string) []string {
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
							"-port=" + strconv.Itoa(port),
							"-service=" + name,
							//peers >= totalPods should be asserted by the container.
							//the nettest container finds peers by looking up list of svc endpoints.
							fmt.Sprintf("-peers=%d", totalPods),
							"-namespace=" + f.Namespace.Name},
						Ports: []api.ContainerPort{{ContainerPort: port}},
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
