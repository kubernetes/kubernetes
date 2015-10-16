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
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/rand"
	"k8s.io/kubernetes/pkg/util/wait"
)

// versions ~ 1.3 (original RO test), 1.6 uses newer services/tokens,...
const nettestVersion = "1.6"

var _ = Describe("Networking", func() {

	f := NewFramework("nettest")
	BeforeEach(func() {
		// Assert basic external connectivity.
		// Since this is not really a test of kubernetes in any way, we
		// leave it as a pre-test assertion, rather than a Ginko test.
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

	// Each tuple defined in this struct array represents
	// a number of services, and a timeout.  So for example,
	// {1, 300} defines a test where 1 service is created and
	// we give it 300 seconds to timeout.  This is how we test
	// services in parallel... we can create a tuple like {5,400}
	// to confirm that services over 5 ports all pass the networking test.
	serviceSoakTests := []struct {
		service        int
		timeoutSeconds time.Duration
	}{
		// Note: On a GCE 3 node cluster, 30 ports is no problem.  > 50 seems to get blocked.
		{3, time.Duration(100 * time.Second)},
	}

	for _, svcSoak := range serviceSoakTests {
		// copy to local to avoid range overwriting
		timeoutSeconds := svcSoak.timeoutSeconds
		serviceNum := svcSoak.service
		It(fmt.Sprintf("should function for intrapod communication between all hosts in %v parallel services [Conformance]", serviceNum),
			func() {
				// Create a list of a few 1000 ports to use as a grab bag.
				minPort := 1000
				maxPort := 10000
				allPorts := rand.Intsn(minPort, 9000, maxPort)
				Logf("Running service test with timeout = %v for %v, ports = ", timeout, serviceNum, allPorts)
				runNetTest(timeoutSeconds, f, allPorts, serviceNum, nettestVersion)
			})
	}
})

type PeerTestResult struct {
	passed    bool
	completed bool
	err       error
}

var netTestDone = false

// pollPeerStatus will either fail, pass, or continue polling.
// returns true if polling succeeds.  writing to "quit" exits polling.
func pollPeerStatus(f *Framework, svc *api.Service, pollTimeoutSeconds time.Duration) PeerTestResult {
	getDetails := func() ([]byte, error) {
		return f.Client.Get().
			Namespace(f.Namespace.Name).
			Prefix("proxy").
			Resource("services").
			Name(svc.Name).
			Suffix("read").
			DoRaw()
	}

	// This can panic if the other tests complete first and the NS is destroyed.
	getStatus := func() ([]byte, error) {
		nsExists := f.Client.Get().Namespace(f.Namespace.Name)
		if nsExists == nil {
			return nil, errors.New("Ns doesnt exist")
		}
		return f.Client.Get().
			Namespace(f.Namespace.Name).
			Prefix("proxy").
			Resource("services").
			Name(svc.Name).
			Suffix("status").
			DoRaw()
	}

	Logf("Begin polling " + svc.Name)

	pollingFunction := func() PeerTestResult {
		body, err := getStatus()
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}

		// Finally, we pass/fail the test based on the container's response body was able to find peers.
		switch {
		case string(body) == "pass":
			return PeerTestResult{true, true, nil}
		case string(body) == "running":
		case string(body) == "fail":
			if body, err = getDetails(); err != nil {
				// Test completed, and it failed, with an error.
				Failf("Failed on attempt. Error reading details: %v", err)
				return PeerTestResult{
					passed:    false,
					completed: true,
					err:       err}
			} else {
				Failf("Failed on attempt. Details:\n%s", string(body))
				// Test completed, and it failed.
				return PeerTestResult{
					passed:    false,
					completed: true,
					err:       err}
			}
		// The below cases all have ambiguous test status: test may or may not be over.
		case netTestDone == true:
		case strings.Contains(string(body), "no endpoints available"):
			Logf("Attempt: waiting on service/endpoints")
		default:
			glog.V(1).Infof("Unexpected response:\n%s", body)
		}
		return PeerTestResult{
			passed:    false,
			completed: false,
			err:       nil}
	}

	var testResult PeerTestResult
	expectNoError(wait.Poll(2*time.Second, pollTimeoutSeconds,
		func() (bool, error) {
			testResult = pollingFunction()
			return testResult.passed, testResult.err
		}))

	if testResult.completed && !testResult.passed {
		if body, err := getDetails(); err != nil {
			Failf("Timed out.  Major error : Couldn't read service details: %v", testResult.err)
		} else {
			Failf("Timed out. Service details :\n%s", string(body))
		}
	}

	// Test result at this point will be either "cut short", or "passed".
	return testResult
}

// runNetTest Creates a single pod on each host which serves
// on a unique port in the cluster.  It then binds a service to
// that port, so that there are "n" nodes to balance traffic to -
// finally, each node reaches out to ping every other node in
// the cluster on the given port.
// The more ports given, the more services will be spun up,
// i.e. one service per port.
// To test basic pod networking, send a single port.
// To soak test the services, we can send a range (i.e. 8000-9000).
func runNetTest(timeoutSeconds time.Duration, f *Framework, portsGrabBag []int, minSvcPorts int, nettestVersion string) {
	// Make sure this is false.
	netTestDone = false
	if len(portsGrabBag) < minSvcPorts {
		panic("This test cannot proceed, the number of valid ports to pick from must be >= the min number of services ports to test.")
	}

	// To be safe, update the signal to all polling goroutines to end when the test is timed out.
	go func() {
		time.Sleep(timeoutSeconds)
		netTestDone = true
	}()

	nodes, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
	expectNoError(err, "Failed to list nodes")

	// previous tests may have caused failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	filterNodes(nodes, func(node api.Node) bool {
		return !node.Spec.Unschedulable && isNodeConditionSetAsExpected(&node, api.NodeReady, true)
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

	// Test doesn't register as passed until all channels report back.
	portCompleteChannel := make(chan string, minSvcPorts)

	// Trial-and-error for all services before we start: So we don't need a stochastic test.
	servicesToTest := createNServices(f, minSvcPorts, portsGrabBag)

	// This is where the actual port connectivity test is invoked.
	for _, service := range servicesToTest {
		go func() {
			passed := CreatePodsAndWait(f, nodes, service, timeoutSeconds)
			if passed {
				portCompleteChannel <- service.Name
			} else {
				// Fail the test and set the done flag, so that all polling stops.
				netTestDone = true
				Failf("Test failed.")
			}
		}()
	}

	for pReturned := 0; pReturned < minSvcPorts; pReturned++ {
		Logf("Waiting on ports to report back.  So far %v have been discovered...", pReturned)
		Logf("... Another service has successfully been discovered: %v ( %v )", pReturned, <-portCompleteChannel)
		Logf("Completion: %v/%v", pReturned, minSvcPorts)
	}

	Logf("Completed test on %v ports", minSvcPorts)
}

// createService creates a service.  Since we might create many service over many ports.
// Given the stochastic nature of open ports, errors aren't necessarily test failures, so we return
// all the data to the caller to decide how to proceed.
func createService(f *Framework, port int) (*api.Service, error) {
	svcname := fmt.Sprintf("nettest-soak-port-%v", port)
	By(fmt.Sprintf("creating a service named %q in namespace %q", svcname, f.Namespace.Name))
	service, err := f.Client.Services(f.Namespace.Name).Create(&api.Service{
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
				TargetPort: intstr.FromInt(port),
			}},
			Selector: map[string]string{
				"name": svcname,
			},
		},
	})

	if err != nil {
		Logf("unable to create test service named [%s] %v on port %v", service.Name, err, port)
		return service, err
		// return the ptr since thats what Get() returns for Get(name string) (*api.Service, error)
	} else {
		Logf("Created service successfully [%s]", service.Name)
		return service, nil
	}
}

// create an array of services, trying out multiple ports in the grab bag.
func createNServices(f *Framework, n int, portsToTry []int) []*api.Service {
	services := make([]*api.Service, n, n)
	for p := 0; p < n; p++ {
		if s, err := createService(f, portsToTry[p]); err == nil {
			services[p] = s
			Logf("Created service %v from element %v out of %v possible ports.", s, p, len(portsToTry))
		} else {
			glog.Warning("Skipping port %v due to error %v.", p, err)
		}
	}
	return services
}

// testPort Runs a test over a single port.  Since this may take a while, it blocks.
// Run it in a goroutine.  When it completes, it will write to the completionChannel.
// Returns wether it passed or failed.
func CreatePodsAndWait(f *Framework, nodes *api.NodeList, service *api.Service, timeout time.Duration) bool {
	defer GinkgoRecover()
	port := service.Spec.Ports[0].Port

	Logf("launching pod for each service/node pair: total pods to launch = %v over service name:%v", nodes, service.Name)
	podNames := CreateAllPods(port, nettestVersion, f, nodes, service.Name)
	Logf("Launched test pods for %v", port)
	By("waiting for all of the webserver pods to transition to Running + reaching the Passing state.")
	for _, podName := range podNames {
		err := f.WaitForPodRunning(podName)
		Expect(err).NotTo(HaveOccurred())
		By(fmt.Sprintf("waiting for connectivity to be verified [ pod = %v ,  port =  %v ] ", podName, port))

		// This is where the actual test occurs: Each pod should pass this test.
		netTestResult := pollPeerStatus(f, service, timeout)
		if !netTestDone && !netTestResult.passed {
			Failf("Failed on pod %v over port %v .", podName, port)
			return false
		}
	}
	// Getting here only happens if all pods can communicate over service, so this service worked on all nodes.
	Logf("Finished test pods for %v", port)
	return true
}

// launchNetTestPodPerNode launches nettest pods, and returns their names.
func CreateAllPods(port int, version string, f *Framework, nodes *api.NodeList, svcName string) []string {
	podNames := []string{}
	Logf("launching net test pod per node %v on with service name %v", len(nodes.Items), svcName)
	totalPods := len(nodes.Items)

	Expect(totalPods).NotTo(Equal(0))

	for _, node := range nodes.Items {
		pod, err := f.Client.Pods(f.Namespace.Name).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				GenerateName: svcName + "-",
				Labels: map[string]string{
					"name": svcName,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "webserver",
						Image: "gcr.io/google_containers/nettest:" + version,
						Args: []string{
							"-port=" + strconv.Itoa(port),
							"-service=" + svcName,
							// peers >= totalPods should be asserted by the container.
							// the nettest container finds peers by looking up list of svc endpoints.
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
