/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

/*
These Network Policy tests create two services A and B.
There is a single pod in the service A , and a single pod on each node in
the service B.

Each pod is running a network monitor container (see test/images/network-monitor):
	-  The A pod uses service discovery to locate the B pods and waits
	   to establish communication (if expected) with those pods.
	-  Each B pod uses service discovery to locate the A pod and waits to
	   establish communication (if expected) with that pod.

We test the following:
	-  Namespace ingress isolation set to DefaultDeny
	- Full TCP isolation
	- Policy rules to allow mono-directional TCP connectivity
	- Policy rules to allow bi-directional TCP connectivity

	TODO:
	-  Test UDP traffic
	-  Test progressively increased isolation (bidir->monodir->fully isolated)
*/

// Summary is the data returned by /summary API on the network monitor container
type Summary struct {
	TCPNumOutboundConnected int
	TCPNumInboundConnected  int
}

const (
	serviceAName        = "service-a"
	serviceBName        = "service-b"
	netMonitorContainer = "gcr.io/google_containers/netmonitor:1.0"
	convergenceTimeout  = 1 // Connection convergence timeout in minutes
)

var _ = framework.KubeDescribe("NetworkPolicy", func() {
	f := framework.NewDefaultFramework("network-policy")

	It("should isolate containers in the same namespace when ingress isolation is DefaultDeny [Feature:NetworkPolicy]", func() {
		nsA := f.Namespace
		runIsolatedToBidirectionalTest(f, nsA, nsA)
	})

	It("should isolate containers in different namespaces when ingress isolation is DefaultDeny [Feature:NetworkPolicy]", func() {
		// This test use two namespaces.  A single namespace is created by
		// default.  Create another.
		nsA := f.Namespace
		nsB, err := f.CreateNamespace(f.BaseName+"-b", map[string]string{
			"e2e-framework": f.BaseName + "-b",
		})
		Expect(err).NotTo(HaveOccurred())
		runIsolatedToBidirectionalTest(f, nsA, nsB)
	})
})

// Run a test that starts with fully isolated pods in two services (in different namespaces) and adds
// policy objects to allow mono-directional and then bi-directional traffic between the pods in the
// two services.
func runIsolatedToBidirectionalTest(f *framework.Framework, nsA, nsB *api.Namespace) {

	var err error

	// Turn on ingress isolation on both namespaces.
	setNamespaceIsolation(f, nsA, "DefaultDeny")
	setNamespaceIsolation(f, nsB, "DefaultDeny")

	// Get the available nodes.
	nodes := framework.GetReadySchedulableNodesOrDie(f.Client)

	if len(nodes.Items) == 1 {
		// In general, the test requires two nodes. But for local development, often a one node cluster
		// is created, for simplicity and speed. We permit one-node test only in local development.
		if !framework.ProviderIs("local") {
			framework.Failf(fmt.Sprintf("The test requires two Ready nodes on %v, but found just one.", framework.TestContext.Provider))
		}
		framework.Logf("Only one ready node is detected. The test has limited scope in such setting. " +
			"Rerun it with at least two nodes to get complete coverage.")
	}

	// Create service A and B in namespaces A and B respectively.  The services are used
	// for pod discovery by the net monitor containers.
	serviceA := createNetworkPolicyService(f, nsA, serviceAName)
	serviceB := createNetworkPolicyService(f, nsB, serviceBName)

	// Clean up services
	defer func() {
		By("Cleaning up the service A")
		if err = f.Client.Services(nsA.Name).Delete(serviceA.Name); err != nil {
			framework.Failf("unable to delete svc %v: %v", serviceA.Name, err)
		}
	}()
	defer func() {
		By("Cleaning up the service B")
		if err = f.Client.Services(nsB.Name).Delete(serviceB.Name); err != nil {
			framework.Failf("unable to delete svc %v: %v", serviceB.Name, err)
		}
	}()

	By("Creating a webserver (pending) pod on each node")
	podA, podBs := launchNetMonitorPods(f, nsA, nsB, nodes)

	// Deferred clean up of the pods.
	defer func() {
		By("Cleaning up the webserver pods")
		if err = f.Client.Pods(nsA.Name).Delete(podA.Name, nil); err != nil {
			framework.Logf("Failed to delete pod %v: %v", podA.Name, err)
		}
		for _, podB := range podBs {
			if err = f.Client.Pods(nsB.Name).Delete(podB.Name, nil); err != nil {
				framework.Logf("Failed to delete pod %v: %v", podB.Name, err)
			}
		}
	}()

	// Wait for all pods to be running.
	By(fmt.Sprintf("Waiting for pod %v to be running", podA.Name))
	err = framework.WaitForPodRunningInNamespace(f.Client, podA)
	Expect(err).NotTo(HaveOccurred())
	for _, podB := range podBs {
		By(fmt.Sprintf("Waiting for pod %v to be running", podB.Name))
		err = framework.WaitForPodRunningInNamespace(f.Client, podB)
		Expect(err).NotTo(HaveOccurred())
	}

	// Open up TCP port used to access the network monitor HTTP API (8080).  Until we do this, all
	// pods in all namespaces are isolated and the test cannot query the netmonitor pods to pull
	// the connectivity information.
	By("Open up monitoring port so that test can pull connectivity data from services")
	addNetworkPolicyOpenPort(f, nsA, "query1", 8080, api.ProtocolTCP)
	addNetworkPolicyOpenPort(f, nsB, "query2", 8080, api.ProtocolTCP)

	// We expect full isolation between all pods in all namespaces.  Monitor, wait for a little and
	// re-check we are still isolated.
	By("Checking full isolation")
	expected := Summary{
		TCPNumOutboundConnected: 0,
		TCPNumInboundConnected:  0,
	}
	monitorConnectivity(f, nsA, serviceA.Name, expected)
	monitorConnectivity(f, nsB, serviceB.Name, expected)

	if nsA != nsB {
		// Add policy to one namespace to accept all traffic to the TCP port used for inter-pod pings (8081).
		// We should see mono-directional traffic.  We only do this if the two namespaces are different - since
		// we open up ports on a per-namespace basis.  If the namespaces are identical we'll end up straight
		// away with bi-directional traffic.
		By("Checking mono-directional TCP between namespaces with isolation enabled and policy applied to one namespace")
		addNetworkPolicyOpenPort(f, nsB, "tcp", 8081, api.ProtocolTCP)

		// We now expect ingress TCP to pods in service B to be allowed.
		expected = Summary{
			TCPNumOutboundConnected: len(nodes.Items),
			TCPNumInboundConnected:  0,
		}
		monitorConnectivity(f, nsA, serviceA.Name, expected)
		expected = Summary{
			TCPNumOutboundConnected: 0,
			TCPNumInboundConnected:  1,
		}
		monitorConnectivity(f, nsB, serviceB.Name, expected)
	}

	// Add policy to other namespace to accept all traffic to the TCP port used for inter-pod pings (8081).
	// We should see bi-directional traffic.
	By("Checking bi-direction TCP between namespaces with isolation enabled and policy applied to both namespaces")
	addNetworkPolicyOpenPort(f, nsA, "tcp", 8081, api.ProtocolTCP)

	// We now expect ingress TCP to pods in both services A and B to be allowed.
	expected = Summary{
		TCPNumOutboundConnected: len(nodes.Items),
		TCPNumInboundConnected:  len(nodes.Items),
	}
	monitorConnectivity(f, nsA, serviceA.Name, expected)
	expected = Summary{
		TCPNumOutboundConnected: 1,
		TCPNumInboundConnected:  1,
	}
	monitorConnectivity(f, nsB, serviceB.Name, expected)
}

// Create a service which exposes TCP ports:
// -  8080: HTTP monitoring API to query inter-pod connectivity
// -  8081: inter-pod connectivity pings
func createNetworkPolicyService(f *framework.Framework, namespace *api.Namespace, name string) *api.Service {
	By(fmt.Sprintf("Creating a service named %v in namespace %v", name, namespace.Name))
	svc, err := f.Client.Services(namespace.Name).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Protocol:   "TCP",
				Port:       8080,
				TargetPort: intstr.FromInt(8080),
				Name:       "net-monitor-query",
			}, {
				Protocol:   "TCP",
				Port:       8081,
				TargetPort: intstr.FromInt(8081),
				Name:       "net-monitor-tcp-ping",
			}},
			Selector: map[string]string{
				"name": name,
			},
		},
	})
	if err != nil {
		framework.Failf("unable to create test service named [%v] %v", svc.Name, err)
	}
	return svc
}

// Launch the required set of network monitor pods for the test.  This creates a single pod
// in namespaceA/serviceA which peers with a pod on each node in namespaceB/serviceB.
func launchNetMonitorPods(f *framework.Framework, namespaceA *api.Namespace, namespaceB *api.Namespace, nodes *api.NodeList) (*api.Pod, []*api.Pod) {
	podBs := []*api.Pod{}

	// Create the A pod on the first node.  It will find all of the B
	// pods (one for each node).
	podA := createNetMonitorPod(f, namespaceA, namespaceB, serviceAName, serviceBName, &nodes.Items[0])

	// Now create the B pods, one on each node - each should just search
	// for the single A pod peer.
	for _, node := range nodes.Items {
		pod := createNetMonitorPod(f, namespaceB, namespaceA, serviceBName, serviceAName, &node)
		podBs = append(podBs, pod)
	}

	return podA, podBs
}

// Create a network monitor pod which peers with other network monitor pods.
func createNetMonitorPod(f *framework.Framework,
	namespace *api.Namespace, peerNamespace *api.Namespace,
	serviceName string, peerServiceName string,
	node *api.Node) *api.Pod {
	pod, err := f.Client.Pods(namespace.Name).Create(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: serviceName + "-",
			Labels: map[string]string{
				"name": serviceName,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "webserver",
					Image: netMonitorContainer,
					Args: []string{
						"--namespace=" + peerNamespace.Name,
						"--service=" + peerServiceName},
					Ports:           []api.ContainerPort{{ContainerPort: 8080}, {ContainerPort: 8081}},
					ImagePullPolicy: api.PullAlways,
				},
			},
			NodeName:      node.Name,
			RestartPolicy: api.RestartPolicyNever,
		},
	})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created pod %v on node %v", pod.ObjectMeta.Name, node.Name)

	return pod
}

// Monitor the connectivity matrix from the network monitor pods, until the returned
// connectivity summary matches the supplied summary.
//
// If convergence does not happen within the required time limit, the test fails.
func monitorConnectivity(f *framework.Framework, namespace *api.Namespace, serviceName string, expected Summary) {
	By(fmt.Sprintf("Verifying expected connectivity on service %v", serviceName))
	passed := false

	// Once response OK, evaluate response body for pass/fail.
	var err error
	var body []byte
	getDetails := func() ([]byte, error) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
		if errProxy != nil {
			return nil, errProxy
		}
		return proxyRequest.Namespace(namespace.Name).
			Name(serviceName + ":net-monitor-query").
			Suffix("details").
			DoRaw()
	}

	getSummary := func() ([]byte, error) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(f.Client, f.Client.Get())
		if errProxy != nil {
			return nil, errProxy
		}
		return proxyRequest.Namespace(namespace.Name).
			Name(serviceName + ":net-monitor-query").
			Suffix("summary").
			DoRaw()
	}

	// The netmonitor container will continuously poll for named service endpoints and then
	// check connectivity between the two specified services.
	timeout := time.Now().Add(convergenceTimeout * time.Minute)
	for i := 0; !passed && timeout.After(time.Now()); i++ {
		time.Sleep(2 * time.Second)
		framework.Logf("About to make a proxy summary call")
		start := time.Now()
		body, err = getSummary()
		framework.Logf("Proxy summary call returned in %v", time.Since(start))
		if err != nil {
			framework.Logf("Attempt %v: service/pod still starting. (error: '%v')", i, err)
			continue
		}

		var summary Summary
		err = json.Unmarshal(body, &summary)
		if err != nil {
			framework.Logf("Warning: unable to unmarshal response (%v): '%v'", string(body), err)
			continue
		}

		framework.Logf("Summary: %v", string(body))
		passed = summary == expected
		if passed {
			break
		}
	}

	if !passed {
		if body, err = getDetails(); err != nil {
			framework.Failf("Timed out. Cleaning up. Error reading details: %v", err)
		} else {
			framework.Failf("Timed out. Cleaning up. Details:\n%s", string(body))
		}
	}
	Expect(passed).To(Equal(true))
}

// Configure namespace network isolation by setting the network-policy annotation
// on the namespace.
func setNamespaceIsolation(f *framework.Framework, namespace *api.Namespace, ingressIsolation string) {
	var annotations = map[string]string{}
	if ingressIsolation != "" {
		By(fmt.Sprintf("Enabling isolation through namespace annotations on namespace %v", namespace.Name))
		policy := fmt.Sprintf(`{"ingress":{"isolation":"%s"}}`, ingressIsolation)
		annotations["net.beta.kubernetes.io/network-policy"] = policy
	} else {
		By(fmt.Sprintf("Disabling isolation through namespace annotations on namespace %v", namespace.Name))
		delete(annotations, "net.beta.kubernetes.io/network-policy")
	}

	// Update the namespace.  We set the resource version to be an empty
	// string, this forces the update.  If we weren't to do this, we would
	// either need to re-query the namespace, or update the namespace
	// references with the one returned by the update.  This approach
	// requires less plumbing.
	namespace.ObjectMeta.Annotations = annotations
	namespace.ObjectMeta.ResourceVersion = ""
	_, err := f.Client.Namespaces().Update(namespace)
	Expect(err).NotTo(HaveOccurred())
}

// Add a network policy object to open up ingress traffic to a specific port on a namespace.
func addNetworkPolicyOpenPort(f *framework.Framework, namespace *api.Namespace, name string, port int32, protocol api.Protocol) {
	By(fmt.Sprintf("Setting network policy to allow proxy traffic for namespace %v", namespace.Name))

	lport := intstr.IntOrString{IntVal: port}

	framework.Logf("Creating policy %v", name)
	_, err := f.Client.NetworkPolicies(namespace.Name).Create(&extensions.NetworkPolicy{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: extensions.NetworkPolicySpec{
			Ingress: []extensions.NetworkPolicyIngressRule{
				{
					Ports: []extensions.NetworkPolicyPort{
						{
							Protocol: &protocol,
							Port:     &lport,
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
}
