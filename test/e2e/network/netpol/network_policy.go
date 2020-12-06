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

package netpol

import (
	"context"
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/util/wait"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/kubernetes/test/e2e/network"

	"github.com/onsi/ginkgo"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	addSCTPContainers = false
	isVerbose         = true

	// useFixedNamespaces is useful when working on these tests: instead of creating new pods and
	//   new namespaces for each test run, it creates a fixed set of namespaces and pods, and then
	//   reuses them for each test case.
	// The result: tests run much faster.  However, this should only be used as a convenience for
	//   working on the tests during development.  It should not be enabled in production.
	useFixedNamespaces = false

	// See https://github.com/kubernetes/kubernetes/issues/95879
	// The semantics of the effect of network policies on loopback calls may be undefined: should
	//   they always be ALLOWED; how do Services affect this?
	//   Calico, Cillium, Antrea seem to do different things.
	// Since different CNIs have different results, that causes tests including loopback to fail
	//   on some CNIs.  So let's just ignore loopback calls for the purposes of deciding test pass/fail.
	ignoreLoopback = true
)

/*
You might be wondering, why are there multiple namespaces used for each test case?

These tests are based on "truth tables" that compare the expected and actual connectivity of each pair of pods.
Since network policies live in namespaces, and peers can be selected by namespace,
howing the connectivity of pods in other namespaces is key information to show whether a network policy is working as intended or not.

We use 3 namespaces each with 3 pods, and probe all combinations ( 9 pods x 9 pods = 81 data points ) -- including cross-namespace calls.

Here's an example of a test run, showing the expected and actual connectivity, along with the differences.  Note how the
visual representation as a truth table greatly aids in understanding what a network policy is intended to do in theory
and what is happening in practice:

		Oct 19 10:34:16.907: INFO: expected:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	X	.	.	.	.	.	.	.	.
		x/b	X	.	.	.	.	.	.	.	.
		x/c	X	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	X	.	.	.	.	.	.	.	.
		z/b	X	.	.	.	.	.	.	.	.
		z/c	X	.	.	.	.	.	.	.	.

		Oct 19 10:34:16.907: INFO: observed:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	X	.	.	.	.	.	.	.	.
		x/b	X	.	.	.	.	.	.	.	.
		x/c	X	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	X	.	.	.	.	.	.	.	.
		z/b	X	.	.	.	.	.	.	.	.
		z/c	X	.	.	.	.	.	.	.	.

		Oct 19 10:34:16.907: INFO: comparison:

		-	x/a	x/b	x/c	y/a	y/b	y/c	z/a	z/b	z/c
		x/a	.	.	.	.	.	.	.	.	.
		x/b	.	.	.	.	.	.	.	.	.
		x/c	.	.	.	.	.	.	.	.	.
		y/a	.	.	.	.	.	.	.	.	.
		y/b	.	.	.	.	.	.	.	.	.
		y/c	.	.	.	.	.	.	.	.	.
		z/a	.	.	.	.	.	.	.	.	.
		z/b	.	.	.	.	.	.	.	.	.
		z/c	.	.	.	.	.	.	.	.	.
*/
var _ = network.SIGDescribe("Netpol [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("netpol")

	ginkgo.Context("NetworkPolicy between server and client", func() {
		ginkgo.BeforeEach(func() {
			if useFixedNamespaces {
				_ = initializeResources(f)

				_, _, _, model, k8s := getK8SModel(f)
				framework.ExpectNoError(k8s.CleanNetworkPolicies(model.NamespaceNames), "unable to clean network policies")
				err := wait.Poll(waitInterval, waitTimeout, func() (done bool, err error) {
					for _, ns := range model.NamespaceNames {
						netpols, err := k8s.ClientSet.NetworkingV1().NetworkPolicies(ns).List(context.TODO(), metav1.ListOptions{})
						framework.ExpectNoError(err, "get network policies from ns %s", ns)
						if len(netpols.Items) > 0 {
							return false, nil
						}
					}
					return true, nil
				})
				framework.ExpectNoError(err, "unable to wait for network policy deletion")
			} else {
				framework.Logf("Using %v as the default dns domain for this cluster... ", framework.TestContext.ClusterDNSDomain)
				framework.ExpectNoError(initializeResources(f), "unable to initialize resources")
			}
		})

		ginkgo.AfterEach(func() {
			if !useFixedNamespaces {
				_, _, _, model, k8s := getK8SModel(f)
				framework.ExpectNoError(k8s.deleteNamespaces(model.NamespaceNames), "unable to clean up netpol namespaces")
			}
		})

		ginkgo.It("should support a 'default-deny-ingress' policy [Feature:NetworkPolicy]", func() {
			nsX, _, _, model, k8s := getK8SModel(f)
			policy := GetDenyIngress("deny-ingress")
			CreatePolicy(k8s, policy, nsX)

			reachability := NewReachability(model.AllPods(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)

			ValidateOrFail(k8s, model, &TestCase{FromPort: 81, ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})

		ginkgo.It("should support a 'default-deny-all' policy [Feature:NetworkPolicy]", func() {
			np := &networkingv1.NetworkPolicy{}
			policy := `
			{
				"kind": "NetworkPolicy",
				"apiVersion": "networking.k8s.io/v1",
				"metadata": {
				   "name": "deny-all-tcp-allow-dns"
				},
				"spec": {
				   "podSelector": {
					  "matchLabels": {}
				   },
				   "ingress": [],
				   "egress": [{
						"ports": [
							{
								"protocol": "UDP",
								"port": 53
							}
						]
					}],
				   "policyTypes": [
					"Ingress",
					"Egress"
				   ]
				}
			 }
			 `
			err := json.Unmarshal([]byte(policy), np)
			framework.ExpectNoError(err, "unmarshal network policy")

			nsX, _, _, model, k8s := getK8SModel(f)
			CreatePolicy(k8s, np, nsX)

			reachability := NewReachability(model.AllPods(), true)
			reachability.ExpectPeer(&Peer{}, &Peer{Namespace: nsX}, false)
			reachability.ExpectPeer(&Peer{Namespace: nsX}, &Peer{}, false)

			ValidateOrFail(k8s, model, &TestCase{FromPort: 81, ToPort: 80, Protocol: v1.ProtocolTCP, Reachability: reachability})
		})
	})
})

// getNamespaces returns the canonical set of namespaces used by this test, taking a root ns as input.  This allows this test to run in parallel.
func getNamespaces(rootNs string) (string, string, string, []string) {
	if useFixedNamespaces {
		rootNs = ""
	} else {
		rootNs = rootNs + "-"
	}
	nsX := fmt.Sprintf("%sx", rootNs)
	nsY := fmt.Sprintf("%sy", rootNs)
	nsZ := fmt.Sprintf("%sz", rootNs)
	return nsX, nsY, nsZ, []string{nsX, nsY, nsZ}
}

// defaultModel creates a new "model" pod system under namespaces (x,y,z) which has pods a, b, and c.  Thus resulting in the
// truth table matrix that is identical for all tests, comprising 81 total connections between 9 pods (x/a, x/b, x/c, ..., z/c).
func defaultModel(namespaces []string, dnsDomain string) *Model {
	protocols := []v1.Protocol{v1.ProtocolTCP, v1.ProtocolUDP}
	if addSCTPContainers {
		protocols = append(protocols, v1.ProtocolSCTP)
	}
	return NewModel(namespaces, []string{"a", "b", "c"}, []int32{80, 81}, protocols, dnsDomain)
}

// getK8sModel uses the e2e framework to create all necessary namespace resources, and returns the default probing model used
// in the scaffold of this test.
func getK8SModel(f *framework.Framework) (string, string, string, *Model, *Scenario) {
	k8s := NewScenario(f)
	rootNs := f.Namespace.GetName()
	nsX, nsY, nsZ, namespaces := getNamespaces(rootNs)

	model := defaultModel(namespaces, framework.TestContext.ClusterDNSDomain)

	return nsX, nsY, nsZ, model, k8s
}

// initializeResources generates a model and then waits for the model to be up-and-running (i.e. all pods are ready and running in their namespaces).
func initializeResources(f *framework.Framework) error {
	_, _, _, model, k8s := getK8SModel(f)

	framework.Logf("initializing cluster: ensuring namespaces, deployments, and pods exist and are ready")

	err := k8s.InitializeCluster(model)
	if err != nil {
		return err
	}

	framework.Logf("finished initializing cluster state")

	return k8s.waitForHTTPServers(model)
}
