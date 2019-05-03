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

package framework

import (
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	e2enet "k8s.io/kubernetes/test/e2e/framework/networking"
)

// CheckReachabilityFromPod checks reachability from the specified pod.
func CheckReachabilityFromPod(expectToBeReachable bool, timeout time.Duration, namespace, pod, target string) {
	cmd := fmt.Sprintf("wget -T 5 -qO- %q", target)
	err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		_, err := RunHostCmd(namespace, pod, cmd)
		if expectToBeReachable && err != nil {
			Logf("Expect target to be reachable. But got err: %v. Retry until timeout", err)
			return false, nil
		}

		if !expectToBeReachable && err == nil {
			Logf("Expect target NOT to be reachable. But it is reachable. Retry until timeout")
			return false, nil
		}
		return true, nil
	})
	ExpectNoError(err)
}

// TestHitNodesFromOutside checkes HTTP connectivity from outside.
func TestHitNodesFromOutside(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String) error {
	return TestHitNodesFromOutsideWithCount(externalIP, httpPort, timeout, expectedHosts, 1)
}

// TestHitNodesFromOutsideWithCount checkes HTTP connectivity from outside with count.
func TestHitNodesFromOutsideWithCount(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String,
	countToSucceed int) error {
	Logf("Waiting up to %v for satisfying expectedHosts for %v times", timeout, countToSucceed)
	hittedHosts := sets.NewString()
	count := 0
	condition := func() (bool, error) {
		result := e2enet.PokeHTTP(externalIP, int(httpPort), "/hostname", &HTTPPokeParams{Timeout: 1 * time.Second})
		if result.Status != HTTPSuccess {
			return false, nil
		}

		hittedHost := strings.TrimSpace(string(result.Body))
		if !expectedHosts.Has(hittedHost) {
			Logf("Error hitting unexpected host: %v, reset counter: %v", hittedHost, count)
			count = 0
			return false, nil
		}
		if !hittedHosts.Has(hittedHost) {
			hittedHosts.Insert(hittedHost)
			Logf("Missing %+v, got %+v", expectedHosts.Difference(hittedHosts), hittedHosts)
		}
		if hittedHosts.Equal(expectedHosts) {
			count++
			if count >= countToSucceed {
				return true, nil
			}
		}
		return false, nil
	}

	if err := wait.Poll(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error waiting for expectedHosts: %v, hittedHosts: %v, count: %v, expected count: %v",
			expectedHosts, hittedHosts, count, countToSucceed)
	}
	return nil
}

// TestUnderTemporaryNetworkFailure blocks outgoing network traffic on 'node'. Then runs testFunc and returns its status.
// At the end (even in case of errors), the network traffic is brought back to normal.
// This function executes commands on a node so it will work only for some
// environments.
func TestUnderTemporaryNetworkFailure(c clientset.Interface, ns string, node *v1.Node, testFunc func()) {
	host, err := GetNodeExternalIP(node)
	if err != nil {
		Failf("Error getting node external ip : %v", err)
	}
	masterAddresses := GetAllMasterAddresses(c)
	ginkgo.By(fmt.Sprintf("block network traffic from node %s to the master", node.Name))
	defer func() {
		// This code will execute even if setting the iptables rule failed.
		// It is on purpose because we may have an error even if the new rule
		// had been inserted. (yes, we could look at the error code and ssh error
		// separately, but I prefer to stay on the safe side).
		ginkgo.By(fmt.Sprintf("Unblock network traffic from node %s to the master", node.Name))
		for _, masterAddress := range masterAddresses {
			UnblockNetwork(host, masterAddress)
		}
	}()

	Logf("Waiting %v to ensure node %s is ready before beginning test...", resizeNodeReadyTimeout, node.Name)
	if !WaitForNodeToBe(c, node.Name, v1.NodeReady, true, resizeNodeReadyTimeout) {
		Failf("Node %s did not become ready within %v", node.Name, resizeNodeReadyTimeout)
	}
	for _, masterAddress := range masterAddresses {
		BlockNetwork(host, masterAddress)
	}

	Logf("Waiting %v for node %s to be not ready after simulated network failure", resizeNodeNotReadyTimeout, node.Name)
	if !WaitForNodeToBe(c, node.Name, v1.NodeReady, false, resizeNodeNotReadyTimeout) {
		Failf("Node %s did not become not-ready within %v", node.Name, resizeNodeNotReadyTimeout)
	}

	testFunc()
	// network traffic is unblocked in a deferred function
}
