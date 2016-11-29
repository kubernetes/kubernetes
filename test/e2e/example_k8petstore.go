/*
Copyright 2015 The Kubernetes Authors.

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
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	k8bpsContainerVersion         = "r.2.8.19"       // Container version, see the examples/k8petstore dockerfiles for details.
	k8bpsThroughputDummy          = "0"              // Polling time = 0, since we framework.Poll in ginkgo rather than using the shell script tests.
	k8bpsRedisSlaves              = "1"              // Number of redis slaves.
	k8bpsDontRunTest              = "0"              // Don't bother embedded test.
	k8bpsStartupTimeout           = 30 * time.Second // Amount of elapsed time before petstore transactions are being stored.
	k8bpsMinTransactionsOnStartup = 3                // Amount of transactions we expect we should have before data generator starts.

	// Constants for the first test. We can make this a hashmap once we add scale tests to it.
	k8bpsSmokeTestFinalTransactions = 50
	k8bpsSmokeTestTimeout           = 60 * time.Second
)

// readTransactions reads # of transactions from the k8petstore web server endpoint.
// for more details see the source of the k8petstore web server.
func readTransactions(c clientset.Interface, ns string) (error, int) {
	proxyRequest, errProxy := framework.GetServicesProxyRequest(c, c.Core().RESTClient().Get())
	if errProxy != nil {
		return errProxy, -1
	}
	body, err := proxyRequest.Namespace(ns).
		Name("frontend").
		Suffix("llen").
		DoRaw()
	if err != nil {
		return err, -1
	} else {
		totalTrans, err := strconv.Atoi(string(body))
		return err, totalTrans
	}
}

// runK8petstore runs the k8petstore application, bound to external nodeport, and
// polls until finalTransactionsExpected transactions are acquired, in a maximum of maxSeconds.
func runK8petstore(restServers int, loadGenerators int, c clientset.Interface, ns string, finalTransactionsExpected int, maxTime time.Duration) {

	var err error = nil
	k8bpsScriptLocation := filepath.Join(framework.TestContext.RepoRoot, "examples/k8petstore/k8petstore-nodeport.sh")

	cmd := exec.Command(
		k8bpsScriptLocation,
		framework.TestContext.KubectlPath,
		k8bpsContainerVersion,
		k8bpsThroughputDummy,
		strconv.Itoa(restServers),
		strconv.Itoa(loadGenerators),
		k8bpsRedisSlaves,
		k8bpsDontRunTest, // Don't bother embedded test.
		ns,
	)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	framework.Logf("Starting k8petstore application....")
	// Run the k8petstore app, and log / fail if it returns any errors.
	// This should return quickly, assuming containers are downloaded.
	if err = cmd.Start(); err != nil {
		framework.Failf("%v", err)
	}
	// Make sure there are no command errors.
	if err = cmd.Wait(); err != nil {
		if exiterr, ok := err.(*exec.ExitError); ok {
			if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
				framework.Logf("Exit Status: %d", status.ExitStatus())
			}
		}
	}
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("... Done starting k8petstore ")

	totalTransactions := 0
	framework.Logf("Start polling, timeout is %v seconds", maxTime)

	// How long until the FIRST transactions are created.
	startupTimeout := time.After(time.Duration(k8bpsStartupTimeout))

	// Maximum time to wait until we reach the nth transaction.
	transactionsCompleteTimeout := time.After(time.Duration(maxTime))
	tick := time.Tick(2 * time.Second)
	var ready = false

	framework.Logf("Now waiting %v seconds to see progress (transactions  > 3)", k8bpsStartupTimeout)
T:
	for {
		select {
		case <-transactionsCompleteTimeout:
			framework.Logf("Completion timeout %v reached, %v transactions not complete.  Breaking!", time.Duration(maxTime), finalTransactionsExpected)
			break T
		case <-tick:
			// Don't fail if there's an error.  We expect a few failures might happen in the cloud.
			err, totalTransactions = readTransactions(c, ns)
			if err == nil {
				framework.Logf("PetStore : Time: %v, %v = total petstore transactions stored into redis.", time.Now(), totalTransactions)
				if totalTransactions >= k8bpsMinTransactionsOnStartup {
					ready = true
				}
				if totalTransactions >= finalTransactionsExpected {
					break T
				}
			} else {
				if ready {
					framework.Logf("Blip: during polling: %v", err)
				} else {
					framework.Logf("Not ready yet: %v", err)
				}
			}
		case <-startupTimeout:
			if !ready {
				framework.Logf("Startup Timeout %v reached: Its been too long and we still haven't started accumulating %v transactions!", startupTimeout, k8bpsMinTransactionsOnStartup)
				break T
			}
		}
	}

	// We should have exceeded the finalTransactionsExpected num of transactions.
	// If this fails, but there are transactions being created, we may need to recalibrate
	// the finalTransactionsExpected value - or else - your cluster is broken/slow !
	Ω(totalTransactions).Should(BeNumerically(">", finalTransactionsExpected))
}

var _ = framework.KubeDescribe("Pet Store [Feature:Example]", func() {

	BeforeEach(func() {
		// The shell scripts in k8petstore break on jenkins... Pure golang rewrite is in progress.
		framework.SkipUnlessProviderIs("local")
	})

	// The number of nodes dictates total number of generators/transaction expectations.
	var nodeCount int
	f := framework.NewDefaultFramework("petstore")

	It(fmt.Sprintf("should scale to persist a nominal number ( %v ) of transactions in %v seconds", k8bpsSmokeTestFinalTransactions, k8bpsSmokeTestTimeout), func() {
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		nodeCount = len(nodes.Items)

		loadGenerators := nodeCount
		restServers := nodeCount
		fmt.Printf("load generators / rest servers [ %v  /  %v ] ", loadGenerators, restServers)
		runK8petstore(restServers, loadGenerators, f.ClientSet, f.Namespace.Name, k8bpsSmokeTestFinalTransactions, k8bpsSmokeTestTimeout)
	})

})
