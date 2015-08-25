/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"time"
)

const (
	k8bpsContainerVersion = "r.2.8.19"       // Container version, see the examples/k8petstore dockerfiles for details.
	k8bpsThroughputDummy  = "0"              // Polling time = 0, since we poll in ginkgo rather than using the shell script tests.
	k8bpsRedisSlaves      = "1"              // Number of redis slaves.
	k8bpsDontRunTest      = "0"              // Don't bother embedded test.
	k8bpsStartupTimeout   = 30 * time.Second // Amount of elapsed time before petstore transactions are being stored.

	// Constants for the first test. We can make this a hashmap once we add scale tests to it.
	k8bpsSmokeTestTransactions = 50
	k8bpsSmokeTestTimeout      = 60 * time.Second
)

// readTransactions reads # of transactions from the k8petstore web server endpoint.
// for more details see the source of the k8petstore web server.
func readTransactions(c *unversioned.Client, ns string) (error, int) {
	body, err := c.Get().
		Namespace(ns).
		Prefix("proxy").
		Resource("services").
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
// polls until minExpected transactions are acquired, in a maximum of maxSeconds.
func runK8petstore(restServers int, loadGenerators int, c *unversioned.Client, ns string, minExpected int, maxTime time.Duration) {

	var err error = nil
	k8bpsScriptLocation := filepath.Join(testContext.RepoRoot, "examples/k8petstore/k8petstore-nodeport.sh")

	cmd := exec.Command(
		k8bpsScriptLocation,
		testContext.KubectlPath,
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

	Logf("Starting k8petstore application....")
	// Run the k8petstore app, and log / fail if it returns any errors.
	// This should return quickly, assuming containers are downloaded.
	if err = cmd.Start(); err != nil {
		log.Fatal(err)
	}
	// Make sure there are no command errors.
	if err = cmd.Wait(); err != nil {
		if exiterr, ok := err.(*exec.ExitError); ok {
			if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
				log.Printf("Exit Status: %d", status.ExitStatus())
			}
		}
	}
	Expect(err).NotTo(HaveOccurred())
	Logf("... Done starting k8petstore ")

	totalTransactions := 0
	Logf("Start polling, timeout is %v seconds", maxTime)

	// How long until the FIRST transactions are created.
	startupTimeout := time.After(time.Duration(k8bpsStartupTimeout))

	// Maximum time to wait until we reach the nth transaction.
	transactionsCompleteTimeout := time.After(time.Duration(maxTime))
	tick := time.Tick(2 * time.Second)
	var ready = false

	Logf("Now waiting %v seconds to see progress (transactions  > 3)", k8bpsStartupTimeout)
T:
	for {
		select {
		case <-transactionsCompleteTimeout:
			Logf("Timeout %v reached, transactions not complete.  Breaking!", tick)
			break T
		case <-startupTimeout:
			err, totalTransactions = readTransactions(c, ns)
			Logf("Timeout %v reached.  Checking if transactions have occured.", startupTimeout)
			// If we don't have 3 transactions, fail the test.
			if err != nil {
				Logf("Failed : Error %v", err)
				break T
			}
			if totalTransactions < 3 {
				break T
			}
			ready = true
		case <-tick:
			// Pass if we've collected enough transactions
			err, totalTransactions = readTransactions(c, ns)
			if ready {
				Expect(err).NotTo(HaveOccurred())
			}
			if totalTransactions > minExpected {
				break T
			}
			Logf("Time: %v, %v = total petstore transactions stored into redis.", time.Now(), totalTransactions)
		}
	}

	// We should have exceeded the minExpected num of transactions.
	// If this fails, but there are transactions being created, we may need to recalibrate
	// the minExpected value - or else - your cluster is broken/slow !
	Ω(totalTransactions).Should(BeNumerically(">", minExpected))
}

var _ = Describe("[Skipped][Example] Pet Store", func() {

	// The number of minions dictates total number of generators/transaction expectations.
	var minionCount int
	f := NewFramework("petstore")

	It(fmt.Sprintf("should scale to persist a nominal number ( %v ) of transactions in %v seconds", k8bpsSmokeTestTransactions, k8bpsSmokeTestTimeout), func() {
		minions, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		minionCount = len(minions.Items)

		loadGenerators := minionCount
		restServers := minionCount
		fmt.Printf("load generators / rest servers [ %v  /  %v ] ", loadGenerators, restServers)
		runK8petstore(restServers, loadGenerators, f.Client, f.Namespace.Name, k8bpsSmokeTestTransactions, k8bpsSmokeTestTimeout)
	})

})
