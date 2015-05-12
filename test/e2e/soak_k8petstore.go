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
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	root0                    = absOrDie(filepath.Clean(filepath.Join(path.Base(os.Args[0]), "..")))
	err       error          = nil
	namespace *api.Namespace = nil
	ns        string         = ""
)

//This must run on a machine which is in the kube-proxy ring.  Otherwise, the publicIP binding will fail.
//The IP Below ~ letter->number cipher for k8petstore (most likely it won't be bound by any other process)
var ip = "165.201.92.15"

// i.e. after 50 trials, expect 3000 transactions... minimum we settle for is 500.
var minionCount int

// readTransactions reads # of transactions from the k8petstore web server endpoint.
// for more details see the source of the k8petstore web server.
func readTransactions(c *client.Client) int {
	resp, err := http.Get("http://165.201.92.15:3000/llen")
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	totalTrans, err := strconv.Atoi(string(body))
	Expect(err).NotTo(HaveOccurred())
	return totalTrans
}

// runK8petstore runs the k8petstore application, bound to external ip "ip", and polls it to assert that "min_expected"
// transactions are acquired in a maximum of "max_seconds".
func runK8petstore(ip string, restServers int, loadGenerators int, c *client.Client, minExpected int, maxSeconds int64) {
	k8bps := filepath.Join(root0, "examples/k8petstore/k8petstore.sh")

	//Get the count of minions.  We'll use this to decide expected throughput and loadgen/REST server count.
	cmd := exec.Command(
		k8bps,
		"kubectl", //for dev, replace w/ "cluster/kubectl.sh"
		"r.2.8.19",
		ip,
		strconv.Itoa(restServers),
		strconv.Itoa(loadGenerators),
		"1", "0", ns) // 1= # slave, 0 = don't bother running the embedded test.

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	Logf("Starting k8petstore application....")
	//Run the k8petstore app, and log / fail if it returns any errors.
	//This should return quickly, assuming containers are downloaded.
	if err = cmd.Start(); err != nil {
		log.Fatal(err)
	}
	//Make sure exit code != 0
	if err = cmd.Wait(); err != nil {
		if exiterr, ok := err.(*exec.ExitError); ok {
			if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
				log.Printf("Exit Status: %d", status.ExitStatus())
			}
		}
	}
	Expect(err).NotTo(HaveOccurred())
	Logf("... Done starting k8petstore successfully.... Now we will poll it...")

	//By now, the original k8petstore app has run and spun up a k8petstore w/ load generators.
	//Lets poll until we reach min_expected transactions
	totalTransactions := 0
	Logf("Start polling, timeout is %v seconds", maxSeconds)

	timeout := time.After(time.Duration(maxSeconds) * time.Second)
	tick := time.Tick(2 * time.Second)

T:
	for {
		select {
		case <-timeout:
			Logf("Timeout %v reached. Breaking!", tick)
			break T
		case <-tick:
			totalTransactions = readTransactions(c)
			Expect(err).NotTo(HaveOccurred())
			if totalTransactions > minExpected {
				break T
			}
			Logf("%v == %v total petstore transactions stored into redis. ==", time.Now(), totalTransactions)
		}
	}

	//Finally! We should have exceeded the min_expected num of transactions.
	//If this fails, but there are transactions being created, we may need to recalibrate
	//the min_expected value - or else - your cluster is broken/slow !
	Ω(totalTransactions).Should(BeNumerically(">", minExpected))
}

var _ = Describe("k8bps", func() {
	BeforeEach(func() {
		By("Creating a kubernetes client")
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		By("Building a namespace api object")
		namespace, err = createTestingNS("k8petstore-soak", c)
		ns = namespace.Name
		Expect(err).NotTo(HaveOccurred())

		//Now get the # of minions and calibrate the test params to them...
		minions, err := c.Nodes().List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		minionCount = len(minions.Items)

		//simple calibration.  TODO, make more ambitious (i.e. 10x density style load generators)...
		//load_generators = minionCount
		//rest_servers = minionCount

	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this test %v", namespace.Name))
		if err := c.Namespaces().Delete(namespace.Name); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	//On a single node cluster, we expect about 10 transactions every second on average.
	//admittedly, this is a very rough estimate given that the ETL is done in batches...
	It(fmt.Sprintf("k8petstore-FUNCTIONAL : Should quickly acquire 500 petstore transactions."), func() {

		//max number of trial before k8petstore.sh dies.
		var loadGenerators int = minionCount
		var restServers int = minionCount

		fmt.Printf("load generators / rest generators [ %v  /  %v ] ", loadGenerators, restServers)

		//At least 500 transactions should be acquired within 30 seconds after startup.
		runK8petstore(ip, restServers, loadGenerators, c, 500, 30)

	})

	//This test takes a few minutes... for simple CI setups testing pure functionality, filter it out.
	It(fmt.Sprintf("k8petstore-SCALE : Should support acquiring up to 5000 petstore transactions per minion"), func() {

		//We double the load generators, to increase the load.  Maybe parameterize this later,
		//the more generators -> the more transactions and the more bursty CPU used per minion.
		var loadGenerators int = minionCount * 2
		var restServers int = minionCount
		//var min_expected int = trials * 10 * minionCount // more minions --> higher expected # of transactions.

		fmt.Printf("load generators / rest generators [ %v  /  %v ] ", loadGenerators, restServers)

		//5000*M transactions in 6 minutes.  That gives 6 minutes of cold-start time.
		runK8petstore(ip, restServers, loadGenerators, c, 5000*minionCount, 60*6)
	})
})
