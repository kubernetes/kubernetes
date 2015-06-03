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
	"math/rand"
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	image             = "gcr.io/google_containers/serve_hostname:1.1"
	simulationTime    = 10 * time.Minute
	smallRCSize       = 5
	mediumRCSize      = 30
	bigRCSize         = 250
	smallRCGroupName  = "load-test-small-rc"
	mediumRCGroupName = "load-test-medium-rc"
	bigRCGroupName    = "load-test-big-rc"
	smallRCBatchSize  = 20
	mediumRCBatchSize = 5
	bigRCBatchSize    = 1
)

// This test suite can take a long time to run, so by default it is added to
// the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
var _ = Describe("Load capacity", func() {
	var c *client.Client
	var nodeCount int
	var ns string
	var smallRCCount, mediumRCCount, bigRCCount int

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())
		nsForTesting, err := createTestingNS("load", c)
		ns = nsForTesting.Name
		expectNoError(err)
	})

	// TODO add flag that allows to skip cleanup on failure
	AfterEach(func() {
		cleanRCGroup(c, ns, smallRCGroupName, smallRCSize, smallRCCount)
		cleanRCGroup(c, ns, mediumRCGroupName, mediumRCSize, mediumRCCount)
		cleanRCGroup(c, ns, bigRCGroupName, bigRCSize, bigRCCount)

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}

		// Verify latency metrics
		// TODO: Update threshold to 1s once we reach this goal
		// TODO: We should reset metrics before the test. Currently previous tests influence latency metrics.
		highLatencyRequests, err := HighLatencyRequests(c, 3*time.Second, util.NewStringSet("events"))
		expectNoError(err, "Too many instances metrics above the threshold")
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0))
	})

	type Load struct {
		podsPerNode int
	}

	loadTests := []Load{
		{podsPerNode: 30},
	}

	for _, testArg := range loadTests {
		name := fmt.Sprintf("[Skipped] should be able to handle %v pods per node", testArg.podsPerNode)

		It(name, func() {
			totalPods := testArg.podsPerNode * nodeCount
			smallRCCount, mediumRCCount, bigRCCount = computeRCCounts(totalPods)
			threads := smallRCCount + mediumRCCount + bigRCCount

			// TODO refactor this code to iterate over slice of RC group description.
			createRCGroup(c, ns, smallRCGroupName, smallRCSize, smallRCCount, smallRCBatchSize)
			createRCGroup(c, ns, mediumRCGroupName, mediumRCSize, mediumRCCount, mediumRCBatchSize)
			createRCGroup(c, ns, bigRCGroupName, bigRCSize, bigRCCount, bigRCBatchSize)

			// TODO add reseting latency metrics here, once it would be supported.

			var wg sync.WaitGroup
			wg.Add(threads)

			// Run RC load for all kinds of RC.
			runRCLoad(c, &wg, ns, smallRCGroupName, smallRCSize, smallRCCount)
			runRCLoad(c, &wg, ns, mediumRCGroupName, mediumRCSize, mediumRCCount)
			runRCLoad(c, &wg, ns, bigRCGroupName, bigRCSize, bigRCCount)

			// Wait for all the pods from all the RC's to return.
			wg.Wait()
		})
	}
})

func computeRCCounts(total int) (int, int, int) {
	// Small RCs owns ~0.5 of total number of pods, medium and big RCs ~0.25 each.
	// For example for 3000 pods (100 nodes, 30 pods per node) there are:
	//  - 500 small RCs each 5 pods
	//  - 25 medium RCs each 30 pods
	//  - 3 big RCs each 250 pods
	bigRCCount := total / 4 / bigRCSize
	mediumRCCount := (total - bigRCCount*bigRCSize) / 3 / mediumRCSize
	smallRCCount := (total - bigRCCount*bigRCSize - mediumRCCount*mediumRCSize) / smallRCSize
	return smallRCCount, mediumRCCount, bigRCCount
}

// The function every few second scales RC to a random size and with 0.1 probability deletes it.
// Assumes that given RC exists.
func playWithRC(c *client.Client, wg *sync.WaitGroup, ns, name string, size int) {
	By(fmt.Sprintf("Playing with Replication Controller %v", name))
	defer GinkgoRecover()
	defer wg.Done()
	// Wait some time to prevent from performing all operations at the same time.
	time.Sleep(time.Duration(rand.Intn(60)) * time.Second)
	rcExist := true
	// Once every 1-2 minutes perform scale of RC.
	for start := time.Now(); time.Since(start) < simulationTime; time.Sleep(time.Duration(60+rand.Intn(60)) * time.Second) {
		if !rcExist {
			config := RCConfig{Client: c,
				Name:      name,
				Namespace: ns,
				Image:     image,
				Replicas:  size,
			}
			expectNoError(RunRC(config), fmt.Sprintf("creating rc %s in namespace %s", name, ns))
			rcExist = true
		}
		// Scale RC to a random size between 0.5x and 1.5x of the original size.
		newSize := uint(rand.Intn(size+1) + size/2)
		expectNoError(ScaleRC(c, ns, name, newSize), fmt.Sprintf("scaling rc %s in namespace %s", name, ns))
		// List all pods within this RC.
		_, err := c.Pods(ns).List(labels.SelectorFromSet(labels.Set(map[string]string{"name": name})), fields.Everything())
		expectNoError(err, fmt.Sprintf("listing pods from rc %v in namespace %v", name, ns))
		// With probability 0.1 remove this RC.
		if rand.Intn(10) == 0 {
			expectNoError(DeleteRC(c, ns, name), fmt.Sprintf("deleting rc %s in namespace %s", name, ns))
			rcExist = false
		}
	}
	if rcExist {
		expectNoError(DeleteRC(c, ns, name), fmt.Sprintf("deleting rc %s in namespace %s after test completion", name, ns))
	}
}

func runRCLoad(c *client.Client, wg *sync.WaitGroup, ns, groupName string, size, count int) {
	for i := 1; i <= count; i++ {
		go playWithRC(c, wg, ns, groupName+"-"+strconv.Itoa(i), size)
	}
}

// Creates <count> RCs with size <size> in namespace <ns>. The requests are sent in batches of size <batchSize>.
func createRCGroup(c *client.Client, ns, groupName string, size, count, batchSize int) {
	By(fmt.Sprintf("Creating %v Replication Controllers with size %v", count, size))
	for i := 1; i <= count; {
		// Create up to <batchSize> RCs in parallel.
		var wg sync.WaitGroup
		for j := 1; j <= batchSize && i <= count; i, j = i+1, j+1 {
			wg.Add(1)
			go func(i int) {
				defer GinkgoRecover()
				defer wg.Done()
				name := groupName + "-" + strconv.Itoa(i)
				config := RCConfig{Client: c,
					Name:      name,
					Namespace: ns,
					Image:     image,
					Replicas:  size,
				}
				expectNoError(RunRC(config), fmt.Sprintf("creating rc %s in namespace %s for the first time", name, ns))
			}(i)
		}
		wg.Wait()
	}
}

// Removes group of RCs if not removed. This function is for cleanup purposes, so ignores errors.
func cleanRCGroup(c *client.Client, ns, groupName string, size, count int) {
	By(fmt.Sprintf("Removing %v Replication Controllers with size %v if not removed", count, size))
	var wg sync.WaitGroup
	wg.Add(count)
	for i := 1; i <= count; i++ {
		go func(i int) {
			defer GinkgoRecover()
			defer wg.Done()
			name := groupName + "-" + strconv.Itoa(i)
			// Since it is cleanup ignore any error.
			DeleteRC(c, name, ns)
		}(i)
	}
	wg.Wait()
}
