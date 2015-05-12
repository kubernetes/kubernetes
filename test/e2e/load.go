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
	image          = "gcr.io/google_containers/serve_hostname:1.1"
	simulationTime = 20 * time.Minute
	smallRCSize    = 5
	mediumRCSize   = 30
	bigRCSize      = 250
)

// This test suite can take a long time to run, so by default it is added to
// the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
var _ = Describe("Load", func() {
	var c *client.Client
	var nodeCount int
	var ns string

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
		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	type Load struct {
		podsPerNode int
	}

	loadTests := []Load{
		{podsPerNode: 30},
	}

	for _, testArg := range loadTests {
		name := fmt.Sprintf("[Performance suite] [Skipped] should be able to handle %v pods per node", testArg.podsPerNode)

		It(name, func() {
			totalPods := testArg.podsPerNode * nodeCount
			smallRCCount, mediumRCCount, bigRCCount := computeRCCounts(totalPods)
			threads := smallRCCount + mediumRCCount + bigRCCount

			var wg sync.WaitGroup
			wg.Add(threads)

			// Run RC load for all kinds of RC.
			runRCLoad(c, &wg, ns, smallRCSize, smallRCCount)
			runRCLoad(c, &wg, ns, mediumRCSize, mediumRCCount)
			runRCLoad(c, &wg, ns, bigRCSize, bigRCCount)

			// Wait for all the pods from all the RC's to return.
			wg.Wait()
			// TODO verify latency metrics
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

// The function creates a RC and then every few second resize it and with 0.1 probability deletes it.
func playWithRC(c *client.Client, wg *sync.WaitGroup, ns string, size int) {
	defer GinkgoRecover()
	defer wg.Done()
	rcExist := false
	var name string
	// Once every 1-2 minutes perform resize of RC.
	for start := time.Now(); time.Since(start) < simulationTime; time.Sleep(time.Duration(60+rand.Intn(60)) * time.Second) {
		if !rcExist {
			name = "load-test-" + string(util.NewUUID())
			expectNoError(RunRC(c, name, ns, image, size))
			rcExist = true
		}
		// Resize RC to a random size between 0.5x and 1.5x of the original size.
		newSize := uint(rand.Intn(size+1) + size/2)
		expectNoError(ResizeRC(c, ns, name, newSize))
		// With probability 0.1 remove this RC.
		if rand.Intn(10) == 0 {
			expectNoError(DeleteRC(c, ns, name))
			rcExist = false
		}
	}
	if rcExist {
		expectNoError(DeleteRC(c, ns, name))
	}
}

func runRCLoad(c *client.Client, wg *sync.WaitGroup, ns string, size, count int) {
	By(fmt.Sprintf("Running %v Replication Controllers with size %v and playing with them", count, size))
	for i := 0; i < count; i++ {
		go playWithRC(c, wg, ns, size)
	}
}
