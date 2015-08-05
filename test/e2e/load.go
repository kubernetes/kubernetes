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

	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	smallRCSize       = 5
	mediumRCSize      = 30
	bigRCSize         = 250
	smallRCGroupName  = "load-test-small-rc"
	mediumRCGroupName = "load-test-medium-rc"
	bigRCGroupName    = "load-test-big-rc"
	smallRCBatchSize  = 30
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
	var configs []*RCConfig

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		// Terminating a namespace (deleting the remaining objects from it - which
		// generally means events) can affect the current run. Thus we wait for all
		// terminating namespace to be finally deleted before starting this test.
		err = deleteTestingNS(c)
		expectNoError(err)

		nsForTesting, err := createTestingNS("load", c)
		ns = nsForTesting.Name
		expectNoError(err)

		expectNoError(resetMetrics(c))
	})

	// TODO add flag that allows to skip cleanup on failure
	AfterEach(func() {
		deleteAllRC(configs)

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}

		// Verify latency metrics
		highLatencyRequests, err := HighLatencyRequests(c, 3*time.Second, util.NewStringSet("events"))
		expectNoError(err, "Too many instances metrics above the threshold")
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0))
	})

	type Load struct {
		podsPerNode int
		image       string
		command     []string
	}

	loadTests := []Load{
		// The container will consume 1 cpu and 512mb of memory.
		{podsPerNode: 3, image: "jess/stress", command: []string{"stress", "-c", "1", "-m", "2"}},
		{podsPerNode: 30, image: "gcr.io/google_containers/serve_hostname:1.1"},
	}

	for _, testArg := range loadTests {
		name := fmt.Sprintf("[Skipped] [Performance suite] should be able to handle %v pods per node", testArg.podsPerNode)
		itArg := testArg

		It(name, func() {
			configs = generateRCConfigs(itArg.podsPerNode*nodeCount, itArg.image, itArg.command, c, ns)

			// Simulate lifetime of RC:
			//  * create with initial size
			//  * scale RC to a random size and list all pods
			//  * scale RC to a random size and list all pods
			//  * delete it
			//
			// This will generate ~5 creations/deletions per second assuming:
			//  - 300 small RCs each 5 pods
			//  - 25 medium RCs each 30 pods
			//  - 3 big RCs each 250 pods
			createAllRC(configs)
			// TODO add reseting latency metrics here, once it would be supported.
			By("============================================================================")
			scaleAllRC(configs)
			By("============================================================================")
			scaleAllRC(configs)
			By("============================================================================")
		})
	}
})

func computeRCCounts(total int) (int, int, int) {
	// Small RCs owns ~0.5 of total number of pods, medium and big RCs ~0.25 each.
	// For example for 3000 pods (100 nodes, 30 pods per node) there are:
	//  - 300 small RCs each 5 pods
	//  - 25 medium RCs each 30 pods
	//  - 3 big RCs each 250 pods
	bigRCCount := total / 4 / bigRCSize
	total -= bigRCCount * bigRCSize
	mediumRCCount := total / 3 / mediumRCSize
	total -= mediumRCCount * mediumRCSize
	smallRCCount := total / smallRCSize
	return smallRCCount, mediumRCCount, bigRCCount
}

func generateRCConfigs(totalPods int, image string, command []string, c *client.Client, ns string) []*RCConfig {
	configs := make([]*RCConfig, 0)

	smallRCCount, mediumRCCount, bigRCCount := computeRCCounts(totalPods)
	configs = append(configs, generateRCConfigsForGroup(c, ns, smallRCGroupName, smallRCSize, smallRCCount, image, command)...)
	configs = append(configs, generateRCConfigsForGroup(c, ns, mediumRCGroupName, mediumRCSize, mediumRCCount, image, command)...)
	configs = append(configs, generateRCConfigsForGroup(c, ns, bigRCGroupName, bigRCSize, bigRCCount, image, command)...)

	return configs
}

func generateRCConfigsForGroup(c *client.Client, ns, groupName string, size, count int, image string, command []string) []*RCConfig {
	configs := make([]*RCConfig, 0, count)
	for i := 1; i <= count; i++ {
		config := &RCConfig{
			Client:    c,
			Name:      groupName + "-" + strconv.Itoa(i),
			Namespace: ns,
			Timeout:   10 * time.Minute,
			Image:     image,
			Command:   command,
			Replicas:  size,
		}
		configs = append(configs, config)
	}
	return configs
}

func sleepUpTo(d time.Duration) {
	time.Sleep(time.Duration(rand.Int63n(d.Nanoseconds())))
}

func createAllRC(configs []*RCConfig) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go createRC(&wg, config)
	}
	wg.Wait()
}

func createRC(wg *sync.WaitGroup, config *RCConfig) {
	defer GinkgoRecover()
	defer wg.Done()
	creatingTime := 10 * time.Minute

	sleepUpTo(creatingTime)
	expectNoError(RunRC(*config), fmt.Sprintf("creating rc %s", config.Name))
}

func scaleAllRC(configs []*RCConfig) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go scaleRC(&wg, config)
	}
	wg.Wait()
}

// Scales RC to a random size within [0.5*size, 1.5*size] and lists all the pods afterwards.
// Scaling happens always based on original size, not the current size.
func scaleRC(wg *sync.WaitGroup, config *RCConfig) {
	defer GinkgoRecover()
	defer wg.Done()
	resizingTime := 3 * time.Minute

	sleepUpTo(resizingTime)
	newSize := uint(rand.Intn(config.Replicas) + config.Replicas/2)
	expectNoError(ScaleRC(config.Client, config.Namespace, config.Name, newSize),
		fmt.Sprintf("scaling rc %s for the first time", config.Name))
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": config.Name}))
	_, err := config.Client.Pods(config.Namespace).List(selector, fields.Everything())
	expectNoError(err, fmt.Sprintf("listing pods from rc %v", config.Name))
}

func deleteAllRC(configs []*RCConfig) {
	var wg sync.WaitGroup
	wg.Add(len(configs))
	for _, config := range configs {
		go deleteRC(&wg, config)
	}
	wg.Wait()
}

func deleteRC(wg *sync.WaitGroup, config *RCConfig) {
	defer GinkgoRecover()
	defer wg.Done()
	deletingTime := 10 * time.Minute

	sleepUpTo(deletingTime)
	expectNoError(DeleteRC(config.Client, config.Namespace, config.Name), fmt.Sprintf("deleting rc %s", config.Name))
}
