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
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This test suite can take a long time to run, so by default it is added to
// the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
var _ = Describe("Scale", func() {
	var c *client.Client
	var minionCount int
	var RCName string
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		minions, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		minionCount = len(minions.Items)
		Expect(minionCount).NotTo(BeZero())
		nsForTesting, err := createTestingNS("scale", c)
		ns = nsForTesting.Name
		expectNoError(err)
	})

	AfterEach(func() {
		// Remove any remaining pods from this test if the
		// replication controller still exists and the replica count
		// isn't 0.  This means the controller wasn't cleaned up
		// during the test so clean it up here
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			By("Cleaning up the replication controller")
			err := DeleteRC(c, ns, RCName)
			expectNoError(err)
		}

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	type Scalability struct {
		skip          bool
		totalPods     int
		podsPerMinion int
		rcsPerThread  int
	}

	scalabilityTests := []Scalability{
		{totalPods: 500, podsPerMinion: 10, rcsPerThread: 5, skip: true},
		{totalPods: 500, podsPerMinion: 10, rcsPerThread: 25, skip: true},
	}

	for _, testArg := range scalabilityTests {
		// # of threads calibrate to totalPods
		threads := (testArg.totalPods / (testArg.podsPerMinion * testArg.rcsPerThread))

		name := fmt.Sprintf(
			"should be able to launch %v pods, %v per minion, in %v rcs/thread.",
			testArg.totalPods, testArg.podsPerMinion, testArg.rcsPerThread)
		if testArg.skip {
			name = "[Skipped] " + name
		}

		itArg := testArg
		It(name, func() {
			podsLaunched := 0
			var wg sync.WaitGroup
			wg.Add(threads)

			// Create queue of pending requests on the api server.
			for i := 0; i < threads; i++ {
				go func() {
					defer GinkgoRecover()
					defer wg.Done()
					for i := 0; i < itArg.rcsPerThread; i++ {
						name := "my-short-lived-pod" + string(util.NewUUID())
						n := itArg.podsPerMinion * minionCount
						expectNoError(RunRC(c, name, ns, "gcr.io/google_containers/pause:go", n))
						podsLaunched += n
						Logf("Launched %v pods so far...", podsLaunched)
						err := DeleteRC(c, ns, name)
						expectNoError(err)
					}
				}()
			}
			// Wait for all the pods from all the RC's to return.
			wg.Wait()
			Logf("%v pods out of %v launched", podsLaunched, itArg.totalPods)
		})
	}
})
