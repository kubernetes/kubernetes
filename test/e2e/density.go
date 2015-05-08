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
	"math"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This test suite can take a long time to run, so by default it is added to
// the ginkgo.skip list (see driver.go).
// To run this suite you must explicitly ask for it by setting the
// -t/--test flag or ginkgo.focus flag.
var _ = Describe("Density", func() {
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
		nsForTesting, err := createTestingNS("density", c)
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

	// Tests with "Skipped" substring in their name will be skipped when running
	// e2e test suite without --ginkgo.focus & --ginkgo.skip flags.
	type Density struct {
		skip          bool
		podsPerMinion int
	}

	densityTests := []Density{
		// This test should always run, even if larger densities are skipped.
		{podsPerMinion: 3, skip: false},
		{podsPerMinion: 30, skip: false},
		// More than 30 pods per node is outside our v1.0 goals.
		// We might want to enable those tests in the future.
		{podsPerMinion: 50, skip: true},
		{podsPerMinion: 100, skip: true},
	}

	for _, testArg := range densityTests {
		name := fmt.Sprintf("should allow starting %d pods per node", testArg.podsPerMinion)
		if testArg.podsPerMinion <= 30 {
			name = "[Performance suite] " + name
		}
		if testArg.skip {
			name = "[Skipped] " + name
		}
		itArg := testArg
		It(name, func() {
			totalPods := itArg.podsPerMinion * minionCount
			nameStr := strconv.Itoa(totalPods) + "-" + string(util.NewUUID())
			RCName = "my-hostname-density" + nameStr

			// Create a listener for events.
			events := make([](*api.Event), 0)
			_, controller := framework.NewInformer(
				&cache.ListWatch{
					ListFunc: func() (runtime.Object, error) {
						return c.Events(ns).List(labels.Everything(), fields.Everything())
					},
					WatchFunc: func(rv string) (watch.Interface, error) {
						return c.Events(ns).Watch(labels.Everything(), fields.Everything(), rv)
					},
				},
				&api.Event{},
				time.Second*10,
				framework.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						events = append(events, obj.(*api.Event))
					},
				},
			)
			stop := make(chan struct{})
			go controller.Run(stop)

			// Start the replication controller.
			startTime := time.Now()
			expectNoError(RunRC(c, RCName, ns, "gcr.io/google_containers/pause:go", totalPods))
			e2eStartupTime := time.Now().Sub(startTime)
			Logf("E2E startup time for %d pods: %v", totalPods, e2eStartupTime)

			By("Waiting for all events to be recorded")
			last := -1
			current := len(events)
			timeout := 10 * time.Minute
			for start := time.Now(); last < current && time.Since(start) < timeout; time.Sleep(10 * time.Second) {
				last = current
				current = len(events)
			}
			close(stop)

			if current != last {
				Logf("Warning: Not all events were recorded after waiting %.2f minutes", timeout.Minutes())
			}
			Logf("Found %d events", current)

			// Tune the threshold for allowed failures.
			badEvents := BadEvents(events)
			Expect(badEvents).NotTo(BeNumerically(">", int(math.Floor(0.01*float64(totalPods)))))

			// Verify latency metrics
			// TODO: Update threshold to 1s once we reach this goal
			// TODO: We should reset metrics before the test. Currently previous tests influence latency metrics.
			highLatencyRequests, err := HighLatencyRequests(c, 10*time.Second, util.NewStringSet("events"))
			expectNoError(err)
			Expect(highLatencyRequests).NotTo(BeNumerically(">", 0))
		})
	}
})
