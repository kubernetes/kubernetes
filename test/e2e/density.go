/*
Copyright 2015 Google Inc. All rights reserved.

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
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Convenient wrapper around listing pods supporting retries.
func listPods(c *client.Client, namespace string, label labels.Selector, field fields.Selector) (*api.PodList, error) {
	maxRetries := 4
	pods, err := c.Pods(namespace).List(label, field)
	for i := 0; i < maxRetries; i++ {
		if err == nil {
			return pods, nil
		}
		pods, err = c.Pods(namespace).List(label, field)
	}
	return pods, err
}

// Delete a Replication Controller and all pods it spawned
func DeleteRC(c *client.Client, ns, name string) error {
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		return fmt.Errorf("Failed to find replication controller %s in namespace %s: %v", name, ns, err)
	}

	rc.Spec.Replicas = 0

	if _, err := c.ReplicationControllers(ns).Update(rc); err != nil {
		return fmt.Errorf("Failed to resize replication controller %s to zero: %v", name, err)
	}

	// Wait up to 20 minutes until all replicas are killed.
	endTime := time.Now().Add(time.Minute * 20)
	for {
		if time.Now().After(endTime) {
			return fmt.Errorf("Timeout while waiting for replication controller %s replicas to 0", name)
		}
		remainingTime := endTime.Sub(time.Now())
		err := wait.Poll(time.Second, remainingTime, client.ControllerHasDesiredReplicas(c, rc))
		if err != nil {
			glog.Errorf("Error while waiting for replication controller %s replicas to read 0: %v", name, err)
		} else {
			break
		}
	}

	// Delete the replication controller.
	if err := c.ReplicationControllers(ns).Delete(name); err != nil {
		return fmt.Errorf("Failed to delete replication controller %s: %v", name, err)
	}
	return nil
}

// Launch a Replication Controller and wait for all pods it spawns
// to become running
func RunRC(c *client.Client, name string, ns, image string, replicas int) {
	defer GinkgoRecover()

	var last int
	current := 0
	same := 0

	defer func() {
		By("Cleaning up the replication controller")
		err := DeleteRC(c, ns, name)
		Expect(err).NotTo(HaveOccurred())
	}()

	By(fmt.Sprintf("Creating replication controller %s", name))
	_, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Making sure all %d replicas exist", replicas))
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	pods, err := listPods(c, ns, label, fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	current = len(pods.Items)
	failCount := 5
	for same < failCount && current < replicas {
		glog.Infof("Controller %s: Found %d pods out of %d", name, current, replicas)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			Failf("Controller %s: Number of submitted pods dropped from %d to %d", last, current)
		}

		if same >= failCount {
			glog.Infof("No pods submitted for the last %d checks", failCount)
		}

		last = current
		time.Sleep(5 * time.Second)
		pods, err = listPods(c, ns, label, fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		current = len(pods.Items)
	}
	Expect(current).To(Equal(replicas))
	glog.Infof("Controller %s: Found %d pods out of %d", name, current, replicas)

	By("Waiting for each pod to be running")
	same = 0
	last = 0
	failCount = 10
	current = 0
	for same < failCount && current < replicas {
		current = 0
		waiting := 0
		pending := 0
		unknown := 0
		time.Sleep(10 * time.Second)

		currentPods, listErr := listPods(c, ns, label, fields.Everything())
		Expect(listErr).NotTo(HaveOccurred())
		if len(currentPods.Items) != len(pods.Items) {
			Failf("Number of reported pods changed: %d vs %d", len(currentPods.Items), len(pods.Items))
		}
		for _, p := range currentPods.Items {
			if p.Status.Phase == api.PodRunning {
				current++
			} else if p.Status.Phase == api.PodPending {
				if p.Spec.Host == "" {
					waiting++
				} else {
					pending++
				}
			} else if p.Status.Phase == api.PodUnknown {
				unknown++
			}
		}
		glog.Infof("Pod States: %d running, %d pending, %d waiting, %d unknown ", current, pending, waiting, unknown)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			Failf("Number of running pods dropped from %d to %d", last, current)
		}
		if same >= failCount {
			glog.Infof("No pods started for the last %d checks", failCount)
		}
		last = current
	}
	Expect(current).To(Equal(replicas))
}

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
		ns = api.NamespaceDefault
	})

	AfterEach(func() {
		// Remove any remaining pods from this test if the
		// replication controller still exists and the replica count
		// isn't 0.  This means the controller wasn't cleaned up
		// during the test so clean it up here
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			DeleteRC(c, ns, RCName)
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
			RCName = "my-hostname-density" + strconv.Itoa(totalPods) + "-" + string(util.NewUUID())
			RunRC(c, RCName, ns, "gcr.io/google_containers/pause:go", totalPods)
		})
	}

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
					defer wg.Done()
					for i := 0; i < itArg.rcsPerThread; i++ {
						name := "my-short-lived-pod" + string(util.NewUUID())
						n := itArg.podsPerMinion * minionCount
						RunRC(c, name, ns, "gcr.io/google_containers/pause:go", n)
						podsLaunched += n
						glog.Info("Launched %v pods so far...", podsLaunched)
					}
				}()
			}
			// Wait for all the pods from all the RC's to return.
			wg.Wait()
			glog.Info("%v pods out of %v launched", podsLaunched, itArg.totalPods)
		})
	}
})
