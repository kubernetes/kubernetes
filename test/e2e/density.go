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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

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

	if err := wait.Poll(time.Second, time.Minute*20, client.ControllerHasDesiredReplicas(c, rc)); err != nil {
		return fmt.Errorf("Error waiting for replication controller %s replicas to reach 0: %v", name, err)
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
	pods, err := c.Pods(ns).List(label)
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
		pods, err = c.Pods(ns).List(label)
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

		currentPods, listErr := c.Pods(ns).List(label)
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
		minions, err := c.Nodes().List()
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
		totalPods     int
		podsPerMinion int
		rcsPerThread  int
	}

	//This test should always run, even if larger densities are skipped.
	d3 := Density{totalPods: 3, podsPerMinion: 0, rcsPerThread: 1, skip: false}

	//These tests are varied and customizable.
	//TODO (wojtek-t):don't skip d30 after #6059
	d30 := Density{totalPods: 30, podsPerMinion: 0, rcsPerThread: 1, skip: true}
	d50 := Density{totalPods: 50, podsPerMinion: 0, rcsPerThread: 1, skip: true}
	d100 := Density{totalPods: 100, podsPerMinion: 0, rcsPerThread: 1, skip: true}
	d500t5 := Density{totalPods: 500, podsPerMinion: 10, rcsPerThread: 5, skip: true}
	d500t25 := Density{totalPods: 500, podsPerMinion: 10, rcsPerThread: 25, skip: true}

	dtests := []Density{d3, d30, d50, d100, d500t5, d500t25}

	//Run each test in the array which isn't skipped.
	for i := range dtests {

		//cannot do a range iterator over structs.
		dtest := dtests[i]

		glog.Info("Density test parameters: %v", dtest)

		//if ppm==0, its a raw density test.
		//otherwise, we continue launching n nodes per pod in threads till we meet the totalPods #.
		if dtest.podsPerMinion == 0 {
			//basic density tests
			name := fmt.Sprintf("should allow starting %d pods per node", dtest.totalPods)

			if dtest.skip {
				name = "[Skipped] " + name
			}
			It(name, func() {
				RCName = "my-hostname-density" + strconv.Itoa(dtest.totalPods) + "-" + string(util.NewUUID())
				RunRC(c, RCName, ns, "gcr.io/google_containers/pause:go", dtest.totalPods)
			})
			glog.Info("moving on, test already finished....")
		} else {
			// # of threads calibrate to totalPods
			threads := (dtest.totalPods / (dtest.podsPerMinion * dtest.rcsPerThread))

			name := fmt.Sprintf(
				"[Skipped] should be able to launch %v pods, %v per minion, in %v rcs/thread.",
				dtest.totalPods, dtest.podsPerMinion, dtest.rcsPerThread)

			if dtest.skip {
				name = "[Skipped] " + name
			}

			podsLaunched := 0
			It(name, func() {

				var wg sync.WaitGroup

				//count down latch.., once all threads are launched, we wait for
				//it to decrement down to zero.
				wg.Add(threads)

				//create queue of pending requests on the api server.
				for i := 0; i < threads; i++ {
					go func() {
						// call to wg.Done will serve as a count down latch.
						defer wg.Done()
						for i := 0; i < dtest.rcsPerThread; i++ {
							name := "my-short-lived-pod" + string(util.NewUUID())
							n := dtest.podsPerMinion * minionCount
							RunRC(c, name, ns, "gcr.io/google_containers/pause:go", n)
							podsLaunched += n
							glog.Info("Launched %v pods so far...", podsLaunched)
						}
					}()
				}
				//Wait for all the pods from all the RC's to return.
				wg.Wait()
				glog.Info("%v pods out of %v launched", podsLaunched, dtest.totalPods)
			})
		}
	}
})
