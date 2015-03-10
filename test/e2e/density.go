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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

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
		Logf("Controller %s: Found %d pods out of %d", name, current, replicas)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			Failf("Controller %s: Number of submitted pods dropped from %d to %d", last, current)
		}

		if same >= failCount {
			Logf("No pods submitted for the last %d checks", failCount)
		}

		last = current
		time.Sleep(5 * time.Second)
		pods, err = c.Pods(ns).List(label)
		Expect(err).NotTo(HaveOccurred())
		current = len(pods.Items)
	}
	Expect(current).To(Equal(replicas))
	Logf("Controller %s: Found %d pods out of %d", name, current, replicas)

	By("Waiting for each pod to be running")
	same = 0
	last = 0
	failCount = 6
	unknown := 0
	pending := 0
	current = 0
	for same < failCount && current < replicas {
		current = 0
		pending = 0
		unknown = 0
		time.Sleep(10 * time.Second)
		for _, pod := range pods.Items {
			p, err := c.Pods(ns).Get(pod.Name)
			Expect(err).NotTo(HaveOccurred())
			if p.Status.Phase == api.PodRunning {
				current++
			} else if p.Status.Phase == api.PodPending {
				pending++
			} else if p.Status.Phase == api.PodUnknown {
				unknown++
			}
		}
		Logf("Pod States: %d running, %d pending, %d unknown ", current, pending, unknown)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			Failf("Number of running pods dropped from %d to %d", last, current)
		}
		if same >= failCount {
			Logf("No pods started for the last %d checks", failCount)
		}
		last = current
	}
	Expect(current).To(Equal(replicas))
}

// This test suite can take a long time to run, so by default it is disabled
// by being marked as Pending.  To enable this suite, remove the P from the
// front of PDescribe (PDescribe->Describe) and then all tests will
// be available
var _ = PDescribe("Density", func() {
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
		// Remove any remaining pods from this test
		DeleteRC(c, ns, RCName)
	})

	It("should allow starting 100 pods per node", func() {
		RCName = "my-hostname-density100-" + string(util.NewUUID())
		RunRC(c, RCName, ns, "dockerfile/nginx", 100*minionCount)
	})

	It("should have master components that can handle many short-lived pods", func() {
		threads := 5
		var wg sync.WaitGroup
		wg.Add(threads)
		for i := 0; i < threads; i++ {
			go func() {
				defer wg.Done()
				for i := 0; i < 10; i++ {
					name := "my-hostname-thrash-" + string(util.NewUUID())
					RunRC(c, name, ns, "kubernetes/pause", 10*minionCount)
				}
			}()
		}
		wg.Wait()
	})
})
