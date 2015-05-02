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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ReplicationController", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should serve a basic image on each replica with a public image", func() {
		ServeImageOrFail(c, "basic", "gcr.io/google_containers/serve_hostname:1.1")
	})

	It("should serve a basic image on each replica with a private image", func() {
		if !providerIs("gce", "gke") {
			By(fmt.Sprintf("Skipping private variant, which is only supported for providers gce and gke (not %s)",
				testContext.Provider))
			return
		}
		ServeImageOrFail(c, "private", "gcr.io/_b_k8s_authenticated_test/serve_hostname:1.1")
	})
})

// A basic test to check the deployment of an image using
// a replication controller. The image serves its hostname
// which is checked for each replica.
func ServeImageOrFail(c *client.Client, test string, image string) {
	ns := api.NamespaceDefault
	name := "my-hostname-" + test + "-" + string(util.NewUUID())
	replicas := 2

	// Create a replication controller for a service
	// that serves its hostname.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	By(fmt.Sprintf("Creating replication controller %s", name))
	controller, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
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
							Ports: []api.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		By("Cleaning up the replication controller")
		rcReaper, err := kubectl.ReaperFor("ReplicationController", c)
		if err != nil {
			Logf("Failed to cleanup replication controller %v: %v.", controller.Name, err)
		}
		if _, err = rcReaper.Stop(ns, controller.Name, nil); err != nil {
			Logf("Failed to stop replication controller %v: %v.", controller.Name, err)
		}
	}()

	// List the pods, making sure we observe all the replicas.
	listTimeout := time.Minute
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	pods, err := c.Pods(ns).List(label, fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	t := time.Now()
	for {
		Logf("Controller %s: Found %d pods out of %d", name, len(pods.Items), replicas)
		if len(pods.Items) == replicas {
			break
		}
		if time.Since(t) > listTimeout {
			Failf("Controller %s: Gave up waiting for %d pods to come up after seeing only %d pods after %v seconds",
				name, replicas, len(pods.Items), time.Since(t).Seconds())
		}
		time.Sleep(5 * time.Second)
		pods, err = c.Pods(ns).List(label, fields.Everything())
		Expect(err).NotTo(HaveOccurred())
	}

	By("Ensuring each pod is running")

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		err = waitForPodRunning(c, pod.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Verify that something is listening.
	By("Trying to dial each unique pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	err = wait.Poll(retryInterval, retryTimeout, responseChecker{c, ns, label, name, pods}.checkAllResponses)
	if err != nil {
		Failf("Did not get expected responses within the timeout period of %.2f seconds.", retryTimeout.Seconds())
	}
}

func isElementOf(podUID types.UID, pods *api.PodList) bool {
	for _, pod := range pods.Items {
		if pod.UID == podUID {
			return true
		}
	}
	return false
}

type responseChecker struct {
	c              *client.Client
	ns             string
	label          labels.Selector
	controllerName string
	pods           *api.PodList
}

func (r responseChecker) checkAllResponses() (done bool, err error) {
	successes := 0
	currentPods, err := r.c.Pods(r.ns).List(r.label, fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("Pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}
		body, err := r.c.Get().
			Prefix("proxy").
			Namespace(api.NamespaceDefault).
			Resource("pods").
			Name(string(pod.Name)).
			Do().
			Raw()
		if err != nil {
			Logf("Controller %s: Failed to GET from replica %d (%s): %v:", r.controllerName, i+1, pod.Name, err)
			continue
		}
		// The body should be the pod name.
		if string(body) != pod.Name {
			Logf("Controller %s: Replica %d expected response %s but got %s", r.controllerName, i+1, pod.Name, string(body))
			continue
		}
		successes++
		Logf("Controller %s: Got expected result from replica %d: %s, %d of %d required successes so far", r.controllerName, i+1, string(body), successes, len(r.pods.Items))
	}
	if successes < len(r.pods.Items) {
		return false, nil
	}
	return true, nil
}
