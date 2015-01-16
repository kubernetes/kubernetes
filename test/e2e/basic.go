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
	"io/ioutil"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// A basic test to check the deployment of an image using
// a replication controller. The image serves its hostname
// which is checked for each replica.
func TestBasicImage(c *client.Client, test string, image string) bool {
	testStart := time.Now()
	ns := api.NamespaceDefault
	name := "my-hostname-" + test + "-" + string(util.NewUUID())
	replicas := 2

	// Create a replication controller for a service
	// that serves its hostname on port 8080.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	glog.Infof("Creating replication controller %s", name)
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
							Ports: []api.Port{{ContainerPort: 9376, HostPort: 8080}},
						},
					},
				},
			},
		},
	})
	if err != nil {
		glog.Infof("Failed to create replication controller %s: %v", name, err)
		return false
	}
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		controller.Spec.Replicas = 0
		if _, err = c.ReplicationControllers(ns).Update(controller); err != nil {
			glog.Errorf("Failed to resize replication controller %s to zero: %v", name, err)
		}

		// Delete the replication controller.
		if err = c.ReplicationControllers(ns).Delete(name); err != nil {
			glog.Errorf("Failed to delete replication controller %s: %v", name, err)
		}

		// Report running time for test.
		glog.Infof("Controller %s test took %v seconds", name, time.Since(testStart).Seconds())
	}()

	// List the pods, making sure we observe all the replicas.
	listTimeout := time.Minute
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	pods, err := c.Pods(ns).List(label)
	if err != nil {
		glog.Errorf("Controller %s: Failed to list pods: %v", name, err)
	}
	t := time.Now()
	for {
		glog.Infof("Controller %s: Found %d pods out of %d", name, len(pods.Items), replicas)
		if len(pods.Items) == replicas {
			break
		}
		if time.Since(t) > listTimeout {
			glog.Errorf("Controller %s: Gave up waiting for %d pods to come up after seeing only %d pods after %v seconds",
				name, replicas, len(pods.Items), time.Since(t).Seconds())
			return false
		}
		time.Sleep(10 * time.Second)
		pods, err = c.Pods(ns).List(label)
		if err != nil {
			glog.Errorf("Controller %s: Failed to list pods: %v", name, err)
		}
	}

	for i, pod := range pods.Items {
		glog.Infof("Controller %s: Replica %d: pod: %s hostPort: %s\n", name, i+1, pod.Name, pod.Status.HostIP)
	}
	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		waitForPodRunning(c, pod.Name)
	}

	// Try to make sure we get a hostIP for each pod.
	hostIPTimeout := 2 * time.Minute
	t = time.Now()
	for i, pod := range pods.Items {
		for {
			p, err := c.Pods(ns).Get(pod.Name)
			if err != nil {
				glog.Errorf("Controller %s: Failed to get status for pod %s: %v", name, pod.Name, err)
				return false
			}
			if p.Status.HostIP != "" {
				glog.Infof("Controller %s: Replica %d has hostIP: %s", name, i+1, p.Status.HostIP)
				break
			}
			if time.Since(t) >= hostIPTimeout {
				glog.Errorf("Controller %s: Gave up waiting for hostIP of replica %d after %v seconds",
					name, i, time.Since(t).Seconds())
				return false
			}
			glog.Infof("Controller %s: Retrying to get the hostIP of replica %d", name, i+1)
			time.Sleep(5 * time.Second)
		}
	}

	// Re-fetch the pod information to update the host port information.
	pods, err = c.Pods(ns).List(label)
	if err != nil {
		glog.Errorf("Controller %s: Failed to list pods: %v", name, err)
	}

	// Verify that something is listening.
	for i, pod := range pods.Items {
		glog.Infof("Pod: %s hostIP: %s", pod.Name, pod.Status.HostIP)
		resp, err := http.Get(fmt.Sprintf("http://%s:8080", pod.Status.HostIP))
		if err != nil {
			glog.Errorf("Controller %s: Failed to GET from replica %d: %v", name, i+1, err)
			return false
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			glog.Errorf("Controller %s: Expected OK status code for replica %d but got %d", name, i+1, resp.StatusCode)
			return false
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			glog.Errorf("Controller %s: Failed to read the body of the GET response from replica %d: %v",
				name, i+1, err)
			return false
		}
		// The body should be the pod name.
		if string(body) != pod.Name {
			glog.Errorf("Controller %s: Replica %d expected response %s but got %s",
				name, i+1, pod.Name, string(body))
			return false
		}
		glog.Infof("Controller %s: Got expected result from replica %d: %s", name, i+1, string(body))
	}

	return true
}

// TestBasic performs the TestBasicImage check with the
// image kubernetes/serve_hostname
func TestBasic(c *client.Client) bool {
	return TestBasicImage(c, "basic", "kubernetes/serve_hostname:1.0")
}
