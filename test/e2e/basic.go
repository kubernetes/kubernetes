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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

// A basic test to check the deployment of an image using
// a replication controller. The image serves its hostname
// which is checked for each replica.
func TestBasicImage(c *client.Client, image string) bool {
	ns := api.NamespaceDefault
	// TODO(satnam6502): Generate a unique name to allow
	//                   parallel executions of this test.
	name := "my-hostname"
	replicas := 2

	// Attmept to delete a controller that might have been
	// left lying around from a previously aborted test.
	controllers, err := c.ReplicationControllers(ns).List(labels.Everything())
	if err != nil {
		glog.Infof("Failed to list replication controllers: %v", err)
		return false
	}
	for _, cnt := range controllers.Items {
		if cnt.Name == name {
			glog.Infof("Found a straggler %s controller", name)
			// Delete any pods controlled by this replicaiton controller.
			cnt.Spec.Replicas = 0
			if _, err := c.ReplicationControllers(ns).Update(&cnt); err != nil {
				glog.Warningf("Failed to resize straggler controller to zero: %v", err)
			}
			// Delete the controller
			if err = c.ReplicationControllers(ns).Delete(name); err != nil {
				glog.Warningf("Failed to delete straggler replicatior controller: %v", err)
			}
			break
		}
	}

	// Create a replication controller  for a service
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
		glog.Infof("Failed to create replication controller for %s: %v", name, err)
		return false
	}

	// List the pods.
	// pods, err := c.Pods(ns).List(labels.Set{"name": name}.AsSelector())
	pods, err := c.Pods(ns).List(labels.SelectorFromSet(labels.Set(map[string]string{"name": name})))
	if err != nil {
		glog.Errorf("Failed to list pods before wait for running check: %v", err)
		return false
	}
	for i, pod := range pods.Items {
		glog.Infof("Replica %d: %s\n", i+1, pod.Name)
	}

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		waitForPodRunning(c, pod.Name)
	}

	// List the pods again to get the host IP information.
	pods, err = c.Pods(ns).List(labels.SelectorFromSet(labels.Set(map[string]string{"name": name})))
	if err != nil {
		glog.Errorf("Failed to list pods after wait for running check: %v", err)
		return false
	}

	// Verify that something is listening.
	for i, pod := range pods.Items {
		glog.Infof("Pod: %s hostIP: %s", pod.Name, pod.Status.HostIP)
		if pod.Status.HostIP == "" {
			glog.Warningf("Skipping test of GET response since hostIP is not available")
			continue
		}
		resp, err := http.Get(fmt.Sprintf("http://%s:8080", pod.Status.HostIP))
		if err != nil {
			glog.Errorf("Failed to GET from replica %d: %v", i+1, err)
			return false
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			glog.Errorf("Expected OK status code for replica %d but got %d", i+1, resp.StatusCode)
			return false
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			glog.Errorf("Failed to read the body of the GET response from replica %d: %v", i+1, err)
			return false
		}
		// The body should be the pod name although we may need to skip a newline
		// character at the end of the response.
		if !strings.HasPrefix(string(body), pod.Name) {
			glog.Errorf("From replica %d expected response %s but got %s", i+1, pod.Name, string(body))
			return false
		}
		glog.Infof("Got expected result from replica %d: %s", i+1, string(body))
	}

	// Resize the replication controller to zero to get rid of pods.
	controller.Spec.Replicas = 0
	if _, err = c.ReplicationControllers(ns).Update(controller); err != nil {
		glog.Errorf("Failed to resize replication controllert to zero: %v", err)
		return false
	}

	// Delete the replication controller.
	if err = c.ReplicationControllers(ns).Delete(name); err != nil {
		glog.Errorf("Failed to delete replication controller %s: %v", name, err)
		return false
	}

	return true
}

// TestBasic performs the TestBasicImage check with the
// image kubernetes/serve_hostname
func TestBasic(c *client.Client) bool {
	return TestBasicImage(c, "kubernetes/serve_hostname")
}
