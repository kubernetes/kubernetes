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
	"bufio"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
)

func testAffinity(c *client.Client, ns string, svc *api.Service, expectedHosts int) {
	name := "server"
	replicas := 3
	// This is the server that will serve traffic, including the hostname.
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: svc.Spec.Selector,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: svc.Spec.Selector,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: "gcr.io/google_containers/nettest:1.3",
							Ports: []api.ContainerPort{{ContainerPort: 8080}},
						},
					},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating rc %s in namespace %s", rc.Name, ns))
	_, err := c.ReplicationControllers(ns).Create(rc)
	expectNoError(err, fmt.Sprintf("creating rc %s", rc.Name))
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		By("Cleaning up the replication controller")
		rcReaper, err := kubectl.ReaperFor("ReplicationController", c)
		if err != nil {
			Logf("Failed to cleanup replication controller %v: %v.", rc.Name, err)
		}
		if _, err = rcReaper.Stop(ns, rc.Name, nil); err != nil {
			Logf("Failed to stop replication controller %v: %v.", rc.Name, err)
		}
	}()

	selector := labels.SelectorFromSet(rc.Spec.Selector)
	By("Waiting for replicas to be created.")
	err = wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {
		list, err := c.Pods(ns).List(selector, fields.Everything())
		if err != nil {
			fmt.Printf("unexpected error listing pods: %v", err)
			return false, nil
		}
		running := 0
		for _, pod := range list.Items {
			if pod.Status.Phase == api.PodRunning {
				running = running + 1
			}
		}
		return running == rc.Spec.Replicas, nil
	})
	expectNoError(err, "waiting for replicas to start running")

	By("Creating service for replicas")
	defer func() {
		err := c.Services(ns).Delete(svc.Name)
		expectNoError(err, "deleting service")
	}()
	_, err = c.Services(ns).Create(svc)
	expectNoError(err, "creating service")

	// This pod repeatedly curls the replicated server via the service, and logs the output.
	testPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "tester",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "tester",
					Image:   "busybox",
					Command: []string{"sh", "-c", "while true; do wget -qO- http://service:8080/read; sleep 1; done"},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating pod %s in namespace %s", testPod.Name, ns))
	_, err = c.Pods(ns).Create(testPod)
	expectNoError(err, fmt.Sprintf("creating pod %s", testPod.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("Deleting the tester pod")
		c.Pods(ns).Delete(testPod.Name, nil)
	}()

	err = waitForPodRunningInNamespace(c, testPod.Name, ns)
	expectNoError(err, "waiting for tester pod to start")

	// Wait for us to accumulate a bunch of logs.
	// Collect the set of hosts that we've been connected to by the load balancer.
	hostsSeen := util.StringSet{}
	// Wait a maximum of 60 seconds
	for i := 0; i < 6; i++ {
		time.Sleep(10 * time.Second)

		// Grab the log files for the pod
		readCloser, err := c.RESTClient.Get().
			Namespace(ns).
			Name(testPod.Name).
			Resource("pods").
			SubResource("log").
			Param("follow", strconv.FormatBool(false)).
			Param("container", "tester").
			Param("previous", strconv.FormatBool(false)).
			Stream()
		expectNoError(err, "getting logs data")

		defer readCloser.Close()
		scanner := bufio.NewScanner(readCloser)

		hostCount := 0
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "Hostname") {
				parts := strings.Split(line, ":")
				hostsSeen.Insert(parts[1])
				hostCount++
			}
		}
		// If we've accumulated enough data, break early.
		if hostCount > 2*expectedHosts {
			break
		}
	}

	// Validate
	if hostsSeen.Len() != expectedHosts {
		Failf("Expected only %d host(s), saw: %v", expectedHosts, hostsSeen.List())
	}
}

var _ = Describe("Affinity", func() {
	f := NewFramework("affinity")

	serviceName := "service"
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: serviceName,
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "hostServer",
			},
			Ports: []api.ServicePort{{
				Port:       8080,
				TargetPort: util.NewIntOrStringFromInt(8080),
			}},
		},
	}

	It("should call multiple hosts", func() {
		testAffinity(f.Client, f.Namespace.Name, svc, 3)
	})

	svc.Spec.SessionAffinity = api.ServiceAffinityClientIP
	It("should have affinity in balancing", func() {
		testAffinity(f.Client, f.Namespace.Name, svc, 1)
	})
})
