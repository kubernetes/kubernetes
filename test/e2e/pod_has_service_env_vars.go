/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TestPodHasServiceEnvVars checks that kubelets and scheduler send events about pods scheduling and running.
func TestPodHasServiceEnvVars(c *client.Client) bool {
	// Make a pod that will be a service.
	// This pod serves its hostname via HTTP.
	serverPod := parsePodOrDie(`{
	  "kind": "Pod",
	  "apiVersion": "v1beta1",
	  "id": "srv",
	  "desiredState": {
		"manifest": {
		  "version": "v1beta1",
		  "id": "srv",
		  "containers": [{
			"name": "srv",
			"image": "kubernetes/serve_hostname",
			"ports": [{
			  "containerPort": 9376,
			  "hostPort": 8080
			}]
		  }]
		}
	  },
	  "labels": {
		"name": "srv"
	  }
	}`)
	_, err := c.Pods(api.NamespaceDefault).Create(serverPod)
	if err != nil {
		glog.Errorf("Failed to create serverPod: %v", err)
		return false
	}
	defer c.Pods(api.NamespaceDefault).Delete(serverPod.Name)
	waitForPodRunning(c, serverPod.Name)

	// This service exposes pod p's port 8080 as a service on port 8765
	svc := parseServiceOrDie(`{
	  "id": "fooservice",
	  "kind": "Service",
	  "apiVersion": "v1beta1",
	  "port": 8765,
	  "containerPort": 8080,
	  "selector": {
		"name": "p"
	  }
	}`)
	if err != nil {
		glog.Errorf("Failed to delete service: %v", err)
		return false
	}
	time.Sleep(2)
	_, err = c.Services(api.NamespaceDefault).Create(svc)
	if err != nil {
		glog.Errorf("Failed to create service: %v", err)
		return false
	}
	defer c.Services(api.NamespaceDefault).Delete(svc.Name)

	// TODO: we don't have a way to wait for a service to be "running".
	// If this proves flaky, then we will need to retry the clientPod or insert a sleep.

	// Make a client pod that verifies that it has the service environment variables.
	clientPod := parsePodOrDie(`{
	  "apiVersion": "v1beta1",
	  "kind": "Pod",
	  "id": "env3",
	  "desiredState": {
		"manifest": {
		  "version": "v1beta1",
		  "id": "env3",
		  "restartPolicy": { "never": {} },
		  "containers": [{
			"name": "env3cont",
			"image": "busybox",
			"command": ["sh", "-c", "env"]
		  }]
		}
	  },
	  "labels": { "name": "env3" }
	}`)
	_, err = c.Pods(api.NamespaceDefault).Create(clientPod)
	if err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	defer c.Pods(api.NamespaceDefault).Delete(clientPod.Name)

	// Wait for client pod to complete.
	success := waitForPodSuccess(c, clientPod.Name, clientPod.Spec.Containers[0].Name)
	if !success {
		glog.Errorf("Failed to run client pod to detect service env vars.")
	}

	// Grab its logs.  Get host first.
	clientPodStatus, err := c.Pods(api.NamespaceDefault).Get(clientPod.Name)
	if err != nil {
		glog.Errorf("Failed to get clientPod to know host: %v", err)
		return false
	}
	glog.Infof("Trying to get logs from host %s pod %s container %s: %v",
		clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err)
	logs, err := c.Get().
		Prefix("proxy").
		Resource("minions").
		Name(clientPodStatus.Status.Host).
		Suffix("containerLogs", api.NamespaceDefault, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name).
		Do().
		Raw()
	if err != nil {
		glog.Errorf("Failed to get logs from host %s pod %s container %s: %v",
			clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err)
		return false
	}
	glog.Info("clientPod logs:", string(logs))

	toFind := []string{
		"FOOSERVICE_SERVICE_HOST=",
		"FOOSERVICE_SERVICE_PORT=",
		"FOOSERVICE_PORT=",
		"FOOSERVICE_PORT_8765_TCP_PORT=",
		"FOOSERVICE_PORT_8765_TCP_PROTO=",
		"FOOSERVICE_PORT_8765_TCP=",
		"FOOSERVICE_PORT_8765_TCP_ADDR=",
	}

	for _, m := range toFind {
		if !strings.Contains(string(logs), m) {
			glog.Errorf("Unable to find env var %q in client env vars.", m)
			success = false
		}
	}

	// We could try a wget the service from the client pod.  But services.sh e2e test covers that pretty well.
	return success
}

var _ = Describe("TestPodHasServiceEnvVars", func() {
	It("should pass", func() {
		// TODO: Instead of OrDie, client should Fail the test if there's a problem.
		// In general tests should Fail() instead of glog.Fatalf().
		Expect(TestPodHasServiceEnvVars(loadClientOrDie())).To(BeTrue())
	})
})
