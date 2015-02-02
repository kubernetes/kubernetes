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
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("TestPodHasServiceEnvVars", func() {
	var (
		c *client.Client
	)

	BeforeEach(func() {
		c = loadClientOrDie()
	})

	It("should contain environment variables for services", func() {
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
			Fail(fmt.Sprintf("Failed to create serverPod: %v", err))
		}
		defer func() {
			defer GinkgoRecover()
			c.Pods(api.NamespaceDefault).Delete(serverPod.Name)
		}()
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
			Fail(fmt.Sprintf("Failed to delete service: %v", err))
		}
		time.Sleep(2)
		_, err = c.Services(api.NamespaceDefault).Create(svc)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create service: %v", err))
		}
		defer func() {
			defer GinkgoRecover()
			c.Services(api.NamespaceDefault).Delete(svc.Name)
		}()

		// TODO: we don't have a way to wait for a service to be "running".  // If this proves flaky, then we will need to retry the clientPod or insert a sleep.

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
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}
		defer func() {
			defer GinkgoRecover()
			c.Pods(api.NamespaceDefault).Delete(clientPod.Name)
		}()

		// Wait for client pod to complete.
		success := waitForPodSuccess(c, clientPod.Name, clientPod.Spec.Containers[0].Name)
		if !success {
			Fail(fmt.Sprintf("Failed to run client pod to detect service env vars."))
		}

		// Grab its logs.  Get host first.
		clientPodStatus, err := c.Pods(api.NamespaceDefault).Get(clientPod.Name)
		if err != nil {
			Fail(fmt.Sprintf("Failed to get clientPod to know host: %v", err))
		}
		By(fmt.Sprintf("Trying to get logs from host %s pod %s container %s: %v",
			clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err))
		logs, err := c.Get().
			Prefix("proxy").
			Resource("minions").
			Name(clientPodStatus.Status.Host).
			Suffix("containerLogs", api.NamespaceDefault, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name).
			Do().
			Raw()
		if err != nil {
			Fail(fmt.Sprintf("Failed to get logs from host %s pod %s container %s: %v",
				clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err))
		}
		fmt.Sprintf("clientPod logs:%v\n", string(logs))

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
			Expect(string(logs)).To(ContainSubstring(m), "%q in client env vars", m)
		}

		// We could try a wget the service from the client pod.  But services.sh e2e test covers that pretty well.
	})
})
