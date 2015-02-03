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
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Pods", func() {
	var (
		c *client.Client
	)

	BeforeEach(func() {
		c = loadClientOrDie()
	})

	It("should be submitted and removed", func() {
		podClient := c.Pods(api.NamespaceDefault)

		By("loading the pod json")
		pod := loadPodOrDie(assetPath("api", "examples", "pod.json"))
		value := strconv.Itoa(time.Now().Nanosecond())
		pod.Name = pod.Name + "-" + randomSuffix()
		pod.Labels["time"] = value
		pod.Spec.Containers[0].Ports[0].HostPort = 0

		By("submitting the pod to kubernetes")
		_, err := podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}
		defer func() {
			// We call defer here in case there is a problem with
			// the test so we can ensure that we clean up after
			// ourselves
			defer GinkgoRecover()
			podClient.Delete(pod.Name)
		}()

		By("verifying the pod is in kubernetes")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		if err != nil {
			Fail(fmt.Sprintf("Failed to query for pods: %v", err))
		}
		Expect(len(pods.Items)).To(Equal(1))

		By("deleting the pod")
		podClient.Delete(pod.Name)
		pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		Expect(len(pods.Items)).To(Equal(0))
	})

	It("should be updated", func() {
		podClient := c.Pods(api.NamespaceDefault)

		By("creating the pod")
		name := "pod-update-" + string(util.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "nginx",
						Image: "dockerfile/nginx",
						Ports: []api.Port{{ContainerPort: 80, HostPort: 8080}},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Path: "/index.html",
									Port: util.NewIntOrStringFromInt(8080),
								},
							},
							InitialDelaySeconds: 30,
						},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		_, err := podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}
		defer func() {
			By("deleting the pod")
			defer GinkgoRecover()
			podClient.Delete(pod.Name)
		}()

		By("waiting for the pod to start running")
		waitForPodRunning(c, pod.Name)

		By("verifying the pod is in kubernetes")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		Expect(len(pods.Items)).To(Equal(1))

		By("retrieving the pod")
		podOut, err := podClient.Get(pod.Name)
		if err != nil {
			Fail(fmt.Sprintf("Failed to get pod: %v", err))
		}

		By("updating the pod")
		value = "time" + value
		pod.Labels["time"] = value
		pod.ResourceVersion = podOut.ResourceVersion
		pod.UID = podOut.UID
		pod, err = podClient.Update(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to update pod: %v", err))
		}

		By("waiting for the updated pod to start running")
		waitForPodRunning(c, pod.Name)

		By("verifying the updated pod is in kubernetes")
		pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		Expect(len(pods.Items)).To(Equal(1))
		fmt.Println("pod update OK")
	})

	It("should contain environment variables for services", func() {
		// Make a pod that will be a service.
		// This pod serves its hostname via HTTP.
		serverName := "server-envvars-" + string(util.NewUUID())
		serverPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   serverName,
				Labels: map[string]string{"name": serverName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "srv",
						Image: "kubernetes/serve_hostname",
						Ports: []api.Port{{ContainerPort: 9376, HostPort: 8080}},
					},
				},
			},
		}
		_, err := c.Pods(api.NamespaceDefault).Create(serverPod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create serverPod: %v", err))
		}
		defer func() {
			defer GinkgoRecover()
			c.Pods(api.NamespaceDefault).Delete(serverPod.Name)
		}()
		waitForPodRunning(c, serverPod.Name)

		// This service exposes port 8080 of the test pod as a service on port 8765
		// TODO(filbranden): We would like to use a unique service name such as:
		//   svcName := "svc-envvars-" + randomSuffix()
		// However, that affects the name of the environment variables which are the capitalized
		// service name, so that breaks this test.  One possibility is to tweak the variable names
		// to match the service.  Another is to rethink environment variable names and possibly
		// allow overriding the prefix in the service manifest.
		svcName := "fooservice"
		svc := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: svcName,
				Labels: map[string]string{
					"name": svcName,
				},
			},
			Spec: api.ServiceSpec{
				Port:          8765,
				ContainerPort: util.NewIntOrStringFromInt(8080),
				Selector: map[string]string{
					"name": serverName,
				},
			},
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
		clientName := "client-envvars-" + string(util.NewUUID())
		clientPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   clientName,
				Labels: map[string]string{"name": clientName},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "env3cont",
						Image:   "busybox",
						Command: []string{"sh", "-c", "env"},
					},
				},
				RestartPolicy: api.RestartPolicy{
					Never: &api.RestartPolicyNever{},
				},
			},
		}
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
