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
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func runLivenessTest(c *client.Client, podDescr *api.Pod) {
	ns := "e2e-test-" + string(util.NewUUID())

	By(fmt.Sprintf("Creating pod %s in namespace %s", podDescr.Name, ns))
	_, err := c.Pods(ns).Create(podDescr)
	expectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("deleting the pod")
		c.Pods(ns).Delete(podDescr.Name)
	}()

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	expectNoError(waitForPodNotPending(c, ns, podDescr.Name),
		fmt.Sprintf("starting pod %s in namespace %s", podDescr.Name, ns))
	By(fmt.Sprintf("Started pod %s in namespace %s", podDescr.Name, ns))

	// Check the pod's current state and verify that restartCount is present.
	By("checking the pod's current state and verifying that restartCount is present")
	pod, err := c.Pods(ns).Get(podDescr.Name)
	expectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", podDescr.Name, ns))
	initialRestartCount := pod.Status.Info["liveness"].RestartCount
	By(fmt.Sprintf("Initial restart count of pod %s is %d", podDescr.Name, initialRestartCount))

	// Wait for at most 48 * 5 = 240s = 4 minutes until restartCount is incremented
	pass := false
	for i := 0; i < 48; i++ {
		// Wait until restartCount is incremented.
		time.Sleep(5 * time.Second)
		pod, err = c.Pods(ns).Get(podDescr.Name)
		expectNoError(err, fmt.Sprintf("getting pod %s", podDescr.Name))
		restartCount := pod.Status.Info["liveness"].RestartCount
		By(fmt.Sprintf("Restart count of pod %s in namespace %s is now %d", podDescr.Name, ns, restartCount))
		if restartCount > initialRestartCount {
			By(fmt.Sprintf("Restart count of pod %s in namespace %s increased from %d to %d during the test", podDescr.Name, ns, initialRestartCount, restartCount))
			pass = true
			break
		}
	}

	if !pass {
		Fail(fmt.Sprintf("Did not see the restart count of pod %s in namespace %s increase from %d during the test", podDescr.Name, ns, initialRestartCount))
	}
}

var _ = Describe("Pods", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("should be submitted and removed", func() {
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
						Ports: []api.ContainerPort{{ContainerPort: 80}},
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

		By("setting up watch")
		pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		if err != nil {
			Fail(fmt.Sprintf("Failed to query for pods: %v", err))
		}
		Expect(len(pods.Items)).To(Equal(0))
		w, err := podClient.Watch(
			labels.SelectorFromSet(labels.Set(map[string]string{"time": value})), labels.Everything(), pods.ListMeta.ResourceVersion)
		if err != nil {
			Fail(fmt.Sprintf("Failed to set up watch: %v", err))
		}

		By("submitting the pod to kubernetes")
		// We call defer here in case there is a problem with
		// the test so we can ensure that we clean up after
		// ourselves
		defer podClient.Delete(pod.Name)
		_, err = podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}

		By("verifying the pod is in kubernetes")
		pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		if err != nil {
			Fail(fmt.Sprintf("Failed to query for pods: %v", err))
		}
		Expect(len(pods.Items)).To(Equal(1))

		By("veryfying pod creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				Fail(fmt.Sprintf("Failed to observe pod creation: %v", event))
			}
		case <-time.After(podStartTimeout):
			Fail("Timeout while waiting for pod creation")
		}

		By("deleting the pod")
		podClient.Delete(pod.Name)
		pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
		if err != nil {
			Fail(fmt.Sprintf("Failed to delete pod: %v", err))
		}
		Expect(len(pods.Items)).To(Equal(0))

		By("veryfying pod deletion was observed")
		deleted := false
		timeout := false
		timer := time.After(podStartTimeout)
		for !deleted && !timeout {
			select {
			case event, _ := <-w.ResultChan():
				if event.Type == watch.Deleted {
					deleted = true
				}
			case <-timer:
				timeout = true
			}
		}
		if !deleted {
			Fail("Failed to observe pod deletion")
		}
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
						Ports: []api.ContainerPort{{ContainerPort: 80}},
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
		defer func() {
			By("deleting the pod")
			podClient.Delete(pod.Name)
		}()
		_, err := podClient.Create(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}

		expectNoError(waitForPodRunning(c, pod.Name))

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
		pod.Spec.Host = podOut.Spec.Host
		pod, err = podClient.Update(pod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to update pod: %v", err))
		}

		expectNoError(waitForPodRunning(c, pod.Name))

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
						Ports: []api.ContainerPort{{ContainerPort: 9376}},
					},
				},
			},
		}
		defer c.Pods(api.NamespaceDefault).Delete(serverPod.Name)
		_, err := c.Pods(api.NamespaceDefault).Create(serverPod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create serverPod: %v", err))
		}
		expectNoError(waitForPodRunning(c, serverPod.Name))

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
		defer c.Services(api.NamespaceDefault).Delete(svc.Name)
		_, err = c.Services(api.NamespaceDefault).Create(svc)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create service: %v", err))
		}

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
						Command: []string{"sh", "-c", "env; sleep 600"},
					},
				},
				RestartPolicy: api.RestartPolicy{
					Never: &api.RestartPolicyNever{},
				},
			},
		}
		defer c.Pods(api.NamespaceDefault).Delete(clientPod.Name)
		_, err = c.Pods(api.NamespaceDefault).Create(clientPod)
		if err != nil {
			Fail(fmt.Sprintf("Failed to create pod: %v", err))
		}

		expectNoError(waitForPodRunning(c, clientPod.Name))

		// Grab its logs.  Get host first.
		clientPodStatus, err := c.Pods(api.NamespaceDefault).Get(clientPod.Name)
		if err != nil {
			Fail(fmt.Sprintf("Failed to get clientPod to know host: %v", err))
		}
		By(fmt.Sprintf("Trying to get logs from host %s pod %s container %s: %v",
			clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err))
		var logs []byte
		start := time.Now()

		// Sometimes the actual containers take a second to get started, try to get logs for 60s
		for time.Now().Sub(start) < (60 * time.Second) {
			logs, err = c.Get().
				Prefix("proxy").
				Resource("minions").
				Name(clientPodStatus.Status.Host).
				Suffix("containerLogs", api.NamespaceDefault, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name).
				Do().
				Raw()
			fmt.Sprintf("clientPod logs:%v\n", string(logs))
			By(fmt.Sprintf("clientPod logs:%v\n", string(logs)))
			if strings.Contains(string(logs), "Internal Error") {
				By(fmt.Sprintf("Failed to get logs from host %s pod %s container %s: %v",
					clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, string(logs)))
				time.Sleep(5 * time.Second)
				continue
			}
			break
		}

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

	It("should be restarted with a docker exec \"cat /tmp/health\" liveness probe", func() {
		runLivenessTest(c, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "liveness-exec",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "liveness",
						Image:   "busybox",
						Command: []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 10; echo fail >/tmp/health; sleep 600"},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								Exec: &api.ExecAction{
									Command: []string{"cat", "/tmp/health"},
								},
							},
							InitialDelaySeconds: 15,
						},
					},
				},
			},
		})
	})

	It("should be restarted with a /healthz http liveness probe", func() {
		runLivenessTest(c, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "liveness",
						Image:   "kubernetes/liveness",
						Command: []string{"/server"},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Path: "/healthz",
									Port: util.NewIntOrStringFromInt(8080),
								},
							},
							InitialDelaySeconds: 15,
						},
					},
				},
			},
		})
	})

	// The following tests for remote command execution and port forwarding are
	// commented out because the GCE environment does not currently have nsenter
	// in the kubelet's PATH, nor does it have socat installed. Once we figure
	// out the best way to have nsenter and socat available in GCE (and hopefully
	// all providers), we can enable these tests.
	/*
		It("should support remote command execution", func() {
			clientConfig, err := loadConfig()
			if err != nil {
				Fail(fmt.Sprintf("Failed to create client config: %v", err))
			}

			podClient := c.Pods(api.NamespaceDefault)

			By("creating the pod")
			name := "pod-exec-" + string(util.NewUUID())
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
						},
					},
				},
			}

			By("submitting the pod to kubernetes")
			_, err = podClient.Create(pod)
			if err != nil {
				Fail(fmt.Sprintf("Failed to create pod: %v", err))
			}
			defer func() {
				// We call defer here in case there is a problem with
				// the test so we can ensure that we clean up after
				// ourselves
				podClient.Delete(pod.Name)
			}()

			By("waiting for the pod to start running")
			expectNoError(waitForPodRunning(c, pod.Name, 300*time.Second))

			By("verifying the pod is in kubernetes")
			pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
			if err != nil {
				Fail(fmt.Sprintf("Failed to query for pods: %v", err))
			}
			Expect(len(pods.Items)).To(Equal(1))

			pod = &pods.Items[0]
			By(fmt.Sprintf("executing command on host %s pod %s in container %s",
				pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name))
			req := c.Get().
				Prefix("proxy").
				Resource("minions").
				Name(pod.Status.Host).
				Suffix("exec", api.NamespaceDefault, pod.Name, pod.Spec.Containers[0].Name)

			out := &bytes.Buffer{}
			e := remotecommand.New(req, clientConfig, []string{"whoami"}, nil, out, nil, false)
			err = e.Execute()
			if err != nil {
				Fail(fmt.Sprintf("Failed to execute command on host %s pod %s in container %s: %v",
					pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name, err))
			}
			if e, a := "root\n", out.String(); e != a {
				Fail(fmt.Sprintf("exec: whoami: expected '%s', got '%s'", e, a))
			}
		})

		It("should support port forwarding", func() {
			clientConfig, err := loadConfig()
			if err != nil {
				Fail(fmt.Sprintf("Failed to create client config: %v", err))
			}

			podClient := c.Pods(api.NamespaceDefault)

			By("creating the pod")
			name := "pod-portforward-" + string(util.NewUUID())
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
							Ports: []api.Port{{ContainerPort: 80}},
						},
					},
				},
			}

			By("submitting the pod to kubernetes")
			_, err = podClient.Create(pod)
			if err != nil {
				Fail(fmt.Sprintf("Failed to create pod: %v", err))
			}
			defer func() {
				// We call defer here in case there is a problem with
				// the test so we can ensure that we clean up after
				// ourselves
				podClient.Delete(pod.Name)
			}()

			By("waiting for the pod to start running")
			expectNoError(waitForPodRunning(c, pod.Name, 300*time.Second))

			By("verifying the pod is in kubernetes")
			pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
			if err != nil {
				Fail(fmt.Sprintf("Failed to query for pods: %v", err))
			}
			Expect(len(pods.Items)).To(Equal(1))

			pod = &pods.Items[0]
			By(fmt.Sprintf("initiating port forwarding to host %s pod %s in container %s",
				pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name))

			req := c.Get().
				Prefix("proxy").
				Resource("minions").
				Name(pod.Status.Host).
				Suffix("portForward", api.NamespaceDefault, pod.Name)

			stopChan := make(chan struct{})
			pf, err := portforward.New(req, clientConfig, []string{"5678:80"}, stopChan)
			if err != nil {
				Fail(fmt.Sprintf("Error creating port forwarder: %s", err))
			}

			errorChan := make(chan error)
			go func() {
				errorChan <- pf.ForwardPorts()
			}()

			// wait for listeners to start
			<-pf.Ready

			resp, err := http.Get("http://localhost:5678/")
			if err != nil {
				Fail(fmt.Sprintf("Error with http get to localhost:5678: %s", err))
			}
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				Fail(fmt.Sprintf("Error reading response body: %s", err))
			}

			titleRegex := regexp.MustCompile("<title>(.+)</title>")
			matches := titleRegex.FindStringSubmatch(string(body))
			if len(matches) != 2 {
				Fail("Unable to locate page title in response HTML")
			}
			if e, a := "Welcome to nginx on Debian!", matches[1]; e != a {
				Fail(fmt.Sprintf("<title>: expected '%s', got '%s'", e, a))
			}
		})
	*/
})
