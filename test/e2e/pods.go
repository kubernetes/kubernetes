/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"strings"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	defaultObservationTimeout = time.Minute * 2
)

var (
	buildBackOffDuration = time.Minute
	syncLoopFrequency    = 10 * time.Second
	maxBackOffTolerance  = time.Duration(1.3 * float64(kubelet.MaxContainerBackOff))
)

func runLivenessTest(c *client.Client, ns string, podDescr *api.Pod, expectNumRestarts int, timeout time.Duration) {
	By(fmt.Sprintf("Creating pod %s in namespace %s", podDescr.Name, ns))
	_, err := c.Pods(ns).Create(podDescr)
	framework.ExpectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("deleting the pod")
		c.Pods(ns).Delete(podDescr.Name, api.NewDeleteOptions(0))
	}()

	// Wait until the pod is not pending. (Here we need to check for something other than
	// 'Pending' other than checking for 'Running', since when failures occur, we go to
	// 'Terminated' which can cause indefinite blocking.)
	framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, podDescr.Name),
		fmt.Sprintf("starting pod %s in namespace %s", podDescr.Name, ns))
	framework.Logf("Started pod %s in namespace %s", podDescr.Name, ns)

	// Check the pod's current state and verify that restartCount is present.
	By("checking the pod's current state and verifying that restartCount is present")
	pod, err := c.Pods(ns).Get(podDescr.Name)
	framework.ExpectNoError(err, fmt.Sprintf("getting pod %s in namespace %s", podDescr.Name, ns))
	initialRestartCount := api.GetExistingContainerStatus(pod.Status.ContainerStatuses, "liveness").RestartCount
	framework.Logf("Initial restart count of pod %s is %d", podDescr.Name, initialRestartCount)

	// Wait for the restart state to be as desired.
	deadline := time.Now().Add(timeout)
	lastRestartCount := initialRestartCount
	observedRestarts := int32(0)
	for start := time.Now(); time.Now().Before(deadline); time.Sleep(2 * time.Second) {
		pod, err = c.Pods(ns).Get(podDescr.Name)
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podDescr.Name))
		restartCount := api.GetExistingContainerStatus(pod.Status.ContainerStatuses, "liveness").RestartCount
		if restartCount != lastRestartCount {
			framework.Logf("Restart count of pod %s/%s is now %d (%v elapsed)",
				ns, podDescr.Name, restartCount, time.Since(start))
			if restartCount < lastRestartCount {
				framework.Failf("Restart count should increment monotonically: restart cont of pod %s/%s changed from %d to %d",
					ns, podDescr.Name, lastRestartCount, restartCount)
			}
		}
		observedRestarts = restartCount - initialRestartCount
		if expectNumRestarts > 0 && int(observedRestarts) >= expectNumRestarts {
			// Stop if we have observed more than expectNumRestarts restarts.
			break
		}
		lastRestartCount = restartCount
	}

	// If we expected 0 restarts, fail if observed any restart.
	// If we expected n restarts (n > 0), fail if we observed < n restarts.
	if (expectNumRestarts == 0 && observedRestarts > 0) || (expectNumRestarts > 0 &&
		int(observedRestarts) < expectNumRestarts) {
		framework.Failf("pod %s/%s - expected number of restarts: %t, found restarts: %t",
			ns, podDescr.Name, expectNumRestarts, observedRestarts)
	}
}

var _ = framework.KubeDescribe("Pods", func() {
	f := framework.NewDefaultFramework("pods")

	It("should *not* be restarted with a /healthz http liveness probe [Conformance]", func() {
		runLivenessTest(f.Client, f.Namespace.Name, &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "liveness-http",
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "liveness",
						Image: "gcr.io/google_containers/nettest:1.7",
						// These args are garbage but the image will exit if they're not there
						// we just care about /read serving a 200, which it always does.
						Args: []string{
							"-service=liveness-http",
							"-peers=1",
							"-namespace=" + f.Namespace.Name},
						Ports: []api.ContainerPort{{ContainerPort: 8080}},
						LivenessProbe: &api.Probe{
							Handler: api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Path: "/read",
									Port: intstr.FromInt(8080),
								},
							},
							InitialDelaySeconds: 15,
							TimeoutSeconds:      10,
							FailureThreshold:    1,
						},
					},
				},
			},
		}, 0, defaultObservationTimeout)
	})

	It("should support remote command execution over websockets", func() {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("Unable to get base config: %v", err)
		}
		podClient := f.Client.Pods(f.Namespace.Name)

		By("creating the pod")
		name := "pod-exec-websocket-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "main",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 600"},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		defer func() {
			By("deleting the pod")
			podClient.Delete(pod.Name, api.NewDeleteOptions(0))
		}()
		pod, err = podClient.Create(pod)
		if err != nil {
			framework.Failf("Failed to create pod: %v", err)
		}

		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		req := f.Client.Get().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			Suffix("exec").
			Param("stderr", "1").
			Param("stdout", "1").
			Param("container", pod.Spec.Containers[0].Name).
			Param("command", "cat").
			Param("command", "/etc/resolv.conf")

		url := req.URL()
		ws, err := framework.OpenWebSocketForURL(url, config, []string{"channel.k8s.io"})
		if err != nil {
			framework.Failf("Failed to open websocket to %s: %v", url.String(), err)
		}
		defer ws.Close()

		buf := &bytes.Buffer{}
		for {
			var msg []byte
			if err := websocket.Message.Receive(ws, &msg); err != nil {
				if err == io.EOF {
					break
				}
				framework.Failf("Failed to read completely from websocket %s: %v", url.String(), err)
			}
			if len(msg) == 0 {
				continue
			}
			if msg[0] != 1 {
				framework.Failf("Got message from server that didn't start with channel 1 (STDOUT): %v", msg)
			}
			buf.Write(msg[1:])
		}
		if buf.Len() == 0 {
			framework.Failf("Unexpected output from server")
		}
		if !strings.Contains(buf.String(), "nameserver") {
			framework.Failf("Expected to find 'nameserver' in %q", buf.String())
		}
	})

	It("should support retrieving logs from the container over websockets", func() {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("Unable to get base config: %v", err)
		}
		podClient := f.Client.Pods(f.Namespace.Name)

		By("creating the pod")
		name := "pod-logs-websocket-" + string(util.NewUUID())
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "main",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 600"},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		defer func() {
			By("deleting the pod")
			podClient.Delete(pod.Name, api.NewDeleteOptions(0))
		}()
		pod, err = podClient.Create(pod)
		if err != nil {
			framework.Failf("Failed to create pod: %v", err)
		}

		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		req := f.Client.Get().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			Suffix("log").
			Param("container", pod.Spec.Containers[0].Name)

		url := req.URL()

		ws, err := framework.OpenWebSocketForURL(url, config, []string{"binary.k8s.io"})
		if err != nil {
			framework.Failf("Failed to open websocket to %s: %v", url.String(), err)
		}
		defer ws.Close()
		buf := &bytes.Buffer{}
		for {
			var msg []byte
			if err := websocket.Message.Receive(ws, &msg); err != nil {
				if err == io.EOF {
					break
				}
				framework.Failf("Failed to read completely from websocket %s: %v", url.String(), err)
			}
			if len(msg) == 0 {
				continue
			}
			buf.Write(msg)
		}
		if buf.String() != "container is alive\n" {
			framework.Failf("Unexpected websocket logs:\n%s", buf.String())
		}
	})

	// The following tests for remote command execution and port forwarding are
	// commented out because the GCE environment does not currently have nsenter
	// in the kubelet's PATH, nor does it have socat installed. Once we figure
	// out the best way to have nsenter and socat available in GCE (and hopefully
	// all providers), we can enable these tests.
	/*
		It("should support remote command execution", func() {
			clientConfig, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("Failed to create client config: %v", err)
			}

			podClient := f.Client.Pods(f.Namespace.Name)

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
							Image: "gcr.io/google_containers/nginx:1.7.9",
						},
					},
				},
			}

			By("submitting the pod to kubernetes")
			_, err = podClient.Create(pod)
			if err != nil {
				framework.Failf("Failed to create pod: %v", err)
			}
			defer func() {
				// We call defer here in case there is a problem with
				// the test so we can ensure that we clean up after
				// ourselves
				podClient.Delete(pod.Name, api.NewDeleteOptions(0))
			}()

			By("waiting for the pod to start running")
			framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

			By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := api.ListOptions{LabelSelector: selector}
			pods, err := podClient.List(options)
			if err != nil {
				framework.Failf("Failed to query for pods: %v", err)
			}
			Expect(len(pods.Items)).To(Equal(1))

			pod = &pods.Items[0]
			By(fmt.Sprintf("executing command on host %s pod %s in container %s",
				pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name))
			req := f.Client.Get().
				Prefix("proxy").
				Resource("nodes").
				Name(pod.Status.Host).
				Suffix("exec", f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)

			out := &bytes.Buffer{}
			e := remotecommand.New(req, clientConfig, []string{"whoami"}, nil, out, nil, false)
			err = e.Execute()
			if err != nil {
				framework.Failf("Failed to execute command on host %s pod %s in container %s: %v",
					pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name, err)
			}
			if e, a := "root\n", out.String(); e != a {
				framework.Failf("exec: whoami: expected '%s', got '%s'", e, a)
			}
		})

		It("should support port forwarding", func() {
			clientConfig, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("Failed to create client config: %v", err)
			}

			podClient := f.Client.Pods(f.Namespace.Name)

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
							Image: "gcr.io/google_containers/nginx:1.7.9",
							Ports: []api.Port{{ContainerPort: 80}},
						},
					},
				},
			}

			By("submitting the pod to kubernetes")
			_, err = podClient.Create(pod)
			if err != nil {
				framework.Failf("Failed to create pod: %v", err)
			}
			defer func() {
				// We call defer here in case there is a problem with
				// the test so we can ensure that we clean up after
				// ourselves
				podClient.Delete(pod.Name, api.NewDeleteOptions(0))
			}()

			By("waiting for the pod to start running")
			framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

			By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := api.ListOptions{LabelSelector: selector}
			pods, err := podClient.List(options)
			if err != nil {
				framework.Failf("Failed to query for pods: %v", err)
			}
			Expect(len(pods.Items)).To(Equal(1))

			pod = &pods.Items[0]
			By(fmt.Sprintf("initiating port forwarding to host %s pod %s in container %s",
				pod.Status.Host, pod.Name, pod.Spec.Containers[0].Name))

			req := f.Client.Get().
				Prefix("proxy").
				Resource("nodes").
				Name(pod.Status.Host).
				Suffix("portForward", f.Namespace.Name, pod.Name)

			stopChan := make(chan struct{})
			pf, err := portforward.New(req, clientConfig, []string{"5678:80"}, stopChan)
			if err != nil {
				framework.Failf("Error creating port forwarder: %s", err)
			}

			errorChan := make(chan error)
			go func() {
				errorChan <- pf.ForwardPorts()
			}()

			// wait for listeners to start
			<-pf.Ready

			resp, err := http.Get("http://localhost:5678/")
			if err != nil {
				framework.Failf("Error with http get to localhost:5678: %s", err)
			}
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				framework.Failf("Error reading response body: %s", err)
			}

			titleRegex := regexp.MustCompile("<title>(.+)</title>")
			matches := titleRegex.FindStringSubmatch(string(body))
			if len(matches) != 2 {
				Fail("Unable to locate page title in response HTML")
			}
			if e, a := "Welcome to nginx on Debian!", matches[1]; e != a {
				framework.Failf("<title>: expected '%s', got '%s'", e, a)
			}
		})
	*/
})
