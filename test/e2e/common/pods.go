/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/websocket"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	buildBackOffDuration = time.Minute
	syncLoopFrequency    = 10 * time.Second
	maxBackOffTolerance  = time.Duration(1.3 * float64(kubelet.MaxContainerBackOff))
)

// testHostIP tests that a pod gets a host IP
func testHostIP(podClient *framework.PodClient, pod *v1.Pod) {
	By("creating pod")
	podClient.CreateSync(pod)

	// Try to make sure we get a hostIP for each pod.
	hostIPTimeout := 2 * time.Minute
	t := time.Now()
	for {
		p, err := podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "Failed to get pod %q", pod.Name)
		if p.Status.HostIP != "" {
			framework.Logf("Pod %s has hostIP: %s", p.Name, p.Status.HostIP)
			break
		}
		if time.Since(t) >= hostIPTimeout {
			framework.Failf("Gave up waiting for hostIP of pod %s after %v seconds",
				p.Name, time.Since(t).Seconds())
		}
		framework.Logf("Retrying to get the hostIP of pod %s", p.Name)
		time.Sleep(5 * time.Second)
	}
}

func startPodAndGetBackOffs(podClient *framework.PodClient, pod *v1.Pod, sleepAmount time.Duration) (time.Duration, time.Duration) {
	podClient.CreateSync(pod)
	time.Sleep(sleepAmount)
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	podName := pod.Name
	containerName := pod.Spec.Containers[0].Name

	By("getting restart delay-0")
	_, err := getRestartDelay(podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}

	By("getting restart delay-1")
	delay1, err := getRestartDelay(podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}

	By("getting restart delay-2")
	delay2, err := getRestartDelay(podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}
	return delay1, delay2
}

func getRestartDelay(podClient *framework.PodClient, podName string, containerName string) (time.Duration, error) {
	beginTime := time.Now()
	for time.Since(beginTime) < (2 * maxBackOffTolerance) { // may just miss the 1st MaxContainerBackOff delay
		time.Sleep(time.Second)
		pod, err := podClient.Get(podName, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podName))
		status, ok := v1.GetContainerStatus(pod.Status.ContainerStatuses, containerName)
		if !ok {
			framework.Logf("getRestartDelay: status missing")
			continue
		}

		if status.State.Waiting == nil && status.State.Running != nil && status.LastTerminationState.Terminated != nil && status.State.Running.StartedAt.Time.After(beginTime) {
			startedAt := status.State.Running.StartedAt.Time
			finishedAt := status.LastTerminationState.Terminated.FinishedAt.Time
			framework.Logf("getRestartDelay: restartCount = %d, finishedAt=%s restartedAt=%s (%s)", status.RestartCount, finishedAt, startedAt, startedAt.Sub(finishedAt))
			return startedAt.Sub(finishedAt), nil
		}
	}
	return 0, fmt.Errorf("timeout getting pod restart delay")
}

var _ = framework.KubeDescribe("Pods", func() {
	f := framework.NewDefaultFramework("pods")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("should get a host IP [Conformance]", func() {
		name := "pod-hostip-" + string(uuid.NewUUID())
		testHostIP(podClient, &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "test",
						Image: framework.GetPauseImageName(f.ClientSet),
					},
				},
			},
		})
	})

	It("should be submitted and removed [Conformance]", func() {
		By("creating the pod")
		name := "pod-submit-remove-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: "gcr.io/google_containers/nginx-slim:0.7",
					},
				},
			},
		}

		By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := v1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(0))
		options = v1.ListOptions{
			LabelSelector:   selector.String(),
			ResourceVersion: pods.ListMeta.ResourceVersion,
		}
		w, err := podClient.Watch(options)
		Expect(err).NotTo(HaveOccurred(), "failed to set up watch")

		By("submitting the pod to kubernetes")
		podClient.Create(pod)

		By("verifying the pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = v1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(1))

		By("verifying pod creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe pod creation: %v", event)
			}
		case <-time.After(framework.PodStartTimeout):
			Fail("Timeout while waiting for pod creation")
		}

		// We need to wait for the pod to be running, otherwise the deletion
		// may be carried out immediately rather than gracefully.
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))
		// save the running pod
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to GET scheduled pod")
		framework.Logf("running pod: %#v", pod)

		By("deleting the pod gracefully")
		err = podClient.Delete(pod.Name, v1.NewDeleteOptions(30))
		Expect(err).NotTo(HaveOccurred(), "failed to delete pod")

		By("verifying the kubelet observed the termination notice")
		Expect(wait.Poll(time.Second*5, time.Second*30, func() (bool, error) {
			podList, err := framework.GetKubeletPods(f.ClientSet, pod.Spec.NodeName)
			if err != nil {
				framework.Logf("Unable to retrieve kubelet pods for node %v: %v", pod.Spec.NodeName, err)
				return false, nil
			}
			for _, kubeletPod := range podList.Items {
				if pod.Name != kubeletPod.Name {
					continue
				}
				if kubeletPod.ObjectMeta.DeletionTimestamp == nil {
					framework.Logf("deletion has not yet been observed")
					return false, nil
				}
				return true, nil
			}
			framework.Logf("no pod exists with the name we were looking for, assuming the termination request was observed and completed")
			return true, nil
		})).NotTo(HaveOccurred(), "kubelet never observed the termination notice")

		By("verifying pod deletion was observed")
		deleted := false
		timeout := false
		var lastPod *v1.Pod
		timer := time.After(30 * time.Second)
		for !deleted && !timeout {
			select {
			case event, _ := <-w.ResultChan():
				if event.Type == watch.Deleted {
					lastPod = event.Object.(*v1.Pod)
					deleted = true
				}
			case <-timer:
				timeout = true
			}
		}
		if !deleted {
			Fail("Failed to observe pod deletion")
		}

		Expect(lastPod.DeletionTimestamp).ToNot(BeNil())
		Expect(lastPod.Spec.TerminationGracePeriodSeconds).ToNot(BeZero())

		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = v1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(0))
	})

	It("should be updated [Conformance]", func() {
		By("creating the pod")
		name := "pod-update-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: "gcr.io/google_containers/nginx-slim:0.7",
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		pod = podClient.CreateSync(pod)

		By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := v1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(1))

		By("updating the pod")
		podClient.Update(name, func(pod *v1.Pod) {
			value = strconv.Itoa(time.Now().Nanosecond())
			pod.Labels["time"] = value
		})

		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("verifying the updated pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = v1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(1))
		framework.Logf("Pod update OK")
	})

	It("should allow activeDeadlineSeconds to be updated [Conformance]", func() {
		By("creating the pod")
		name := "pod-update-activedeadlineseconds-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: "gcr.io/google_containers/nginx-slim:0.7",
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		podClient.CreateSync(pod)

		By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := v1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pods")
		Expect(len(pods.Items)).To(Equal(1))

		By("updating the pod")
		podClient.Update(name, func(pod *v1.Pod) {
			newDeadline := int64(5)
			pod.Spec.ActiveDeadlineSeconds = &newDeadline
		})

		framework.ExpectNoError(f.WaitForPodTerminated(pod.Name, "DeadlineExceeded"))
	})

	It("should contain environment variables for services [Conformance]", func() {
		// Make a pod that will be a service.
		// This pod serves its hostname via HTTP.
		serverName := "server-envvars-" + string(uuid.NewUUID())
		serverPod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name:   serverName,
				Labels: map[string]string{"name": serverName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "srv",
						Image: "gcr.io/google_containers/serve_hostname:v1.4",
						Ports: []v1.ContainerPort{{ContainerPort: 9376}},
					},
				},
			},
		}
		podClient.CreateSync(serverPod)

		// This service exposes port 8080 of the test pod as a service on port 8765
		// TODO(filbranden): We would like to use a unique service name such as:
		//   svcName := "svc-envvars-" + randomSuffix()
		// However, that affects the name of the environment variables which are the capitalized
		// service name, so that breaks this test.  One possibility is to tweak the variable names
		// to match the service.  Another is to rethink environment variable names and possibly
		// allow overriding the prefix in the service manifest.
		svcName := "fooservice"
		svc := &v1.Service{
			ObjectMeta: v1.ObjectMeta{
				Name: svcName,
				Labels: map[string]string{
					"name": svcName,
				},
			},
			Spec: v1.ServiceSpec{
				Ports: []v1.ServicePort{{
					Port:       8765,
					TargetPort: intstr.FromInt(8080),
				}},
				Selector: map[string]string{
					"name": serverName,
				},
			},
		}
		_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(svc)
		Expect(err).NotTo(HaveOccurred(), "failed to create service")

		// Make a client pod that verifies that it has the service environment variables.
		podName := "client-envvars-" + string(uuid.NewUUID())
		const containerName = "env3cont"
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"sh", "-c", "env"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		// It's possible for the Pod to be created before the Kubelet is updated with the new
		// service. In that case, we just retry.
		const maxRetries = 3
		expectedVars := []string{
			"FOOSERVICE_SERVICE_HOST=",
			"FOOSERVICE_SERVICE_PORT=",
			"FOOSERVICE_PORT=",
			"FOOSERVICE_PORT_8765_TCP_PORT=",
			"FOOSERVICE_PORT_8765_TCP_PROTO=",
			"FOOSERVICE_PORT_8765_TCP=",
			"FOOSERVICE_PORT_8765_TCP_ADDR=",
		}
		framework.ExpectNoErrorWithRetries(func() error {
			return f.MatchContainerOutput(pod, containerName, expectedVars, ContainSubstring)
		}, maxRetries, "Container should have service environment variables set")
	})

	It("should support remote command execution over websockets", func() {
		config, err := framework.LoadConfig()
		Expect(err).NotTo(HaveOccurred(), "unable to get base config")

		By("creating the pod")
		name := "pod-exec-websocket-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "main",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 600"},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		pod = podClient.CreateSync(pod)

		req := f.ClientSet.Core().RESTClient().Get().
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
		Eventually(func() error {
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
				return fmt.Errorf("Unexpected output from server")
			}
			if !strings.Contains(buf.String(), "nameserver") {
				return fmt.Errorf("Expected to find 'nameserver' in %q", buf.String())
			}
			return nil
		}, time.Minute, 10*time.Second).Should(BeNil())
	})

	It("should support retrieving logs from the container over websockets", func() {
		config, err := framework.LoadConfig()
		Expect(err).NotTo(HaveOccurred(), "unable to get base config")

		By("creating the pod")
		name := "pod-logs-websocket-" + string(uuid.NewUUID())
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "main",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 10000"},
					},
				},
			},
		}

		By("submitting the pod to kubernetes")
		podClient.CreateSync(pod)

		req := f.ClientSet.Core().RESTClient().Get().
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
			if len(strings.TrimSpace(string(msg))) == 0 {
				continue
			}
			buf.Write(msg)
		}
		if buf.String() != "container is alive\n" {
			framework.Failf("Unexpected websocket logs:\n%s", buf.String())
		}
	})

	It("should have their auto-restart back-off timer reset on image update [Slow]", func() {
		podName := "pod-back-off-image"
		containerName := "back-off"
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "back-off-image"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "sleep 5", "/crash/missing"},
					},
				},
			},
		}

		delay1, delay2 := startPodAndGetBackOffs(podClient, pod, buildBackOffDuration)

		By("updating the image")
		podClient.Update(podName, func(pod *v1.Pod) {
			pod.Spec.Containers[0].Image = "gcr.io/google_containers/nginx-slim:0.7"
		})

		time.Sleep(syncLoopFrequency)
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("get restart delay after image update")
		delayAfterUpdate, err := getRestartDelay(podClient, podName, containerName)
		if err != nil {
			framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
		}

		if delayAfterUpdate > 2*delay2 || delayAfterUpdate > 2*delay1 {
			framework.Failf("updating image did not reset the back-off value in pod=%s/%s d3=%s d2=%s d1=%s", podName, containerName, delayAfterUpdate, delay1, delay2)
		}
	})

	// Slow issue #19027 (20 mins)
	It("should cap back-off at MaxContainerBackOff [Slow]", func() {
		podName := "back-off-cap"
		containerName := "back-off-cap"
		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"/bin/sh", "-c", "sleep 5", "/crash/missing"},
					},
				},
			},
		}

		podClient.CreateSync(pod)
		time.Sleep(2 * kubelet.MaxContainerBackOff) // it takes slightly more than 2*x to get to a back-off of x

		// wait for a delay == capped delay of MaxContainerBackOff
		By("geting restart delay when capped")
		var (
			delay1 time.Duration
			err    error
		)
		for i := 0; i < 3; i++ {
			delay1, err = getRestartDelay(podClient, podName, containerName)
			if err != nil {
				framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
			}

			if delay1 < kubelet.MaxContainerBackOff {
				continue
			}
		}

		if (delay1 < kubelet.MaxContainerBackOff) || (delay1 > maxBackOffTolerance) {
			framework.Failf("expected %s back-off got=%s in delay1", kubelet.MaxContainerBackOff, delay1)
		}

		By("getting restart delay after a capped delay")
		delay2, err := getRestartDelay(podClient, podName, containerName)
		if err != nil {
			framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
		}

		if delay2 < kubelet.MaxContainerBackOff || delay2 > maxBackOffTolerance { // syncloop cumulative drift
			framework.Failf("expected %s back-off got=%s on delay2", kubelet.MaxContainerBackOff, delay2)
		}
	})
})
