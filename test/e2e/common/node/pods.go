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

package node

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"k8s.io/client-go/util/retry"

	"golang.org/x/net/websocket"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubectl/pkg/util/podutils"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2ewebsocket "k8s.io/kubernetes/test/e2e/framework/websocket"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	buildBackOffDuration = time.Minute
	syncLoopFrequency    = 10 * time.Second
	maxBackOffTolerance  = time.Duration(1.3 * float64(kubelet.MaxContainerBackOff))
	podRetryPeriod       = 1 * time.Second
)

// testHostIP tests that a pod gets a host IP
func testHostIP(ctx context.Context, podClient *e2epod.PodClient, pod *v1.Pod) {
	ginkgo.By("creating pod")
	podClient.CreateSync(ctx, pod)

	// Try to make sure we get a hostIP for each pod.
	hostIPTimeout := 2 * time.Minute
	t := time.Now()
	for {
		p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get pod %q", pod.Name)
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

func startPodAndGetBackOffs(ctx context.Context, podClient *e2epod.PodClient, pod *v1.Pod, sleepAmount time.Duration) (time.Duration, time.Duration) {
	podClient.CreateSync(ctx, pod)
	time.Sleep(sleepAmount)
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())
	podName := pod.Name
	containerName := pod.Spec.Containers[0].Name

	ginkgo.By("getting restart delay-0")
	_, err := getRestartDelay(ctx, podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}

	ginkgo.By("getting restart delay-1")
	delay1, err := getRestartDelay(ctx, podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}

	ginkgo.By("getting restart delay-2")
	delay2, err := getRestartDelay(ctx, podClient, podName, containerName)
	if err != nil {
		framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
	}
	return delay1, delay2
}

func getRestartDelay(ctx context.Context, podClient *e2epod.PodClient, podName string, containerName string) (time.Duration, error) {
	beginTime := time.Now()
	var previousRestartCount int32 = -1
	var previousFinishedAt time.Time
	for time.Since(beginTime) < (2 * maxBackOffTolerance) { // may just miss the 1st MaxContainerBackOff delay
		time.Sleep(time.Second)
		pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
		framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podName))
		status, ok := podutil.GetContainerStatus(pod.Status.ContainerStatuses, containerName)
		if !ok {
			framework.Logf("getRestartDelay: status missing")
			continue
		}

		// the only case this happens is if this is the first time the Pod is running and there is no "Last State".
		if status.LastTerminationState.Terminated == nil {
			framework.Logf("Container's last state is not \"Terminated\".")
			continue
		}

		if previousRestartCount == -1 {
			if status.State.Running != nil {
				// container is still Running, there is no "FinishedAt" time.
				continue
			} else if status.State.Terminated != nil {
				previousFinishedAt = status.State.Terminated.FinishedAt.Time
			} else {
				previousFinishedAt = status.LastTerminationState.Terminated.FinishedAt.Time
			}
			previousRestartCount = status.RestartCount
		}

		// when the RestartCount is changed, the Containers will be in one of the following states:
		//Running, Terminated, Waiting (it already is waiting for the backoff period to expire, and the last state details have been stored into status.LastTerminationState).
		if status.RestartCount > previousRestartCount {
			var startedAt time.Time
			if status.State.Running != nil {
				startedAt = status.State.Running.StartedAt.Time
			} else if status.State.Terminated != nil {
				startedAt = status.State.Terminated.StartedAt.Time
			} else {
				startedAt = status.LastTerminationState.Terminated.StartedAt.Time
			}
			framework.Logf("getRestartDelay: restartCount = %d, finishedAt=%s restartedAt=%s (%s)", status.RestartCount, previousFinishedAt, startedAt, startedAt.Sub(previousFinishedAt))
			return startedAt.Sub(previousFinishedAt), nil
		}
	}
	return 0, fmt.Errorf("timeout getting pod restart delay")
}

// expectNoErrorWithRetries checks if an error occurs with the given retry count.
func expectNoErrorWithRetries(fn func() error, maxRetries int, explain ...interface{}) {
	// TODO (pohly): replace the entire function with gomege.Eventually.
	var err error
	for i := 0; i < maxRetries; i++ {
		err = fn()
		if err == nil {
			return
		}
		framework.Logf("(Attempt %d of %d) Unexpected error occurred: %v", i+1, maxRetries, err)
	}
	if err != nil {
		debug.PrintStack()
	}
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), explain...)
}

var _ = SIGDescribe("Pods", func() {
	f := framework.NewDefaultFramework("pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelRestricted
	var podClient *e2epod.PodClient
	var dc dynamic.Interface

	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
		dc = f.DynamicClient
	})

	/*
		Release: v1.9
		Testname: Pods, assigned hostip
		Description: Create a Pod. Pod status MUST return successfully and contains a valid IP address.
	*/
	framework.ConformanceIt("should get a host IP", f.WithNodeConformance(), func(ctx context.Context) {
		name := "pod-hostip-" + string(uuid.NewUUID())
		testHostIP(ctx, podClient, e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "test",
						Image: imageutils.GetPauseImageName(),
					},
				},
			},
		}))
	})

	/*
		Release: v1.9
		Testname: Pods, lifecycle
		Description: A Pod is created with a unique label. Pod MUST be accessible when queried using the label selector upon creation. Add a watch, check if the Pod is running. Pod then deleted, The pod deletion timestamp is observed. The watch MUST return the pod deleted event. Query with the original selector for the Pod MUST return empty list.
	*/
	framework.ConformanceIt("should be submitted and removed", f.WithNodeConformance(), func(ctx context.Context) {
		ginkgo.By("creating the pod")
		name := "pod-submit-remove-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: imageutils.GetPauseImageName(),
					},
				},
			},
		})

		ginkgo.By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.BeEmpty())

		lw := &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = selector.String()
				podList, err := podClient.List(ctx, options)
				return podList, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = selector.String()
				return podClient.Watch(ctx, options)
			},
		}
		_, informer, w, _ := watchtools.NewIndexerInformerWatcher(lw, &v1.Pod{})
		defer w.Stop()

		ctxUntil, cancelCtx := context.WithTimeout(ctx, wait.ForeverTestTimeout)
		defer cancelCtx()
		if !cache.WaitForCacheSync(ctxUntil.Done(), informer.HasSynced) {
			framework.Failf("Timeout while waiting to Pod informer to sync")
		}

		ginkgo.By("submitting the pod to kubernetes")
		podClient.Create(ctx, pod)

		ginkgo.By("verifying the pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = metav1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))

		ginkgo.By("verifying pod creation was observed")
		select {
		case event := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe pod creation: %v", event)
			}
		case <-time.After(framework.PodStartTimeout):
			framework.Failf("Timeout while waiting for pod creation")
		}

		// We need to wait for the pod to be running, otherwise the deletion
		// may be carried out immediately rather than gracefully.
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
		// save the running pod
		pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to GET scheduled pod")

		ginkgo.By("deleting the pod gracefully")
		err = podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err, "failed to delete pod")

		ginkgo.By("verifying pod deletion was observed")
		deleted := false
		var lastPod *v1.Pod
		timer := time.After(e2epod.DefaultPodDeletionTimeout)
		for !deleted {
			select {
			case event := <-w.ResultChan():
				switch event.Type {
				case watch.Deleted:
					lastPod = event.Object.(*v1.Pod)
					deleted = true
				case watch.Error:
					framework.Logf("received a watch error: %v", event.Object)
					framework.Failf("watch closed with error")
				}
			case <-timer:
				framework.Failf("timed out waiting for pod deletion")
			}
		}
		if !deleted {
			framework.Failf("Failed to observe pod deletion")
		}

		gomega.Expect(lastPod.DeletionTimestamp).ToNot(gomega.BeNil())
		gomega.Expect(lastPod.Spec.TerminationGracePeriodSeconds).ToNot(gomega.BeZero())

		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = metav1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.BeEmpty())
	})

	/*
		Release: v1.9
		Testname: Pods, update
		Description: Create a Pod with a unique label. Query for the Pod with the label as selector MUST be successful. Update the pod to change the value of the Label. Query for the Pod with the new value for the label MUST be successful.
	*/
	framework.ConformanceIt("should be updated", f.WithNodeConformance(), func(ctx context.Context) {
		ginkgo.By("creating the pod")
		name := "pod-update-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: imageutils.GetPauseImageName(),
					},
				},
			},
		})

		ginkgo.By("submitting the pod to kubernetes")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))

		ginkgo.By("updating the pod")
		podClient.Update(ctx, name, func(pod *v1.Pod) {
			value = strconv.Itoa(time.Now().Nanosecond())
			pod.Labels["time"] = value
		})

		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

		ginkgo.By("verifying the updated pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = metav1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))
		framework.Logf("Pod update OK")
	})

	/*
		Release: v1.9
		Testname: Pods, ActiveDeadlineSeconds
		Description: Create a Pod with a unique label. Query for the Pod with the label as selector MUST be successful. The Pod is updated with ActiveDeadlineSeconds set on the Pod spec. Pod MUST terminate of the specified time elapses.
	*/
	framework.ConformanceIt("should allow activeDeadlineSeconds to be updated", f.WithNodeConformance(), func(ctx context.Context) {
		ginkgo.By("creating the pod")
		name := "pod-update-activedeadlineseconds-" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": "foo",
					"time": value,
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: imageutils.GetPauseImageName(),
					},
				},
			},
		})

		ginkgo.By("submitting the pod to kubernetes")
		podClient.CreateSync(ctx, pod)

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set{"time": value})
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(ctx, options)
		framework.ExpectNoError(err, "failed to query for pods")
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))

		ginkgo.By("updating the pod")
		podClient.Update(ctx, name, func(pod *v1.Pod) {
			newDeadline := int64(5)
			pod.Spec.ActiveDeadlineSeconds = &newDeadline
		})

		framework.ExpectNoError(e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "DeadlineExceeded", f.Namespace.Name))
	})

	/*
		Release: v1.9
		Testname: Pods, service environment variables
		Description: Create a server Pod listening on port 9376. A Service called fooservice is created for the server Pod listening on port 8765 targeting port 8080. If a new Pod is created in the cluster then the Pod MUST have the fooservice environment variables available from this new Pod. The new create Pod MUST have environment variables such as FOOSERVICE_SERVICE_HOST, FOOSERVICE_SERVICE_PORT, FOOSERVICE_PORT, FOOSERVICE_PORT_8765_TCP_PORT, FOOSERVICE_PORT_8765_TCP_PROTO, FOOSERVICE_PORT_8765_TCP and FOOSERVICE_PORT_8765_TCP_ADDR that are populated with proper values.
	*/
	framework.ConformanceIt("should contain environment variables for services", f.WithNodeConformance(), func(ctx context.Context) {
		// Make a pod that will be a service.
		// This pod serves its hostname via HTTP.
		serverName := "server-envvars-" + string(uuid.NewUUID())
		serverPod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   serverName,
				Labels: map[string]string{"name": serverName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "srv",
						Image: framework.ServeHostnameImage,
						Ports: []v1.ContainerPort{{ContainerPort: 9376}},
					},
				},
			},
		})
		podClient.CreateSync(ctx, serverPod)

		// This service exposes port 8080 of the test pod as a service on port 8765
		// TODO(filbranden): We would like to use a unique service name such as:
		//   svcName := "svc-envvars-" + randomSuffix()
		// However, that affects the name of the environment variables which are the capitalized
		// service name, so that breaks this test.  One possibility is to tweak the variable names
		// to match the service.  Another is to rethink environment variable names and possibly
		// allow overriding the prefix in the service manifest.
		svcName := "fooservice"
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: svcName,
				Labels: map[string]string{
					"name": svcName,
				},
			},
			Spec: v1.ServiceSpec{
				Ports: []v1.ServicePort{{
					Port:       8765,
					TargetPort: intstr.FromInt32(8080),
				}},
				Selector: map[string]string{
					"name": serverName,
				},
			},
		}
		_, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create service")

		// Make a client pod that verifies that it has the service environment variables.
		podName := "client-envvars-" + string(uuid.NewUUID())
		const containerName = "env3cont"
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": podName},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		})

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
		expectNoErrorWithRetries(func() error {
			return e2epodoutput.MatchContainerOutput(ctx, f, pod, containerName, expectedVars, gomega.ContainSubstring)
		}, maxRetries, "Container should have service environment variables set")
	})

	/*
		Release: v1.13
		Testname: Pods, remote command execution over websocket
		Description: A Pod is created. Websocket is created to retrieve exec command output from this pod.
		Message retrieved form Websocket MUST match with expected exec command output.
	*/
	framework.ConformanceIt("should support remote command execution over websockets", f.WithNodeConformance(), func(ctx context.Context) {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "unable to get base config")

		ginkgo.By("creating the pod")
		name := "pod-exec-websocket-" + string(uuid.NewUUID())
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "main",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 600"},
					},
				},
			},
		})

		ginkgo.By("submitting the pod to kubernetes")
		pod = podClient.CreateSync(ctx, pod)

		req := f.ClientSet.CoreV1().RESTClient().Get().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			Suffix("exec").
			Param("stderr", "1").
			Param("stdout", "1").
			Param("container", pod.Spec.Containers[0].Name).
			Param("command", "echo").
			Param("command", "remote execution test")

		url := req.URL()
		ws, err := e2ewebsocket.OpenWebSocketForURL(url, config, []string{"channel.k8s.io"})
		if err != nil {
			framework.Failf("Failed to open websocket to %s: %v", url.String(), err)
		}
		defer ws.Close()

		buf := &bytes.Buffer{}
		gomega.Eventually(ctx, func() error {
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
					if len(msg) == 1 {
						// skip an empty message on stream other than stdout
						continue
					} else {
						framework.Failf("Got message from server that didn't start with channel 1 (STDOUT): %v", msg)
					}

				}
				buf.Write(msg[1:])
			}
			if buf.Len() == 0 {
				return fmt.Errorf("unexpected output from server")
			}
			if !strings.Contains(buf.String(), "remote execution test") {
				return fmt.Errorf("expected to find 'remote execution test' in %q", buf.String())
			}
			return nil
		}, time.Minute, 10*time.Second).Should(gomega.BeNil())
	})

	/*
		Release: v1.13
		Testname: Pods, logs from websockets
		Description: A Pod is created. Websocket is created to retrieve log of a container from this pod.
		Message retrieved form Websocket MUST match with container's output.
	*/
	framework.ConformanceIt("should support retrieving logs from the container over websockets", f.WithNodeConformance(), func(ctx context.Context) {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "unable to get base config")

		ginkgo.By("creating the pod")
		name := "pod-logs-websocket-" + string(uuid.NewUUID())
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "main",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 10000"},
					},
				},
			},
		})

		ginkgo.By("submitting the pod to kubernetes")
		podClient.CreateSync(ctx, pod)

		req := f.ClientSet.CoreV1().RESTClient().Get().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			Suffix("log").
			Param("container", pod.Spec.Containers[0].Name)

		url := req.URL()

		ws, err := e2ewebsocket.OpenWebSocketForURL(url, config, []string{"binary.k8s.io"})
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

	// Slow (~7 mins)
	f.It("should have their auto-restart back-off timer reset on image update", f.WithSlow(), f.WithNodeConformance(), func(ctx context.Context) {
		podName := "pod-back-off-image"
		containerName := "back-off"
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "back-off-image"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", "sleep 5", "/crash/missing"},
					},
				},
			},
		})

		delay1, delay2 := startPodAndGetBackOffs(ctx, podClient, pod, buildBackOffDuration)

		ginkgo.By("updating the image")
		podClient.Update(ctx, podName, func(pod *v1.Pod) {
			pod.Spec.Containers[0].Image = imageutils.GetE2EImage(imageutils.Nginx)
		})

		time.Sleep(syncLoopFrequency)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

		ginkgo.By("get restart delay after image update")
		delayAfterUpdate, err := getRestartDelay(ctx, podClient, podName, containerName)
		if err != nil {
			framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
		}

		if delayAfterUpdate > 2*delay2 || delayAfterUpdate > 2*delay1 {
			framework.Failf("updating image did not reset the back-off value in pod=%s/%s d3=%s d2=%s d1=%s", podName, containerName, delayAfterUpdate, delay1, delay2)
		}
	})

	// Slow by design (~27 mins) issue #19027
	f.It("should cap back-off at MaxContainerBackOff", f.WithSlow(), f.WithNodeConformance(), func(ctx context.Context) {
		podName := "back-off-cap"
		containerName := "back-off-cap"
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "liveness"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", "sleep 5", "/crash/missing"},
					},
				},
			},
		})

		podClient.CreateSync(ctx, pod)
		time.Sleep(2 * kubelet.MaxContainerBackOff) // it takes slightly more than 2*x to get to a back-off of x

		// wait for a delay == capped delay of MaxContainerBackOff
		ginkgo.By("getting restart delay when capped")
		var (
			delay1 time.Duration
			err    error
		)
		for i := 0; i < 3; i++ {
			delay1, err = getRestartDelay(ctx, podClient, podName, containerName)
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

		ginkgo.By("getting restart delay after a capped delay")
		delay2, err := getRestartDelay(ctx, podClient, podName, containerName)
		if err != nil {
			framework.Failf("timed out waiting for container restart in pod=%s/%s", podName, containerName)
		}

		if delay2 < kubelet.MaxContainerBackOff || delay2 > maxBackOffTolerance { // syncloop cumulative drift
			framework.Failf("expected %s back-off got=%s on delay2", kubelet.MaxContainerBackOff, delay2)
		}
	})

	f.It("should support pod readiness gates", f.WithNodeConformance(), func(ctx context.Context) {
		podName := "pod-ready"
		readinessGate1 := "k8s.io/test-condition1"
		readinessGate2 := "k8s.io/test-condition2"
		patchStatusFmt := `{"status":{"conditions":[{"type":%q, "status":%q}]}}`
		pod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"test": "pod-readiness-gate"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "pod-readiness-gate",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh", "-c", "echo container is alive; sleep 10000"},
					},
				},
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType(readinessGate1)},
					{ConditionType: v1.PodConditionType(readinessGate2)},
				},
			},
		})

		validatePodReadiness := func(expectReady bool) {
			err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				podReady := podutils.IsPodReady(pod)
				res := expectReady == podReady
				if !res {
					framework.Logf("Expect the Ready condition of pod %q to be %v, but got %v (pod status %#v)", podName, expectReady, podReady, pod.Status)
				}
				return res, nil
			})
			framework.ExpectNoError(err)
		}

		ginkgo.By("submitting the pod to kubernetes")
		e2epod.NewPodClient(f).Create(ctx, pod)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
		if podClient.PodIsReady(ctx, podName) {
			framework.Failf("Expect pod(%s/%s)'s Ready condition to be false initially.", f.Namespace.Name, pod.Name)
		}

		ginkgo.By(fmt.Sprintf("patching pod status with condition %q to true", readinessGate1))
		_, err := podClient.Patch(ctx, podName, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate1, "True")), metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		// Sleep for 10 seconds.
		time.Sleep(syncLoopFrequency)
		// Verify the pod is still not ready
		if podClient.PodIsReady(ctx, podName) {
			framework.Failf("Expect pod(%s/%s)'s Ready condition to be false with only one condition in readinessGates equal to True", f.Namespace.Name, pod.Name)
		}

		ginkgo.By(fmt.Sprintf("patching pod status with condition %q to true", readinessGate2))
		_, err = podClient.Patch(ctx, podName, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate2, "True")), metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		validatePodReadiness(true)

		ginkgo.By(fmt.Sprintf("patching pod status with condition %q to false", readinessGate1))
		_, err = podClient.Patch(ctx, podName, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate1, "False")), metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		validatePodReadiness(false)

	})

	/*
		Release: v1.19
		Testname: Pods, delete a collection
		Description: A set of pods is created with a label selector which MUST be found when listed.
		The set of pods is deleted and MUST NOT show up when listed by its label selector.
	*/
	framework.ConformanceIt("should delete a collection of pods", func(ctx context.Context) {
		podTestNames := []string{"test-pod-1", "test-pod-2", "test-pod-3"}

		one := int64(1)

		ginkgo.By("Create set of pods")
		// create a set of pods in test namespace
		for _, podTestName := range podTestNames {
			_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx,
				e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podTestName,
						Labels: map[string]string{
							"type": "Testing",
						},
					},
					Spec: v1.PodSpec{
						TerminationGracePeriodSeconds: &one,
						Containers: []v1.Container{{
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Name:  "token-test",
						}},
						RestartPolicy: v1.RestartPolicyNever,
					}}), metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod")
			framework.Logf("created %v", podTestName)
		}

		// wait as required for all 3 pods to be running
		ginkgo.By("waiting for all 3 pods to be running")
		err := e2epod.WaitForPodsRunningReady(ctx, f.ClientSet, f.Namespace.Name, 3, 0, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "3 pods not found running.")

		// delete Collection of pods with a label in the current namespace
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).DeleteCollection(ctx, metav1.DeleteOptions{GracePeriodSeconds: &one}, metav1.ListOptions{
			LabelSelector: "type=Testing"})
		framework.ExpectNoError(err, "failed to delete collection of pods")

		// wait for all pods to be deleted
		ginkgo.By("waiting for all pods to be deleted")
		err = wait.PollUntilContextTimeout(ctx, podRetryPeriod, f.Timeouts.PodDelete, true, checkPodListQuantity(f, "type=Testing", 0))
		framework.ExpectNoError(err, "found a pod(s)")
	})

	/*
		Release: v1.20
		Testname: Pods, completes the lifecycle of a Pod and the PodStatus
		Description: A Pod is created with a static label which MUST succeed. It MUST succeed when
		patching the label and the pod data. When checking and replacing the PodStatus it MUST
		succeed. It MUST succeed when deleting the Pod.
	*/
	framework.ConformanceIt("should run through the lifecycle of Pods and PodStatus", func(ctx context.Context) {
		podResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
		testNamespaceName := f.Namespace.Name
		testPodName := "pod-test"
		testPodImage := imageutils.GetE2EImage(imageutils.Agnhost)
		testPodImage2 := imageutils.GetE2EImage(imageutils.Httpd)
		testPodLabels := map[string]string{"test-pod-static": "true"}
		testPodLabelsFlat := "test-pod-static=true"
		one := int64(1)

		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = testPodLabelsFlat
				return f.ClientSet.CoreV1().Pods(testNamespaceName).Watch(ctx, options)
			},
		}
		podsList, err := f.ClientSet.CoreV1().Pods("").List(ctx, metav1.ListOptions{LabelSelector: testPodLabelsFlat})
		framework.ExpectNoError(err, "failed to list Pods")

		testPod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   testPodName,
				Labels: testPodLabels,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: &one,
				Containers: []v1.Container{
					{
						Name:  testPodName,
						Image: testPodImage,
					},
				},
			},
		})
		ginkgo.By("creating a Pod with a static label")
		_, err = f.ClientSet.CoreV1().Pods(testNamespaceName).Create(ctx, testPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Pod %v in namespace %v", testPod.ObjectMeta.Name, testNamespaceName)

		ginkgo.By("watching for Pod to be ready")
		ctxUntil, cancel := context.WithTimeout(ctx, f.Timeouts.PodStart)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, podsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if pod, ok := event.Object.(*v1.Pod); ok {
				found := pod.ObjectMeta.Name == testPod.ObjectMeta.Name &&
					pod.ObjectMeta.Namespace == testNamespaceName &&
					pod.Labels["test-pod-static"] == "true" &&
					pod.Status.Phase == v1.PodRunning
				if !found {
					framework.Logf("observed Pod %v in namespace %v in phase %v with labels: %v & conditions %v", pod.ObjectMeta.Name, pod.ObjectMeta.Namespace, pod.Status.Phase, pod.Labels, pod.Status.Conditions)
					return false, nil
				}
				framework.Logf("Found Pod %v in namespace %v in phase %v with labels: %v & conditions %v", pod.ObjectMeta.Name, pod.ObjectMeta.Namespace, pod.Status.Phase, pod.Labels, pod.Status.Conditions)
				return found, nil
			}
			framework.Logf("Observed event: %+v", event.Object)
			return false, nil
		})
		if err != nil {
			framework.Logf("failed to see event that pod is created: %v", err)
		}
		p, err := f.ClientSet.CoreV1().Pods(testNamespaceName).Get(ctx, testPodName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get Pod %v in namespace %v", testPodName, testNamespaceName)
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodRunning), "failed to see Pod %v in namespace %v running", p.ObjectMeta.Name, testNamespaceName)

		ginkgo.By("patching the Pod with a new Label and updated data")
		prePatchResourceVersion := p.ResourceVersion
		podPatch, err := json.Marshal(v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{"test-pod": "patched"},
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: &one,
				Containers: []v1.Container{{
					Name:  testPodName,
					Image: testPodImage2,
				}},
			},
		})
		framework.ExpectNoError(err, "failed to marshal JSON patch for Pod")
		_, err = f.ClientSet.CoreV1().Pods(testNamespaceName).Patch(ctx, testPodName, types.StrategicMergePatchType, []byte(podPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Pod %s in namespace %s", testPodName, testNamespaceName)
		ctxUntil, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, prePatchResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if pod, ok := event.Object.(*v1.Pod); ok {
					found := pod.ObjectMeta.Name == pod.Name &&
						pod.Labels["test-pod-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		if err != nil {
			framework.Logf("failed to see %v event: %v", watch.Modified, err)
		}

		ginkgo.By("getting the Pod and ensuring that it's patched")
		pod, err := f.ClientSet.CoreV1().Pods(testNamespaceName).Get(ctx, testPodName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Pod %s in namespace %s", testPodName, testNamespaceName)
		gomega.Expect(pod.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("test-pod", "patched"), "failed to patch Pod - missing label")
		gomega.Expect(pod.Spec.Containers[0].Image).To(gomega.Equal(testPodImage2), "failed to patch Pod - wrong image")

		ginkgo.By("replacing the Pod's status Ready condition to False")
		var podStatusUpdate *v1.Pod

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			podStatusUnstructured, err := dc.Resource(podResource).Namespace(testNamespaceName).Get(ctx, testPodName, metav1.GetOptions{}, "status")
			framework.ExpectNoError(err, "failed to fetch PodStatus of Pod %s in namespace %s", testPodName, testNamespaceName)
			podStatusBytes, err := json.Marshal(podStatusUnstructured)
			framework.ExpectNoError(err, "failed to marshal unstructured response")
			var podStatus v1.Pod
			err = json.Unmarshal(podStatusBytes, &podStatus)
			framework.ExpectNoError(err, "failed to unmarshal JSON bytes to a Pod object type")
			podStatusUpdated := podStatus
			podStatusFieldPatchCount := 0
			podStatusFieldPatchCountTotal := 2
			for pos, cond := range podStatusUpdated.Status.Conditions {
				if (cond.Type == v1.PodReady && cond.Status == v1.ConditionTrue) || (cond.Type == v1.ContainersReady && cond.Status == v1.ConditionTrue) {
					podStatusUpdated.Status.Conditions[pos].Status = v1.ConditionFalse
					podStatusFieldPatchCount++
				}
			}
			gomega.Expect(podStatusFieldPatchCount).To(gomega.Equal(podStatusFieldPatchCountTotal), "failed to patch all relevant Pod conditions")
			podStatusUpdate, err = f.ClientSet.CoreV1().Pods(testNamespaceName).UpdateStatus(ctx, &podStatusUpdated, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update PodStatus of Pod %s in namespace %s", testPodName, testNamespaceName)

		ginkgo.By("check the Pod again to ensure its Ready conditions are False")
		podStatusFieldPatchCount := 0
		podStatusFieldPatchCountTotal := 2
		for _, cond := range podStatusUpdate.Status.Conditions {
			if (cond.Type == v1.PodReady && cond.Status == v1.ConditionFalse) || (cond.Type == v1.ContainersReady && cond.Status == v1.ConditionFalse) {
				podStatusFieldPatchCount++
			}
		}
		gomega.Expect(podStatusFieldPatchCount).To(gomega.Equal(podStatusFieldPatchCountTotal), "failed to update PodStatus - field patch count doesn't match the total")

		ginkgo.By("deleting the Pod via a Collection with a LabelSelector")
		preDeleteResourceVersion := podStatusUpdate.ResourceVersion
		err = f.ClientSet.CoreV1().Pods(testNamespaceName).DeleteCollection(ctx, metav1.DeleteOptions{GracePeriodSeconds: &one}, metav1.ListOptions{LabelSelector: testPodLabelsFlat})
		framework.ExpectNoError(err, "failed to delete Pod by collection")

		ginkgo.By("watching for the Pod to be deleted")
		ctxUntil, cancel = context.WithTimeout(ctx, f.Timeouts.PodDelete)
		defer cancel()
		_, err = watchtools.Until(ctxUntil, preDeleteResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				if pod, ok := event.Object.(*v1.Pod); ok {
					found := pod.ObjectMeta.Name == pod.Name &&
						pod.Labels["test-pod-static"] == "true"
					return found, nil
				} else {
					framework.Logf("observed event type %v that was not a pod: %T", event.Type, event.Object)
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		if err != nil {
			framework.Logf("failed to see %v event: %v", watch.Deleted, err)
		}
		postDeletePod, err := f.ClientSet.CoreV1().Pods(testNamespaceName).Get(ctx, testPodName, metav1.GetOptions{})
		var postDeletePodJSON []byte
		if postDeletePod != nil {
			postDeletePodJSON, _ = json.Marshal(postDeletePod)
		}
		gomega.Expect(err).To(gomega.HaveOccurred(), "pod %v found in namespace %v, but it should be deleted: %s", testPodName, testNamespaceName, string(postDeletePodJSON))
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected IsNotFound error, got %#v", err)
		}
	})

	/*
		Release: v1.25
		Testname: Pods, patching status
		Description: A pod is created which MUST succeed
		and be found running. The pod status when patched
		MUST succeed. Given the patching of the pod status,
		the fields MUST equal the new values.
	*/
	framework.ConformanceIt("should patch a pod status", func(ctx context.Context) {
		ns := f.Namespace.Name
		podClient := f.ClientSet.CoreV1().Pods(ns)
		podName := "pod-" + utilrand.String(5)
		label := map[string]string{"e2e": podName}

		ginkgo.By("Create a pod")
		testPod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   podName,
				Labels: label,
			},
			Spec: v1.PodSpec{
				TerminationGracePeriodSeconds: pointer.Int64(1),
				Containers: []v1.Container{
					{
						Name:  "agnhost",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		})
		pod, err := podClient.Create(ctx, testPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Pod %v in namespace %v", testPod.ObjectMeta.Name, ns)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "Pod didn't start within time out period")

		ginkgo.By("patching /status")
		podStatus := v1.PodStatus{
			Message: "Patched by e2e test",
			Reason:  "E2E",
		}
		pStatusJSON, err := json.Marshal(podStatus)
		framework.ExpectNoError(err, "Failed to marshal. %v", podStatus)

		pStatus, err := podClient.Patch(ctx, podName, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":`+string(pStatusJSON)+`}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err, "failed to patch pod: %q", podName)
		gomega.Expect(pStatus.Status.Message).To(gomega.Equal("Patched by e2e test"), "Status.Message for %q was %q but expected it to be \"Patched by e2e test\"", podName, pStatus.Status.Message)
		gomega.Expect(pStatus.Status.Reason).To(gomega.Equal("E2E"), "Status.Reason for %q was %q but expected it to be \"E2E\"", podName, pStatus.Status.Reason)
		framework.Logf("Status Message: %q and Reason: %q", pStatus.Status.Message, pStatus.Status.Reason)
	})
})

func checkPodListQuantity(f *framework.Framework, label string, quantity int) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		var err error

		list, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: label})

		if err != nil {
			return false, err
		}

		if len(list.Items) != quantity {
			framework.Logf("Pod quantity %d is different from expected quantity %d", len(list.Items), quantity)
			return false, err
		}
		return true, nil
	}
}
