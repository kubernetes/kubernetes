/*
Copyright The Kubernetes Authors.

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

package podlogs

import (
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// listCounter counts non-watch "list pods" requests seen by a client, by
// wrapping its transport. That lets the test detect a watch reconnect that
// busy-loops on List calls.
type listCounter struct {
	count atomic.Int64
}

func (l *listCounter) wrap(rt http.RoundTripper) http.RoundTripper {
	return roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Method == http.MethodGet &&
			strings.HasSuffix(req.URL.Path, "/pods") &&
			req.URL.Query().Get("watch") != "true" {
			l.count.Add(1)
		}
		return rt.RoundTrip(req)
	})
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// createRunningPod creates a pod with a single container and then directly
// sets its status to "running", without any kubelet actually running it.
// That is good enough for exercising CopyPodLogs's pod watching logic, even
// though the container's logs cannot actually be retrieved (there is no real
// kubelet or container to get them from).
func createRunningPod(tCtx ktesting.TContext, cs kubernetes.Interface, ns, name string) {
	tCtx.Helper()

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: corev1.PodSpec{
			NodeName: "fake-node",
			Containers: []corev1.Container{
				{
					Name:  "the-container",
					Image: "does-not-matter:latest",
				},
			},
		},
	}
	created, err := cs.CoreV1().Pods(ns).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("create pod %s: %v", name, err)
	}

	created.Status = corev1.PodStatus{
		Phase: corev1.PodRunning,
		ContainerStatuses: []corev1.ContainerStatus{
			{
				Name:        "the-container",
				ContainerID: "fake://" + name,
				State: corev1.ContainerState{
					Running: &corev1.ContainerStateRunning{StartedAt: metav1.Now()},
				},
			},
		},
	}
	if _, err := cs.CoreV1().Pods(ns).UpdateStatus(tCtx, created, metav1.UpdateOptions{}); err != nil {
		tCtx.Fatalf("update status of pod %s: %v", name, err)
	}
}

// TestCopyPodLogsAPIServerTemporarilyDown simulates the apiserver
// repeatedly dropping podlogs.CopyAllLogs's Pods watch, as happens during a
// connectivity blip or an apiserver restart, and verifies that:
//   - it does not busy-loop while reconnecting, and
//   - it still picks up on new pods.
//
// This reproduces the scenario described in
// https://github.com/kubernetes/kubernetes/pull/142026: a broken watch
// causing a burst of List calls against the apiserver.
//
// The apiserver is configured with a very short --min-request-timeout so
// that it closes watches on its own every 1-2s. That repeatedly triggers
// the exact code path a connectivity blip would (the watch's result channel
// closing), quickly and deterministically, without having to simulate an
// actual network outage (which, tried directly, turned out to leave the
// client's HTTP/2 transport in a state that took a long time to be
// garbage-collected, without actually exercising the reconnect logic any
// better).
func TestCopyPodLogsAPIServerTemporarilyDown(t *testing.T) {
	tCtx := ktesting.Init(t)

	flags := append(append([]string{}, framework.DefaultTestServerFlags()...), "--min-request-timeout=1")
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	defer server.TearDownFn()

	var lists listCounter
	clientConfig := restclient.CopyConfig(server.ClientConfig)
	clientConfig.WrapTransport = lists.wrap
	clientset, err := kubernetes.NewForConfig(clientConfig)
	tCtx.ExpectNoError(err, "create client")

	ns := framework.CreateNamespaceOrDie(clientset, "podlogs-outage", t)
	defer framework.DeleteNamespaceOrDie(clientset, ns, t)

	createRunningPod(tCtx, clientset, ns.Name, "pod-a")

	seen := make(chan string, 10)
	to := podlogs.LogOutput{
		LogOpen: func(podName, containerName string) io.Writer {
			select {
			case <-tCtx.Done():
			case seen <- podName:
			}
			return io.Discard
		},
	}
	copyCtx := tCtx.WithCancel()
	defer copyCtx.Cancel("stopping log analysis")
	tCtx.ExpectNoError(podlogs.CopyAllLogs(copyCtx, clientset, ns.Name, to), "CopyAllLogs")

	waitForPodLog := func(want string) {
		tCtx.Helper()
		deadline := time.After(30 * time.Second)
		for {
			select {
			case podName := <-seen:
				if podName == want {
					return
				}
			case <-deadline:
				tCtx.Fatalf("timed out waiting for logs to be opened for pod %q", want)
			}
		}
	}
	waitForPodLog("pod-a")

	// The apiserver closes watches on its own every 1-2s (--min-request-timeout=1).
	// Without rate-limited reconnects, each such closure causes a busy loop
	// of List calls (bounded only by the client's QPS limiter, i.e. several
	// per second, forever). A properly rate-limited watch reconnects a
	// handful of times at most and only calls List again for actual pod
	// events, none of which happen here.
	const quiet = 5 * time.Second
	listsBefore := lists.count.Load()
	time.Sleep(quiet)
	listsDuringQuiet := lists.count.Load() - listsBefore
	tCtx.Logf("list calls while quiet for %s (watches keep getting closed by the apiserver): %d", quiet, listsDuringQuiet)
	if max := int64(3); listsDuringQuiet > max {
		tCtx.Fatalf("got %d list calls while quiet for %s, expected at most %d: watch reconnects are not being rate-limited (busy loop)", listsDuringQuiet, quiet, max)
	}

	// If the pod watch keeps recovering correctly from these repeated
	// drops, a new pod should still be noticed.
	createRunningPod(tCtx, clientset, ns.Name, "pod-b")
	waitForPodLog("pod-b")
}
