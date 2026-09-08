/*
Copyright 2026 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/common/model"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// A watch client that stops reading its response body cannot be distinguished
// from a healthy one by any pre-existing metric: the apiserver keeps encoding
// events and simply blocks in the transport. This test drives exactly that,
// all the way to the cacher terminating the watcher, and asserts that the new
// per-stage histograms attribute the stall to the transport rather than to
// serialization.
//
// Note the client must eventually resume reading. The duration histograms are
// observed when a write completes, so a stall that never ends is never
// recorded.
func TestWatchTransportStallMetrics(t *testing.T) {
	testCases := []struct {
		name string
		// annotationLen pads each pod so a stall can be provoked with few events.
		annotationLen int
		podCount      int
		streamWindow  int
		// expectStage is the histogram that should hold most of the transport
		// time. Objects larger than HTTP/2's 4KiB write buffer push bytes to the
		// wire from inside Write; smaller ones only do so on Flush.
		expectStage string
	}{
		{
			name:          "large events stall in write",
			annotationLen: 128 << 10,
			podCount:      100,
			expectStage:   "apiserver_watch_event_write_duration_seconds",
		},
		{
			name:         "realistically sized events stall in flush",
			podCount:     200,
			streamWindow: 64 << 10,
			expectStage:  "apiserver_watch_event_flush_duration_seconds",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			runWatchTransportStall(t, tc.annotationLen, tc.podCount, tc.streamWindow, tc.expectStage)
		})
	}
}

func runWatchTransportStall(t *testing.T, annotationLen, podCount, streamWindow int, expectStage string) {
	tCtx := ktesting.Init(t)

	clientSet, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{})
	defer tearDownFn()

	// The default client rate limit would make generating the load below take
	// tens of seconds.
	loadConfig := restclient.CopyConfig(kubeConfig)
	loadConfig.QPS = 1000
	loadConfig.Burst = 2000

	const namespace = "watch-stall"
	if _, err := clientSet.CoreV1().Namespaces().Create(tCtx, &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	// The token controller that would normally create this is not running.
	if _, err := clientSet.CoreV1().ServiceAccounts(namespace).Create(tCtx, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default"},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create default service account: %v", err)
	}

	// A cluster-scoped watch with no field selector does not use the trigger
	// index, so it sees every pod event. A spec.nodeName-filtered watch (what a
	// kubelet opens) would only see one node's worth and could not fill the
	// buffers.
	stalledBody := openUnreadPodWatch(tCtx, t, kubeConfig, streamWindow)
	defer func() {
		_ = stalledBody.Close()
	}()

	before := scrapeMetrics(tCtx, t, kubeConfig)

	generatePodEvents(tCtx, t, loadConfig, namespace, podCount, annotationLen)

	// The stalled serve loop stops draining cacheWatcher.result, which backs up
	// into cacheWatcher.input, which the cacher reacts to by killing the watcher.
	terminatedBefore := counterValue(t, before, "apiserver_terminated_watchers_total", map[string]string{"resource": "pods"})
	if err := wait.PollUntilContextTimeout(tCtx, 250*time.Millisecond, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		current := scrapeMetrics(ctx, t, kubeConfig)
		return counterValue(t, current, "apiserver_terminated_watchers_total", map[string]string{"resource": "pods"}) > terminatedBefore, nil
	}); err != nil {
		t.Fatalf("watcher was never terminated, the stall did not reproduce: %v", err)
	}

	// Let the blocked write finish so its duration is actually observed.
	_, _ = io.Copy(io.Discard, stalledBody)

	podLabels := map[string]string{"group": "", "version": "v1", "resource": "pods"}
	transportSum := func(metrics testutil.Metrics) float64 {
		return histogramSum(t, metrics, "apiserver_watch_event_write_duration_seconds", podLabels) +
			histogramSum(t, metrics, "apiserver_watch_event_flush_duration_seconds", podLabels)
	}
	// The stall is only observed once the blocked write returns, so this waits
	// for the drain above to be reflected in the histograms. The threshold is
	// deliberately far below the stall the test produces; the assertions that
	// carry the signal are the ratios below.
	transportBefore := transportSum(before)
	var after testutil.Metrics
	if err := wait.PollUntilContextTimeout(tCtx, 250*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
		after = scrapeMetrics(ctx, t, kubeConfig)
		return transportSum(after)-transportBefore > 0.1, nil
	}); err != nil {
		t.Fatalf("no transport stall was recorded: %v", err)
	}

	// The cacher noticed that the serve loop was not consuming.
	handoffLabels := map[string]string{"group": "", "resource": "pods", "stage": "cache_to_watcher"}
	handoff := histogramSum(t, after, "apiserver_watch_events_dispatch_duration_seconds", handoffLabels) -
		histogramSum(t, before, "apiserver_watch_events_dispatch_duration_seconds", handoffLabels)
	if handoff <= 0 {
		t.Error("expected the cache_to_watcher dispatch stage to record the blocked handoff")
	}

	// The stall must land in the transport, not in serialization. encode covers
	// serialization plus the writes it issues, so encode-write is the time spent
	// serializing.
	delta := func(name string) float64 {
		return histogramSum(t, after, name, podLabels) - histogramSum(t, before, name, podLabels)
	}
	encode := delta("apiserver_watch_event_encode_duration_seconds")
	write := delta("apiserver_watch_event_write_duration_seconds")
	flush := delta("apiserver_watch_event_flush_duration_seconds")
	serialization := encode - write
	transport := write + flush

	t.Logf("serialization=%.3fs transport=%.3fs (write=%.3fs flush=%.3fs) cache_to_watcher=%.3fs",
		serialization, transport, write, flush, handoff)

	if transport <= serialization {
		t.Errorf("expected the transport to dominate, got serialization=%.3fs transport=%.3fs", serialization, transport)
	}
	if stage := delta(expectStage); stage <= transport/2 {
		t.Errorf("expected %s to hold most of the %.3fs transport time, got %.3fs", expectStage, transport, stage)
	}
}

// openUnreadPodWatch starts a watch on pods and deliberately never reads the
// returned body, so the apiserver blocks writing to it. A non-zero streamWindow
// shrinks the client's HTTP/2 per-stream flow control window from the 4MiB
// default, so that a stall can be provoked with realistically sized objects
// instead of thousands of them.
func openUnreadPodWatch(ctx context.Context, t *testing.T, kubeConfig *restclient.Config, streamWindow int) io.ReadCloser {
	t.Helper()

	config := withCoreV1(kubeConfig)
	if streamWindow > 0 {
		tlsConfig, err := restclient.TLSConfigFor(config)
		if err != nil {
			t.Fatalf("failed to build TLS config: %v", err)
		}
		config.TLSClientConfig = restclient.TLSClientConfig{}
		config.Transport = &http.Transport{
			TLSClientConfig:   tlsConfig,
			ForceAttemptHTTP2: true,
			HTTP2:             &http.HTTP2Config{MaxReceiveBufferPerStream: streamWindow},
		}
	}

	client, err := restclient.RESTClientFor(config)
	if err != nil {
		t.Fatalf("failed to build client: %v", err)
	}

	body, err := client.Get().
		Resource("pods").
		VersionedParams(&metav1.ListOptions{Watch: true}, scheme.ParameterCodec).
		Stream(ctx)
	if err != nil {
		t.Fatalf("failed to open watch: %v", err)
	}
	return body
}

func generatePodEvents(ctx context.Context, t *testing.T, config *restclient.Config, namespace string, count, annotationLen int) {
	t.Helper()

	client, err := restclient.RESTClientFor(withCoreV1(config))
	if err != nil {
		t.Fatalf("failed to build client: %v", err)
	}

	filler := strings.Repeat("x", annotationLen)
	for i := range count {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "stall-",
				Annotations:  map[string]string{"stall.k8s.io/filler": filler},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "c", Image: "image"}},
			},
		}
		result := &v1.Pod{}
		if err := client.Post().Namespace(namespace).Resource("pods").Body(pod).Do(ctx).Into(result); err != nil {
			t.Fatalf("failed to create pod %d: %v", i, err)
		}
	}
}

func withCoreV1(config *restclient.Config) *restclient.Config {
	out := restclient.CopyConfig(config)
	out.GroupVersion = &v1.SchemeGroupVersion
	out.APIPath = "/api"
	out.NegotiatedSerializer = scheme.Codecs.WithoutConversion()
	return out
}

func scrapeMetrics(ctx context.Context, t *testing.T, kubeConfig *restclient.Config) testutil.Metrics {
	t.Helper()

	client, err := restclient.RESTClientFor(withCoreV1(kubeConfig))
	if err != nil {
		t.Fatalf("failed to build client: %v", err)
	}
	raw, err := client.Get().AbsPath("/metrics").DoRaw(ctx)
	if err != nil {
		t.Fatalf("failed to scrape metrics: %v", err)
	}
	parsed := testutil.NewMetrics()
	if err := testutil.ParseMetrics(string(raw), &parsed); err != nil {
		t.Fatalf("failed to parse metrics: %v", err)
	}
	return parsed
}

func matches(sample *testutil.Sample, labels map[string]string) bool {
	for name, want := range labels {
		if string(sample.Metric[model.LabelName(name)]) != want {
			return false
		}
	}
	return true
}

func counterValue(t *testing.T, metrics testutil.Metrics, name string, labels map[string]string) float64 {
	t.Helper()
	var total float64
	for _, sample := range metrics[name] {
		if matches(sample, labels) {
			total += float64(sample.Value)
		}
	}
	return total
}

func histogramSum(t *testing.T, metrics testutil.Metrics, name string, labels map[string]string) float64 {
	t.Helper()
	return counterValue(t, metrics, name+"_sum", labels)
}
