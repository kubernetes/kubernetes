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

package apiserver

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestSlowReaderSurvivesChurnBurst stops reading a watch while more data is
// written than the HTTP/2 stream window and the watcher channels hold. With
// WatchCacheStallResume the client gets every event in order and the watch
// stays open. With the gate off, the control, the server closes the watch.
func TestSlowReaderSurvivesChurnBurst(t *testing.T) {
	for _, gateOn := range []bool{true, false} {
		t.Run(fmt.Sprintf("WatchCacheStallResume=%v", gateOn), func(t *testing.T) {
			testSlowReaderSurvivesChurnBurst(t, gateOn)
		})
	}
}

func testSlowReaderSurvivesChurnBurst(t *testing.T, gateOn bool) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, gateOn)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	ns := framework.CreateNamespaceOrDie(client, "stall-resume", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	seed, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "seed"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	terminatedBefore := terminatedWatchers(ctx, t, client)

	w, err := client.CoreV1().ConfigMaps(ns.Name).Watch(ctx, metav1.ListOptions{ResourceVersion: seed.ResourceVersion, AllowWatchBookmarks: true})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	// About 20 MB of events, far more than the window of about 4 MB and the
	// channels hold. The client reads nothing until the writer is done.
	const total = 200
	pad := strings.Repeat("x", 100*1024)
	writerDone := make(chan error, 1)
	go func() {
		for i := range total {
			cm := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("cm-%04d", i)}, Data: map[string]string{"pad": pad}}
			if _, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm, metav1.CreateOptions{}); err != nil {
				writerDone <- err
				return
			}
		}
		writerDone <- nil
	}()
	select {
	case err := <-writerDone:
		if err != nil {
			t.Fatalf("the writer failed: %v", err)
		}
	case <-time.After(90 * time.Second):
		t.Fatal("timed out waiting for the writer")
	}
	if !gateOn {
		// The dispatcher waits for a blocked watcher up to its time budget, so
		// a client that reads too early is never closed.
		if err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			return terminatedWatchers(ctx, t, client) > terminatedBefore, nil
		}); err != nil {
			t.Fatalf("the server did not terminate the watcher: %v", err)
		}
	}

	received := 0
	prev := ""
	closed := false
	deadline := time.After(90 * time.Second)
	for !closed && received < total {
		select {
		case ev, ok := <-w.ResultChan():
			switch {
			case !ok:
				closed = true
			case ev.Type == watch.Added:
				cm := ev.Object.(*v1.ConfigMap)
				if cm.Name <= prev {
					t.Errorf("out of order delivery: %q after %q", cm.Name, prev)
				}
				prev = cm.Name
				received++
			case ev.Type != watch.Bookmark:
				t.Fatalf("unexpected event: %#v", ev)
			}
		case <-deadline:
			t.Fatalf("timed out reading the watch: %d of %d events", received, total)
		}
	}
	terminatedAfter := terminatedWatchers(ctx, t, client)

	if gateOn {
		if closed {
			t.Fatalf("the watch closed after %d of %d events", received, total)
		}
		if terminatedAfter != terminatedBefore {
			t.Errorf("expected no terminated watchers, the counter went from %v to %v", terminatedBefore, terminatedAfter)
		}
		return
	}
	if !closed {
		t.Fatalf("expected the server to close the watch, but it delivered all %d events", total)
	}
	if terminatedAfter <= terminatedBefore {
		t.Errorf("expected terminated watchers, the counter went from %v to %v", terminatedBefore, terminatedAfter)
	}
}

// terminatedWatchers sums apiserver_terminated_watchers_total for configmaps
// over all reasons.
func terminatedWatchers(ctx context.Context, t *testing.T, client clientset.Interface) float64 {
	t.Helper()
	body, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(ctx)
	if err != nil {
		t.Fatalf("fetching /metrics: %v", err)
	}
	metrics := testutil.NewMetrics()
	if err := testutil.ParseMetrics(string(body), &metrics); err != nil {
		t.Fatalf("parsing /metrics: %v", err)
	}
	sum := 0.0
	for _, sample := range metrics["apiserver_terminated_watchers_total"] {
		if sample.Metric["resource"] == "configmaps" {
			sum += float64(sample.Value)
		}
	}
	return sum
}
