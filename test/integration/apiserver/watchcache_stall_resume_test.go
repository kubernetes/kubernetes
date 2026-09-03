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
	"bufio"
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestSlowReaderSurvivesChurnBurst drives a real kube-apiserver watch whose
// client stops reading mid-stream while far more data than the HTTP/2
// stream window plus the watcher's buffers can hold is written. With the
// WatchCacheStallResume feature gate enabled the same watch delivers every
// event in order once the client resumes and stays open; with the gate
// disabled (the control that proves the schedule really overflows the
// watcher) the server terminates the watch during the stall.
func TestSlowReaderSurvivesChurnBurst(t *testing.T) {
	for _, gateOn := range []bool{true, false} {
		t.Run(fmt.Sprintf("WatchCacheStallResume=%v", gateOn), func(t *testing.T) {
			testSlowReaderSurvivesChurnBurst(t, gateOn)
		})
	}
}

func testSlowReaderSurvivesChurnBurst(t *testing.T, gateOn bool) {
	flags := framework.DefaultTestServerFlags()
	flags = append(flags, fmt.Sprintf("--feature-gates=WatchCacheStallResume=%v", gateOn))
	// The gate is registered at 1.38; run the server (and this process's
	// gate) at 1.38 so it is settable while DefaultKubeBinaryVersion is 1.37.
	// TODO: drop the version overrides once DefaultKubeBinaryVersion is 1.38
	// (https://github.com/kubernetes/kubernetes/pull/140764).
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.38"))
	server := kubeapiservertesting.StartTestServerOrDie(t, &kubeapiservertesting.TestServerInstanceOptions{BinaryVersion: "1.38"}, flags, framework.SharedEtcd())
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
	terminatedBefore := terminatedWatchersTotal(ctx, t, client)

	w, err := client.CoreV1().ConfigMaps(ns.Name).Watch(ctx, metav1.ListOptions{ResourceVersion: seed.ResourceVersion, AllowWatchBookmarks: true})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	// The consumer hiccups (does not read) while ~20 MB of watch events are
	// produced: 200 configmaps of ~100 KB each, well past the ~4 MB stream
	// window plus the configmaps watcher's channels (2 x 10 events).
	const total = 200
	pad := strings.Repeat("x", 100*1024)
	writerDone := make(chan error, 1)
	go func() {
		defer close(writerDone)
		for i := range total {
			cm := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("cm-%04d", i)}, Data: map[string]string{"pad": pad}}
			if _, err := client.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm, metav1.CreateOptions{}); err != nil {
				writerDone <- err
				return
			}
		}
	}()

	time.Sleep(6 * time.Second) // the client-side hiccup

	names := 0
	prev := ""
	closed := false
	deadline := time.After(90 * time.Second)
drain:
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				closed = true
				break drain
			}
			switch ev.Type {
			case watch.Error:
				t.Fatalf("unexpected error event: %#v", ev.Object)
			case watch.Bookmark:
				continue
			case watch.Added:
				cm := ev.Object.(*v1.ConfigMap)
				if !strings.HasPrefix(cm.Name, "cm-") {
					continue
				}
				if cm.Name <= prev {
					t.Errorf("out-of-order delivery: %q after %q", cm.Name, prev)
				}
				prev = cm.Name
				names++
				if names == total {
					break drain
				}
			}
		case <-deadline:
			t.Fatalf("timed out draining the watch: %d/%d events", names, total)
		}
	}
	if err := <-writerDone; err != nil {
		t.Fatalf("writer failed: %v", err)
	}
	terminatedAfter := terminatedWatchersTotal(ctx, t, client)

	if gateOn {
		if closed {
			t.Fatalf("gate on: the watch was closed after %d/%d events; expected it to survive the stall", names, total)
		}
		if names != total {
			t.Fatalf("gate on: expected all %d events, got %d", total, names)
		}
		if terminatedAfter != terminatedBefore {
			t.Errorf("gate on: expected no watcher terminations, terminated_watchers_total went %v -> %v", terminatedBefore, terminatedAfter)
		}
		return
	}
	// Gate off (control): today's behavior is a server-side termination that
	// the client observes as a clean close before receiving everything.
	if !closed {
		t.Fatalf("gate off: expected the watch to be closed during the stall, but it delivered all %d events", names)
	}
	if terminatedAfter <= terminatedBefore {
		t.Errorf("gate off: expected terminated_watchers_total to increase, got %v -> %v", terminatedBefore, terminatedAfter)
	}
}

// terminatedWatchersTotal sums apiserver_terminated_watchers_total for
// configmaps across all reason label values, from the server's /metrics.
func terminatedWatchersTotal(ctx context.Context, t *testing.T, client clientset.Interface) float64 {
	t.Helper()
	body, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(ctx)
	if err != nil {
		t.Fatalf("fetching /metrics: %v", err)
	}
	sum := 0.0
	scanner := bufio.NewScanner(strings.NewReader(string(body)))
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "apiserver_terminated_watchers_total") || !strings.Contains(line, `resource="configmaps"`) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		v, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			t.Fatalf("parsing %q: %v", line, err)
		}
		sum += v
	}
	return sum
}
