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

package cacher

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	goruntime "runtime"
	rtmetrics "runtime/metrics"
	"runtime/pprof"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
)

// Tier-2 isolation benchmark for the apiserver watch-cache "fan-out wake-storm".
//
// Hypothesis: at a CONSTANT pod count, increasing the number of registered
// watchers N makes a concurrent large LIST slower and raises Go scheduler
// latency, WHILE CPU stays below saturation (idle cores). No etcd, no HTTP,
// no cluster: a pure cacher-package benchmark. One watchCache.Update fans out
// to all N cacheWatcher goroutines via Cacher.dispatchEvent -> nonblockingAdd,
// which is the wake-storm we are isolating.

const (
	// wakestormPodCount is the constant number of fat pods preloaded into the
	// cacher. Held fixed across the watcher sweep on purpose: only N varies.
	wakestormPodCount = 50000

	// wakestormEventRateHz is the target steady event rate (events/sec) the
	// background driver pushes through watchCache.Update to generate continuous
	// fan-out during the measured LIST.
	wakestormEventRateHz = 3000

	// wakestormFatLabelBytes is the size of the random label blob per pod, so
	// the LIST encode is non-trivial (~2KB like BenchmarkCacher_GetList).
	wakestormFatLabelBytes = 2 * 1024
)

// wakestormEventRVBase is the starting resourceVersion for driven events. It
// must exceed the fake-storage list RV (12345) so processEvent accepts them.
const wakestormEventRVBase = 1_000_000

// wakestormListRV is the resourceVersion the fake backing storage reports for
// the preload LIST. Driven events use RVs above wakestormEventRVBase.
const wakestormListRV = 12345

// newWakestormStorage returns a fake backing storage that preloads the given
// pods. It uses cachertesting.MockStorage (which implements the full
// storage.Interface, including the newer EnableResourceSizeEstimation and
// CompactRevision methods the cacher's background goroutines call) and only
// overrides GetList to return the preloaded slice and GetCurrentResourceVersion
// to stay consistent with the list RV.
func newWakestormStorage(pods []example.Pod) *cachertesting.MockStorage {
	return &cachertesting.MockStorage{
		GetListFn: func(_ context.Context, _ string, _ storage.ListOptions, listObj runtime.Object) error {
			podList := listObj.(*example.PodList)
			podList.ListMeta = metav1.ListMeta{ResourceVersion: strconv.Itoa(wakestormListRV)}
			podList.Items = pods
			return nil
		},
		GetRVFn: func(_ context.Context) (uint64, error) { return wakestormListRV, nil },
	}
}

func makeWakestormPods(n int) []example.Pod {
	pods := make([]example.Pod, n)
	for i := range pods {
		pods[i].Namespace = "default"
		pods[i].Name = fmt.Sprintf("pod-%d", i)
		pods[i].ResourceVersion = strconv.Itoa(i)
		pods[i].Spec.NodeName = "node-0"
		data := make([]byte, wakestormFatLabelBytes)
		_, _ = rand.Read(data)
		pods[i].Spec.NodeSelector = map[string]string{"key": string(data)}
	}
	return pods
}

// schedLatencyP99Ms reads the /sched/latencies:seconds histogram and returns an
// approximate p99 in milliseconds, computed over the delta between two reads.
func readSchedHist() *rtmetrics.Float64Histogram {
	const name = "/sched/latencies:seconds"
	samples := []rtmetrics.Sample{{Name: name}}
	rtmetrics.Read(samples)
	if samples[0].Value.Kind() != rtmetrics.KindFloat64Histogram {
		return nil
	}
	// Copy so a later Read doesn't mutate the captured snapshot.
	src := samples[0].Value.Float64Histogram()
	dst := &rtmetrics.Float64Histogram{
		Counts:  append([]uint64(nil), src.Counts...),
		Buckets: append([]float64(nil), src.Buckets...),
	}
	return dst
}

func schedHistPercentileMs(before, after *rtmetrics.Float64Histogram, q float64) float64 {
	if before == nil || after == nil || len(after.Counts) == 0 {
		return 0
	}
	var total uint64
	delta := make([]uint64, len(after.Counts))
	for i := range after.Counts {
		var b uint64
		if i < len(before.Counts) {
			b = before.Counts[i]
		}
		if after.Counts[i] >= b {
			delta[i] = after.Counts[i] - b
		}
		total += delta[i]
	}
	if total == 0 {
		return 0
	}
	target := uint64(float64(total) * q)
	var cum uint64
	for i, c := range delta {
		cum += c
		if cum >= target {
			// Bucket i spans Buckets[i]..Buckets[i+1] seconds.
			hi := after.Buckets[i+1]
			if i+1 >= len(after.Buckets) {
				hi = after.Buckets[len(after.Buckets)-1]
			}
			return hi * 1000.0
		}
	}
	return 0
}

// TestWakestormSanity asserts the harness actually fans out: a single watcher
// registered against the cluster-scope /pods/ watch sees an event driven via
// watchCache.Update. If this fails, the bigger sweep measures nothing.
func TestWakestormSanity(t *testing.T) {
	pods := makeWakestormPods(10)
	store := newWakestormStorage(pods)
	cacher, _, err := newTestCacher(store)
	if err != nil {
		t.Fatalf("new cacher: %v", err)
	}
	defer cacher.Stop()

	ctx := context.Background()
	w, err := cacher.Watch(ctx, "/pods/", storage.ListOptions{
		ResourceVersion: strconv.Itoa(wakestormListRV),
		Predicate:       storage.Everything,
		Recursive:       true,
	})
	if err != nil {
		t.Fatalf("watch: %v", err)
	}
	defer w.Stop()

	// Drive one Update with an RV beyond the fake list RV.
	p := pods[0]
	p.ResourceVersion = strconv.Itoa(wakestormEventRVBase)
	if err := cacher.watchCache.Update(&p); err != nil {
		t.Fatalf("update: %v", err)
	}

	// A watch from RV "0" first replays existing pods as ADDED; the event we
	// drove arrives as a MODIFIED. Confirm the fan-out delivers it.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("watch channel closed before the driven event")
			}
			if ev.Type == watch.Modified {
				return
			}
		case <-deadline:
			t.Fatal("timed out waiting for the watcher to see the driven event")
		}
	}
}

func BenchmarkCacherWakestorm(b *testing.B) {
	watcherCounts := []int{0, 100, 1000, 5000, 20000}

	pods := makeWakestormPods(wakestormPodCount)

	for _, n := range watcherCounts {
		b.Run(fmt.Sprintf("watchers=%d", n), func(b *testing.B) {
			runWakestorm(b, pods, n)
		})
	}
}

func runWakestorm(b *testing.B, pods []example.Pod, nWatchers int) {
	gomaxprocs := goruntime.GOMAXPROCS(0)

	setupStart := time.Now()
	store := newWakestormStorage(pods)
	cacher, _, err := newTestCacher(store)
	if err != nil {
		b.Fatalf("new cacher: %v", err)
	}
	defer cacher.Stop()
	delegator := NewCacheDelegator(cacher, store)
	defer delegator.Stop()
	setupDur := time.Since(setupStart)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Register N watchers, each draining its ResultChan in a goroutine so they
	// stay parked-and-wakeable rather than stuck on a full channel.
	var drainWG sync.WaitGroup
	watchers := make([]watch.Interface, 0, nWatchers)
	for i := 0; i < nWatchers; i++ {
		w, werr := cacher.Watch(ctx, "/pods/", storage.ListOptions{
			ResourceVersion: strconv.Itoa(wakestormListRV),
			Predicate:       storage.Everything,
			Recursive:       true,
		})
		if werr != nil {
			b.Fatalf("watch %d: %v", i, werr)
		}
		watchers = append(watchers, w)
		drainWG.Add(1)
		go func(ch <-chan watch.Event) {
			defer drainWG.Done()
			for range ch {
			}
		}(w.ResultChan())
	}
	defer func() {
		for _, w := range watchers {
			w.Stop()
		}
		drainWG.Wait()
	}()

	// Background event driver: steady stream of Updates on rotating pods to
	// generate continuous fan-out across all registered watchers.
	var rvCounter int64 = wakestormEventRVBase
	var eventsSent int64
	driverDone := make(chan struct{})
	go func() {
		defer close(driverDone)
		interval := time.Second / time.Duration(wakestormEventRateHz)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		idx := 0
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				p := pods[idx%len(pods)]
				idx++
				p.ResourceVersion = strconv.FormatInt(atomic.AddInt64(&rvCounter, 1), 10)
				if uerr := cacher.watchCache.Update(&p); uerr != nil {
					return
				}
				atomic.AddInt64(&eventsSent, 1)
			}
		}
	}()

	// Field selector that matches everything (every pod has NodeName node-0),
	// so the victim LIST returns all preloaded pods (large encode).
	parsedField, err := fields.ParseSelector("spec.nodeName=node-0")
	if err != nil {
		b.Fatalf("parse selector: %v", err)
	}
	pred := storage.SelectionPredicate{Label: labels.Everything(), Field: parsedField}

	// Let the driver warm up so fan-out is active before the timed section.
	time.Sleep(200 * time.Millisecond)

	highest := nWatchers == 20000
	if highest {
		goruntime.SetMutexProfileFraction(1)
		goruntime.SetBlockProfileRate(1)
	}

	schedBefore := readSchedHist()
	var cpuFile *os.File
	if highest {
		cpuFile, _ = os.Create("/tmp/wakestorm-cpu-N20000.pprof")
		if cpuFile != nil {
			_ = pprof.StartCPUProfile(cpuFile)
		}
	}

	latencies := make([]time.Duration, 0, b.N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := &example.PodList{}
		callStart := time.Now()
		err = delegator.GetList(ctx, "/pods/", storage.ListOptions{
			Predicate:       pred,
			Recursive:       true,
			ResourceVersion: "0",
		}, result)
		d := time.Since(callStart)
		if err != nil {
			b.Fatalf("GetList: %v", err)
		}
		if len(result.Items) == 0 {
			b.Fatalf("victim LIST returned 0 items")
		}
		latencies = append(latencies, d)
	}
	b.StopTimer()

	if highest && cpuFile != nil {
		pprof.StopCPUProfile()
		_ = cpuFile.Close()
	}
	schedAfter := readSchedHist()

	if highest {
		writeProfile("/tmp/wakestorm-mutex-N20000.pprof", "mutex")
		writeProfile("/tmp/wakestorm-block-N20000.pprof", "block")
		goruntime.SetMutexProfileFraction(0)
		goruntime.SetBlockProfileRate(0)
	}

	// Custom metrics.
	b.ReportMetric(listP99Ms(latencies), "list-p99-ms")
	b.ReportMetric(schedHistPercentileMs(schedBefore, schedAfter, 0.99), "sched-p99-ms")
	b.ReportMetric(float64(gomaxprocs), "gomaxprocs")
	b.ReportMetric(float64(len(pods)), "pods")
	b.ReportMetric(float64(atomic.LoadInt64(&eventsSent)), "events")
	b.Logf("watchers=%d pods=%d gomaxprocs=%d setup=%s events=%d",
		nWatchers, len(pods), gomaxprocs, setupDur, atomic.LoadInt64(&eventsSent))
}

func listP99Ms(latencies []time.Duration) float64 {
	if len(latencies) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), latencies...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(float64(len(s)) * 0.99)
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return float64(s[idx].Microseconds()) / 1000.0
}

func writeProfile(path, name string) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	if p := pprof.Lookup(name); p != nil {
		_ = p.WriteTo(f, 0)
	}
}
