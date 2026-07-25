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
	"fmt"
	goruntime "runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// meanBatchSize reports the mean size of the fan-out passes recorded so far, and
// how many passes that was. Benchmarks read it as a delta around the measured
// section so that a latency number can be tied to the batch size the load
// actually produced: batching is opportunistic, so an arm that shows no win
// because no batch ever formed and an arm that shows no win despite batching are
// otherwise indistinguishable.
func meanBatchSize() (sum float64, count uint64) {
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		return 0, 0
	}
	for _, f := range families {
		if f.GetName() != "apiserver_watch_dispatch_batch_size" {
			continue
		}
		// Summed across label sets: the benchmark builds a fresh cacher per
		// iteration, and they all share one group/resource.
		for _, m := range f.GetMetric() {
			if h := m.GetHistogram(); h != nil {
				sum += h.GetSampleSum()
				count += h.GetSampleCount()
			}
		}
	}
	return sum, count
}

// BenchmarkDispatchBurst measures the full dispatcher path -- processEvent ->
// c.incoming -> dispatchEvents -> fan-out -> cacheWatcher.process -> result --
// under bursts that outrun the dispatcher, which is the regime opportunistic
// batching targets. Unlike the nonblockingAdd microbenchmarks it exercises
// dispatchEvent, blockedWatchers and the termination path.
//
// Each round injects burst events back-to-back (so several are queued in
// c.incoming at once and a batch can form) and then waits for every watcher to
// receive all of them. ns/op is per event, i.e. the time to get one event to
// all N watchers, amortized over the burst.
func BenchmarkDispatchBurst(b *testing.B) {
	for _, nWatchers := range []int{100, 1000, 5000} {
		for _, burst := range []int{1, 2, 3, 8, 32} {
			b.Run(fmt.Sprintf("watchers=%d/burst=%d", nWatchers, burst), func(b *testing.B) {
				runDispatchBurst(b, nWatchers, burst)
			})
		}
	}
}

func runDispatchBurst(b *testing.B, nWatchers, burst int) {
	// newTestCacher does not register the cacher metrics, so without this the
	// batch-size histogram gathers nothing. A real apiserver registers them from
	// pkg/server/routes/metrics.go.
	metrics.Register()
	pods := makeWakestormPods(64)
	store := newWakestormStorage(pods)
	cacher, _, err := newTestCacher(store)
	if err != nil {
		b.Fatalf("new cacher: %v", err)
	}
	defer cacher.Stop()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var delivered atomic.Int64
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
			for ev := range ch {
				if ev.Type == watch.Modified {
					delivered.Add(1)
				}
			}
		}(w.ResultChan())
	}
	defer func() {
		for _, w := range watchers {
			w.Stop()
		}
		drainWG.Wait()
	}()

	// Let the initial-list replay drain so the measured section only sees the
	// driven MODIFIED events.
	time.Sleep(500 * time.Millisecond)
	delivered.Store(0)

	rv := int64(wakestormEventRVBase)

	sum0, count0 := meanBatchSize()

	b.ResetTimer()
	for i := 0; i < b.N; i += burst {
		n := burst
		if i+n > b.N {
			n = b.N - i
		}
		base := delivered.Load()
		for j := 0; j < n; j++ {
			p := pods[(i+j)%len(pods)]
			rv++
			p.ResourceVersion = strconv.FormatInt(rv, 10)
			if uerr := cacher.watchCache.Update(&p); uerr != nil {
				b.Fatalf("update: %v", uerr)
			}
		}
		want := base + int64(n)*int64(nWatchers)
		deadline := time.Now().Add(60 * time.Second)
		for delivered.Load() < want {
			if time.Now().After(deadline) {
				b.Fatalf("timed out: delivered %d/%d (watchers may have been terminated)", delivered.Load(), want)
			}
			goruntime.Gosched()
		}
	}
	b.StopTimer()

	if sum1, count1 := meanBatchSize(); count1 > count0 {
		b.ReportMetric((sum1-sum0)/float64(count1-count0), "batch-size")
	}
	b.ReportMetric(float64(registeredWatchers(cacher)), "alive-watchers")
}

// registeredWatchers reports how many watchers are still registered, so a run
// that silently killed watchers is visible in the benchmark output.
func registeredWatchers(c *Cacher) int {
	c.Lock()
	defer c.Unlock()
	n := 0
	for _, ws := range c.watchers.allWatchers {
		n += len(ws)
	}
	return n
}

var _ = example.Pod{}
