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

package cache

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fcache "k8s.io/client-go/tools/cache/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

const handlerWaitTime = time.Millisecond

func BenchmarkAddWithSlowHandlers(b *testing.B) {
	for _, unlockWhileProcessing := range []bool{false, true} {
		b.Run(fmt.Sprintf("unlockWhileProcessing=%t", unlockWhileProcessing), func(b *testing.B) {
			logger, ctx := ktesting.NewTestContext(b)
			ctx, cancel := context.WithCancel(ctx)
			source := fcache.NewFakeControllerSource()
			b.Cleanup(func() {
				cancel()
				source.Shutdown()
			})
			testIDs := []string{"a-hello"}
			source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: testIDs[0]}})

			store := NewStore(DeletionHandlingMetaNamespaceKeyFunc)
			fifoOptions := RealFIFOOptions{}
			if unlockWhileProcessing {
				fifoOptions.UnlockWhileProcessing = true
				fifoOptions.AtomicEvents = true
			} else {
				fifoOptions.KnownObjects = store
			}
			fifo := NewRealFIFOWithOptions(fifoOptions)
			handler := ResourceEventHandlerFuncs{
				UpdateFunc: func(oldObj, newObj interface{}) {
					time.Sleep(handlerWaitTime)
				},
			}

			cfg := &Config{
				Queue:            fifo,
				ListerWatcher:    source,
				ObjectType:       &v1.Pod{},
				FullResyncPeriod: 0,

				Process: func(obj interface{}, isInInitialList bool) error {
					if deltas, ok := obj.(Deltas); ok {
						return processDeltas(logger, handler, store, deltas, isInInitialList, DeletionHandlingMetaNamespaceKeyFunc)
					}
					return errors.New("object given as Process argument is not Deltas")
				},
				ProcessBatch: func(deltaList []Delta, isInInitialList bool) error {
					return processDeltasInBatch(logger, handler, store, deltaList, isInInitialList, DeletionHandlingMetaNamespaceKeyFunc)
				},
			}
			c := New(cfg)
			go c.RunWithContext(ctx)
			if !WaitForCacheSync(ctx.Done(), c.HasSynced) {
				b.Fatal("Timed out waiting for cache sync")
			}

			// Producer: Modify object as fast as the handler can process it. This should ensure that the process func is always running.
			go func() {
				// Stop when the benchmark context is cancelled.
				ticker := time.NewTicker(handlerWaitTime)
				defer ticker.Stop()
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						source.Modify(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: testIDs[0]}})
					}
				}
			}()
			benchmarkReflectorWithSlowHandlers(b, fifo)
		})
	}
}

func benchmarkReflectorWithSlowHandlers(b *testing.B, fifo *RealFIFO) {
	b.ResetTimer()
	// Try adding an object to the queue, while the controller is processing other events.
	for i := 0; i < b.N; i++ {
		if err := fifo.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "b-hello"}}); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkRealFIFOBatchingAndBackpressure measures throughput and batch size
// when a burst of updates to a small set of active pods is processed while
// concurrent readers call indexer.List().
//
// TransactionStore.Transaction acquires the store's exclusive write lock once
// per batch to amortize RWMutex contention against concurrent List() readers.
// When PopBatch breaks early on duplicate keys, batches collapse to at most
// activePods items, multiplying contended write-lock acquisitions while holding
// the FIFO lock under backpressure (see RealFIFO.whileProcessing_locked).
func BenchmarkRealFIFOBatchingAndBackpressure(b *testing.B) {
	const (
		totalStorePods = 5000
		activePods     = 25
		burstSize      = 3 * defaultBatchSize // exceeds 2*batchSize in RealFIFO.whileProcessing_locked to trigger backpressure
	)

	logger := klog.Background()
	indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		NamespaceIndex: MetaNamespaceIndexFunc,
	})

	// Populate the store so concurrent indexer.List() calls hold RLock for a
	// realistic duration, creating RWMutex contention per Transaction call.
	for i := 0; i < totalStorePods; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:       fmt.Sprintf("ns-%d", i%50),
				Name:            fmt.Sprintf("pod-%d", i),
				ResourceVersion: "1",
			},
		}
		if err := indexer.Add(pod); err != nil {
			b.Fatal(err)
		}
	}

	// Create two versions per active pod so every update modifies an existing store item.
	pods := make([][2]*v1.Pod, activePods)
	for i := 0; i < activePods; i++ {
		for v := 0; v < 2; v++ {
			pods[i][v] = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       fmt.Sprintf("ns-%d", i%50),
					Name:            fmt.Sprintf("pod-%d", i),
					ResourceVersion: fmt.Sprintf("%d", v+2),
				},
			}
		}
	}

	// Run background List() readers to contend on indexer's RWMutex, which is the
	// lock contention that PopBatch + TransactionStore.Transaction is designed to amortize.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for r := 0; r < 4; r++ {
		go func() {
			for ctx.Err() == nil {
				_ = indexer.List()
				time.Sleep(50 * time.Microsecond)
			}
		}()
	}

	fifo := NewRealFIFOWithOptions(RealFIFOOptions{
		KeyFunction:           MetaNamespaceKeyFunc,
		AtomicEvents:          true,
		UnlockWhileProcessing: true,
	})

	var processedEvents int64
	var processedBatches int64
	handler := ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { processedEvents++ },
		UpdateFunc: func(oldObj, newObj interface{}) { processedEvents++ },
		DeleteFunc: func(obj interface{}) { processedEvents++ },
	}

	b.ReportAllocs()
	b.ResetTimer()

	for processedEvents < int64(b.N) {
		for i := 0; i < burstSize; i++ {
			_ = fifo.Update(pods[i%activePods][(i/activePods)%2])
		}

		for drained := 0; drained < burstSize; {
			var popped int
			err := fifo.PopBatch(
				func(deltas []Delta, isInInitialList bool) error {
					popped = len(deltas)
					processedBatches++
					return processDeltasInBatch(logger, handler, indexer, deltas, isInInitialList, MetaNamespaceKeyFunc)
				},
				func(obj interface{}, isInInitialList bool) error {
					popped = 1
					processedBatches++
					if deltas, ok := obj.(Deltas); ok {
						return processDeltas(logger, handler, indexer, deltas, isInInitialList, MetaNamespaceKeyFunc)
					}
					return nil
				},
			)
			if err != nil {
				b.Fatal(err)
			}
			drained += popped
		}
	}

	b.StopTimer()
	if processedBatches > 0 {
		b.ReportMetric(float64(processedEvents)/float64(processedBatches), "batch_size_mean")
	}
}

