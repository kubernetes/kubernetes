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
