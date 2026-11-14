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

package cacher_test

import (
	"fmt"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/consistency"
	"k8s.io/apiserver/pkg/storage/cacher/testing/setup"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// All changes of one etcd transaction share a resourceVersion.
// An open watch must deliver every one of them.
func TestWatchMultiKeyTransaction(t *testing.T) {
	if consistency.PanicOnCacheInconsistency {
		t.Skip("the consistency checker panics on events that share a resourceVersion")
	}
	const pods = 5
	for _, watchCacheStallResume := range []bool{false, true} {
		for _, tc := range []struct {
			name      string
			eventType watch.EventType
			change    func(keys []string, values []string) []clientv3.Op
		}{
			{
				name:      "DeleteRange",
				eventType: watch.Deleted,
				change: func(keys, _ []string) []clientv3.Op {
					return []clientv3.Op{clientv3.OpDelete(keys[0], clientv3.WithRange(keys[len(keys)-1]+"\x00"))}
				},
			},
			{
				name:      "TxnPut",
				eventType: watch.Modified,
				change: func(keys, values []string) (ops []clientv3.Op) {
					for i := range keys {
						ops = append(ops, clientv3.OpPut(keys[i], values[i]))
					}
					return ops
				},
			},
		} {
			t.Run(fmt.Sprintf("WatchCacheStallResume=%v/%s", watchCacheStallResume, tc.name), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, watchCacheStallResume)
				ctx, store, server, terminate := setup.SetupWithEtcdServer(t)
				t.Cleanup(terminate)

				out := &example.Pod{}
				want := sets.New[string]()
				for i := 0; i < pods; i++ {
					pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns", Name: fmt.Sprintf("pod-%d", i)}}
					if err := store.Create(ctx, "/pods/test-ns/"+pod.Name, pod, out, 0); err != nil {
						t.Fatalf("Failed to create pod: %v", err)
					}
					want.Insert(pod.Name)
				}
				w, err := store.Watch(ctx, "/pods", storage.ListOptions{ResourceVersion: out.ResourceVersion, Predicate: storage.Everything, Recursive: true})
				if err != nil {
					t.Fatalf("Failed to watch: %v", err)
				}
				defer w.Stop()

				resp, err := server.V3Client.Client.Get(ctx, "/", clientv3.WithPrefix())
				if err != nil {
					t.Fatalf("Failed to read etcd: %v", err)
				}
				var keys, values []string
				for _, kv := range resp.Kvs {
					keys, values = append(keys, string(kv.Key)), append(values, string(kv.Value))
				}
				if len(keys) != pods {
					t.Fatalf("Expected %d keys in etcd, got %v", pods, keys)
				}
				if _, err := server.V3Client.Client.Txn(ctx).Then(tc.change(keys, values)...).Commit(); err != nil {
					t.Fatalf("Failed to commit transaction: %v", err)
				}

				// The next write has a higher resourceVersion, so its event must come last.
				last := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "test-ns", Name: "last"}}
				if err := store.Create(ctx, "/pods/test-ns/last", last, out, 0); err != nil {
					t.Fatalf("Failed to create pod: %v", err)
				}

				got := sets.New[string]()
				timeout := time.After(wait.ForeverTestTimeout)
				for {
					select {
					case event, ok := <-w.ResultChan():
						if !ok {
							t.Fatal("Watch closed")
						}
						object := event.Object
						if cacheable, ok := object.(runtime.CacheableObject); ok {
							object = cacheable.GetObject()
						}
						name := object.(*example.Pod).Name
						if name == last.Name {
							if !got.Equal(want) {
								t.Fatalf("Open watch missed %d of %d events of one transaction: %v", want.Difference(got).Len(), want.Len(), sets.List(want.Difference(got)))
							}
							return
						}
						if event.Type != tc.eventType {
							t.Fatalf("Unexpected event type %v, expected %v", event.Type, tc.eventType)
						}
						if got.Has(name) {
							t.Fatalf("Got a second event for %s", name)
						}
						got.Insert(name)
					case <-timeout:
						t.Fatalf("Timed out, received %v", sets.List(got))
					}
				}
			})
		}
	}
}
