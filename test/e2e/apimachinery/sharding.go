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

package apimachinery

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("Server-Side Sharded List and Watch", framework.WithFeatureGate(features.ShardedListAndWatch), func() {
	f := framework.NewDefaultFramework("sharding")

	ginkgo.It("should partition LIST results across 4 UID shards with disjoint completeness, pagination, and ShardInfo echo", func(ctx context.Context) {
		ns := f.Namespace.Name
		cmClient := f.ClientSet.CoreV1().ConfigMaps(ns)

		const numObjects = 24
		createdUIDs := make(map[string]string, numObjects)
		for i := range numObjects {
			cm, err := cmClient.Create(ctx, &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:   fmt.Sprintf("shard-uid-cm-%02d", i),
					Labels: map[string]string{"e2e-sharding": "uid-list"},
				},
				Data: map[string]string{"index": fmt.Sprintf("%d", i)},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create ConfigMap %d", i)
			createdUIDs[string(cm.UID)] = cm.Name
		}

		const numShards = 4
		seenUnpaginated := make(map[string]int, numObjects)
		seenPaginated := make(map[string]int, numObjects)
		seenShardedLW := make(map[string]int, numObjects)

		baseLW := &cache.ListWatch{
			ListWithContextFunc: func(ctx context.Context, opts metav1.ListOptions) (runtime.Object, error) {
				opts.LabelSelector = "e2e-sharding=uid-list"
				return cmClient.List(ctx, opts)
			},
			WatchFuncWithContext: func(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
				opts.LabelSelector = "e2e-sharding=uid-list"
				return cmClient.Watch(ctx, opts)
			},
		}

		for shard := range numShards {
			sel, err := sharding.NewShardRangeSelector("object.metadata.uid", shard, numShards)
			framework.ExpectNoError(err)

			list, err := cmClient.List(ctx, metav1.ListOptions{
				LabelSelector: "e2e-sharding=uid-list",
				ShardSelector: sel.String(),
			})
			framework.ExpectNoError(err, "failed to list shard %d", shard)
			gomega.Expect(list.ShardInfo).ToNot(gomega.BeNil())
			gomega.Expect(list.ShardInfo.Selector).To(gomega.Equal(sel.String()))

			for i := range list.Items {
				cm := &list.Items[i]
				uid := string(cm.UID)
				matched, err := sel.Matches(cm)
				framework.ExpectNoError(err)
				gomega.Expect(matched).To(gomega.BeTrueBecause("shard %d returned out-of-shard ConfigMap %s", shard, cm.Name))
				gomega.Expect(seenUnpaginated).ToNot(gomega.HaveKey(uid))
				seenUnpaginated[uid] = shard
			}

			shardedLW := cache.ToListerWatcherWithContext(cache.NewShardedListWatch(baseLW, sel))
			lwObj, err := shardedLW.ListWithContext(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			lwList := lwObj.(*v1.ConfigMapList)
			gomega.Expect(lwList.ShardInfo).ToNot(gomega.BeNil())
			for i := range lwList.Items {
				seenShardedLW[string(lwList.Items[i].UID)] = shard
			}

			opts := metav1.ListOptions{
				LabelSelector: "e2e-sharding=uid-list",
				ShardSelector: sel.String(),
				Limit:         2,
			}
			for {
				page, err := cmClient.List(ctx, opts)
				framework.ExpectNoError(err, "failed paginated list for shard %d", shard)
				gomega.Expect(page.ShardInfo).ToNot(gomega.BeNil())
				gomega.Expect(page.ShardInfo.Selector).To(gomega.Equal(sel.String()))
				gomega.Expect(len(page.Items)).To(gomega.BeNumerically("<=", opts.Limit))

				for i := range page.Items {
					cm := &page.Items[i]
					uid := string(cm.UID)
					matched, err := sel.Matches(cm)
					framework.ExpectNoError(err)
					gomega.Expect(matched).To(gomega.BeTrueBecause("shard %d paginated list returned out-of-shard ConfigMap %s", shard, cm.Name))
					gomega.Expect(seenPaginated).ToNot(gomega.HaveKey(uid))
					seenPaginated[uid] = shard
				}
				if page.Continue == "" {
					break
				}
				opts.Continue = page.Continue
			}
		}

		gomega.Expect(seenUnpaginated).To(gomega.HaveLen(numObjects))
		gomega.Expect(seenPaginated).To(gomega.Equal(seenUnpaginated))
		gomega.Expect(seenShardedLW).To(gomega.Equal(seenUnpaginated))
	})

	ginkgo.It("should filter LIST and WATCH events by object.metadata.uid and object.metadata.namespace", func(ctx context.Context) {
		ns := f.Namespace.Name
		cmClient := f.ClientSet.CoreV1().ConfigMaps(ns)

		initialList, err := cmClient.List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		rv := initialList.ResourceVersion

		const numShards = 2
		uidSelectors := make([]sharding.Selector, numShards)
		nsSelectors := make([]sharding.Selector, numShards)
		uidWatchers := make([]watch.Interface, numShards)
		nsWatchers := make([]watch.Interface, numShards)

		for shard := range numShards {
			uidSelectors[shard], err = sharding.NewShardRangeSelector("object.metadata.uid", shard, numShards)
			framework.ExpectNoError(err)
			nsSelectors[shard], err = sharding.NewShardRangeSelector("object.metadata.namespace", shard, numShards)
			framework.ExpectNoError(err)

			uidWatchers[shard], err = cmClient.Watch(ctx, metav1.ListOptions{
				ResourceVersion: rv,
				LabelSelector:   "e2e-sharding=watch-test",
				ShardSelector:   uidSelectors[shard].String(),
			})
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(uidWatchers[shard].Stop)

			nsWatchers[shard], err = cmClient.Watch(ctx, metav1.ListOptions{
				ResourceVersion: rv,
				LabelSelector:   "e2e-sharding=watch-test",
				ShardSelector:   nsSelectors[shard].String(),
			})
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(nsWatchers[shard].Stop)
		}

		const numObjects = 12
		created := make([]*v1.ConfigMap, 0, numObjects)
		for i := range numObjects {
			cm, err := cmClient.Create(ctx, &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:   fmt.Sprintf("shard-watch-cm-%02d", i),
					Labels: map[string]string{"e2e-sharding": "watch-test"},
				},
				Data: map[string]string{"step": "created"},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			created = append(created, cm)
		}

		// Verify namespace-sharded LIST returns all objects on the matching shard and 0 on the complement.
		for shard := range numShards {
			list, err := cmClient.List(ctx, metav1.ListOptions{
				LabelSelector: "e2e-sharding=watch-test",
				ShardSelector: nsSelectors[shard].String(),
			})
			framework.ExpectNoError(err)
			gomega.Expect(list.ShardInfo).ToNot(gomega.BeNil())
			matched, err := nsSelectors[shard].Matches(created[0])
			framework.ExpectNoError(err)
			if matched {
				gomega.Expect(list.Items).To(gomega.HaveLen(numObjects))
			} else {
				gomega.Expect(list.Items).To(gomega.BeEmpty())
			}
		}

		verifyPhaseEvents := func(watchers []watch.Interface, selectors []sharding.Selector, wantType watch.EventType) {
			timer := time.NewTimer(wait.ForeverTestTimeout)
			defer timer.Stop()

			totalSeen := 0
			for shard, w := range watchers {
				expected := make(map[string]bool)
				for _, cm := range created {
					matched, err := selectors[shard].Matches(cm)
					framework.ExpectNoError(err)
					if matched {
						expected[string(cm.UID)] = true
					}
				}
				seen := make(map[string]bool, len(expected))
				for len(seen) < len(expected) {
					select {
					case <-ctx.Done():
						framework.Failf("context canceled waiting for %s events on shard %d: %v", wantType, shard, ctx.Err())
					case <-timer.C:
						framework.Failf("timed out waiting for %s events on shard %d: got %d/%d", wantType, shard, len(seen), len(expected))
					case evt, ok := <-w.ResultChan():
						if !ok {
							framework.Failf("watch channel closed unexpectedly on shard %d waiting for %s", shard, wantType)
						}
						gomega.Expect(evt.Type).To(gomega.Equal(wantType), "unexpected watch event type on shard %d: %#v", shard, evt.Object)
						cm, ok := evt.Object.(*v1.ConfigMap)
						gomega.Expect(ok).To(gomega.BeTrueBecause("expected *v1.ConfigMap, got %T", evt.Object))
						uid := string(cm.UID)
						gomega.Expect(expected).To(gomega.HaveKey(uid), "shard %d received out-of-shard %s event for %s (%s)", shard, wantType, cm.Name, uid)
						gomega.Expect(seen).ToNot(gomega.HaveKey(uid), "shard %d received duplicate %s event for %s", shard, wantType, uid)
						seen[uid] = true
						totalSeen++
					}
				}
			}
			gomega.Expect(totalSeen).To(gomega.Equal(numObjects))

			// Verify no extra or out-of-shard events are queued on any shard.
			for shard, w := range watchers {
				select {
				case evt := <-w.ResultChan():
					framework.Failf("shard %d received unexpected extra event %s: %#v", shard, evt.Type, evt.Object)
				default:
				}
			}
		}

		verifyPhaseEvents(uidWatchers, uidSelectors, watch.Added)
		verifyPhaseEvents(nsWatchers, nsSelectors, watch.Added)

		for i, cm := range created {
			cm.Data["step"] = "modified"
			updated, err := cmClient.Update(ctx, cm, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
			created[i] = updated
		}
		verifyPhaseEvents(uidWatchers, uidSelectors, watch.Modified)
		verifyPhaseEvents(nsWatchers, nsSelectors, watch.Modified)

		for _, cm := range created {
			framework.ExpectNoError(cmClient.Delete(ctx, cm.Name, metav1.DeleteOptions{}))
		}
		verifyPhaseEvents(uidWatchers, uidSelectors, watch.Deleted)
		verifyPhaseEvents(nsWatchers, nsSelectors, watch.Deleted)
	})
})
