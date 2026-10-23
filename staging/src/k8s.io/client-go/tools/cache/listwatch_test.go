/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"slices"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/util/watchlist"
)

type fakeWatchListClient struct {
	unSupportedWatchListSemantics bool
}

func (f fakeWatchListClient) IsWatchListSemanticsUnSupported() bool {
	return f.unSupportedWatchListSemantics
}

func TestToListWatcherWithWatchListSemantics(t *testing.T) {
	scenarios := []struct {
		name                                string
		client                              any
		expectUnSupportedWatchListSemantics bool
	}{
		{
			name:                                "client which doesn't implement the interface supports WatchList semantics",
			client:                              nil,
			expectUnSupportedWatchListSemantics: false,
		},
		{
			name:                                "client does not support WatchList semantics",
			client:                              fakeWatchListClient{unSupportedWatchListSemantics: true},
			expectUnSupportedWatchListSemantics: true,
		},
		{
			name:                                "client supports WatchList semantics",
			client:                              fakeWatchListClient{unSupportedWatchListSemantics: false},
			expectUnSupportedWatchListSemantics: false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			target := ToListWatcherWithWatchListSemantics(&ListWatch{}, scenario.client)

			if got := watchlist.DoesClientNotSupportWatchListSemantics(target); got != scenario.expectUnSupportedWatchListSemantics {
				t.Fatalf("DoesClientNotSupportWatchListSemantics returned: %v, want: %v", got, scenario.expectUnSupportedWatchListSemantics)
			}
		})
	}
}

func TestNewShardedListWatch(t *testing.T) {
	base := ToListWatcherWithWatchListSemantics(&ListWatch{}, fakeWatchListClient{unSupportedWatchListSemantics: true})
	for _, emptySel := range []sharding.Selector{nil, sharding.Everything()} {
		passthrough := NewShardedListWatch(base, emptySel)
		if !watchlist.DoesClientNotSupportWatchListSemantics(passthrough) {
			t.Errorf("expected DoesClientNotSupportWatchListSemantics to be preserved for empty selector %v", emptySel)
		}
	}

	sel0, err := sharding.NewShardRangeSelector("object.metadata.uid", 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	sel1, err := sharding.NewShardRangeSelector("object.metadata.uid", 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	conflictLW := NewShardedListWatch(&ListWatch{
		ListWithContextFunc: func(_ context.Context, _ metav1.ListOptions) (runtime.Object, error) {
			return &v1.PodList{
				ListMeta: metav1.ListMeta{
					ResourceVersion: "10",
					ShardInfo:       &metav1.ShardInfo{Selector: sel1.String()},
				},
			}, nil
		},
		WatchFuncWithContext: func(_ context.Context, _ metav1.ListOptions) (watch.Interface, error) {
			return watch.NewFake(), nil
		},
	}, sel0)
	if _, err := conflictLW.List(metav1.ListOptions{}); err == nil {
		t.Errorf("expected error when server ShardInfo.Selector mismatches configured selector")
	}
	if _, err := conflictLW.List(metav1.ListOptions{ShardSelector: sel1.String()}); err == nil {
		t.Errorf("expected error when ListOptions.ShardSelector conflicts with configured selector")
	}
	if _, err := conflictLW.Watch(metav1.ListOptions{ShardSelector: sel1.String()}); err == nil {
		t.Errorf("expected error when Watch ListOptions.ShardSelector conflicts with configured selector")
	}

	pods := make([]v1.Pod, 16)
	for i := range pods {
		pods[i] = v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "default",
				UID:             types.UID(fmt.Sprintf("uid-%d", i)),
				ResourceVersion: "10",
			},
		}
	}

	for _, serverSharded := range []bool{true, false} {
		for totalShards := 1; totalShards <= 7; totalShards++ {
			t.Run(fmt.Sprintf("serverSharded=%v/totalShards=%d", serverSharded, totalShards), func(t *testing.T) {
				seen := make(map[string]int)
				seenPaginated := make(map[string]int)
				seenWatchList := make(map[string]int)

				for shard := range totalShards {
					sel, err := sharding.NewShardRangeSelector("object.metadata.uid", shard, totalShards)
					if err != nil {
						t.Fatalf("NewShardRangeSelector: %v", err)
					}

					var gotListSelector, gotWatchSelector string
					baseLW := &ListWatch{
						ListWithContextFunc: func(_ context.Context, opts metav1.ListOptions) (runtime.Object, error) {
							gotListSelector = opts.ShardSelector
							var source []v1.Pod
							if serverSharded {
								for i := range pods {
									if ok, _ := sel.Matches(&pods[i]); ok {
										source = append(source, pods[i])
									}
								}
							} else {
								source = pods
							}

							start := 0
							if opts.Continue != "" {
								if _, err := fmt.Sscanf(opts.Continue, "%d", &start); err != nil {
									return nil, err
								}
							}
							end := len(source)
							nextContinue := ""
							if opts.Limit > 0 && start+int(opts.Limit) < len(source) {
								end = start + int(opts.Limit)
								nextContinue = fmt.Sprintf("%d", end)
							}

							out := &v1.PodList{
								ListMeta: metav1.ListMeta{
									ResourceVersion: "10",
									Continue:        nextContinue,
								},
								Items: slices.Clone(source[start:end]),
							}
							if serverSharded {
								out.ShardInfo = &metav1.ShardInfo{Selector: opts.ShardSelector}
							} else if nextContinue != "" {
								rem := int64(len(source) - end)
								out.RemainingItemCount = &rem
							}
							return out, nil
						},
						WatchFuncWithContext: func(_ context.Context, opts metav1.ListOptions) (watch.Interface, error) {
							gotWatchSelector = opts.ShardSelector
							fw := watch.NewFake()
							if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
								go func() {
									for i := range pods {
										if !serverSharded {
											fw.Add(&pods[i])
											continue
										}
										if ok, _ := sel.Matches(&pods[i]); ok {
											fw.Add(&pods[i])
										}
									}
									// Terminate initial events stream with the synthetic empty-UID bookmark.
									fw.Action(watch.Bookmark, &v1.Pod{
										ObjectMeta: metav1.ObjectMeta{
											ResourceVersion: "10",
											Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
										},
									})
								}()
							}
							return fw, nil
						},
					}

					// Path 1: Legacy paginated LIST (WatchListSemantics unsupported).
					legacyShardedLW := NewShardedListWatch(ToListWatcherWithWatchListSemantics(baseLW, fakeWatchListClient{unSupportedWatchListSemantics: true}), sel)
					if !watchlist.DoesClientNotSupportWatchListSemantics(legacyShardedLW) {
						t.Errorf("expected DoesClientNotSupportWatchListSemantics to be preserved")
					}
					shardedLWWithCtx := ToListerWatcherWithContext(legacyShardedLW)

					listObj, err := shardedLWWithCtx.ListWithContext(context.Background(), metav1.ListOptions{})
					if err != nil {
						t.Fatalf("ListWithContext: %v", err)
					}
					if gotListSelector != sel.String() {
						t.Errorf("expected List ShardSelector %q, got %q", sel.String(), gotListSelector)
					}

					podList := listObj.(*v1.PodList)
					if !serverSharded && podList.RemainingItemCount != nil {
						t.Errorf("expected RemainingItemCount to be cleared on client-filtered list")
					}
					for _, p := range podList.Items {
						if prev, dup := seen[p.Name]; dup {
							t.Errorf("pod %s seen in both shard %d and %d", p.Name, prev, shard)
						}
						seen[p.Name] = shard
					}

					// Verify Reflector.list pagination (with small page size so some unsharded pages filter to 0 items).
					store := NewStore(MetaNamespaceKeyFunc)
					reflector := NewReflector(legacyShardedLW, &v1.Pod{}, store, 0)
					reflector.WatchListPageSize = 2
					if err := reflector.list(context.Background()); err != nil {
						t.Fatalf("reflector.list: %v", err)
					}
					if got := len(store.List()); got != len(podList.Items) {
						t.Errorf("shard %d paginated reflector store count %d != unpaginated count %d", shard, got, len(podList.Items))
					}
					for _, item := range store.List() {
						p := item.(*v1.Pod)
						seenPaginated[p.Name] = shard
					}

					// Path 2: Default WatchListClient streaming initial events (SendInitialEvents=true).
					watchListShardedLW := NewShardedListWatch(baseLW, sel)
					if watchlist.DoesClientNotSupportWatchListSemantics(watchListShardedLW) {
						t.Errorf("expected WatchListSemantics to be supported by default")
					}
					watchListStore := NewStore(MetaNamespaceKeyFunc)
					watchListReflector := NewReflector(watchListShardedLW, &v1.Pod{}, watchListStore, 0)
					watchListReflector.useWatchList = true
					wlWatcher, err := watchListReflector.watchList(context.Background())
					if err != nil {
						t.Fatalf("reflector.watchList: %v", err)
					}
					wlWatcher.Stop()
					if got := len(watchListStore.List()); got != len(podList.Items) {
						t.Errorf("shard %d watchList reflector store count %d != list count %d", shard, got, len(podList.Items))
					}
					for _, item := range watchListStore.List() {
						p := item.(*v1.Pod)
						if prev, dup := seenWatchList[p.Name]; dup {
							t.Errorf("pod %s seen in watchList store for both shard %d and %d", p.Name, prev, shard)
						}
						seenWatchList[p.Name] = shard
					}

					// Live watch stream (Added + Bookmark + Error).
					liveFakeWatch := watch.NewFake()
					liveLW := NewShardedListWatch(&ListWatch{
						WatchFuncWithContext: func(_ context.Context, opts metav1.ListOptions) (watch.Interface, error) {
							gotWatchSelector = opts.ShardSelector
							return liveFakeWatch, nil
						},
					}, sel)
					w, err := ToListerWatcherWithContext(liveLW).WatchWithContext(context.Background(), metav1.ListOptions{})
					if err != nil {
						t.Fatalf("WatchWithContext: %v", err)
					}
					if gotWatchSelector != sel.String() {
						t.Errorf("expected Watch ShardSelector %q, got %q", sel.String(), gotWatchSelector)
					}
					go func() {
						// Even when LIST hit a serverSharded replica, send all pods on WATCH to
						// verify WatchWithContext filters out-of-shard events under HA server skew.
						for i := range pods {
							liveFakeWatch.Add(&pods[i])
						}
						// Bookmark (empty UID) and Error (*metav1.Status) must pass through on every shard.
						liveFakeWatch.Action(watch.Bookmark, &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "11"}})
						liveFakeWatch.Error(&metav1.Status{Status: metav1.StatusFailure, Message: "synthetic error"})
						liveFakeWatch.Stop()
					}()
					watchCount := 0
					bookmarks := 0
					errorsSeen := 0
					for ev := range w.ResultChan() {
						switch ev.Type {
						case watch.Bookmark:
							bookmarks++
						case watch.Error:
							errorsSeen++
						default:
							p := ev.Object.(*v1.Pod)
							if ok, _ := sel.Matches(p); !ok {
								t.Errorf("shard %d received out-of-shard watch event for %s", shard, p.Name)
							}
							watchCount++
						}
					}
					if watchCount != len(podList.Items) {
						t.Errorf("shard %d watch count %d != list count %d", shard, watchCount, len(podList.Items))
					}
					if bookmarks != 1 {
						t.Errorf("shard %d expected 1 bookmark event, got %d", shard, bookmarks)
					}
					if errorsSeen != 1 {
						t.Errorf("shard %d expected 1 error event, got %d", shard, errorsSeen)
					}
				}

				if len(seen) != len(pods) {
					t.Errorf("expected %d total pods across shards, got %d", len(pods), len(seen))
				}
				if len(seenPaginated) != len(pods) {
					t.Errorf("expected %d total paginated pods across shards, got %d", len(pods), len(seenPaginated))
				}
				if len(seenWatchList) != len(pods) {
					t.Errorf("expected %d total watchList pods across shards, got %d", len(pods), len(seenWatchList))
				}
			})
		}
	}
}

func TestNewShardedListWatchUnstructured(t *testing.T) {
	sel, err := sharding.NewShardRangeSelector("object.metadata.uid", 0, 2)
	if err != nil {
		t.Fatal(err)
	}

	makeItem := func(uid string) unstructured.Unstructured {
		u := unstructured.Unstructured{Object: map[string]any{}}
		u.SetUID(types.UID(uid))
		u.SetName(uid)
		return u
	}
	items := []unstructured.Unstructured{makeItem("uid-a"), makeItem("uid-b"), makeItem("uid-c"), makeItem("uid-d")}

	for _, serverSharded := range []bool{true, false} {
		t.Run(fmt.Sprintf("serverSharded=%v", serverSharded), func(t *testing.T) {
			lw := NewShardedListWatch(&ListWatch{
				ListWithContextFunc: func(_ context.Context, opts metav1.ListOptions) (runtime.Object, error) {
					out := &unstructured.UnstructuredList{Object: map[string]any{}}
					out.SetResourceVersion("1")
					if serverSharded {
						out.SetShardInfo(&metav1.ShardInfo{Selector: opts.ShardSelector})
						for i := range items {
							if ok, _ := sel.Matches(&items[i]); ok {
								out.Items = append(out.Items, items[i])
							}
						}
					} else {
						out.Items = append(out.Items, items...)
					}
					return out, nil
				},
			}, sel)

			obj, err := lw.List(metav1.ListOptions{})
			if err != nil {
				t.Fatalf("List: %v", err)
			}
			uList := obj.(*unstructured.UnstructuredList)
			for i := range uList.Items {
				if ok, _ := sel.Matches(&uList.Items[i]); !ok {
					t.Errorf("unexpected out-of-shard unstructured item %s", uList.Items[i].GetName())
				}
			}
		})
	}
}

func TestSharedIndexInformerShardSelector(t *testing.T) {
	pods := make([]v1.Pod, 12)
	for i := range pods {
		pods[i] = v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "default",
				UID:             types.UID(fmt.Sprintf("uid-%d", i)),
				ResourceVersion: "10",
			},
		}
	}

	const totalShards = 3
	seen := make(map[string]int, len(pods))
	for shard := range totalShards {
		sel, err := sharding.NewShardRangeSelector("object.metadata.uid", shard, totalShards)
		if err != nil {
			t.Fatalf("NewShardRangeSelector: %v", err)
		}

		var gotListSelector string
		lw := &ListWatch{
			ListWithContextFunc: func(_ context.Context, opts metav1.ListOptions) (runtime.Object, error) {
				gotListSelector = opts.ShardSelector
				return &v1.PodList{
					ListMeta: metav1.ListMeta{ResourceVersion: "10"},
					Items:    slices.Clone(pods),
				}, nil
			},
			WatchFuncWithContext: func(_ context.Context, _ metav1.ListOptions) (watch.Interface, error) {
				return watch.NewFake(), nil
			},
		}

		informer := NewSharedIndexInformer(ToListWatcherWithWatchListSemantics(lw, fakeWatchListClient{unSupportedWatchListSemantics: true}), &v1.Pod{}, 0, Indexers{})
		if err := informer.SetShardSelector(sel); err != nil {
			t.Fatalf("SetShardSelector before start: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		go informer.RunWithContext(ctx)
		if !WaitForCacheSync(ctx.Done(), informer.HasSynced) {
			cancel()
			t.Fatalf("shard %d failed to sync cache", shard)
		}
		if err := informer.SetShardSelector(sel); err == nil {
			cancel()
			t.Fatalf("expected error calling SetShardSelector after informer started")
		}
		cancel()

		if gotListSelector != sel.String() {
			t.Errorf("shard %d expected List ShardSelector %q, got %q", shard, sel.String(), gotListSelector)
		}
		for _, item := range informer.GetStore().List() {
			p := item.(*v1.Pod)
			if ok, _ := sel.Matches(p); !ok {
				t.Errorf("shard %d store contains out-of-shard pod %s", shard, p.Name)
			}
			if prev, dup := seen[p.Name]; dup {
				t.Errorf("pod %s seen in both shard %d and %d", p.Name, prev, shard)
			}
			seen[p.Name] = shard
		}
	}
	if len(seen) != len(pods) {
		t.Errorf("expected %d total pods across sharded informers, got %d", len(pods), len(seen))
	}
}
