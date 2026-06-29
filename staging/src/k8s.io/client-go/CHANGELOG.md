This file documents Go API changes in client-go.

Go API changes are typically not included in the Kubernetes release notes, so
noteworthy Go API changes *may* be documented here. This is currently not
*required*, so consult the git history to see all changes.

### mutation cache: support informer events

Calling OnAddOrUpdate and OnDelete from event handlers is optional. Calling
them has the advantage that the mutation cache is updated sooner. It also fixes
incorrectly reporting a locally updated item when a) using "include adds" and
b) the updated item got deleted in the apiserver and informer cache.

Although formally a Go API break because the MutationCache interface gets
extended, in practice that interface is expected to have only the single
implementation in client-go itself, so no-one should be affected by this.

```
- ./tools/cache.MutationCache.OnAddOrUpdate: added
- ./tools/cache.MutationCache.OnDelete: added
- ./tools/cache.MutationCache.internal: added unexported method
```

### restmapper + discovery: add context-aware APIs

See [PR #129109](https://github.com/kubernetes/kubernetes/pull/129109).

```
- ./kubernetes.(*Clientset).Discovery: changed from func() k8s.io/client-go/discovery.DiscoveryInterface to func() k8s.io/client-go/discovery.DiscoveryInterfaces
- ./kubernetes.Interface.Discovery: changed from func() k8s.io/client-go/discovery.DiscoveryInterface to func() k8s.io/client-go/discovery.DiscoveryInterfaces
- ./kubernetes/fake.(*Clientset).Discovery: changed from func() k8s.io/client-go/discovery.DiscoveryInterface to func() k8s.io/client-go/discovery.DiscoveryInterfaces
```

### Remove v2beta1 aggregated discovery support from clients

See [PR #138271](https://github.com/kubernetes/kubernetes/pull/138271).

```
- ./discovery.AcceptV2Beta1: removed
- ./discovery.SplitGroupsAndResourcesV2Beta1: removed
```

### Add GC to client-go TLS cache

See [PR #136355](https://github.com/kubernetes/kubernetes/pull/136355).

```
- ./transport.DialerStopCh: removed
```

### Add metric tracking the latest cached rv of informers

See [PR #137419](https://github.com/kubernetes/kubernetes/pull/137419).

```
- ./tools/cache.FIFOMetricsProvider: removed
- ./tools/cache.InformerMetricsProvider.NewStoreResourceVersionMetric: added
- ./tools/cache.InformerOptions.FIFOMetricsProvider: removed
- ./tools/cache.NewIndexer: changed from func(KeyFunc, Indexers) Indexer to func(KeyFunc, Indexers, ...StoreOption) Indexer
- ./tools/cache.NewThreadSafeStore: changed from func(Indexers, Indices) ThreadSafeStore to func(Indexers, Indices, ...ThreadSafeStoreOption) ThreadSafeStore
- ./tools/cache.SetFIFOMetricsProvider: removed
- ./tools/cache.SharedIndexInformerOptions.FIFOMetricsProvider: removed
```

### Rename `name` to `command` in kuberc credentialPluginAllowlist entries

See [PR #137272](https://github.com/kubernetes/kubernetes/pull/137272).

```
- ./tools/clientcmd/api.AllowlistEntry.Name: removed
```

### Add Resource Version query and Bookmarks to thread safe store

See [PR #134827](https://github.com/kubernetes/kubernetes/pull/134827).

```
- ./tools/cache.Store.Bookmark: added
- ./tools/cache.Store.LastStoreSyncResourceVersion: added
- ./tools/cache.ThreadSafeStore.Bookmark: added
- ./tools/cache.ThreadSafeStore.DeleteWithObject: added
- ./tools/cache.ThreadSafeStore.LastStoreSyncResourceVersion: added
```

### apimachinery + client-go + device taint eviction unit test: context-aware Start/WaitFor, waiting through channels

See [PR #135395](https://github.com/kubernetes/kubernetes/pull/135395).

```
- ./informers.SharedInformerFactory.StartWithContext: added
- ./informers.SharedInformerFactory.WaitForCacheSyncWithContext: added
- ./tools/cache.Controller.HasSyncedChecker: added
- ./tools/cache.Queue.HasSyncedChecker: added
- ./tools/cache.ResourceEventHandlerRegistration.HasSyncedChecker: added
- ./tools/cache.SharedInformer.HasSyncedChecker: added
- ./tools/cache/synctrack.AsyncTracker.UpstreamHasSynced: removed
- ./tools/cache/synctrack.SingleFileTracker.UpstreamHasSynced: removed
```

### Add identifier-based queue depth metrics for RealFIFO

See [PR #135782](https://github.com/kubernetes/kubernetes/pull/135782).

```
- ./informers/internalinterfaces.SharedInformerFactory.InformerName: added
- ./informers/internalinterfaces.SharedInformerFactory.InformerName: added
```

### Ensure that processing does not block queue writers in RealFIFO

See [PR #136264](https://github.com/kubernetes/kubernetes/pull/136264).

```
- ./tools/cache.Pop: removed
```

### Add atomic replace in client-go

See [PR #135462](https://github.com/kubernetes/kubernetes/pull/135462).

```
- ./tools/cache.(*RealFIFO).PopBatch: changed from func(ProcessBatchFunc) error to func(ProcessBatchFunc, PopProcessFunc) error
- ./tools/cache.QueueWithBatch.PopBatch: changed from func(ProcessBatchFunc) error to func(ProcessBatchFunc, PopProcessFunc) error
```

### Embed proper interface in TransformingStore to ensure DeltaFIFO and RealFIFO are implementing it

See [PR #135580](https://github.com/kubernetes/kubernetes/pull/135580).

```
- ./tools/cache.Store.Get, method set of TransformingStore: removed
- ./tools/cache.Store.GetByKey, method set of TransformingStore: removed
- ./tools/cache.Store.List, method set of TransformingStore: removed
- ./tools/cache.Store.ListKeys, method set of TransformingStore: removed
- ./tools/cache.TransformingStore: no longer implements ./tools/cache.KeyLister
- ./util/consistencydetector.CheckDataConsistency: changed from func(context.Context, string, string, ListFunc[T], k8s.io/apimachinery/pkg/apis/meta/v1.ListOptions, RetrieveItemsFunc[U]) to func(context.Context, string, string, ListFunc[T], TransformFunc, k8s.io/apimachinery/pkg/apis/meta/v1.ListOptions, RetrieveItemsFunc[U])
```

### Replace deprecated sets.String with sets.Set in client-go/tools/*

See [PR #133923](https://github.com/kubernetes/kubernetes/pull/133923).

```
- ./tools/cache.FakeExpirationPolicy.NeverExpire: changed from k8s.io/apimachinery/pkg/util/sets.String to k8s.io/apimachinery/pkg/util/sets.Set[string]
- ./tools/cache.Index: removed
```

### Changes for Kubernetes <= 1.34

For older changes refer to the commit messages and PR descriptions.
