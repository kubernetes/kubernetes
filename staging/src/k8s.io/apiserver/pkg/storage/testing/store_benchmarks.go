/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"context"
	_ "embed"
	"fmt"
	goruntime "runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"
	"sigs.k8s.io/yaml"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
)

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

type scope string

var (
	cluster   scope = "Cluster"
	node      scope = "Node"
	namespace scope = "Namespace"
)

const (
	loadNone               = "None"
	loadWatcher            = "Watcher"
	loadLister             = "Lister"
	loadListerExactRV      = "ListerExactRV"
	loadListerNotOlderThan = "ListerNotOlderThan"
	loadWatchList          = "WatchList"
	trafficDeleteCreate    = "DeleteCreate"
	trafficPatch           = "Patch"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(metav1.AddMetaToScheme(scheme))
	scheme.AddUnversionedTypes(corev1.SchemeGroupVersion, &metav1.Status{})
	pb := protobuf.NewSerializer(scheme, scheme)
	corev1ProtoCodec = codecs.CodecForVersions(pb, pb, schema.GroupVersions{corev1.SchemeGroupVersion}, schema.GroupVersions{corev1.SchemeGroupVersion})
}

var (
	corev1ProtoCodec runtime.Codec
)

// StoreConfig is subset of Storage configuration that is shared between cacher and etcd3. Used to ensure consistent config for setting up a store for benchmarks.
type StoreConfig struct {
	Versioner      storage.Versioner
	GroupResource  schema.GroupResource
	ResourcePrefix string
	KeyFunc        func(runtime.Object) (string, error)
	GetAttrsFunc   func(runtime.Object) (label labels.Set, field fields.Set, err error)
	NewFunc        func() runtime.Object
	NewListFunc    func() runtime.Object
	Codec          runtime.Codec
}

func StoreConfigForBenchmarks() StoreConfig {
	prefix := "/pods/"
	return StoreConfig{
		Versioner:      storage.APIObjectVersioner{},
		GroupResource:  schema.GroupResource{Resource: "pods"},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   getCorev1PodAttrs,
		NewFunc:        func() runtime.Object { return &corev1.Pod{} },
		NewListFunc:    func() runtime.Object { return &corev1.PodList{} },
		Codec:          corev1ProtoCodec,
	}
}

func getCorev1PodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	fs := fields.Set{
		"metadata.name":      pod.Name,
		"metadata.namespace": pod.Namespace,
		"spec.nodeName":      pod.Spec.NodeName,
		"spec.restartPolicy": string(pod.Spec.RestartPolicy),
		"status.phase":       string(pod.Status.Phase),
	}
	return labels.Set(pod.Labels), fs, nil
}

func RunBenchmarkWriteThroughput(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, hasIndex bool, tracker *WatchLatencyTracker) {
	require.NoError(b, PrecreateBenchmarkPods(ctx, store, data))
	require.NoError(b, waitForConsistent(ctx, store))

	for _, trafficType := range []string{trafficDeleteCreate, trafficPatch} {
		b.Run(fmt.Sprintf("Traffic=%s", trafficType), func(b *testing.B) {
			for _, parallelism := range []int{25} {
				b.Run(fmt.Sprintf("Parallelism=%d", parallelism), func(b *testing.B) {
					for _, loadType := range []string{loadNone, loadWatcher, loadLister, loadListerExactRV, loadListerNotOlderThan, loadWatchList} {
						useIndexOptions := []bool{false}
						if hasIndex && loadType != loadNone {
							useIndexOptions = []bool{false, true}
						}
						for _, readIndexed := range useIndexOptions {
							b.Run(fmt.Sprintf("Background=%s/UseIndex=%v", loadType, readIndexed), func(b *testing.B) {
								b.SetParallelism(parallelism)
								if tracker != nil {
									tracker.Reset()
								}
								runBenchmarkWriteThroughput(ctx, b, store, data, trafficType, loadType, readIndexed, tracker)
							})
						}
					}
				})
			}
		})
	}
}

func runBenchmarkWriteThroughput(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, trafficType string, loadType string, readIndexed bool, tracker *WatchLatencyTracker) {
	stopBackgroundLoadCh := make(chan struct{})
	var workersWg sync.WaitGroup
	var stopOnce sync.Once
	stopBackgroundLoad := func() {
		stopOnce.Do(func() {
			close(stopBackgroundLoadCh)
			workersWg.Wait()
		})
	}
	defer stopBackgroundLoad()

	var writes atomic.Uint64
	var watchEvents atomic.Uint64
	var listCalls atomic.Uint64
	var listObjects atomic.Uint64
	var index atomic.Uint64
	var latestRV atomic.Pointer[string]
	initialRV := "0"
	latestRV.Store(&initialRV)

	switch loadType {
	case loadNone:
	case loadWatcher:
		startBackgroundWatchers(ctx, store, data, 10, readIndexed, &workersWg, stopBackgroundLoadCh, &watchEvents)
	case loadLister:
		startBackgroundListers(ctx, store, data, 1, readIndexed, &workersWg, stopBackgroundLoadCh, &listCalls, &listObjects, "", &latestRV)
	case loadListerExactRV:
		startBackgroundListers(ctx, store, data, 1, readIndexed, &workersWg, stopBackgroundLoadCh, &listCalls, &listObjects, metav1.ResourceVersionMatchExact, &latestRV)
	case loadListerNotOlderThan:
		startBackgroundListers(ctx, store, data, 1, readIndexed, &workersWg, stopBackgroundLoadCh, &listCalls, &listObjects, metav1.ResourceVersionMatchNotOlderThan, &latestRV)
	case loadWatchList:
		startBackgroundWatchListers(ctx, store, data, 1, readIndexed, &workersWg, stopBackgroundLoadCh, &listCalls, &listObjects)
	default:
		panic(fmt.Sprintf("Unknown load type: %s", loadType))
	}
	writes.Store(0)
	watchEvents.Store(0)
	listCalls.Store(0)
	listObjects.Store(0)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(index.Add(1)) % len(data.PodKeys)
			writes.Add(runTraffic(ctx, b, store, data, trafficType, i, &latestRV, tracker))
		}
	})
	elapsedSeconds := b.Elapsed().Seconds()
	require.NoError(b, waitForConsistent(ctx, store))
	b.ReportMetric(float64(writes.Load())/elapsedSeconds, "writes/s")

	stopBackgroundLoad()

	switch loadType {
	case loadWatcher:
		b.ReportMetric(float64(watchEvents.Load())/elapsedSeconds, "watch-events/s")
	case loadLister, loadListerExactRV, loadListerNotOlderThan, loadWatchList:
		b.ReportMetric(float64(listCalls.Load())/elapsedSeconds, "list-calls/s")
		b.ReportMetric(float64(listObjects.Load())/elapsedSeconds, "list-objs/s")
	}

	if tracker != nil {
		if p99 := tracker.GetP99Latency(); p99 > 0 {
			b.ReportMetric(p99.Seconds(), "watch-latency-p99-s")
		}
	}
}

func waitForConsistent(ctx context.Context, store storage.Interface) error {
	listOut := &corev1.PodList{}
	err := store.GetList(ctx, "/pods/", storage.ListOptions{
		Recursive: true,
		Predicate: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.Everything(),
			Limit: 1,
		},
	}, listOut)
	if err != nil {
		return fmt.Errorf("unexpected error waiting for consistency: %w", err)
	}
	return nil
}

func runTraffic(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, trafficType string, index int, latestRV *atomic.Pointer[string], tracker *WatchLatencyTracker) (writes uint64) {
	var podOut *corev1.Pod
	switch trafficType {
	case trafficDeleteCreate:
		podOut = &corev1.Pod{}
		err := store.Delete(ctx, data.PodKeys[index], podOut, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
		if err == nil {
			writes += 1
		} else if !storage.IsNotFound(err) {
			panic(fmt.Sprintf("Unexpected error on Delete %q: %v", data.PodKeys[index], err))
		}
		pod := data.Pods[index].DeepCopy()
		if tracker != nil {
			tracker.RecordWrite(pod)
		}
		podOut = &corev1.Pod{}
		err = store.Create(ctx, data.PodKeys[index], pod, podOut, 0)
		if err == nil {
			writes += 1
			latestRV.Store(&podOut.ResourceVersion)
		} else if !storage.IsExist(err) {
			panic(fmt.Sprintf("Unexpected error on Create %q: %v", data.PodKeys[index], err))
		}
	case trafficPatch:
		podOut = &corev1.Pod{}
		err := store.GuaranteedUpdate(ctx, data.PodKeys[index], podOut, false, nil, patchFunc(index, tracker), nil)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error on Patch %q: %v", data.PodKeys[index], err))
		} else {
			writes += 1
			latestRV.Store(&podOut.ResourceVersion)
		}
	default:
		panic(fmt.Sprintf("Unknown traffic type: %s", trafficType))
	}
	return writes
}

func patchFunc(i int, tracker *WatchLatencyTracker) func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
	return func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		curr := input.(*corev1.Pod)
		if curr.Annotations == nil {
			curr.Annotations = make(map[string]string)
		}
		curr.Annotations["updated-by-benchmark"] = strconv.Itoa(i)
		if tracker != nil {
			tracker.RecordWrite(curr)
		}
		return curr, nil, nil
	}
}

func startBackgroundWatchers(ctx context.Context, store storage.Interface, data BenchmarkData, count int, readIndexed bool, wg *sync.WaitGroup, stopCh <-chan struct{}, eventCounter *atomic.Uint64) {
	for i := range count {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			opts := storage.ListOptions{
				Recursive: true,
				Predicate: storage.Everything,
			}
			if readIndexed {
				nodeName := "default-node"
				if len(data.NodeNames) > 0 {
					nodeName = data.NodeNames[i%len(data.NodeNames)]
				}
				opts.Predicate.GetAttrs = podAttr
				opts.Predicate.IndexFields = []string{"spec.nodeName"}
				opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
			}
			w, err := store.Watch(ctx, "/pods/", opts)
			if err != nil {
				return
			}
			defer w.Stop()
			for {
				select {
				case <-stopCh:
					return
				case <-ctx.Done():
					return
				case ev, ok := <-w.ResultChan():
					if !ok {
						return
					}
					eventCounter.Add(1)
					_ = ev
				}
			}
		}(i)
	}
}

func startBackgroundListers(ctx context.Context, store storage.Interface, data BenchmarkData, count int, readIndexed bool, wg *sync.WaitGroup, stopCh <-chan struct{}, listCounter *atomic.Uint64, objCounter *atomic.Uint64, rvMatch metav1.ResourceVersionMatch, latestRV *atomic.Pointer[string]) {
	for i := range count {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			listOut := &corev1.PodList{}
			ticker := time.NewTicker(10 * time.Millisecond)
			defer ticker.Stop()
			for {
				select {
				case <-stopCh:
					return
				case <-ctx.Done():
					return
				case <-ticker.C:
					opts := storage.ListOptions{
						Recursive:            true,
						ResourceVersionMatch: rvMatch,
						Predicate:            storage.Everything,
					}
					switch rvMatch {
					case metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan:
						rv := *latestRV.Load()
						if rv == "0" || rv == "" {
							continue
						}
						opts.ResourceVersion = rv
					case "":
					default:
						panic(fmt.Sprintf("Unknown rvMatch: %s", rvMatch))
					}
					if readIndexed {
						nodeName := "default-node"
						if len(data.NodeNames) > 0 {
							nodeName = data.NodeNames[i%len(data.NodeNames)]
						}
						opts.Predicate.GetAttrs = podAttr
						opts.Predicate.IndexFields = []string{"spec.nodeName"}
						opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
					}
					err := store.GetList(ctx, "/pods/", opts, listOut)
					if err == nil {
						listCounter.Add(1)
						objCounter.Add(uint64(len(listOut.Items)))
					}
				}
			}
		}(i)
	}
}

func startBackgroundWatchListers(ctx context.Context, store storage.Interface, data BenchmarkData, count int, readIndexed bool, wg *sync.WaitGroup, stopCh <-chan struct{}, listCounter *atomic.Uint64, objCounter *atomic.Uint64) {
	for i := range count {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			opts := storage.ListOptions{
				Recursive:         true,
				Predicate:         storage.Everything,
				SendInitialEvents: new(true),
			}
			opts.Predicate.AllowWatchBookmarks = true

			if readIndexed {
				nodeName := "default-node"
				if len(data.NodeNames) > 0 {
					nodeName = data.NodeNames[i%len(data.NodeNames)]
				}
				opts.Predicate.GetAttrs = podAttr
				opts.Predicate.IndexFields = []string{"spec.nodeName"}
				opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
			}

			for {
				select {
				case <-stopCh:
					return
				case <-ctx.Done():
					return
				default:
				}

				w, err := store.Watch(ctx, "/pods/", opts)
				if err != nil {
					time.Sleep(10 * time.Millisecond)
					continue
				}

				initialFinished := false
				for !initialFinished {
					select {
					case <-stopCh:
						w.Stop()
						return
					case <-ctx.Done():
						w.Stop()
						return
					case ev, ok := <-w.ResultChan():
						if !ok {
							initialFinished = true
							break
						}
						switch ev.Type {
						case watch.Bookmark:
							pod, ok := ev.Object.(*corev1.Pod)
							if !ok {
								panic("Unexpected type in event")
							}
							if pod.Annotations != nil && pod.Annotations[metav1.InitialEventsAnnotationKey] == "true" {
								initialFinished = true
							}
						default:
							objCounter.Add(1)
						}
					}
				}
				w.Stop()
				listCounter.Add(1)
			}
		}(i)
	}
}

func RunBenchmarkStoreList(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, useIndex bool) {
	for _, rvm := range []metav1.ResourceVersionMatch{"", metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan} {
		b.Run(fmt.Sprintf("RV=%s", rvm), func(b *testing.B) {
			for _, scope := range []scope{cluster, node, namespace} {
				b.Run(fmt.Sprintf("Scope=%s", scope), func(b *testing.B) {
					var expectedElements int
					switch scope {
					case namespace:
						expectedElements = len(data.Pods) / len(data.NamespaceNames)
					case node:
						expectedElements = len(data.Pods) / len(data.NodeNames)
					case cluster:
						expectedElements = len(data.Pods)
					}
					limitOptions := []int64{0}
					switch {
					case expectedElements > 1000:
						limitOptions = append(limitOptions, 1000)
					case expectedElements > 100:
						limitOptions = append(limitOptions, 100)
					}
					for _, limit := range limitOptions {
						b.Run(fmt.Sprintf("Paginate=%v", limit), func(b *testing.B) {
							runBenchmarkStoreList(ctx, b, store, limit, rvm, scope, data, useIndex)
						})
					}
				})
			}
		})
	}
}

func runBenchmarkStoreList(ctx context.Context, b *testing.B, store storage.Interface, limit int64, match metav1.ResourceVersionMatch, scope scope, data BenchmarkData, useIndex bool) {
	objectCount := atomic.Uint64{}
	listCount := atomic.Uint64{}
	var index atomic.Uint64

	b.SetParallelism(4)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(index.Add(1))
			resourceVersion := ""
			switch match {
			case metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan:
				maxRevision := 1 + len(data.Pods)
				resourceVersion = fmt.Sprintf("%d", maxRevision-99+i%100)
			}
			nodeName := data.NodeNames[i%len(data.NodeNames)]
			namespaceName := data.NamespaceNames[i%len(data.NamespaceNames)]

			opts := storage.ListOptions{
				Recursive:            true,
				ResourceVersion:      resourceVersion,
				ResourceVersionMatch: match,
				Predicate: storage.SelectionPredicate{
					GetAttrs: podAttr,
					Label:    labels.Everything(),
					Field:    fields.Everything(),
					Limit:    limit,
				},
			}
			switch scope {
			case cluster:
				objects, lists := paginateList(ctx, store, "/pods/", opts)
				objectCount.Add(uint64(objects))
				listCount.Add(uint64(lists))
			case node:
				opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
				if useIndex {
					opts.Predicate.IndexFields = []string{"spec.nodeName"}
				}
				objects, lists := paginateList(ctx, store, "/pods/", opts)
				if objects == 0 {
					b.Errorf("Scope=Node list for node %q returned no objects", nodeName)
				}
				objectCount.Add(uint64(objects))
				listCount.Add(uint64(lists))
			case namespace:
				ctx := ctx
				if useIndex {
					opts.Predicate.IndexFields = []string{"metadata.namespace"}
					ctx = request.WithRequestInfo(ctx, &request.RequestInfo{Namespace: namespaceName})
				}
				objects, lists := paginateList(ctx, store, "/pods/"+namespaceName, opts)
				objectCount.Add(uint64(objects))
				listCount.Add(uint64(lists))
			}
		}
	})
	elapsedSeconds := b.Elapsed().Seconds()
	b.ReportMetric(float64(objectCount.Load())/elapsedSeconds, "list-objs/s")
	b.ReportMetric(float64(listCount.Load())/elapsedSeconds, "list-calls/s")
}

func paginateList(ctx context.Context, store storage.Interface, key string, opts storage.ListOptions) (objectCount int, listCount int) {
	listOut := &corev1.PodList{}
	err := store.GetList(ctx, key, opts, listOut)
	if err != nil {
		panic(fmt.Sprintf("Unexpected error %s", err))
	}
	opts.Predicate.Continue = listOut.Continue
	opts.ResourceVersion = ""
	opts.ResourceVersionMatch = ""
	listCount += 1
	objectCount += len(listOut.Items)
	for opts.Predicate.Continue != "" {
		listOut := &corev1.PodList{}
		err := store.GetList(ctx, key, opts, listOut)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error %s", err))
		}
		opts.Predicate.Continue = listOut.Continue
		listCount += 1
		objectCount += len(listOut.Items)
	}
	return objectCount, listCount
}

func podAttr(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod := obj.(*corev1.Pod)
	return nil, fields.Set{
		"spec.nodeName":      pod.Spec.NodeName,
		"metadata.namespace": pod.Namespace,
	}, nil
}

func PrepareBenchmarkData(namespaceCount, podPerNamespaceCount, nodeCount int) (data BenchmarkData) {
	exemplar := loadExemplarPod()
	data.NodeNames = make([]string, nodeCount)
	for i := 0; i < nodeCount; i++ {
		data.NodeNames[i] = rand.String(10)
	}
	data.NamespaceNames = make([]string, namespaceCount)
	for i := 0; i < namespaceCount; i++ {
		namespace := rand.String(10)
		data.NamespaceNames[i] = namespace
		for j := 0; j < podPerNamespaceCount; j++ {
			p := exemplar.DeepCopy()
			nodeIdx := (i*podPerNamespaceCount + j) % nodeCount
			randomizePod(p, namespace, data.NodeNames[nodeIdx])
			data.Pods = append(data.Pods, p)
			data.PodKeys = append(data.PodKeys, computePodKey(p))
		}
	}
	return data
}

func RunBenchmarkStoreWatch(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, useIndex bool) {
	require.NoError(b, waitForConsistent(ctx, store))
	const watchersCount = 10
	for _, initMode := range []string{"SendInitialEvents", "RV0", "HistoryRV1"} {
		b.Run(fmt.Sprintf("Mode=%s", initMode), func(b *testing.B) {
			for _, scope := range []scope{cluster, node, namespace} {
				b.Run(fmt.Sprintf("Scope=%s", scope), func(b *testing.B) {
					runBenchmarkStoreWatch(ctx, b, store, watchersCount, initMode, scope, data, useIndex)
				})
			}
		})
	}
}

func runBenchmarkStoreWatch(ctx context.Context, b *testing.B, store storage.Interface, watchersCount int, initMode string, scope scope, data BenchmarkData, useIndex bool) {
	var expectedElements int
	switch scope {
	case namespace:
		expectedElements = len(data.Pods) / len(data.NamespaceNames)
	case node:
		expectedElements = len(data.Pods) / len(data.NodeNames)
	case cluster:
		expectedElements = len(data.Pods)
	}

	var totalEvents atomic.Uint64
	var totalWatches atomic.Uint64

	var memBefore, memAfter goruntime.MemStats
	b.ReportAllocs()
	b.ResetTimer()
	goruntime.ReadMemStats(&memBefore)

	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		wg.Add(watchersCount)
		for wIdx := range watchersCount {
			go func(wIdx int) {
				defer wg.Done()

				nodeName := data.NodeNames[wIdx%len(data.NodeNames)]
				namespaceName := data.NamespaceNames[wIdx%len(data.NamespaceNames)]

				opts := storage.ListOptions{
					Recursive: true,
					Predicate: storage.SelectionPredicate{
						GetAttrs: podAttr,
						Label:    labels.Everything(),
						Field:    fields.Everything(),
					},
				}

				switch initMode {
				case "SendInitialEvents":
					sendInitial := true
					opts.SendInitialEvents = &sendInitial
					opts.Predicate.AllowWatchBookmarks = true
				case "RV0":
					opts.ResourceVersion = "0"
				case "HistoryRV1":
					opts.ResourceVersion = "1"
				default:
					panic(fmt.Sprintf("unknown initMode: %s", initMode))
				}

				watchKey := "/pods/"
				watchCtx := ctx
				switch scope {
				case cluster:
				case node:
					opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
					if useIndex {
						opts.Predicate.IndexFields = []string{"spec.nodeName"}
					}
				case namespace:
					watchKey = "/pods/" + namespaceName
					if useIndex {
						opts.Predicate.IndexFields = []string{"metadata.namespace"}
						watchCtx = request.WithRequestInfo(ctx, &request.RequestInfo{Namespace: namespaceName})
					}
				}

				w, err := store.Watch(watchCtx, watchKey, opts)
				if err != nil {
					b.Errorf("Watch failed: %v", err)
					return
				}
				defer w.Stop()

				received := 0
				for received < expectedElements {
					select {
					case <-ctx.Done():
						return
					case ev, ok := <-w.ResultChan():
						if !ok {
							b.Errorf("Watch channel closed early after %d/%d events", received, expectedElements)
							return
						}
						switch ev.Type {
						case watch.Bookmark:
							pod, ok := ev.Object.(*example.Pod)
							if ok && pod.Annotations != nil && pod.Annotations[metav1.InitialEventsAnnotationKey] == "true" {
								if received != expectedElements {
									b.Errorf("InitialEvents bookmark received after %d events, expected %d", received, expectedElements)
								}
								totalEvents.Add(uint64(received))
								totalWatches.Add(1)
								return
							}
						case watch.Added, watch.Modified, watch.Deleted:
							received++
						case watch.Error:
							b.Errorf("Unexpected watch error event: %#v", ev.Object)
							return
						}
					}
				}

				if initMode == "SendInitialEvents" {
					// Drain the initial-events-end bookmark as part of full initial stream delivery.
					for {
						select {
						case <-ctx.Done():
							return
						case ev, ok := <-w.ResultChan():
							if !ok {
								return
							}
							if ev.Type == watch.Bookmark {
								if pod, ok := ev.Object.(*example.Pod); ok && pod.Annotations != nil && pod.Annotations[metav1.InitialEventsAnnotationKey] == "true" {
									totalEvents.Add(uint64(received))
									totalWatches.Add(1)
									return
								}
							}
						}
					}
				}

				totalEvents.Add(uint64(received))
				totalWatches.Add(1)
			}(wIdx)
		}
		wg.Wait()
	}

	goruntime.ReadMemStats(&memAfter)
	b.StopTimer()

	elapsedSeconds := b.Elapsed().Seconds()
	watches := totalWatches.Load()
	events := totalEvents.Load()
	totalAllocs := memAfter.Mallocs - memBefore.Mallocs
	totalBytes := memAfter.TotalAlloc - memBefore.TotalAlloc

	if watches > 0 {
		b.ReportMetric(float64(totalAllocs)/float64(watches), "allocs/watch")
		b.ReportMetric(float64(totalBytes)/float64(watches), "B/watch")
	}
	if events > 0 {
		b.ReportMetric(float64(totalAllocs)/float64(events), "allocs/event")
	}
	if b.N > 0 {
		b.ReportMetric(elapsedSeconds*1000/float64(b.N), "ms/10-watches")
	}
	if elapsedSeconds > 0 {
		b.ReportMetric(float64(events)/elapsedSeconds, "watch-events/s")
	}
}

func PrecreateBenchmarkPods(ctx context.Context, store storage.Interface, data BenchmarkData) error {
	podOut := &corev1.Pod{}
	for _, pod := range data.Pods {
		key := computePodKey(pod)
		err := store.Create(ctx, key, pod, podOut, 0)
		if err != nil && !storage.IsExist(err) {
			return fmt.Errorf("unexpected error pre-creating pod %q: %w", key, err)
		}
	}
	return nil
}

func PrecreateBenchmarkPodsParallel(ctx context.Context, store storage.Interface, data BenchmarkData) error {
	const workers = 32
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	var idx atomic.Int64

	for range workers {
		wg.Go(func() {
			podOut := &example.Pod{}
			for {
				i := int(idx.Add(1) - 1)
				if i >= len(data.Pods) {
					return
				}
				pod := data.Pods[i]
				key := computePodKey(pod)
				err := store.Create(ctx, key, pod, podOut, 0)
				if err != nil && !storage.IsExist(err) {
					select {
					case errCh <- fmt.Errorf("unexpected error pre-creating pod %q: %w", key, err):
					default:
					}
					return
				}
			}
		})
	}
	wg.Wait()
	close(errCh)
	if err := <-errCh; err != nil {
		return err
	}
	return nil
}

type BenchmarkData struct {
	Pods           []*corev1.Pod
	PodKeys        []string
	NamespaceNames []string
	NodeNames      []string
}

func loadExemplarPod() *corev1.Pod {
	var pod corev1.Pod
	if len(exemplarPodYAML) == 0 {
		panic("exemplar pod empty")
	}
	if err := yaml.UnmarshalStrict(exemplarPodYAML, &pod); err != nil {
		panic(fmt.Sprintf("decode exemplar pod: %v", err))
	}
	return &pod
}

func randomizePod(pod *corev1.Pod, ns string, nodeName string) {
	pod.Namespace = ns
	pod.Name = pod.GenerateName + rand.String(10)
	pod.UID = types.UID(rand.String(36))
	pod.ResourceVersion = ""
	pod.Spec.NodeName = nodeName
}

func RunBenchmarkStoreStats(ctx context.Context, b *testing.B, store storage.Interface) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := store.Stats(ctx)
		if err != nil {
			b.Fatal(err)
		}
	}
}

const latencyTimestampAnnotation = "watch-latency-timestamp"

type WatchLatencyTracker struct {
	clock     clock.Clock
	mu        sync.Mutex
	durations []time.Duration
}

func NewWatchLatencyTracker(clk clock.Clock) *WatchLatencyTracker {
	return &WatchLatencyTracker{
		clock: clk,
	}
}

func (t *WatchLatencyTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.durations = nil
}

func (t *WatchLatencyTracker) RecordWrite(obj interface{}) {
	metaObj, ok := obj.(metav1.Object)
	if !ok {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	annotations := metaObj.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[latencyTimestampAnnotation] = serializeTimestamp(t.clock.Now())
	metaObj.SetAnnotations(annotations)
}

func (t *WatchLatencyTracker) HandleEvent(obj interface{}) {
	metaObj, ok := obj.(metav1.Object)
	if !ok {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	annotations := metaObj.GetAnnotations()
	if annotations == nil {
		return
	}
	tStr, ok := annotations[latencyTimestampAnnotation]
	if !ok {
		return
	}
	writeTime, err := parseTimestamp(tStr)
	if err != nil {
		return
	}
	delay := t.clock.Since(writeTime)
	t.durations = append(t.durations, delay)
}

func (t *WatchLatencyTracker) GetP99Latency() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.durations) < 100 {
		return 0
	}
	slices.Sort(t.durations)
	idx := len(t.durations)*99/100 - 1
	return t.durations[idx]
}

func serializeTimestamp(t time.Time) string {
	return strconv.FormatInt(t.UnixNano(), 10)
}

func parseTimestamp(s string) (time.Time, error) {
	tNano, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	return time.Unix(0, tNano), nil
}
