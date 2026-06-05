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
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/yaml"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
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
	loadNone            = "None"
	loadWatcher         = "Watcher"
	loadLister          = "Lister"
	loadWatchList       = "WatchList"
	trafficDeleteCreate = "DeleteCreate"
	trafficPatch        = "Patch"
)

func RunBenchmarkWriteThroughput(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, hasIndex bool) {
	require.NoError(b, PrecreateBenchmarkPods(ctx, store, data))
	require.NoError(b, waitForConsistent(ctx, store))

	for _, trafficType := range []string{trafficDeleteCreate, trafficPatch} {
		b.Run(fmt.Sprintf("Traffic=%s", trafficType), func(b *testing.B) {
			for _, parallelism := range []int{25} {
				b.Run(fmt.Sprintf("Parallelism=%d", parallelism), func(b *testing.B) {
					for _, loadType := range []string{loadNone, loadWatcher, loadLister, loadWatchList} {
						useIndexOptions := []bool{false}
						if hasIndex && loadType != loadNone {
							useIndexOptions = []bool{false, true}
						}
						for _, readIndexed := range useIndexOptions {
							b.Run(fmt.Sprintf("Background=%s/UseIndex=%v", loadType, readIndexed), func(b *testing.B) {
								b.SetParallelism(parallelism)
								runBenchmarkWriteThroughput(ctx, b, store, data, trafficType, loadType, readIndexed)
							})
						}
					}
				})
			}
		})
	}
}

func runBenchmarkWriteThroughput(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, trafficType string, loadType string, readIndexed bool) {
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

	switch loadType {
	case loadNone:
	case loadWatcher:
		startBackgroundWatchers(ctx, store, data, 10, readIndexed, &workersWg, stopBackgroundLoadCh, &watchEvents)
	case loadLister:
		startBackgroundListers(ctx, store, data, 1, readIndexed, &workersWg, stopBackgroundLoadCh, &listCalls, &listObjects)
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
	start := time.Now()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(index.Add(1)) % len(data.PodKeys)
			writes.Add(runTraffic(ctx, b, store, data, trafficType, i))
		}
	})
	end := time.Now()
	elapsedSeconds := end.Sub(start).Seconds()
	require.NoError(b, waitForConsistent(ctx, store))
	consistentDelaySeconds := float64(time.Since(end).Nanoseconds()) / float64(time.Second.Nanoseconds())
	b.ReportMetric(consistentDelaySeconds, "seconds-delay")
	b.ReportMetric(float64(writes.Load())/elapsedSeconds, "writes/s")

	stopBackgroundLoad()

	switch loadType {
	case loadWatcher:
		b.ReportMetric(float64(watchEvents.Load())/elapsedSeconds, "watch-events/s")
	case loadLister, loadWatchList:
		b.ReportMetric(float64(listCalls.Load())/elapsedSeconds, "list-calls/s")
		b.ReportMetric(float64(listObjects.Load())/elapsedSeconds, "list-objs/s")
	}
}

func waitForConsistent(ctx context.Context, store storage.Interface) error {
	listOut := &example.PodList{}
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

func runTraffic(ctx context.Context, b *testing.B, store storage.Interface, data BenchmarkData, trafficType string, index int) (writes uint64) {
	var podOut *example.Pod
	switch trafficType {
	case trafficDeleteCreate:
		podOut = &example.Pod{}
		err := store.Delete(ctx, data.PodKeys[index], podOut, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
		if err == nil {
			writes += 1
		} else if !storage.IsNotFound(err) {
			panic(fmt.Sprintf("Unexpected error on Delete %q: %v", data.PodKeys[index], err))
		}
		pod := data.Pods[index]
		podOut = &example.Pod{}
		err = store.Create(ctx, data.PodKeys[index], pod, podOut, 0)
		if err == nil {
			writes += 1
		} else if !storage.IsExist(err) {
			panic(fmt.Sprintf("Unexpected error on Create %q: %v", data.PodKeys[index], err))
		}
	case trafficPatch:
		podOut = &example.Pod{}
		err := store.GuaranteedUpdate(ctx, data.PodKeys[index], podOut, false, nil, patchFunc(index), nil)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error on Patch %q: %v", data.PodKeys[index], err))
		} else {
			writes += 1
		}
	default:
		panic(fmt.Sprintf("Unknown traffic type: %s", trafficType))
	}
	return writes
}

func patchFunc(i int) func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
	return func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		curr := input.(*example.Pod)
		if curr.Annotations == nil {
			curr.Annotations = make(map[string]string)
		}
		curr.Annotations["updated-by-benchmark"] = strconv.Itoa(i)
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

func startBackgroundListers(ctx context.Context, store storage.Interface, data BenchmarkData, count int, readIndexed bool, wg *sync.WaitGroup, stopCh <-chan struct{}, listCounter *atomic.Uint64, objCounter *atomic.Uint64) {
	for i := range count {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			listOut := &example.PodList{}
			ticker := time.NewTicker(10 * time.Millisecond)
			defer ticker.Stop()
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
			for {
				select {
				case <-stopCh:
					return
				case <-ctx.Done():
					return
				case <-ticker.C:
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
							pod, ok := ev.Object.(*example.Pod)
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
	wg := sync.WaitGroup{}
	objectCount := atomic.Uint64{}
	pageCount := atomic.Uint64{}
	for i := 0; i < b.N; i++ {
		wg.Add(1)
		resourceVersion := ""
		switch match {
		case metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan:
			maxRevision := 1 + len(data.Pods)
			resourceVersion = fmt.Sprintf("%d", maxRevision-99+i%100)
		}
		go func(resourceVersion, nodeName, namespaceName string) {
			defer wg.Done()
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
				objects, pages := paginateList(ctx, store, "/pods/", opts)
				objectCount.Add(uint64(objects))
				pageCount.Add(uint64(pages))
			case node:
				if useIndex {
					opts.Predicate.GetAttrs = podAttr
					opts.Predicate.IndexFields = []string{"spec.nodeName"}
					opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
				}
				objects, pages := paginateList(ctx, store, "/pods/", opts)
				objectCount.Add(uint64(objects))
				pageCount.Add(uint64(pages))
			case namespace:
				ctx := ctx
				if useIndex {
					opts.Predicate.IndexFields = []string{"metadata.namespace"}
					ctx = request.WithRequestInfo(ctx, &request.RequestInfo{Namespace: namespaceName})
				}
				objects, pages := paginateList(ctx, store, "/pods/"+namespaceName, opts)
				objectCount.Add(uint64(objects))
				pageCount.Add(uint64(pages))
			}
		}(resourceVersion, data.NodeNames[i%len(data.NodeNames)], data.NamespaceNames[i%len(data.NamespaceNames)])
	}
	wg.Wait()
	b.ReportMetric(float64(objectCount.Load())/float64(b.N), "objects/op")
	b.ReportMetric(float64(pageCount.Load())/float64(b.N), "pages/op")
}

func paginateList(ctx context.Context, store storage.Interface, key string, opts storage.ListOptions) (objectCount int, pageCount int) {
	listOut := &example.PodList{}
	err := store.GetList(ctx, key, opts, listOut)
	if err != nil {
		panic(fmt.Sprintf("Unexpected error %s", err))
	}
	opts.Predicate.Continue = listOut.Continue
	opts.ResourceVersion = ""
	opts.ResourceVersionMatch = ""
	pageCount += 1
	objectCount += len(listOut.Items)
	for opts.Predicate.Continue != "" {
		listOut := &example.PodList{}
		err := store.GetList(ctx, key, opts, listOut)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error %s", err))
		}
		opts.Predicate.Continue = listOut.Continue
		pageCount += 1
		objectCount += len(listOut.Items)
	}
	return objectCount, pageCount
}

func podAttr(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod := obj.(*example.Pod)
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
			randomizePod(p, namespace, data.NodeNames[rand.Intn(nodeCount)])
			data.Pods = append(data.Pods, p)
			data.PodKeys = append(data.PodKeys, computePodKey(p))
		}
	}
	return data
}

func PrecreateBenchmarkPods(ctx context.Context, store storage.Interface, data BenchmarkData) error {
	podOut := &example.Pod{}
	for _, pod := range data.Pods {
		key := computePodKey(pod)
		err := store.Create(ctx, key, pod, podOut, 0)
		if err != nil && !storage.IsExist(err) {
			return fmt.Errorf("unexpected error pre-creating pod %q: %w", key, err)
		}
	}
	return nil
}

type BenchmarkData struct {
	Pods           []*example.Pod
	PodKeys        []string
	NamespaceNames []string
	NodeNames      []string
}

func loadExemplarPod() *example.Pod {
	var pod example.Pod
	if len(exemplarPodYAML) == 0 {
		panic("exemplar pod empty")
	}
	if err := yaml.Unmarshal(exemplarPodYAML, &pod); err != nil {
		panic(fmt.Sprintf("decode exemplar pod: %v", err))
	}
	return &pod
}

func randomizePod(pod *example.Pod, ns string, nodeName string) {
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
