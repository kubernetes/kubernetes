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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func RunBenchmarkStoreListCreate(ctx context.Context, b *testing.B, store storage.Interface, match metav1.ResourceVersionMatch) {
	objectCount := atomic.Uint64{}
	pods := []*example.Pod{}
	for i := 0; i < b.N; i++ {
		name := rand.String(100)
		pods = append(pods, &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name}})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pod := pods[i]
		podOut := &example.Pod{}
		err := store.Create(ctx, computePodKey(pod), pod, podOut, 0)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error %s", err))
		}
		listOut := &example.PodList{}
		err = store.GetList(ctx, "/pods", storage.ListOptions{
			Recursive:            true,
			ResourceVersion:      podOut.ResourceVersion,
			ResourceVersionMatch: match,
			Predicate: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
		}, listOut)
		if err != nil {
			panic(fmt.Sprintf("Unexpected error %s", err))
		}
		if len(listOut.Items) != 1 {
			b.Errorf("Expected to get 1 element, got %d", len(listOut.Items))
		}
		objectCount.Add(uint64(len(listOut.Items)))
	}
	b.ReportMetric(float64(objectCount.Load())/float64(b.N), "objects/op")
}

func RunBenchmarkStoreList(ctx context.Context, b *testing.B, store storage.Interface) {
	namespaceCount := 100
	podPerNamespaceCount := 100
	var paginateLimit int64 = 100
	nodeCount := 100
	namespacedNames, nodeNames := prepareBenchchmarkData(ctx, store, namespaceCount, podPerNamespaceCount, nodeCount)
	b.ResetTimer()
	maxRevision := 1 + namespaceCount*podPerNamespaceCount
	cases := []struct {
		name  string
		match metav1.ResourceVersionMatch
	}{
		{
			name:  "RV=Empty",
			match: "",
		},
		{
			name:  "RV=NotOlderThan",
			match: metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:  "RV=MatchExact",
			match: metav1.ResourceVersionMatchExact,
		},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			runBenchmarkStoreList(ctx, b, store, 0, maxRevision, c.match, false, nodeNames)
		})
	}
	b.Run("Paginate", func(b *testing.B) {
		for _, c := range cases {
			b.Run(c.name, func(b *testing.B) {
				runBenchmarkStoreList(ctx, b, store, paginateLimit, maxRevision, c.match, false, nodeNames)
			})
		}
	})
	b.Run("NodeIndexed", func(b *testing.B) {
		for _, c := range cases {
			b.Run(c.name, func(b *testing.B) {
				runBenchmarkStoreList(ctx, b, store, 0, maxRevision, c.match, true, nodeNames)
			})
		}
		b.Run("Paginate", func(b *testing.B) {
			for _, c := range cases {
				b.Run(c.name, func(b *testing.B) {
					runBenchmarkStoreList(ctx, b, store, paginateLimit, maxRevision, c.match, true, nodeNames)
				})
			}
		})
	})
	b.Run("Namespace", func(b *testing.B) {
		for _, c := range cases {
			b.Run(c.name, func(b *testing.B) {
				runBenchmarkStoreListNamespace(ctx, b, store, maxRevision, c.match, namespacedNames)
			})
		}
	})
}

func runBenchmarkStoreListNamespace(ctx context.Context, b *testing.B, store storage.Interface, maxRV int, match metav1.ResourceVersionMatch, namespaceNames []string) {
	wg := sync.WaitGroup{}
	objectCount := atomic.Uint64{}
	pageCount := atomic.Uint64{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg.Add(1)
		resourceVersion := ""
		switch match {
		case metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan:
			resourceVersion = fmt.Sprintf("%d", maxRV-99+i%100)
		}
		go func(resourceVersion string) {
			defer wg.Done()
			opts := storage.ListOptions{
				Recursive:            true,
				ResourceVersion:      resourceVersion,
				ResourceVersionMatch: match,
				Predicate:            storage.Everything,
			}
			for j := 0; j < len(namespaceNames); j++ {
				objects, pages := paginate(ctx, store, "/pods/"+namespaceNames[j], opts)
				objectCount.Add(uint64(objects))
				pageCount.Add(uint64(pages))
			}
		}(resourceVersion)
	}
	wg.Wait()
	b.ReportMetric(float64(objectCount.Load())/float64(b.N), "objects/op")
	b.ReportMetric(float64(pageCount.Load())/float64(b.N), "pages/op")
}

func runBenchmarkStoreList(ctx context.Context, b *testing.B, store storage.Interface, limit int64, maxRV int, match metav1.ResourceVersionMatch, perNode bool, nodeNames []string) {
	wg := sync.WaitGroup{}
	objectCount := atomic.Uint64{}
	pageCount := atomic.Uint64{}
	for i := 0; i < b.N; i++ {
		wg.Add(1)
		resourceVersion := ""
		switch match {
		case metav1.ResourceVersionMatchExact, metav1.ResourceVersionMatchNotOlderThan:
			resourceVersion = fmt.Sprintf("%d", maxRV-99+i%100)
		}
		go func(resourceVersion string) {
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
			if perNode {
				for _, nodeName := range nodeNames {
					opts.Predicate.GetAttrs = podAttr
					opts.Predicate.IndexFields = []string{"spec.nodeName"}
					opts.Predicate.Field = fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
					objects, pages := paginate(ctx, store, "/pods/", opts)
					objectCount.Add(uint64(objects))
					pageCount.Add(uint64(pages))
				}
			} else {
				objects, pages := paginate(ctx, store, "/pods/", opts)
				objectCount.Add(uint64(objects))
				pageCount.Add(uint64(pages))
			}
		}(resourceVersion)
	}
	wg.Wait()
	b.ReportMetric(float64(objectCount.Load())/float64(b.N), "objects/op")
	b.ReportMetric(float64(pageCount.Load())/float64(b.N), "pages/op")
}

func paginate(ctx context.Context, store storage.Interface, key string, opts storage.ListOptions) (objectCount int, pageCount int) {
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
		"spec.nodeName": pod.Spec.NodeName,
	}, nil
}

func prepareBenchchmarkData(ctx context.Context, store storage.Interface, namespaceCount, podPerNamespaceCount, nodeCount int) (namespaceNames, nodeNames []string) {
	nodeNames = make([]string, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodeNames[i] = rand.String(100)
	}
	namespaceNames = make([]string, nodeCount)
	out := &example.Pod{}
	for i := 0; i < namespaceCount; i++ {
		namespace := rand.String(100)
		namespaceNames[i] = namespace
		for j := 0; j < podPerNamespaceCount; j++ {
			name := rand.String(100)
			pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}, Spec: example.PodSpec{NodeName: nodeNames[rand.Intn(nodeCount)]}}
			err := store.Create(ctx, computePodKey(pod), pod, out, 0)
			if err != nil {
				panic(fmt.Sprintf("Unexpected error %s", err))
			}
		}
	}
	return namespaceNames, nodeNames
}
