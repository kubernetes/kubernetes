/*
Copyright 2019 The Kubernetes Authors.

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

package selectorspread

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	tests = []struct {
		name            string
		existingPodsNum int
		allNodesNum     int
	}{
		{
			name:            "100nodes",
			existingPodsNum: 1000,
			allNodesNum:     100,
		},
		{
			name:            "1000nodes",
			existingPodsNum: 10000,
			allNodesNum:     1000,
		},
	}
)

func BenchmarkTestSelectorSpreadPriority(b *testing.B) {
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			pod := st.MakePod().Name("p").Label("foo", "").Obj()
			existingPods, allNodes, filteredNodes := st.MakeNodesAndPodsForEvenPodsSpread(pod.Labels, tt.existingPodsNum, tt.allNodesNum, tt.allNodesNum)
			snapshot := cache.NewSnapshot(existingPods, allNodes)
			client := fake.NewSimpleClientset(
				&v1.Service{Spec: v1.ServiceSpec{Selector: map[string]string{"foo": ""}}},
			)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			_ = informerFactory.Core().V1().Services().Lister()
			informerFactory.Start(ctx.Done())
			caches := informerFactory.WaitForCacheSync(ctx.Done())
			for _, synced := range caches {
				if !synced {
					b.Errorf("error waiting for informer cache sync")
				}
			}
			fh, _ := runtime.NewFramework(nil, nil, ctx.Done(), runtime.WithSnapshotSharedLister(snapshot), runtime.WithInformerFactory(informerFactory))
			pl, err := New(nil, fh)
			if err != nil {
				b.Fatal(err)
			}
			plugin := pl.(*SelectorSpread)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				state := framework.NewCycleState()
				status := plugin.PreScore(ctx, state, pod, allNodes)
				if !status.IsSuccess() {
					b.Fatalf("unexpected error: %v", status)
				}
				gotList := make(framework.NodeScoreList, len(filteredNodes))
				scoreNode := func(i int) {
					n := filteredNodes[i]
					score, _ := plugin.Score(ctx, state, pod, n.Name)
					gotList[i] = framework.NodeScore{Name: n.Name, Score: score}
				}
				parallelizer := parallelize.NewParallelizer(parallelize.DefaultParallelism)
				parallelizer.Until(ctx, len(filteredNodes), scoreNode, "")
				status = plugin.NormalizeScore(ctx, state, pod, gotList)
				if !status.IsSuccess() {
					b.Fatal(status)
				}
			}
		})
	}
}
