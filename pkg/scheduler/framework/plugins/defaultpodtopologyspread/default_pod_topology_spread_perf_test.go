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

package defaultpodtopologyspread

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
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
			services := &v1.ServiceList{
				Items: []v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"foo": ""}}}},
			}
			client := clientsetfake.NewSimpleClientset(services)
			ctx := context.Background()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			_ = informerFactory.Core().V1().Services().Lister()
			informerFactory.Start(ctx.Done())
			caches := informerFactory.WaitForCacheSync(ctx.Done())
			for _, synced := range caches {
				if !synced {
					b.Errorf("error waiting for informer cache sync")
				}
			}
			fh, _ := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(snapshot), framework.WithInformerFactory(informerFactory))
			plugin := &DefaultPodTopologySpread{handle: fh}
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				state := framework.NewCycleState()
				status := plugin.PostFilter(ctx, state, pod, allNodes, nil)
				if !status.IsSuccess() {
					b.Fatalf("unexpected error: %v", status)
				}
				for _, node := range filteredNodes {
					_, status := plugin.Score(ctx, state, pod, node.Name)
					if !status.IsSuccess() {
						b.Errorf("unexpected error: %v", status)
					}
				}
			}
		})
	}
}
