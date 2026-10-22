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

package controller

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"
)

func BenchmarkFilterPodsByOwner(b *testing.B) {
	ownerKind := "ReplicaSet"
	ownerName := "owner"
	owner := &metav1.ObjectMeta{
		Namespace: "ns1",
		Name:      ownerName,
		UID:       types.UID("owner-uid"),
	}
	controllerRef := metav1.OwnerReference{
		UID:        owner.UID,
		Kind:       ownerKind,
		Name:       ownerName,
		Controller: ptr.To(true),
	}

	cases := []struct {
		name              string
		ownedPods         int
		orphanPods        int
		includeOrphanPods bool
	}{
		{name: "10-owned", ownedPods: 10},
		{name: "10000-owned", ownedPods: 10000},
		{name: "10000-owned-10000-orphan", ownedPods: 10000, orphanPods: 10000, includeOrphanPods: true},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			fakeClient := fake.NewSimpleClientset()
			sharedInformers := informers.NewSharedInformerFactory(fakeClient, 0)
			podInformer := sharedInformers.Core().V1().Pods()
			if err := AddPodControllerIndexer(podInformer.Informer()); err != nil {
				b.Fatalf("failed to register indexer: %v", err)
			}
			podIndexer := podInformer.Informer().GetIndexer()

			for i := 0; i < tc.ownedPods; i++ {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:            fmt.Sprintf("owned-%d", i),
						Namespace:       owner.Namespace,
						OwnerReferences: []metav1.OwnerReference{controllerRef},
					},
				}
				if err := podIndexer.Add(pod); err != nil {
					b.Fatalf("failed adding Pod to indexer: %v", err)
				}
			}
			for i := 0; i < tc.orphanPods; i++ {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("orphan-%d", i),
						Namespace: owner.Namespace,
					},
				}
				if err := podIndexer.Add(pod); err != nil {
					b.Fatalf("failed adding Pod to indexer: %v", err)
				}
			}

			want := tc.ownedPods
			if tc.includeOrphanPods {
				want += tc.orphanPods
			}

			b.ReportAllocs()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				pods, err := FilterPodsByOwner(podIndexer, owner, ownerKind, tc.includeOrphanPods)
				if err != nil {
					b.Fatalf("FilterPodsByOwner() error: %v", err)
				}
				if len(pods) != want {
					b.Fatalf("expected %d pods, got %d", want, len(pods))
				}
			}
		})
	}
}
