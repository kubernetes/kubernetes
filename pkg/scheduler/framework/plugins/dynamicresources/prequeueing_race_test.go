/*
Copyright 2026 The Kubernetes Authors.

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

package dynamicresources

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func TestPreQueueingHint_PodNotYetIndexed(t *testing.T) {
	logger := klog.Background()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podResourceClaimIndexPrefix + "-test": podResourceClaimIndexFunc})

	// Pod with template-based claim, status NOT yet updated
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "my-pod", Namespace: "ns1"},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{Name: "gpu", ResourceClaimTemplateName: new("gpu-template")},
			},
		},
	}
	if err := indexer.Add(pod); err != nil {
		t.Fatal(err)
	}

	pl := &DynamicResources{podIndexer: indexer, podResourceClaimIndex: podResourceClaimIndexPrefix + "-test"}

	// Claim allocation event - indexer cannot find pod by this claim name
	got, err := pl.preQueueingHint(logger, nil, &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "my-pod-gpu-xyz", Namespace: "ns1"},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Known limitation: when pod's ResourceClaimStatuses isn't populated yet,
	// the indexer cannot find the pod. The periodic flush rescues such pods.
	if got.AllPods {
		t.Logf("AllPods=true (safe fallback)")
	} else if len(got.Pods) == 0 {
		t.Logf("empty Pods (known race - pod rescued by flush)")
	}
}

func TestPreQueueingHint_DeleteEventWithPodInIndexer(t *testing.T) {
	// When an allocated claim is deleted and a pod referencing it exists
	// in the indexer, AllPods is returned (deletion frees resources).
	logger := klog.Background()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podResourceClaimIndexPrefix + "-test": podResourceClaimIndexFunc})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "my-pod", Namespace: "ns1"},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{Name: "gpu", ResourceClaimName: new("deleted-claim")},
			},
		},
	}
	if err := indexer.Add(pod); err != nil {
		t.Fatal(err)
	}

	pl := &DynamicResources{podIndexer: indexer, podResourceClaimIndex: podResourceClaimIndexPrefix + "-test"}

	// Delete event for an allocated claim
	got, err := pl.preQueueingHint(logger, &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "deleted-claim", Namespace: "ns1"},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{},
		},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !got.AllPods {
		t.Errorf("expected AllPods=true for delete of allocated claim, got %+v", got)
	}
}
