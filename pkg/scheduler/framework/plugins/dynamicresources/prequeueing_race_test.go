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

	// When a template pod's ResourceClaimStatuses isn't populated yet, the
	// generated claim name isn't known, so the indexer cannot map the claim to
	// the pod and this claim event narrows the pod out. That is expected and
	// safe: populating ResourceClaimStatuses is itself a pod update that emits a
	// TargetPod/UpdatePodGeneratedResourceClaim event (see
	// isSchedulableAfterTargetPodUpdate), which requeues the pod. The rescue does
	// not depend on the periodic flush.
	if got.AllPods {
		t.Errorf("expected AllPods=false when the pod is not yet indexed, got true")
	}
	if len(got.Pods) != 0 {
		t.Errorf("expected no pods when the pod is not yet indexed, got %v", got.Pods)
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
