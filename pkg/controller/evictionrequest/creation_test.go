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

package evictionrequest

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

func TestEvictionCreationPreservesExistingOwners(t *testing.T) {
	ctx := t.Context()
	client := fake.NewClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	evictions := factory.Lifecycle().V1alpha1().Evictions()
	requests := factory.Lifecycle().V1alpha1().EvictionRequests()
	pods := factory.Core().V1().Pods()
	c, err := NewController(ctx, "evictionrequest-controller", evictions, requests, pods, client)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		c.evictionQueue.ShutDown()
		c.evictionRequestQueue.ShutDown()
		c.evictionMetaRefreshQueue.ShutDown()
	})
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "target", Namespace: "ns", UID: "pod-uid"}}
	if err := pods.Informer().GetStore().Add(pod); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a", "b"} {
		request := &lifecyclev1alpha1.EvictionRequest{
			ObjectMeta: metav1.ObjectMeta{
				Name: name, Namespace: "ns", UID: types.UID(name), Generation: 1,
				CreationTimestamp: metav1.NewTime(time.Now().Add(-time.Minute)),
			},
			Spec: lifecyclev1alpha1.EvictionRequestSpec{
				Requester: name + ".example.com/requester",
				Intent:    lifecyclev1alpha1.EvictionRequestIntentEviction,
				Target:    lifecyclev1alpha1.EvictionRequestTarget{Pod: &lifecyclev1alpha1.EvictionRequestPodReference{Name: pod.Name, UID: pod.UID}},
			},
		}
		if err := requests.Informer().GetStore().Add(request); err != nil {
			t.Fatal(err)
		}
		if _, err := client.LifecycleV1alpha1().EvictionRequests("ns").Create(ctx, request, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	// Keep the informer behind the API, as can happen between two workers.
	if err := c.syncEvictionRequest(ctx, "ns/a"); err != nil {
		t.Fatal(err)
	}
	before, err := client.LifecycleV1alpha1().Evictions("ns").Get(ctx, "pod-1-target", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(before.OwnerReferences) != 1 || before.OwnerReferences[0].Name != "a" {
		t.Fatalf("invalid initial owners: %v", before.OwnerReferences)
	}
	if err := c.syncEvictionRequest(ctx, "ns/b"); err != nil {
		t.Fatal(err)
	}
	after, err := client.LifecycleV1alpha1().Evictions("ns").Get(ctx, before.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(before, after); diff != "" {
		t.Errorf("creation collision changed the existing eviction (-before,+after):\n%s", diff)
	}
	if err := evictions.Informer().GetStore().Add(after); err != nil {
		t.Fatal(err)
	}
	if err := c.syncEvictionMetaRefresh(ctx, "ns/"+after.Name); err != nil {
		t.Fatal(err)
	}
	reconciled, err := client.LifecycleV1alpha1().Evictions("ns").Get(ctx, after.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(reconciled.OwnerReferences) != 2 || reconciled.Labels["a.example.com/requester"] != "requester" || reconciled.Labels["b.example.com/requester"] != "requester" {
		t.Fatalf("metadata reconciliation did not attach both requesters: %#v", reconciled)
	}
}
