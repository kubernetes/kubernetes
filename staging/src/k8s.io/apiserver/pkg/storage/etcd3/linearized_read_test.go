/*
Copyright 2022 The Kubernetes Authors.

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

package etcd3

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestLinearizedReadRevisionInvariant(t *testing.T) {
	// The etcd documentation [1] states that "linearized requests must go through the Raft consensus process."
	// A full round of Raft consensus adds a new item to the Raft log, some of which is surfaced by etcd as a
	// higher store revision in the response header. Kubernetes exposes this header revision in e.g. List calls,
	// so it is ultimately client-facing. By default, all the requests that our *etcd3.store{} issues are
	// linearized. However, this also includes *read* requests, and we would not expect non-mutating requests
	// against etcd to, by "go[ing] through the Raft consensus process," result in a higher resource version on
	// List calls. Today, the mechanism etcd uses to increment the store revision ensures that linearized reads
	// do *not* bump the key-value store revision. This test exists to ensure that we notice if this implementation
	// detail ever changes.
	// [1] https://etcd.io/docs/v3.5/learning/api_guarantees/#isolation-level-and-consistency-of-replicas
	ctx, store, etcdClient := testSetup(t)

	dir := "/testing"
	key := dir + "/testkey"
	out := &example.Pod{}
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}

	if err := store.Create(ctx, key, obj, out, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	originalRevision := out.ResourceVersion

	for i := 0; i < 5; i++ {
		if _, err := etcdClient.KV.Get(ctx, key); err != nil { // this is by default linearizable, the only option the client library exposes is WithSerializable() to make it *not* a linearized read
			t.Fatalf("failed to get key: %v", err)
		}
	}

	list := &example.PodList{}
	if err := store.GetList(ctx, dir, storage.ListOptions{Predicate: storage.Everything, Recursive: true}, list); err != nil {
		t.Errorf("Unexpected List error: %v", err)
	}
	finalRevision := list.ResourceVersion

	if originalRevision != finalRevision {
		t.Fatalf("original revision (%s) did not match final revision after linearized reads (%s)", originalRevision, finalRevision)
	}
}
