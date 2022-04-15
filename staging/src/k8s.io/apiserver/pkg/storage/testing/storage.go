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

package testing

import (
	"context"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

type ObjectInvariantsFunc func(context.Context, *testing.T, storage.Interface, string)

func StorageInterfaceCreateTest(t *testing.T, store storage.Interface, checkInvariants ObjectInvariantsFunc) {
	ctx := context.Background()
	key := "/testkey"
	out := &example.Pod{}
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}

	// verify that object doesn't exist before set.
	if err := store.Get(ctx, key, storage.GetOptions{}, out); !storage.IsNotFound(err) {
		t.Errorf("no object expected, got: %#v, err: %v", out, err)
	}

	if err := store.Create(ctx, key, obj, out, 0); err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	// basic tests of the output
	if obj.ObjectMeta.Name != out.ObjectMeta.Name {
		t.Errorf("pod name want=%s, get=%s", obj.ObjectMeta.Name, out.ObjectMeta.Name)
	}
	if out.ResourceVersion == "" {
		t.Errorf("output should have non-empty resource version")
	}
	if out.SelfLink != "" {
		t.Errorf("output should have empty selfLink")
	}
	checkInvariants(ctx, t, store, key)

	// Ensure that second creation fails.
	if err := store.Create(ctx, key, obj, out, 0); !storage.IsExist(err) {
		t.Errorf("expecting key exists error, got: %v", err)
	}
}

func StorageInterfaceCreateWithTTLTest(t *testing.T, store storage.Interface) {
	ctx := context.Background()
	key := "/testkey"
	out := &example.Pod{}
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}

	// verify that object doesn't exist before set.
	if err := store.Get(ctx, key, storage.GetOptions{}, out); !storage.IsNotFound(err) {
		t.Errorf("no object expected, got: %#v, err: %v", out, err)
	}

	if err := store.Create(ctx, key, obj, out, 5); err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	if err := store.Get(ctx, key, storage.GetOptions{}, out); err != nil {
		t.Fatalf("Failed to get object after creation: %v", err)
	}

	// Ensure that object get automatically deleted.
	// It should be deleted after 5s, but to accomodate resource starvation
	// in test environment, we wait for it with much larger buffer.
	objectExists := func() (bool, error) {
		return storage.IsNotFound(store.Get(ctx, key, storage.GetOptions{}, out)), nil
	}
	if err := wait.Poll(time.Second, wait.ForeverTestTimeout, objectExists); err != nil {
		t.Errorf("Failed to wait for object deletion: %v", err)
	}
}
