/*
Copyright 2014 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

var testTTL uint64 = 60

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "events",
	}
	rest, err := NewREST(restOptions, testTTL)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return rest, server
}

func validNewEvent(namespace string) *api.Event {
	someTime := metav1.MicroTime{Time: time.Unix(1505828956, 0)}
	return &api.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
		InvolvedObject: api.ObjectReference{
			Name:      "bar",
			Namespace: namespace,
		},
		EventTime:           someTime,
		ReportingController: "test-controller",
		ReportingInstance:   "test-node",
		Action:              "Do",
		Reason:              "forTesting",
		Type:                "Normal",
		Series: &api.EventSeries{
			Count:            2,
			LastObservedTime: someTime,
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	event := validNewEvent(test.TestNamespace())
	event.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		event,
		// invalid
		&api.Event{},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestUpdate(
		// valid
		validNewEvent(test.TestNamespace()),
		// valid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Event)
			object.Series.Count = 100
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Event)
			object.ReportingController = ""
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(validNewEvent(test.TestNamespace()))
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"ev"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestWatchWithStaleResourceVersion(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	// Create an event first to get the current resource version
	event := validNewNewEvent("test-namespace")
	_, err := storage.Store.Create(context.TODO(), event, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create event: %v", err)
	}

	// Test with resourceVersion="0" - should succeed
	t.Run("resourceVersion zero", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		opts := &metainternalversion.ListOptions{ResourceVersion: "0"}
		watcher, err := storage.Watch(ctx, opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		defer watcher.Stop()

		// Should be able to receive at least a bookmark event
		select {
		case event, ok := <-watcher.ResultChan():
			if !ok {
				t.Fatal("watch channel closed unexpectedly")
			}
			// Event type should not be Error
			if event.Type == watch.Error {
				t.Fatalf("unexpected error event: %v", event.Object)
			}
		case <-ctx.Done():
			t.Fatal("timeout waiting for watch event")
		}
	})

	// Test with very old resourceVersion - should return 410 error
	t.Run("stale resourceVersion", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Use resourceVersion "1" which is definitely too old
		opts := &metainternalversion.ListOptions{ResourceVersion: "1"}
		watcher, err := storage.Watch(ctx, opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		defer watcher.Stop()

		// Should receive an error event
		select {
		case event, ok := <-watcher.ResultChan():
			if !ok {
				t.Fatal("watch channel closed without returning error event")
			}
			// Should be an error event
			if event.Type != watch.Error {
				t.Fatalf("expected error event, got %v", event.Type)
			}

			// Check the error is a 410 Gone
			status, ok := event.Object.(*metav1.Status)
			if !ok {
				t.Fatalf("expected *metav1.Status, got %T", event.Object)
			}
			if status.Code != 410 {
				t.Fatalf("expected status code 410, got %d", status.Code)
			}
			if status.Reason != metav1.StatusReasonExpired {
				t.Fatalf("expected reason %q, got %q", metav1.StatusReasonExpired, status.Reason)
			}
		case <-ctx.Done():
			t.Fatal("timeout waiting for error event")
		}
	})

	// Test with invalid resourceVersion - should return BadRequest
	t.Run("invalid resourceVersion", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		opts := &metainternalversion.ListOptions{ResourceVersion: "invalid"}
		_, err := storage.Watch(ctx, opts)
		if err == nil {
			t.Fatal("expected error for invalid resourceVersion")
		}
		if !errors.IsBadRequest(err) {
			t.Fatalf("expected BadRequest error, got %v", err)
		}
	})
}

func validNewNewEvent(namespace string) *api.Event {
	someTime := metav1.NowMicro()
	return &api.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("test-event-%d", time.Now().UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: api.ObjectReference{
			Name:      "test-pod",
			Namespace: namespace,
		},
		EventTime:           someTime,
		ReportingController: "test-controller",
		ReportingInstance:   "test-node",
		Action:              "Do",
		Reason:              "forTesting",
		Type:                "Normal",
	}
}
