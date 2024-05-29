/*
Copyright 2016 The Kubernetes Authors.

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

package watch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
)

type fakePod struct {
}

func (obj *fakePod) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *fakePod) DeepCopyObject() runtime.Object   { panic("DeepCopyObject not supported by fakePod") }

func TestUntil(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *fakePod
		fw.Add(obj)
		fw.Modify(obj)
		<-fw.StopChan()
		fw.Close()
	}()
	conditions := []ConditionFunc{
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
		func(event watch.Event) (bool, error) { return event.Type == watch.Modified, nil },
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	lastEvent, err := UntilWithoutRetry(ctx, fw, conditions...)
	if err != nil {
		t.Fatalf("expected nil error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Modified {
		t.Fatalf("expected MODIFIED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*fakePod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}

	// Validate the UntilWithoutRetry stopped the watcher on condition or error
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestUntilMultipleConditions(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *fakePod
		fw.Add(obj)
		<-fw.StopChan()
		fw.Close()
	}()
	conditions := []ConditionFunc{
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	lastEvent, err := UntilWithoutRetry(ctx, fw, conditions...)
	if err != nil {
		t.Fatalf("expected nil error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Added {
		t.Fatalf("expected MODIFIED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*fakePod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}

	// Validate the UntilWithoutRetry stopped the watcher on condition or error
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestUntilMultipleConditionsFail(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *fakePod
		fw.Add(obj)
		<-fw.StopChan()
		fw.Close()
	}()
	conditions := []ConditionFunc{
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
		func(event watch.Event) (bool, error) { return event.Type == watch.Deleted, nil },
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	lastEvent, err := UntilWithoutRetry(ctx, fw, conditions...)
	if err != wait.ErrWaitTimeout {
		t.Fatalf("expected ErrWaitTimeout error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Added {
		t.Fatalf("expected ADDED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*fakePod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}

	// Validate the UntilWithoutRetry stopped the watcher on condition or error
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestUntilTimeout(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *fakePod
		fw.Add(obj)
		fw.Modify(obj)
		<-fw.StopChan()
		fw.Close()
	}()
	conditions := []ConditionFunc{
		func(event watch.Event) (bool, error) {
			return event.Type == watch.Added, nil
		},
		func(event watch.Event) (bool, error) {
			return event.Type == watch.Modified, nil
		},
	}

	lastEvent, err := UntilWithoutRetry(context.Background(), fw, conditions...)
	if err != nil {
		t.Fatalf("expected nil error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Modified {
		t.Fatalf("expected MODIFIED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*fakePod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}

	// Validate the UntilWithoutRetry stopped the watcher on condition or error
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestUntilErrorCondition(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *fakePod
		fw.Add(obj)
		<-fw.StopChan()
		fw.Close()
	}()
	expected := "something bad"
	conditions := []ConditionFunc{
		func(event watch.Event) (bool, error) { return event.Type == watch.Added, nil },
		func(event watch.Event) (bool, error) { return false, errors.New(expected) },
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	_, err := UntilWithoutRetry(ctx, fw, conditions...)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), expected) {
		t.Fatalf("expected %q in error string, got %q", expected, err.Error())
	}

	// Validate the UntilWithoutRetry stopped the watcher on condition or error
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestUntilWithSync(t *testing.T) {
	// FIXME: test preconditions
	tt := []struct {
		name             string
		lw               *cache.ListWatch
		preconditionFunc PreconditionFunc
		conditionFunc    ConditionFunc
		expectedErr      error
		expectedEvent    *watch.Event
	}{
		{
			name: "doesn't wait for sync with no precondition",
			lw: &cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					select {}
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					select {}
				},
			},
			preconditionFunc: nil,
			conditionFunc: func(e watch.Event) (bool, error) {
				return true, nil
			},
			expectedErr:   wait.ErrWaitTimeout,
			expectedEvent: nil,
		},
		{
			name: "waits indefinitely with precondition if it can't sync",
			lw: &cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					select {}
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					select {}
				},
			},
			preconditionFunc: func(store cache.Store) (bool, error) {
				return true, nil
			},
			conditionFunc: func(e watch.Event) (bool, error) {
				return true, nil
			},
			expectedErr:   fmt.Errorf("UntilWithSync: unable to sync caches: %w", context.DeadlineExceeded),
			expectedEvent: nil,
		},
		{
			name: "precondition can stop the loop",
			lw: func() *cache.ListWatch {
				fakeclient := fakeclient.NewSimpleClientset(&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "first"}})

				return &cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						return fakeclient.CoreV1().Secrets("").List(context.TODO(), options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return fakeclient.CoreV1().Secrets("").Watch(context.TODO(), options)
					},
				}
			}(),
			preconditionFunc: func(store cache.Store) (bool, error) {
				_, exists, err := store.Get(&metav1.ObjectMeta{Namespace: "", Name: "first"})
				if err != nil {
					return true, err
				}
				if exists {
					return true, nil
				}
				return false, nil
			},
			conditionFunc: func(e watch.Event) (bool, error) {
				return true, errors.New("should never reach this")
			},
			expectedErr:   nil,
			expectedEvent: nil,
		},
		{
			name: "precondition lets it proceed to regular condition",
			lw: func() *cache.ListWatch {
				fakeclient := fakeclient.NewSimpleClientset(&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "first"}})

				return &cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						return fakeclient.CoreV1().Secrets("").List(context.TODO(), options)
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return fakeclient.CoreV1().Secrets("").Watch(context.TODO(), options)
					},
				}
			}(),
			preconditionFunc: func(store cache.Store) (bool, error) {
				return false, nil
			},
			conditionFunc: func(e watch.Event) (bool, error) {
				if e.Type == watch.Added {
					return true, nil
				}
				panic("no other events are expected")
			},
			expectedErr:   nil,
			expectedEvent: &watch.Event{Type: watch.Added, Object: &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "first"}}},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			// Informer waits for caches to sync by polling in 100ms intervals,
			// timeout needs to be reasonably higher
			ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
			defer cancel()

			event, err := UntilWithSync(ctx, tc.lw, &corev1.Secret{}, tc.preconditionFunc, tc.conditionFunc)

			if !reflect.DeepEqual(err, tc.expectedErr) {
				t.Errorf("expected error %#v, got %#v", tc.expectedErr, err)
			}

			if !reflect.DeepEqual(event, tc.expectedEvent) {
				t.Errorf("expected event %#v, got %#v", tc.expectedEvent, event)
			}
		})
	}
}
