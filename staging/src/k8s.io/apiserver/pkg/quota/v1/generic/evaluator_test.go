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

package generic

import (
	"errors"
	"testing"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
)

func TestCachedHasSynced(t *testing.T) {

	called := 0
	result := false
	cachedFunc := cachedHasSynced(func() bool {
		called++
		return result
	})

	if cachedFunc() {
		t.Fatal("expected false")
	}
	if called != 1 {
		t.Fatalf("expected called=1, got %d", called)
	}

	if cachedFunc() {
		t.Fatal("expected false")
	}
	if called != 2 {
		t.Fatalf("expected called=2, got %d", called)
	}

	result = true
	if !cachedFunc() {
		t.Fatal("expected true")
	}
	if called != 3 {
		t.Fatalf("expected called=3, got %d", called)
	}

	if !cachedFunc() {
		t.Fatal("expected true")
	}
	if called != 3 {
		// no more calls once we return true
		t.Fatalf("expected called=3, got %d", called)
	}
}

func TestProtectedLister(t *testing.T) {

	hasSynced := false
	notReadyErr := errors.New("not ready")
	fake := &fakeLister{}
	l := &protectedLister{
		hasSynced:   func() bool { return hasSynced },
		notReadyErr: notReadyErr,
		delegate:    fake,
	}
	if _, err := l.List(nil); err != notReadyErr {
		t.Fatalf("expected %v, got %v", notReadyErr, err)
	}
	if _, err := l.Get(""); err != notReadyErr {
		t.Fatalf("expected %v, got %v", notReadyErr, err)
	}
	if fake.called != 0 {
		t.Fatalf("expected called=0, got %d", fake.called)
	}
	fake.called = 0

	hasSynced = true

	if _, err := l.List(nil); err != errFakeLister {
		t.Fatalf("expected %v, got %v", errFakeLister, err)
	}
	if _, err := l.Get(""); err != errFakeLister {
		t.Fatalf("expected %v, got %v", errFakeLister, err)
	}
	if fake.called != 2 {
		t.Fatalf("expected called=2, got %d", fake.called)
	}
	fake.called = 0

	hasSynced = false

	if _, err := l.List(nil); err != notReadyErr {
		t.Fatalf("expected %v, got %v", notReadyErr, err)
	}
	if _, err := l.Get(""); err != notReadyErr {
		t.Fatalf("expected %v, got %v", notReadyErr, err)
	}
	if fake.called != 0 {
		t.Fatalf("expected called=2, got %d", fake.called)
	}
}

var errFakeLister = errors.New("errFakeLister")

type fakeLister struct {
	called int
}

func (f *fakeLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	f.called++
	return nil, errFakeLister
}
func (f *fakeLister) Get(name string) (runtime.Object, error) {
	f.called++
	return nil, errFakeLister
}
func (f *fakeLister) ByNamespace(namespace string) cache.GenericNamespaceLister {
	panic("not implemented")
}
