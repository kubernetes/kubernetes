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

package coverage

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// fakeStorage implements storage.Interface, returning canned results/errors
// for Get/GetList/Watch/Delete. Every other method is left unimplemented
// (nil embedded Interface) since the tests below never call them - Wrap must
// not need to touch them to satisfy the interface.
type fakeStorage struct {
	storage.Interface
	err error
}

func (f *fakeStorage) Versioner() storage.Versioner {
	return storage.APIObjectVersioner{}
}

func (f *fakeStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	return f.err
}

func (f *fakeStorage) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	return f.err
}

func (f *fakeStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	if f.err != nil {
		return nil, f.err
	}
	return watch.NewFake(), nil
}

func (f *fakeStorage) Delete(
	ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions,
	validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions,
) error {
	return f.err
}

func TestWrapClassifiesCalls(t *testing.T) {
	rec := NewRecorder()
	w := Wrap(&fakeStorage{}, rec)

	if err := w.Get(context.Background(), "/key", storage.GetOptions{ResourceVersion: "0"}, nil); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if err := w.GetList(context.Background(), "/prefix", storage.ListOptions{Recursive: true, Predicate: storage.SelectionPredicate{Limit: 10}}, nil); err != nil {
		t.Fatalf("GetList: %v", err)
	}
	if _, err := w.Watch(context.Background(), "/prefix", storage.ListOptions{Recursive: true}); err != nil {
		t.Fatalf("Watch: %v", err)
	}
	if err := w.Delete(context.Background(), "/key", nil, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	want := []State{
		{Verb: VerbGet, ResourceVersion: RVZero, IgnoreNotFound: false, Outcome: OutcomeSuccess},
		{Verb: VerbList, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset, Recursive: true, Pagination: PaginationLimit, Selector: SelectorNone, Outcome: OutcomeSuccess},
		{Verb: VerbWatch, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset, Recursive: true, Selector: SelectorNone, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess},
		{Verb: VerbDelete, Preconditions: PreconditionNone, Outcome: OutcomeSuccess},
	}
	counts := rec.Counts()
	for _, state := range want {
		if counts[state] != 1 {
			t.Errorf("expected exactly one observation of %s, got %d (all counts: %v)", state, counts[state], counts)
		}
	}
	if len(counts) != len(want) {
		t.Errorf("expected %d distinct observed states, got %d: %v", len(want), len(counts), counts)
	}
}

func TestWrapPassthroughDoesNotPanic(t *testing.T) {
	inner := &fakeStorage{}
	w := Wrap(inner, NewRecorder())
	if w.Versioner() == nil {
		t.Errorf("expected Versioner() to pass through to the inner storage.Interface")
	}
}

func TestWrapRecordsOutcome(t *testing.T) {
	rec := NewRecorder()
	w := Wrap(&fakeStorage{err: storage.NewKeyNotFoundError("/key", 0)}, rec)

	_ = w.Get(context.Background(), "/key", storage.GetOptions{}, nil)

	want := State{Verb: VerbGet, ResourceVersion: RVUnset, IgnoreNotFound: false, Outcome: OutcomeNotFound}
	if counts := rec.Counts(); counts[want] != 1 {
		t.Errorf("expected %s to be observed once, got counts: %v", want, counts)
	}
}
