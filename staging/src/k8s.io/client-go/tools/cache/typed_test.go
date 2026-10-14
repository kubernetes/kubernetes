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

package cache

import (
	"fmt"
	"testing"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/ktesting"
)

// recordingTypedHandler implements TypedResourceEventHandler[*v1.Pod] and
// records all invocations for later inspection.
type recordingTypedHandler struct {
	added        []*v1.Pod
	addedInitial []bool
	updatedOld   []*v1.Pod
	updatedNew   []*v1.Pod
	deleted      []DeletedObject[*v1.Pod]
}

func (r *recordingTypedHandler) OnAdd(obj *v1.Pod, isInInitialList bool) {
	r.added = append(r.added, obj)
	r.addedInitial = append(r.addedInitial, isInInitialList)
}

func (r *recordingTypedHandler) OnUpdate(oldObj, newObj *v1.Pod) {
	r.updatedOld = append(r.updatedOld, oldObj)
	r.updatedNew = append(r.updatedNew, newObj)
}

func (r *recordingTypedHandler) OnDelete(obj DeletedObject[*v1.Pod]) {
	r.deleted = append(r.deleted, obj)
}

func TestTypedResourceEventHandlerFuncs(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}
	oldPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old-pod"}}

	t.Run("callbacks-are-invoked", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		var added []*v1.Pod
		var addedInitial bool
		var updatedOld, updatedNew *v1.Pod
		var deleted DeletedObject[*v1.Pod]

		funcs := TypedResourceEventHandlerFuncs[*v1.Pod]{
			AddFunc: func(obj *v1.Pod) {
				added = append(added, obj)
			},
			UpdateFunc: func(o, n *v1.Pod) {
				updatedOld, updatedNew = o, n
			},
			DeleteFunc: func(obj DeletedObject[*v1.Pod]) {
				deleted = obj
			},
		}

		funcs.OnAdd(pod, true)
		tCtx.Assert(added).Should(gomega.Equal([]*v1.Pod{pod}))
		addedInitial = true // AddFunc doesn't get isInInitialList, only presence is checked above
		_ = addedInitial

		funcs.OnUpdate(oldPod, pod)
		tCtx.Assert(updatedOld).Should(gomega.Equal(oldPod))
		tCtx.Assert(updatedNew).Should(gomega.Equal(pod))

		funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		tCtx.Assert(deleted.OptionalObj).Should(gomega.Equal(pod))
	})

	t.Run("nil-callbacks-do-not-panic", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		var funcs TypedResourceEventHandlerFuncs[*v1.Pod]
		tCtx.Assert(func() {
			funcs.OnAdd(pod, false)
			funcs.OnUpdate(oldPod, pod)
			funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		}).ShouldNot(gomega.Panic())
	})

	// Compile-time and runtime check that the type implements the interface.
	var _ TypedResourceEventHandler[*v1.Pod] = TypedResourceEventHandlerFuncs[*v1.Pod]{}
}

func TestTypedResourceEventHandlerDetailedFuncs(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}
	oldPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old-pod"}}

	t.Run("callbacks-are-invoked-including-isInInitialList", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		var added *v1.Pod
		var addedInitial bool
		var updatedOld, updatedNew *v1.Pod
		var deleted DeletedObject[*v1.Pod]

		funcs := TypedResourceEventHandlerDetailedFuncs[*v1.Pod]{
			AddFunc: func(obj *v1.Pod, isInInitialList bool) {
				added = obj
				addedInitial = isInInitialList
			},
			UpdateFunc: func(o, n *v1.Pod) {
				updatedOld, updatedNew = o, n
			},
			DeleteFunc: func(obj DeletedObject[*v1.Pod]) {
				deleted = obj
			},
		}

		funcs.OnAdd(pod, true)
		tCtx.Assert(added).Should(gomega.Equal(pod))
		tCtx.Assert(addedInitial).Should(gomega.BeTrueBecause("isInInitialList was passed as true"))

		funcs.OnAdd(pod, false)
		tCtx.Assert(addedInitial).Should(gomega.BeFalseBecause("isInInitialList was passed as false"))

		funcs.OnUpdate(oldPod, pod)
		tCtx.Assert(updatedOld).Should(gomega.Equal(oldPod))
		tCtx.Assert(updatedNew).Should(gomega.Equal(pod))

		funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		tCtx.Assert(deleted.OptionalObj).Should(gomega.Equal(pod))
	})

	t.Run("nil-callbacks-do-not-panic", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		var funcs TypedResourceEventHandlerDetailedFuncs[*v1.Pod]
		tCtx.Assert(func() {
			funcs.OnAdd(pod, true)
			funcs.OnUpdate(oldPod, pod)
			funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		}).ShouldNot(gomega.Panic())
	})

	var _ TypedResourceEventHandler[*v1.Pod] = TypedResourceEventHandlerDetailedFuncs[*v1.Pod]{}
}

func TestTypedFilteringResourceEventHandler(t *testing.T) {
	oldPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old"}}
	newPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new"}}

	t.Run("OnAdd", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			passes bool
		}{
			{name: "filter-passes", passes: true},
			{name: "filter-fails", passes: false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				tCtx := ktesting.Init(t)
				handler := &recordingTypedHandler{}
				r := TypedFilteringResourceEventHandler[*v1.Pod]{
					FilterFunc: func(obj *v1.Pod) bool { return tc.passes },
					Handler:    handler,
				}
				r.OnAdd(newPod, true)
				if tc.passes {
					tCtx.Require(handler.added).Should(gomega.HaveLen(1))
					tCtx.Assert(handler.added[0]).Should(gomega.Equal(newPod))
					tCtx.Assert(handler.addedInitial).Should(gomega.Equal([]bool{true}))
				} else {
					tCtx.Assert(handler.added).Should(gomega.BeEmpty())
				}
			})
		}
	})

	t.Run("OnDelete", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			passes bool
		}{
			{name: "filter-passes", passes: true},
			{name: "filter-fails", passes: false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				tCtx := ktesting.Init(t)
				handler := &recordingTypedHandler{}
				r := TypedFilteringResourceEventHandler[*v1.Pod]{
					FilterFunc: func(obj *v1.Pod) bool { return tc.passes },
					Handler:    handler,
				}
				deleted := DeletedObject[*v1.Pod]{OptionalObj: oldPod}
				r.OnDelete(deleted)
				if tc.passes {
					tCtx.Require(handler.deleted).Should(gomega.HaveLen(1))
					tCtx.Assert(handler.deleted[0]).Should(gomega.Equal(deleted))
				} else {
					tCtx.Assert(handler.deleted).Should(gomega.BeEmpty())
				}
			})
		}
	})

	t.Run("OnUpdate", func(t *testing.T) {
		for _, tc := range []struct {
			name         string
			older, newer bool
			wantAdd      bool
			wantUpdate   bool
			wantDelete   bool
		}{
			{name: "still-passing-to-update", older: true, newer: true, wantUpdate: true},
			{name: "starts-passing-to-add", older: false, newer: true, wantAdd: true},
			{name: "stops-passing-to-delete", older: true, newer: false, wantDelete: true},
			{name: "still-failing-to-nothing", older: false, newer: false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				tCtx := ktesting.Init(t)
				handler := &recordingTypedHandler{}
				r := TypedFilteringResourceEventHandler[*v1.Pod]{
					FilterFunc: func(obj *v1.Pod) bool {
						switch obj {
						case oldPod:
							return tc.older
						case newPod:
							return tc.newer
						}
						t.Fatalf("unexpected object %v", obj)
						return false
					},
					Handler: handler,
				}
				r.OnUpdate(oldPod, newPod)

				if tc.wantAdd {
					tCtx.Require(handler.added).Should(gomega.HaveLen(1))
					tCtx.Assert(handler.added[0]).Should(gomega.Equal(newPod))
					tCtx.Assert(handler.addedInitial).Should(gomega.Equal([]bool{false}))
				} else {
					tCtx.Assert(handler.added).Should(gomega.BeEmpty())
				}

				if tc.wantUpdate {
					tCtx.Require(handler.updatedNew).Should(gomega.HaveLen(1))
					tCtx.Assert(handler.updatedOld[0]).Should(gomega.Equal(oldPod))
					tCtx.Assert(handler.updatedNew[0]).Should(gomega.Equal(newPod))
				} else {
					tCtx.Assert(handler.updatedNew).Should(gomega.BeEmpty())
				}

				if tc.wantDelete {
					tCtx.Require(handler.deleted).Should(gomega.HaveLen(1))
					tCtx.Assert(handler.deleted[0].OptionalObj).Should(gomega.Equal(oldPod))
				} else {
					tCtx.Assert(handler.deleted).Should(gomega.BeEmpty())
				}
			})
		}
	})

	// Compile-time and runtime check that the type implements the interface.
	var _ TypedResourceEventHandler[*v1.Pod] = TypedFilteringResourceEventHandler[*v1.Pod]{}
}

func TestTypedResourceEventHandlerAdapter(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}
	oldPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old-pod"}}

	t.Run("OnAdd", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnAdd(pod, true)
		tCtx.Require(handler.added).Should(gomega.HaveLen(1))
		tCtx.Assert(handler.added[0]).Should(gomega.Equal(pod))
		tCtx.Assert(handler.addedInitial).Should(gomega.Equal([]bool{true}))
	})

	t.Run("OnUpdate", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnUpdate(oldPod, pod)
		tCtx.Require(handler.updatedNew).Should(gomega.HaveLen(1))
		tCtx.Assert(handler.updatedOld[0]).Should(gomega.Equal(oldPod))
		tCtx.Assert(handler.updatedNew[0]).Should(gomega.Equal(pod))
	})

	t.Run("OnDelete-with-a-plain-object", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnDelete(pod)
		tCtx.Require(handler.deleted).Should(gomega.HaveLen(1))
		tCtx.Assert(handler.deleted[0].OptionalObj).Should(gomega.Equal(pod))
		tCtx.Assert(handler.deleted[0].FinalStateUnknown).Should(gomega.BeNil())
	})

	t.Run("OnDelete-with-DeletedFinalStateUnknown-carrying-an-object", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		tomb := DeletedFinalStateUnknown{Key: "default/pod", Obj: pod}
		adapter.OnDelete(tomb)
		tCtx.Require(handler.deleted).Should(gomega.HaveLen(1))
		tCtx.Assert(handler.deleted[0].OptionalObj).Should(gomega.Equal(pod))
		tCtx.Require(handler.deleted[0].FinalStateUnknown).ShouldNot(gomega.BeNil())
		tCtx.Assert(handler.deleted[0].FinalStateUnknown.Key).Should(gomega.Equal("default/pod"))
	})

	t.Run("OnDelete-with-DeletedFinalStateUnknown-without-an-object", func(t *testing.T) {
		tCtx := ktesting.Init(t)
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		tomb := DeletedFinalStateUnknown{Key: "default/pod", Obj: nil}
		adapter.OnDelete(tomb)
		tCtx.Require(handler.deleted).Should(gomega.HaveLen(1))
		var zero *v1.Pod
		tCtx.Assert(handler.deleted[0].OptionalObj).Should(gomega.Equal(zero))
		tCtx.Require(handler.deleted[0].FinalStateUnknown).ShouldNot(gomega.BeNil())
		tCtx.Assert(handler.deleted[0].FinalStateUnknown.Key).Should(gomega.Equal("default/pod"))
	})

	var _ ResourceEventHandler = &typedResourceEventHandler[*v1.Pod]{}
}

func TestTypedIndexer(t *testing.T) {
	tCtx := ktesting.Init(t)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1", Labels: map[string]string{"foo": "bar"}}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod2", Labels: map[string]string{"foo": "bar"}}}

	raw := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		"foo": func(obj interface{}) ([]string, error) {
			return []string{obj.(*v1.Pod).Labels["foo"]}, nil
		},
	})
	tCtx.ExpectNoError(raw.Add(pod1))
	tCtx.ExpectNoError(raw.Add(pod2))

	typed := typedIndexer[*v1.Pod]{Indexer: raw}

	tCtx.Run("TypedIndex", func(tCtx ktesting.TContext) {
		res, err := typed.TypedIndex("foo", pod1)
		tCtx.ExpectNoError(err)
		tCtx.Assert(res).Should(gomega.ConsistOf(pod1, pod2))
	})

	tCtx.Run("ByTypedIndex", func(tCtx ktesting.TContext) {
		res, err := typed.ByTypedIndex("foo", "bar")
		tCtx.ExpectNoError(err)
		tCtx.Assert(res).Should(gomega.ConsistOf(pod1, pod2))
	})

	tCtx.Run("ByTypedIndex-propagates-errors", func(tCtx ktesting.TContext) {
		_, err := typed.ByTypedIndex("does-not-exist", "bar")
		tCtx.Require(err).Should(gomega.HaveOccurred())
	})

	tCtx.Run("AddTypedIndexers", func(tCtx ktesting.TContext) {
		err := typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{
			"name": func(obj *v1.Pod) ([]string, error) {
				return []string{obj.Name}, nil
			},
		})
		tCtx.ExpectNoError(err)

		res, err := raw.ByIndex("name", "pod1")
		tCtx.ExpectNoError(err)
		tCtx.Assert(res).Should(gomega.Equal([]interface{}{pod1}))
	})

	var _ TypedIndexer[*v1.Pod] = typedIndexer[*v1.Pod]{}
}

// fakeSharedIndexInformer is a minimal SharedIndexInformer stand-in that
// records calls made to it by typedSharedIndexInformer, without needing a
// fully functional informer.
type fakeSharedIndexInformer struct {
	SharedIndexInformer // nil: only the methods overridden below may be called.

	handlerOptions []HandlerOptions

	addedIndexers  Indexers
	addIndexersErr error

	indexer Indexer
}

func (f *fakeSharedIndexInformer) AddEventHandlerWithOptions(handler ResourceEventHandler, options HandlerOptions) (ResourceEventHandlerRegistration, error) {
	f.handlerOptions = append(f.handlerOptions, options)
	return nil, nil
}

func (f *fakeSharedIndexInformer) AddIndexers(indexers Indexers) error {
	f.addedIndexers = indexers
	return f.addIndexersErr
}

func (f *fakeSharedIndexInformer) GetIndexer() Indexer {
	return f.indexer
}

func TestTypedSharedIndexInformerAddTypedEventHandler(t *testing.T) {
	period := time.Minute

	for _, tc := range []struct {
		name    string
		options []HandlerOptions
		want    HandlerOptions
		wantErr bool
	}{
		{
			name: "no-options-uses-the-default",
			want: HandlerOptions{},
		},
		{
			name:    "one-option-is-passed-through",
			options: []HandlerOptions{{ResyncPeriod: &period}},
			want:    HandlerOptions{ResyncPeriod: &period},
		},
		{
			name:    "two-options-are-rejected",
			options: []HandlerOptions{{}, {}},
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fake := &fakeSharedIndexInformer{}
			typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

			_, err := typed.AddTypedEventHandler(TypedResourceEventHandlerFuncs[*v1.Pod]{}, tc.options...)
			if tc.wantErr {
				tCtx.Require(err).Should(gomega.HaveOccurred())
				tCtx.Assert(fake.handlerOptions).Should(gomega.BeEmpty())
				return
			}
			tCtx.ExpectNoError(err)
			tCtx.Require(fake.handlerOptions).Should(gomega.HaveLen(1))
			tCtx.Assert(fake.handlerOptions[0]).Should(gomega.Equal(tc.want))
		})
	}
}

func TestTypedSharedIndexInformerAddTypedIndexers(t *testing.T) {
	tCtx := ktesting.Init(t)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}

	fake := &fakeSharedIndexInformer{}
	typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

	err := typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{
		"name": func(obj *v1.Pod) ([]string, error) {
			return []string{obj.Name}, nil
		},
	})
	tCtx.ExpectNoError(err)
	tCtx.Require(fake.addedIndexers).Should(gomega.HaveKey("name"))

	values, err := fake.addedIndexers["name"](pod)
	tCtx.ExpectNoError(err)
	tCtx.Assert(values).Should(gomega.Equal([]string{"pod"}))

	fake.addIndexersErr = fmt.Errorf("boom")
	err = typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{})
	tCtx.Assert(err).Should(gomega.MatchError("boom"))
}

func TestTypedSharedIndexInformerGetTypedIndexer(t *testing.T) {
	tCtx := ktesting.Init(t)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod"}}
	raw := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		"name": func(obj interface{}) ([]string, error) {
			return []string{obj.(*v1.Pod).Name}, nil
		},
	})
	tCtx.ExpectNoError(raw.Add(pod))

	fake := &fakeSharedIndexInformer{indexer: raw}
	typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

	typedIndexer := typed.GetTypedIndexer()
	res, err := typedIndexer.TypedIndex("name", pod)
	tCtx.ExpectNoError(err)
	tCtx.Assert(res).Should(gomega.Equal([]*v1.Pod{pod}))

	res, err = typedIndexer.ByTypedIndex("name", "pod")
	tCtx.ExpectNoError(err)
	tCtx.Assert(res).Should(gomega.Equal([]*v1.Pod{pod}))
}

func TestTypedIndexersToIndexers(t *testing.T) {
	tCtx := ktesting.Init(t)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}

	untyped := TypedIndexersToIndexers(TypedIndexers[*v1.Pod]{
		"name": func(obj *v1.Pod) ([]string, error) {
			return []string{obj.Name}, nil
		},
	})
	tCtx.Require(untyped).Should(gomega.HaveKey("name"))
	values, err := untyped["name"](pod)
	tCtx.ExpectNoError(err)
	tCtx.Assert(values).Should(gomega.Equal([]string{"pod"}))
}
