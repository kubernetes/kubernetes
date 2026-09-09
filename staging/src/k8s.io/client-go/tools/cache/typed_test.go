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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		assert.Equal(t, []*v1.Pod{pod}, added)
		addedInitial = true // AddFunc doesn't get isInInitialList, only presence is checked above
		_ = addedInitial

		funcs.OnUpdate(oldPod, pod)
		assert.Equal(t, oldPod, updatedOld)
		assert.Equal(t, pod, updatedNew)

		funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		assert.Equal(t, pod, deleted.OptionalObj)
	})

	t.Run("nil-callbacks-do-not-panic", func(t *testing.T) {
		var funcs TypedResourceEventHandlerFuncs[*v1.Pod]
		assert.NotPanics(t, func() {
			funcs.OnAdd(pod, false)
			funcs.OnUpdate(oldPod, pod)
			funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		})
	})

	// Compile-time and runtime check that the type implements the interface.
	var _ TypedResourceEventHandler[*v1.Pod] = TypedResourceEventHandlerFuncs[*v1.Pod]{}
}

func TestTypedResourceEventHandlerDetailedFuncs(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}
	oldPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old-pod"}}

	t.Run("callbacks-are-invoked-including-isInInitialList", func(t *testing.T) {
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
		assert.Equal(t, pod, added)
		assert.True(t, addedInitial)

		funcs.OnAdd(pod, false)
		assert.False(t, addedInitial)

		funcs.OnUpdate(oldPod, pod)
		assert.Equal(t, oldPod, updatedOld)
		assert.Equal(t, pod, updatedNew)

		funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		assert.Equal(t, pod, deleted.OptionalObj)
	})

	t.Run("nil-callbacks-do-not-panic", func(t *testing.T) {
		var funcs TypedResourceEventHandlerDetailedFuncs[*v1.Pod]
		assert.NotPanics(t, func() {
			funcs.OnAdd(pod, true)
			funcs.OnUpdate(oldPod, pod)
			funcs.OnDelete(DeletedObject[*v1.Pod]{OptionalObj: pod})
		})
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
				handler := &recordingTypedHandler{}
				r := TypedFilteringResourceEventHandler[*v1.Pod]{
					FilterFunc: func(obj *v1.Pod) bool { return tc.passes },
					Handler:    handler,
				}
				r.OnAdd(newPod, true)
				if tc.passes {
					require.Len(t, handler.added, 1)
					assert.Equal(t, newPod, handler.added[0])
					assert.Equal(t, []bool{true}, handler.addedInitial)
				} else {
					assert.Empty(t, handler.added)
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
				handler := &recordingTypedHandler{}
				r := TypedFilteringResourceEventHandler[*v1.Pod]{
					FilterFunc: func(obj *v1.Pod) bool { return tc.passes },
					Handler:    handler,
				}
				deleted := DeletedObject[*v1.Pod]{OptionalObj: oldPod}
				r.OnDelete(deleted)
				if tc.passes {
					require.Len(t, handler.deleted, 1)
					assert.Equal(t, deleted, handler.deleted[0])
				} else {
					assert.Empty(t, handler.deleted)
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
					require.Len(t, handler.added, 1)
					assert.Equal(t, newPod, handler.added[0])
					assert.Equal(t, []bool{false}, handler.addedInitial)
				} else {
					assert.Empty(t, handler.added)
				}

				if tc.wantUpdate {
					require.Len(t, handler.updatedNew, 1)
					assert.Equal(t, oldPod, handler.updatedOld[0])
					assert.Equal(t, newPod, handler.updatedNew[0])
				} else {
					assert.Empty(t, handler.updatedNew)
				}

				if tc.wantDelete {
					require.Len(t, handler.deleted, 1)
					assert.Equal(t, oldPod, handler.deleted[0].OptionalObj)
				} else {
					assert.Empty(t, handler.deleted)
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
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnAdd(pod, true)
		require.Len(t, handler.added, 1)
		assert.Equal(t, pod, handler.added[0])
		assert.Equal(t, []bool{true}, handler.addedInitial)
	})

	t.Run("OnUpdate", func(t *testing.T) {
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnUpdate(oldPod, pod)
		require.Len(t, handler.updatedNew, 1)
		assert.Equal(t, oldPod, handler.updatedOld[0])
		assert.Equal(t, pod, handler.updatedNew[0])
	})

	t.Run("OnDelete-with-a-plain-object", func(t *testing.T) {
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		adapter.OnDelete(pod)
		require.Len(t, handler.deleted, 1)
		assert.Equal(t, pod, handler.deleted[0].OptionalObj)
		assert.Nil(t, handler.deleted[0].FinalStateUnknown)
	})

	t.Run("OnDelete-with-DeletedFinalStateUnknown-carrying-an-object", func(t *testing.T) {
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		tomb := DeletedFinalStateUnknown{Key: "default/pod", Obj: pod}
		adapter.OnDelete(tomb)
		require.Len(t, handler.deleted, 1)
		assert.Equal(t, pod, handler.deleted[0].OptionalObj)
		require.NotNil(t, handler.deleted[0].FinalStateUnknown)
		assert.Equal(t, "default/pod", handler.deleted[0].FinalStateUnknown.Key)
	})

	t.Run("OnDelete-with-DeletedFinalStateUnknown-without-an-object", func(t *testing.T) {
		handler := &recordingTypedHandler{}
		adapter := &typedResourceEventHandler[*v1.Pod]{handler: handler}
		tomb := DeletedFinalStateUnknown{Key: "default/pod", Obj: nil}
		adapter.OnDelete(tomb)
		require.Len(t, handler.deleted, 1)
		var zero *v1.Pod
		assert.Equal(t, zero, handler.deleted[0].OptionalObj)
		require.NotNil(t, handler.deleted[0].FinalStateUnknown)
		assert.Equal(t, "default/pod", handler.deleted[0].FinalStateUnknown.Key)
	})

	var _ ResourceEventHandler = &typedResourceEventHandler[*v1.Pod]{}
}

func TestTypedIndexer(t *testing.T) {
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1", Labels: map[string]string{"foo": "bar"}}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod2", Labels: map[string]string{"foo": "bar"}}}

	raw := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		"foo": func(obj interface{}) ([]string, error) {
			return []string{obj.(*v1.Pod).Labels["foo"]}, nil
		},
	})
	require.NoError(t, raw.Add(pod1))
	require.NoError(t, raw.Add(pod2))

	typed := typedIndexer[*v1.Pod]{Indexer: raw}

	t.Run("TypedIndex", func(t *testing.T) {
		res, err := typed.TypedIndex("foo", pod1)
		require.NoError(t, err)
		assert.ElementsMatch(t, []*v1.Pod{pod1, pod2}, res)
	})

	t.Run("ByTypedIndex", func(t *testing.T) {
		res, err := typed.ByTypedIndex("foo", "bar")
		require.NoError(t, err)
		assert.ElementsMatch(t, []*v1.Pod{pod1, pod2}, res)
	})

	t.Run("ByTypedIndex-propagates-errors", func(t *testing.T) {
		_, err := typed.ByTypedIndex("does-not-exist", "bar")
		require.Error(t, err)
	})

	t.Run("AddTypedIndexers", func(t *testing.T) {
		err := typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{
			"name": func(obj *v1.Pod) ([]string, error) {
				return []string{obj.Name}, nil
			},
		})
		require.NoError(t, err)

		res, err := raw.ByIndex("name", "pod1")
		require.NoError(t, err)
		assert.Equal(t, []interface{}{pod1}, res)
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
			fake := &fakeSharedIndexInformer{}
			typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

			_, err := typed.AddTypedEventHandler(TypedResourceEventHandlerFuncs[*v1.Pod]{}, tc.options...)
			if tc.wantErr {
				require.Error(t, err)
				assert.Empty(t, fake.handlerOptions)
				return
			}
			require.NoError(t, err)
			require.Len(t, fake.handlerOptions, 1)
			assert.Equal(t, tc.want, fake.handlerOptions[0])
		})
	}
}

func TestTypedSharedIndexInformerAddTypedIndexers(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}

	fake := &fakeSharedIndexInformer{}
	typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

	err := typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{
		"name": func(obj *v1.Pod) ([]string, error) {
			return []string{obj.Name}, nil
		},
	})
	require.NoError(t, err)
	require.Contains(t, fake.addedIndexers, "name")

	values, err := fake.addedIndexers["name"](pod)
	require.NoError(t, err)
	assert.Equal(t, []string{"pod"}, values)

	fake.addIndexersErr = fmt.Errorf("boom")
	err = typed.AddTypedIndexers(TypedIndexers[*v1.Pod]{})
	assert.EqualError(t, err, "boom")
}

func TestTypedSharedIndexInformerGetTypedIndexer(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod"}}
	raw := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		"name": func(obj interface{}) ([]string, error) {
			return []string{obj.(*v1.Pod).Name}, nil
		},
	})
	require.NoError(t, raw.Add(pod))

	fake := &fakeSharedIndexInformer{indexer: raw}
	typed := NewTypedSharedIndexInformer[*v1.Pod](fake)

	typedIndexer := typed.GetTypedIndexer()
	res, err := typedIndexer.TypedIndex("name", pod)
	require.NoError(t, err)
	assert.Equal(t, []*v1.Pod{pod}, res)

	res, err = typedIndexer.ByTypedIndex("name", "pod")
	require.NoError(t, err)
	assert.Equal(t, []*v1.Pod{pod}, res)
}

func TestTypedIndexersToIndexers(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod"}}

	untyped := TypedIndexersToIndexers(TypedIndexers[*v1.Pod]{
		"name": func(obj *v1.Pod) ([]string, error) {
			return []string{obj.Name}, nil
		},
	})
	require.Contains(t, untyped, "name")
	values, err := untyped["name"](pod)
	require.NoError(t, err)
	assert.Equal(t, []string{"pod"}, values)
}
