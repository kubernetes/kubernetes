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

package etcd3

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestWatchWithoutPrevKVValidation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		configure func(*watcher, *storage.ListOptions)
		wantError string
	}{
		{
			name:      "missing object constructor",
			configure: func(w *watcher, _ *storage.ListOptions) { w.newFunc = nil },
			wantError: "requires a newFunc",
		},
		{
			name: "label selector",
			configure: func(_ *watcher, opts *storage.ListOptions) {
				opts.Predicate.Label = labels.SelectorFromSet(labels.Set{"app": "test"})
			},
			wantError: "requires an empty predicate",
		},
		{
			name: "field selector",
			configure: func(_ *watcher, opts *storage.ListOptions) {
				opts.Predicate.Field = fields.OneTermEqualSelector("metadata.name", "pod")
			},
			wantError: "requires an empty predicate",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := &watcher{
				newFunc:        newPod,
				reverseKeyFunc: func(storageKey) (string, string, error) { return "pod", "ns", nil },
			}
			opts := storage.ListOptions{WatchWithoutPrevKV: true, Predicate: storage.Everything}
			tc.configure(w, &opts)
			result, err := w.Watch(context.Background(), "/pods/ns/pod", 1, opts)
			require.Nil(t, result)
			require.ErrorContains(t, err, tc.wantError)
			require.True(t, apierrors.IsInternalError(err))
		})
	}
}

func TestWatchWithoutPrevKVReverseError(t *testing.T) {
	wantErr := errors.New("invalid resource key")
	wc := &watchChan{
		watchWithoutPrevKV: true,
		watcher: &watcher{
			reverseKeyFunc: func(storageKey) (string, string, error) { return "", "", wantErr },
		},
	}
	_, _, err := wc.prepareObjs(&event{key: "/registry/pods", isDeleted: true, rev: 2})
	require.ErrorIs(t, err, wantErr)
}

type prevKVRecordingWatcher struct {
	clientv3.Watcher
	options chan bool
}

func (w *prevKVRecordingWatcher) Watch(ctx context.Context, key string, opts ...clientv3.OpOption) clientv3.WatchChan {
	w.options <- clientv3.OpGet(key, opts...).IsPrevKV()
	return w.Watcher.Watch(ctx, key, opts...)
}

func TestWatchWithoutPrevKV(t *testing.T) {
	for _, tc := range []struct {
		name, prefix, namespace string
		withoutReverseKeyFunc   bool
	}{
		{name: "namespaced", prefix: "/registry", namespace: "ns"},
		{name: "root prefix", prefix: "/", namespace: "ns"},
		{name: "cluster scoped", prefix: "/custom/prefix"},
		{name: "custom key without reverse mapping", prefix: "/registry", namespace: "ns", withoutReverseKeyFunc: true},
	} {
		for _, withoutPrevKV := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/withoutPrevKV=%t", tc.name, withoutPrevKV), func(t *testing.T) {
				key := "/pods/pod"
				if tc.namespace != "" {
					key = "/pods/" + tc.namespace + "/pod"
				}
				reverse := func(resourceKey string) (string, string, error) {
					if resourceKey != key {
						return "", "", fmt.Errorf("expected resource key %q, got %q", key, resourceKey)
					}
					return "pod", tc.namespace, nil
				}
				if tc.withoutReverseKeyFunc {
					reverse = nil
				}
				expectPrevKV := !withoutPrevKV || tc.withoutReverseKeyFunc
				ctx, store, client := testSetup(t, withPrefix(tc.prefix), withResourcePrefix("/pods"), withReverseKeyFunc(reverse))
				recording := &prevKVRecordingWatcher{Watcher: client.Watcher, options: make(chan bool, 1)}
				client.Watcher = recording
				pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{
					Name: "pod", Namespace: tc.namespace, Labels: map[string]string{"app": "test"},
				}}
				created := &example.Pod{}
				require.NoError(t, store.Create(ctx, key, pod, created, 0))
				w, err := store.Watch(ctx, "/pods", storage.ListOptions{
					ResourceVersion: "0", Recursive: true, Predicate: storage.Everything, WatchWithoutPrevKV: withoutPrevKV,
				})
				require.NoError(t, err)
				defer w.Stop()
				checkEvent := func(eventType watch.EventType, object runtime.Object) {
					t.Helper()
					select {
					case e, ok := <-w.ResultChan():
						require.True(t, ok, "watch closed unexpectedly")
						require.Equal(t, watch.Event{Type: eventType, Object: object}, e)
					case <-time.After(10 * time.Second):
						t.Fatal("timed out waiting for watch event")
					}
				}
				checkEvent(watch.Added, created)
				select {
				case prevKV := <-recording.options:
					require.Equal(t, expectPrevKV, prevKV, "etcd watch PrevKV option")
				case <-time.After(10 * time.Second):
					t.Fatal("etcd watch was not started")
				}
				updated := &example.Pod{}
				require.NoError(t, store.GuaranteedUpdate(ctx, key, updated, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					obj.(*example.Pod).Annotations = map[string]string{"updated": "true"}
					return obj, nil
				}), nil))
				checkEvent(watch.Modified, updated)
				deleted := &example.Pod{}
				require.NoError(t, store.Delete(ctx, key, deleted, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{}))
				if !expectPrevKV {
					deleted = &example.Pod{ObjectMeta: metav1.ObjectMeta{
						Name: pod.Name, Namespace: pod.Namespace, ResourceVersion: deleted.ResourceVersion,
					}}
				}
				checkEvent(watch.Deleted, deleted)
			})
		}
	}
}
