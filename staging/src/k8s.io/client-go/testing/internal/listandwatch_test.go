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

package internal

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init" // for -testing.v
)

// TestListAndWatch mirrors how fake client-go is often used with real
// informers. It enforces a timing such that List completes, a new
// object gets created because of the completed cache sync, and only
// then is the Watch call in the reflector's "ListAndWatch" allowed to
// continue.
//
// The fake Watch implementation then must use the ResourceVersion to
// detect that it must send some (but not all!) objects to the new watch.
//
// This runs in a synctest bubble, therefore time is virtual.
func TestListAndWatch(t *testing.T) { synctest.Test(t, testListAndWatch) }
func testListAndWatch(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cm1",
			Namespace: "default",
		},
	}
	client := fake.NewClientset(cm)
	createDone := make(chan struct{})

	f := informers.NewSharedInformerFactory(client, 0)
	configMapInformer := f.InformerFor(&v1.ConfigMap{}, func(client kubernetes.Interface, defaultEventHandlerResyncPeriod time.Duration) cache.SharedIndexInformer {

		return cache.NewSharedIndexInformer(cache.ToListWatcherWithWatchListSemantics(&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				objs, err := client.CoreV1().ConfigMaps("").List(context.Background(), options)
				logger.Info("Listed", "configMaps", objs, "err", err)
				if err != nil {
					t.Errorf("Unexpected List error: %v", err)
				} else if objs.ResourceVersion != "2" {
					t.Errorf("Expected ListMeta ResourceVersion 2, got %q", objs.ResourceVersion)
				}
				return objs, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				if options.ResourceVersion != "2" {
					t.Errorf("Expected ListOptions ResourceVersion 2, got %q", options.ResourceVersion)
				}
				logger.Info("Delaying Watch...")
				<-createDone
				logger.Info("Continuing Watch...")
				return client.CoreV1().ConfigMaps("").Watch(context.Background(), options)
			},
		}, client), &v1.ConfigMap{}, defaultEventHandlerResyncPeriod, nil)
	})

	var adds, updates, deletes int
	handle, err := configMapInformer.AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(_ any) { adds++ },
		UpdateFunc: func(_, _ any) { updates++ },
		DeleteFunc: func(_ any) { deletes++ },
	}, cache.HandlerOptions{Logger: &logger})
	if err != nil {
		t.Fatalf("Unexpected error adding event handler: %v", err)
	}
	defer configMapInformer.RemoveEventHandler(handle)

	configMapStore := configMapInformer.GetStore()
	f.StartWithContext(ctx)
	f.WaitForCacheSyncWithContext(ctx)
	logger.Info("Caches synced")

	objs := configMapStore.List()
	if len(objs) != 1 {
		t.Fatalf("Unexpected item(s) in informer cache, want 1, got %d = %v", len(objs), objs)
	}

	cm = &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cm2",
			Namespace: "default",
		},
	}
	_, err = client.CoreV1().ConfigMaps(cm.Namespace).Create(ctx, cm, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating ConfigMap: %v", err)
	}
	logger.Info("Created second ConfigMap")
	close(createDone)

	// Wait for watch setup and event processing.
	synctest.Wait()

	objs = configMapStore.List()
	if len(objs) != 2 {
		t.Errorf("Unexpected item(s) in informer cache, want 2, got %d = %v", len(objs), objs)
	}

	if !handle.HasSynced() {
		t.Error("Expected event handler to have synced, it didn't")
	}
	if adds != 2 || updates != 0 || deletes != 0 {
		t.Errorf("Expected two new objects, got adds/updates/deletes %d/%d/%d", adds, updates, deletes)
	}
}

// TestListAndWatchDelete is like TestListAndWatch, but the objects get deleted
// instead of a new one being created. The Watch must then deliver the deletion,
// or fail such that the reflector lists again. Otherwise the informer cache
// keeps the objects forever.
func TestListAndWatchDelete(t *testing.T) {
	for name, tc := range map[string]struct {
		initial int
		// changeInGap runs after the caches have synced and before the Watch call gets to continue.
		changeInGap func(ctx context.Context, t *testing.T, client kubernetes.Interface)
		// expectedNames are the objects that must be in the informer cache in the end.
		expectedNames []string
		// expectedEvents are the number of added, updated and deleted events in the end.
		expectedAdds, expectedUpdates, expectedDeletes int
	}{
		"deleted": {
			initial: 1,
			changeInGap: func(ctx context.Context, t *testing.T, client kubernetes.Interface) {
				deleteConfigMap(ctx, t, client, "cm0")
			},
			expectedAdds:    1,
			expectedDeletes: 1,
		},
		"deleted-and-created-again": {
			initial: 1,
			changeInGap: func(ctx context.Context, t *testing.T, client kubernetes.Interface) {
				deleteConfigMap(ctx, t, client, "cm0")
				createConfigMap(ctx, t, client, "cm0")
			},
			expectedNames: []string{"cm0"},
			expectedAdds:  2,
			// Between the first and second object with the same name.
			expectedDeletes: 1,
		},
		// The fake client only remembers a limited number of deletions. When
		// the history is incomplete, the reflector must list again.
		"deleted-more-than-remembered": {
			initial: 150,
			changeInGap: func(ctx context.Context, t *testing.T, client kubernetes.Interface) {
				for i := 0; i < 150; i++ {
					deleteConfigMap(ctx, t, client, fmt.Sprintf("cm%d", i))
				}
			},
			expectedAdds:    150,
			expectedDeletes: 150,
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				var initial []runtime.Object
				for i := 0; i < tc.initial; i++ {
					initial = append(initial, newConfigMap(fmt.Sprintf("cm%d", i)))
				}
				client := fake.NewClientset(initial...)
				changeDone := make(chan struct{})

				f := informers.NewSharedInformerFactory(client, 0)
				configMapInformer := f.InformerFor(&v1.ConfigMap{}, func(client kubernetes.Interface, defaultEventHandlerResyncPeriod time.Duration) cache.SharedIndexInformer {
					return cache.NewSharedIndexInformer(cache.ToListWatcherWithWatchListSemantics(&cache.ListWatch{
						ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
							return client.CoreV1().ConfigMaps("").List(context.Background(), options)
						},
						WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
							logger.Info("Delaying Watch...")
							<-changeDone
							logger.Info("Continuing Watch...")
							return client.CoreV1().ConfigMaps("").Watch(context.Background(), options)
						},
					}, client), &v1.ConfigMap{}, defaultEventHandlerResyncPeriod, nil)
				})

				var adds, updates, deletes int
				handle, err := configMapInformer.AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
					AddFunc:    func(_ any) { adds++ },
					UpdateFunc: func(_, _ any) { updates++ },
					DeleteFunc: func(_ any) { deletes++ },
				}, cache.HandlerOptions{Logger: &logger})
				if err != nil {
					t.Fatalf("Unexpected error adding event handler: %v", err)
				}
				defer configMapInformer.RemoveEventHandler(handle)

				configMapStore := configMapInformer.GetStore()
				f.StartWithContext(ctx)
				f.WaitForCacheSyncWithContext(ctx)
				logger.Info("Caches synced")

				if objs := configMapStore.List(); len(objs) != tc.initial {
					t.Fatalf("Unexpected item(s) in informer cache, want %d, got %d", tc.initial, len(objs))
				}

				tc.changeInGap(ctx, t, client)
				close(changeDone)

				// Wait for watch setup and event processing. If the Watch call
				// fails, the reflector waits a bit before it lists again.
				time.Sleep(time.Minute)
				synctest.Wait()

				var names []string
				for _, obj := range configMapStore.List() {
					names = append(names, obj.(*v1.ConfigMap).Name)
				}
				slices.Sort(names)
				if !slices.Equal(names, tc.expectedNames) {
					t.Errorf("Unexpected items in informer cache, want %v, got %v", tc.expectedNames, names)
				}
				if adds != tc.expectedAdds || updates != tc.expectedUpdates || deletes != tc.expectedDeletes {
					t.Errorf("Expected adds/updates/deletes %d/%d/%d, got %d/%d/%d", tc.expectedAdds, tc.expectedUpdates, tc.expectedDeletes, adds, updates, deletes)
				}
			})
		})
	}
}

func newConfigMap(name string) *v1.ConfigMap {
	return &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"}}
}

func createConfigMap(ctx context.Context, t *testing.T, client kubernetes.Interface, name string) {
	t.Helper()
	if _, err := client.CoreV1().ConfigMaps("default").Create(ctx, newConfigMap(name), metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error creating ConfigMap %s: %v", name, err)
	}
}

func deleteConfigMap(ctx context.Context, t *testing.T, client kubernetes.Interface, name string) {
	t.Helper()
	if err := client.CoreV1().ConfigMaps("default").Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Unexpected error deleting ConfigMap %s: %v", name, err)
	}
}
