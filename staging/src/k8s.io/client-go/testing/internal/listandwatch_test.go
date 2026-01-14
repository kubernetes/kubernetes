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
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cm1",
			Namespace: "default",
		},
	}
	client := fake.NewClientset(cm)
	stopCh := make(chan struct{})
	defer close(stopCh)
	createDone := make(chan struct{})

	f := informers.NewSharedInformerFactory(client, 0)
	configMapInformer := f.InformerFor(&v1.ConfigMap{}, func(client kubernetes.Interface, defaultEventHandlerResyncPeriod time.Duration) cache.SharedIndexInformer {

		return cache.NewSharedIndexInformer(cache.ToListWatcherWithWatchListSemantics(&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				objs, err := client.CoreV1().ConfigMaps("").List(context.Background(), options)
				logger.Info("Listed", "configMaps", objs, "err", err)
				if err != nil {
					t.Errorf("Unexpected List error: %v", err)
				} else if objs.ResourceVersion != "1" {
					t.Errorf("Expected ListMeta ResourceVersion 1, got %q", objs.ResourceVersion)
				}
				return objs, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				if options.ResourceVersion != "1" {
					t.Errorf("Expected ListOptions ResourceVersion 1, got %q", options.ResourceVersion)
				}
				logger.Info("Delaying Watch...")
				<-createDone
				logger.Info("Continuing Watch...")
				return client.CoreV1().ConfigMaps("").Watch(context.Background(), options)
			},
		}, client), &v1.ConfigMap{}, defaultEventHandlerResyncPeriod, nil)
	})

	var adds, updates, deletes int
	handle, err := configMapInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(_ any) { adds++ },
		UpdateFunc: func(_, _ any) { updates++ },
		DeleteFunc: func(_ any) { deletes++ },
	})
	if err != nil {
		t.Fatalf("Unexpected error adding event handler: %v", err)
	}
	defer configMapInformer.RemoveEventHandler(handle)

	configMapStore := configMapInformer.GetStore()
	f.Start(stopCh)
	f.WaitForCacheSync(stopCh)
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
