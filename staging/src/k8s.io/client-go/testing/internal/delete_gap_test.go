/*
Copyright 2026 The Kubernetes Authors.

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

// TestListAndWatchDelete mirrors TestListAndWatch for deletions: the object
// is deleted after the cache sync and before the Watch call in the
// reflector's ListAndWatch is allowed to continue.
//
// Unlike a lost create (covered by TestListAndWatch), a lost delete is
// irrecoverable: the informer's store would keep a phantom object forever
// because no relist happens and no DeleteFunc ever fires. The fake client
// must converge on the tracker's contents regardless of the timing of
// Watch, for deletions as it now does for creations — either the deletion
// is delivered, or the watch is rejected so that the reflector relists.
//
// This runs in a synctest bubble, therefore time is virtual.
func TestListAndWatchDelete(t *testing.T) { synctest.Test(t, testListAndWatchDelete) }
func testListAndWatchDelete(t *testing.T) {
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
	deleteDone := make(chan struct{})

	f := informers.NewSharedInformerFactory(client, 0)
	configMapInformer := f.InformerFor(&v1.ConfigMap{}, func(client kubernetes.Interface, defaultEventHandlerResyncPeriod time.Duration) cache.SharedIndexInformer {

		return cache.NewSharedIndexInformer(cache.ToListWatcherWithWatchListSemantics(&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				objs, err := client.CoreV1().ConfigMaps("").List(context.Background(), options)
				logger.Info("Listed", "configMaps", objs, "err", err)
				return objs, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				logger.Info("Delaying Watch...")
				<-deleteDone
				logger.Info("Continuing Watch...")
				return client.CoreV1().ConfigMaps("").Watch(context.Background(), options)
			},
		}, client), &v1.ConfigMap{}, defaultEventHandlerResyncPeriod, nil)
	})

	var adds, updates, deletes int
	if _, err := configMapInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(_ any) { adds++ },
		UpdateFunc: func(_, _ any) { updates++ },
		DeleteFunc: func(_ any) { deletes++ },
	}); err != nil {
		t.Fatalf("Unexpected error adding event handler: %v", err)
	}

	store := configMapInformer.GetStore()
	f.Start(stopCh)
	f.WaitForCacheSync(stopCh)
	logger.Info("Caches synced")

	if err := client.CoreV1().ConfigMaps("default").Delete(ctx, "cm1", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Unexpected error deleting ConfigMap: %v", err)
	}
	logger.Info("Deleted the ConfigMap")
	close(deleteDone)

	// Wait for the watch setup, its rejection and the re-list triggered by
	// the expired resource version, then event processing.
	synctest.Wait()

	if objs := store.List(); len(objs) != 0 {
		t.Errorf("Unexpected item(s) in informer cache, want 0, got %d = %v", len(objs), objs)
	}
	if adds != 1 || updates != 0 || deletes != 1 {
		t.Errorf("Expected the object to be added and deleted, got adds/updates/deletes %d/%d/%d", adds, updates, deletes)
	}
}