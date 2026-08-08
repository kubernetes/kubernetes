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

package network

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestServiceListResourceVersionContract pins the two properties the NodePort
// repair loop relies on before it releases a port that no Service in its
// informer references:
//
//  1. A LIST of Services with resourceVersion="" returns a collection
//     resourceVersion at least as new as every Service write that completed
//     before the request, even when Limit truncates the response to one item.
//  2. Once the informer store's LastStoreSyncResourceVersion is at least that
//     resourceVersion, the lister reflects every such write: created Services
//     are present and deleted Services are gone.
//
// The watch cache computes the collection resourceVersion differently from
// the etcd path, so both are exercised.
func TestServiceListResourceVersionContract(t *testing.T) {
	// LastStoreSyncResourceVersion is only reported with AtomicFIFO on.
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.AtomicFIFO, true)

	testcases := []struct {
		name   string
		create int
		delete int
		limit  int64
	}{
		{name: "single-service", create: 1, limit: 1},
		{name: "burst-larger-than-page", create: 25, limit: 1},
		{name: "unpaginated", create: 5, limit: 0},
		{name: "deletes", create: 6, delete: 4, limit: 1},
	}

	for _, watchCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("watchCache=%t", watchCache), func(t *testing.T) {
			tCtx := ktesting.Init(t)
			client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
					opts.Etcd.EnableWatchCache = watchCache
				},
			})
			defer tearDownFn()

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			serviceInformer := informerFactory.Core().V1().Services()
			store := serviceInformer.Informer().GetStore()
			lister := serviceInformer.Lister()
			informerFactory.Start(tCtx.Done())
			if !cache.WaitForCacheSync(tCtx.Done(), serviceInformer.Informer().HasSynced) {
				t.Fatal("failed to sync Service informer")
			}
			if store.LastStoreSyncResourceVersion() == "" {
				t.Fatal("informer store reports no resource version after sync")
			}

			for _, tc := range testcases {
				t.Run(tc.name, func(t *testing.T) {
					ns := framework.CreateNamespaceOrDie(client, "svc-rv-"+tc.name, t)
					defer framework.DeleteNamespaceOrDie(client, ns, t)

					var maxWriteRV string
					created := make([]string, 0, tc.create)
					for i := 0; i < tc.create; i++ {
						svc, err := client.CoreV1().Services(ns.Name).Create(tCtx, newClusterIPService(fmt.Sprintf("svc-%d", i)), metav1.CreateOptions{})
						if err != nil {
							t.Fatalf("failed to create service: %v", err)
						}
						created = append(created, svc.Name)
						maxWriteRV = maxResourceVersion(t, maxWriteRV, svc.ResourceVersion)
					}
					deleted := created[:tc.delete]
					remaining := created[tc.delete:]
					for _, name := range deleted {
						if err := client.CoreV1().Services(ns.Name).Delete(tCtx, name, metav1.DeleteOptions{}); err != nil {
							t.Fatalf("failed to delete service %s: %v", name, err)
						}
					}

					// Every write above completed before this request was issued.
					list, err := client.CoreV1().Services(metav1.NamespaceAll).List(tCtx, metav1.ListOptions{Limit: tc.limit})
					if err != nil {
						t.Fatalf("failed to list services: %v", err)
					}
					if tc.limit > 0 && int64(len(list.Items)) > tc.limit {
						t.Fatalf("expected at most %d items, got %d", tc.limit, len(list.Items))
					}
					if cmp := compareResourceVersion(t, list.ResourceVersion, maxWriteRV); cmp < 0 {
						t.Fatalf("collection resourceVersion %s is older than a Service write %s that completed before the list", list.ResourceVersion, maxWriteRV)
					}

					// The store only advances on Service events or bookmarks; issue one
					// more write so the test does not wait on the bookmark interval.
					if _, err := client.CoreV1().Services(ns.Name).Create(tCtx, newClusterIPService("nudge"), metav1.CreateOptions{}); err != nil {
						t.Fatalf("failed to create nudge service: %v", err)
					}
					if err := wait.PollUntilContextTimeout(tCtx, 50*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
						return compareResourceVersion(t, store.LastStoreSyncResourceVersion(), list.ResourceVersion) >= 0, nil
					}); err != nil {
						t.Fatalf("informer store never reached the collection resourceVersion %s (last seen %s): %v", list.ResourceVersion, store.LastStoreSyncResourceVersion(), err)
					}

					// The lister is now at least as new as the collection was, so it must
					// reflect every write that completed before the list.
					for _, name := range remaining {
						if _, err := lister.Services(ns.Name).Get(name); err != nil {
							t.Errorf("service %s/%s was created before the list but is missing from the lister at store resourceVersion %s: %v", ns.Name, name, store.LastStoreSyncResourceVersion(), err)
						}
					}
					for _, name := range deleted {
						if _, err := lister.Services(ns.Name).Get(name); !apierrors.IsNotFound(err) {
							t.Errorf("service %s/%s was deleted before the list but is still in the lister at store resourceVersion %s (err=%v)", ns.Name, name, store.LastStoreSyncResourceVersion(), err)
						}
					}
				})
			}
		})
	}
}

func newClusterIPService(name string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.ServiceSpec{
			Type:  v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{{Port: 80}},
		},
	}
}

func compareResourceVersion(t *testing.T, a, b string) int {
	t.Helper()
	cmp, err := resourceversion.CompareResourceVersion(a, b)
	if err != nil {
		t.Fatalf("resource versions %q and %q are not comparable: %v", a, b, err)
	}
	return cmp
}

func maxResourceVersion(t *testing.T, current, candidate string) string {
	t.Helper()
	if current == "" || compareResourceVersion(t, candidate, current) > 0 {
		return candidate
	}
	return current
}
