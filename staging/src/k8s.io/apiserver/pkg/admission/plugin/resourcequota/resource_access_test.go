/*
Copyright 2020 The Kubernetes Authors.

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

package resourcequota

import (
	"fmt"
	core "k8s.io/client-go/testing"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/lru"
)

func TestLRUCacheLookup(t *testing.T) {
	namespace := "foo"
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
	}

	testcases := []struct {
		description   string
		cacheInput    []*corev1.ResourceQuota
		clientInput   []runtime.Object
		ttl           time.Duration
		namespace     string
		expectedQuota *corev1.ResourceQuota
	}{
		{
			description:   "object is found via cache",
			cacheInput:    []*corev1.ResourceQuota{resourceQuota},
			ttl:           30 * time.Second,
			namespace:     namespace,
			expectedQuota: resourceQuota,
		},
		{
			description:   "object is outdated and not found with client",
			cacheInput:    []*corev1.ResourceQuota{resourceQuota},
			ttl:           -30 * time.Second,
			namespace:     namespace,
			expectedQuota: nil,
		},
		{
			description:   "object is outdated but is found with client",
			cacheInput:    []*corev1.ResourceQuota{resourceQuota},
			clientInput:   []runtime.Object{resourceQuota},
			ttl:           -30 * time.Second,
			namespace:     namespace,
			expectedQuota: resourceQuota,
		},
		{
			description:   "object does not exist in cache and is not found with client",
			cacheInput:    []*corev1.ResourceQuota{resourceQuota},
			ttl:           30 * time.Second,
			expectedQuota: nil,
		},
		{
			description:   "object does not exist in cache and is found with client",
			cacheInput:    []*corev1.ResourceQuota{},
			clientInput:   []runtime.Object{resourceQuota},
			namespace:     namespace,
			expectedQuota: resourceQuota,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			liveLookupCache := lru.New(1)
			kubeClient := fake.NewSimpleClientset(tc.clientInput...)
			informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

			accessor, _ := newQuotaAccessor()
			accessor.client = kubeClient
			accessor.lister = informerFactory.Core().V1().ResourceQuotas().Lister()
			accessor.liveLookupCache = liveLookupCache

			for _, q := range tc.cacheInput {
				quota := q
				liveLookupCache.Add(quota.Namespace, liveLookupEntry{expiry: time.Now().Add(tc.ttl), items: []*corev1.ResourceQuota{quota}})
			}

			quotas, err := accessor.GetQuotas(tc.namespace)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if tc.expectedQuota != nil {
				if count := len(quotas); count != 1 {
					t.Fatalf("Expected 1 object but got %d", count)
				}

				if !reflect.DeepEqual(quotas[0], *tc.expectedQuota) {
					t.Errorf("Retrieved object does not match")
				}
				return
			}

			if count := len(quotas); count > 0 {
				t.Errorf("Expected 0 objects but got %d", count)
			}
		})
	}

}

// Fixed #22422
func TestGetQuotas(t *testing.T) {
	namespace := "test"
	namespace1 := "test1"
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}

	resourceQuotas := []*corev1.ResourceQuota{resourceQuota}

	liveLookupCache := lru.New(10000)
	kubeClient := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	accessor, _ := newQuotaAccessor()
	accessor.client = kubeClient
	accessor.lister = informerFactory.Core().V1().ResourceQuotas().Lister()
	accessor.liveLookupCache = liveLookupCache

	var (
		testCount  int64
		test1Count int64
	)
	kubeClient.AddReactor("list", "resourcequotas", func(action core.Action) (bool, runtime.Object, error) {
		switch action.GetNamespace() {
		case "test":
			atomic.AddInt64(&testCount, 1)
		case "test1":
			atomic.AddInt64(&test1Count, 1)
		default:
			t.Error("unexpected namespace")
		}

		resourceQuotaList := &corev1.ResourceQuotaList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: fmt.Sprintf("%d", len(resourceQuotas)),
			},
		}
		for index, value := range resourceQuotas {
			value.ResourceVersion = fmt.Sprintf("%d", index)
			value.Namespace = action.GetNamespace()
			resourceQuotaList.Items = append(resourceQuotaList.Items, *value)
		}
		// make the handler slow so concurrent calls exercise the singleflight
		time.Sleep(time.Second)
		return true, resourceQuotaList, nil
	})

	wg := sync.WaitGroup{}
	for i := 0; i < 10; i++ {
		wg.Add(2)
		// simulating concurrent calls after a cache failure
		go func() {
			defer wg.Done()
			ret, err := accessor.GetQuotas(namespace)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, c := range ret {
				if c.Namespace != namespace {
					t.Errorf("Expected %s namespace, got %s", namespace, c.Namespace)
				}
			}
		}()

		// simulation of different namespaces is not a call
		go func() {
			defer wg.Done()
			ret, err := accessor.GetQuotas(namespace1)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, c := range ret {
				if c.Namespace != namespace1 {
					t.Errorf("Expected %s namespace, got %s", namespace1, c.Namespace)
				}
			}
		}()
	}

	// and here we wait for all the goroutines
	wg.Wait()
	// since all the calls with the same namespace will be holded, they must be catched on the singleflight group,
	// There are two different sets of namespace calls
	// hence only 2
	if testCount != 1 {
		t.Errorf("Expected 1 resource quota call, got %d", testCount)
	}
	if test1Count != 1 {
		t.Errorf("Expected 1 resource quota call, got %d", test1Count)
	}

	// invalidate the cache
	accessor.liveLookupCache.Remove(namespace)
	_, err := accessor.GetQuotas(namespace)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if testCount != 2 {
		t.Errorf("Expected 2 resource quota call, got %d", testCount)
	}
	if test1Count != 1 {
		t.Errorf("Expected 1 resource quota call, got %d", test1Count)
	}
}
