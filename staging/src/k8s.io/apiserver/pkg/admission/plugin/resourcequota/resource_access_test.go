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
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	lru "github.com/hashicorp/golang-lru"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
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
			liveLookupCache, err := lru.New(1)
			if err != nil {
				t.Fatal(err)
			}
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
