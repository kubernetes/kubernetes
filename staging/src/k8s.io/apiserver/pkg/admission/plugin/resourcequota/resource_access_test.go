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

	lru "github.com/hashicorp/golang-lru"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

func TestLRUCacheLookup(t *testing.T) {
	liveLookupCache, err := lru.New(100)
	if err != nil {
		t.Fatal(err)
	}
	kubeClient := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	accessor, _ := newQuotaAccessor()
	accessor.client = kubeClient
	accessor.lister = informerFactory.Core().V1().ResourceQuotas().Lister()
	accessor.liveLookupCache = liveLookupCache

	namespace := "foo"
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
	}

	liveLookupCache.Add(resourceQuota.Namespace, liveLookupEntry{expiry: time.Now().Add(30 * time.Second), items: []*corev1.ResourceQuota{
		resourceQuota,
	}})

	quotas, err := accessor.GetQuotas(resourceQuota.Namespace)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if count := len(quotas); count != 1 {
		t.Errorf("Expected 1 object but got %d", count)
	}

	if !reflect.DeepEqual(quotas[0], *resourceQuota) {
		t.Errorf("Retrieved object does not match")
	}
}
