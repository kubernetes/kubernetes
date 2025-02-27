/*
Copyright 2025 The Kubernetes Authors.

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

package metrics

import (
	"strings"
	"testing"

	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestStorageDigest(t *testing.T) {
	registry := k8smetrics.NewKubeRegistry()
	registry.CustomMustRegister(storageDigest)

	if err := testutil.GatherAndCompare(registry, strings.NewReader(``), "apiserver_storage_digest"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	RecordStorageDigest("pods", StorageDigest{ResourceVersion: "42", CacheDigest: "123", EtcdDigest: "123"})
	if err := testutil.GatherAndCompare(registry, strings.NewReader(`
# HELP apiserver_storage_digest [ALPHA] Gage exposing digest of storages to validate consistency between. Algorithm to calculate digest might change, so it should not be compared accross apiservers.
# TYPE apiserver_storage_digest gauge
apiserver_storage_digest{digest="123",resource="pods",resource_version="42",storage="cache"} 1
apiserver_storage_digest{digest="123",resource="pods",resource_version="42",storage="etcd"} 1
	`), "apiserver_storage_digest"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	RecordStorageDigest("replicaSets", StorageDigest{ResourceVersion: "43", CacheDigest: "234", EtcdDigest: "234"})
	if err := testutil.GatherAndCompare(registry, strings.NewReader(`
# HELP apiserver_storage_digest [ALPHA] Gage exposing digest of storages to validate consistency between. Algorithm to calculate digest might change, so it should not be compared accross apiservers.
# TYPE apiserver_storage_digest gauge
apiserver_storage_digest{digest="123",resource="pods",resource_version="42",storage="cache"} 1
apiserver_storage_digest{digest="234",resource="replicaSets",resource_version="43",storage="cache"} 1
apiserver_storage_digest{digest="234",resource="replicaSets",resource_version="43",storage="etcd"} 1
apiserver_storage_digest{digest="123",resource="pods",resource_version="42",storage="etcd"} 1
	`), "apiserver_storage_digest"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	RecordStorageDigest("pods", StorageDigest{ResourceVersion: "44", CacheDigest: "123", EtcdDigest: "345"})
	if err := testutil.GatherAndCompare(registry, strings.NewReader(`
# HELP apiserver_storage_digest [ALPHA] Gage exposing digest of storages to validate consistency between. Algorithm to calculate digest might change, so it should not be compared accross apiservers.
# TYPE apiserver_storage_digest gauge
apiserver_storage_digest{digest="123",resource="pods",resource_version="44",storage="cache"} 1
apiserver_storage_digest{digest="234",resource="replicaSets",resource_version="43",storage="cache"} 1
apiserver_storage_digest{digest="234",resource="replicaSets",resource_version="43",storage="etcd"} 1
apiserver_storage_digest{digest="345",resource="pods",resource_version="44",storage="etcd"} 1
	`), "apiserver_storage_digest"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
