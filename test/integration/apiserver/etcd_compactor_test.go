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

package apiserver

import (
	"context"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

const compactRevKey = "compact_rev_key"

func TestAPIServerCompactorHandlesMissingCompactRevKey(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	etcdURL, stopEtcd, err := framework.RunCustomEtcd(logger, "etcd_compactor", nil)
	if err != nil {
		t.Fatalf("Couldn't start etcd: %v", err)
	}
	t.Cleanup(stopEtcd)

	etcdOptions := framework.DefaultEtcdOptions()
	etcdOptions.StorageConfig.Transport.ServerList = []string{etcdURL}

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, append(framework.DefaultTestServerFlags(), "--etcd-compaction-interval=100ms"), &etcdOptions.StorageConfig)
	t.Cleanup(server.TearDownFn)

	clientSet, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	kvClient := server.EtcdClient.KV
	waitForCompactRevKeyVersion(ctx, t, kvClient, 2)

	if _, err := kvClient.Delete(ctx, compactRevKey); err != nil {
		t.Fatalf("failed to delete %q: %v", compactRevKey, err)
	}

	// Before the fix, the next compaction cycle panicked in the apiserver
	// goroutine because the fallback Get response had no Kvs entry.
	waitForCompactRevKeyVersion(ctx, t, kvClient, 1)

	if _, err := clientSet.CoreV1().Namespaces().Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "compactor-still-running"}}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("apiserver did not stay healthy after compact_rev_key was recreated: %v", err)
	}
}

func waitForCompactRevKeyVersion(ctx context.Context, t *testing.T, kvClient clientv3.KV, version int64) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(ctx, 20*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		resp, err := kvClient.Get(ctx, compactRevKey)
		if err != nil {
			return false, err
		}
		if len(resp.Kvs) == 0 {
			return false, nil
		}
		return resp.Kvs[0].Version >= version, nil
	}); err != nil {
		t.Fatalf("timed out waiting for %q version >= %d: %v", compactRevKey, version, err)
	}
}
