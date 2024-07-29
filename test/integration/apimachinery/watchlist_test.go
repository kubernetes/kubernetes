/*
Copyright 2023 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestReflectorWatchListFallback(t *testing.T) {
	ctx := context.TODO()

	t.Log("Starting etcd that will be used by two different instances of kube-apiserver")
	etcdURL, etcdTearDownFn, err := framework.RunCustomEtcd("etcd_watchlist", []string{"--experimental-watch-progress-notify-interval", "1s"}, nil)
	require.NoError(t, err)
	defer etcdTearDownFn()
	etcdOptions := framework.DefaultEtcdOptions()
	etcdOptions.StorageConfig.Transport.ServerList = []string{etcdURL}

	t.Log("Starting the first server with the WatchList feature enabled")
	server1 := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--feature-gates=WatchList=true"}, &etcdOptions.StorageConfig)
	defer server1.TearDownFn()
	clientSet, err := kubernetes.NewForConfig(server1.ClientConfig)
	require.NoError(t, err)

	ns := framework.CreateNamespaceOrDie(clientSet, "reflector-fallback-watchlist", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

	t.Logf("Adding 5 secrets to %s namespace", ns.Name)
	for i := 1; i <= 5; i++ {
		_, err := clientSet.CoreV1().Secrets(ns.Name).Create(ctx, newSecret(fmt.Sprintf("secret-%d", i)), metav1.CreateOptions{})
		require.NoError(t, err)
	}

	t.Log("Creating a secret reflector that will use the WatchList feature to synchronise the store")
	store := &wrappedStore{Store: cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)}
	lw := &wrappedListWatch{&cache.ListWatch{}}
	lw.SetClient(ctx, clientSet, ns)
	target := cache.NewReflector(lw, &v1.Secret{}, store, time.Duration(0))
	target.UseWatchList = ptr.To(true)

	t.Log("Waiting until the secret reflector synchronises to the store (call to the Replace method)")
	reflectorCtx, reflectorCtxCancel := context.WithCancel(context.Background())
	defer reflectorCtxCancel()
	store.setCancelOnReplace(reflectorCtxCancel)
	err = target.ListAndWatchWithContext(reflectorCtx)
	require.NoError(t, err)

	t.Log("Verifying if the secret reflector was properly synchronised")
	verifyStore(t, ctx, clientSet, store, ns)

	t.Log("Starting the second server with the WatchList feature disabled")
	server2 := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--feature-gates=WatchList=false"}, &etcdOptions.StorageConfig)
	defer server2.TearDownFn()
	clientSet2, err := kubernetes.NewForConfig(server2.ClientConfig)
	require.NoError(t, err)

	t.Log("Pointing the ListWatcher used by the secret reflector to the second server (with the WatchList feature disabled)")
	lw.SetClient(ctx, clientSet2, ns)
	reflectorCtx, reflectorCtxCancel = context.WithCancel(context.Background())
	defer reflectorCtxCancel()
	store.setCancelOnReplace(reflectorCtxCancel)
	err = target.ListAndWatchWithContext(reflectorCtx)
	require.NoError(t, err)

	t.Log("Verifying if the secret reflector was properly synchronised")
	verifyStore(t, ctx, clientSet, store, ns)
}

// TODO(#115478): refactor with e2e/apimachinery/watchlist
func verifyStore(t *testing.T, ctx context.Context, clientSet kubernetes.Interface, store cache.Store, namespace *v1.Namespace) {
	t.Logf("Listing secrets directly from the server from %s namespace", namespace.Name)
	expectedSecretsList, err := clientSet.CoreV1().Secrets(namespace.Name).List(ctx, metav1.ListOptions{})
	require.NoError(t, err)
	expectedSecrets := expectedSecretsList.Items

	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (done bool, err error) {
		t.Log("Comparing secrets retrieved directly from the server with the ones that have been streamed to the secret reflector")
		rawStreamedSecrets := store.List()
		streamedSecrets := make([]v1.Secret, 0, len(rawStreamedSecrets))
		for _, rawSecret := range rawStreamedSecrets {
			streamedSecrets = append(streamedSecrets, *rawSecret.(*v1.Secret))
		}
		sort.Sort(byName(expectedSecrets))
		sort.Sort(byName(streamedSecrets))
		return cmp.Equal(expectedSecrets, streamedSecrets), nil
	})
	require.NoError(t, err)
}

type byName []v1.Secret

func (a byName) Len() int           { return len(a) }
func (a byName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func newSecret(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

type wrappedStore struct {
	cache.Store
	ctxCancel context.CancelFunc
}

func (s *wrappedStore) Replace(items []interface{}, rv string) error {
	s.ctxCancel()
	return s.Store.Replace(items, rv)
}

func (s *wrappedStore) setCancelOnReplace(ctxCancel context.CancelFunc) {
	s.ctxCancel = ctxCancel
}

type wrappedListWatch struct {
	*cache.ListWatch
}

func (lw *wrappedListWatch) SetClient(ctx context.Context, clientSet kubernetes.Interface, ns *v1.Namespace) {
	lw.ListFunc = func(options metav1.ListOptions) (runtime.Object, error) {
		return clientSet.CoreV1().Secrets(ns.Name).List(ctx, options)
	}
	lw.WatchFunc = func(options metav1.ListOptions) (watch.Interface, error) {
		return clientSet.CoreV1().Secrets(ns.Name).Watch(ctx, options)
	}
}
