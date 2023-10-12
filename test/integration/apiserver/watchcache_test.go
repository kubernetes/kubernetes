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

package apiserver

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// setup create kube-apiserver backed up by two separate etcds,
// with one of them containing events and the other all other objects.
func multiEtcdSetup(ctx context.Context, t *testing.T) (clientset.Interface, framework.TearDownFunc) {
	etcdArgs := []string{"--experimental-watch-progress-notify-interval", "1s"}
	etcd0URL, stopEtcd0, err := framework.RunCustomEtcd("etcd_watchcache0", etcdArgs, nil)
	if err != nil {
		t.Fatalf("Couldn't start etcd: %v", err)
	}

	etcd1URL, stopEtcd1, err := framework.RunCustomEtcd("etcd_watchcache1", etcdArgs, nil)
	if err != nil {
		t.Fatalf("Couldn't start etcd: %v", err)
	}

	etcdOptions := framework.DefaultEtcdOptions()
	// Overwrite etcd setup to our custom etcd instances.
	etcdOptions.StorageConfig.Transport.ServerList = []string{etcd0URL}
	etcdOptions.EtcdServersOverrides = []string{fmt.Sprintf("/events#%s", etcd1URL)}
	etcdOptions.EnableWatchCache = true

	clientSet, _, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Ensure we're using the same etcd across apiserver restarts.
			opts.Etcd = etcdOptions
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			// Switch off endpoints reconciler to avoid unnecessary operations.
			config.Extra.EndpointReconcilerType = reconcilers.NoneEndpointReconcilerType
		},
	})

	closeFn := func() {
		tearDownFn()
		stopEtcd1()
		stopEtcd0()
	}

	// Wait for apiserver to be stabilized.
	// Everything but default service creation is checked in StartTestServer above by
	// waiting for post start hooks, so we just wait for default service to exist.
	// TODO(wojtek-t): Figure out less fragile way.
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientSet.CoreV1().Services("default").Get(ctx, "kubernetes", metav1.GetOptions{})
		return err == nil, nil
	}); err != nil {
		t.Fatalf("Failed to wait for kubernetes service: %v:", err)
	}
	return clientSet, closeFn
}

func TestWatchCacheUpdatedByEtcd(t *testing.T) {
	tCtx := ktesting.Init(t)
	c, closeFn := multiEtcdSetup(tCtx, t)
	defer closeFn()

	makeConfigMap := func(name string) *v1.ConfigMap {
		return &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name}}
	}
	makeSecret := func(name string) *v1.Secret {
		return &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: name}}
	}
	makeEvent := func(name string) *v1.Event {
		return &v1.Event{ObjectMeta: metav1.ObjectMeta{Name: name}}
	}

	cm, err := c.CoreV1().ConfigMaps("default").Create(tCtx, makeConfigMap("name"), metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Couldn't create configmap: %v", err)
	}
	ev, err := c.CoreV1().Events("default").Create(tCtx, makeEvent("name"), metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Couldn't create event: %v", err)
	}

	listOptions := metav1.ListOptions{
		ResourceVersion:      "0",
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
	}

	// Wait until listing from cache returns resource version of corresponding
	// resources (being the last updates).
	t.Logf("Waiting for configmaps watchcache synced to %s", cm.ResourceVersion)
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		res, err := c.CoreV1().ConfigMaps("default").List(tCtx, listOptions)
		if err != nil {
			return false, nil
		}
		return res.ResourceVersion == cm.ResourceVersion, nil
	}); err != nil {
		t.Errorf("Failed to wait for configmaps watchcache synced: %v", err)
	}
	t.Logf("Waiting for events watchcache synced to %s", ev.ResourceVersion)
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		res, err := c.CoreV1().Events("default").List(tCtx, listOptions)
		if err != nil {
			return false, nil
		}
		return res.ResourceVersion == ev.ResourceVersion, nil
	}); err != nil {
		t.Errorf("Failed to wait for events watchcache synced: %v", err)
	}

	// Create a secret, that is stored in the same etcd as configmap, but
	// different than events.
	se, err := c.CoreV1().Secrets("default").Create(tCtx, makeSecret("name"), metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Couldn't create secret: %v", err)
	}

	t.Logf("Waiting for configmaps watchcache synced to %s", se.ResourceVersion)
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		res, err := c.CoreV1().ConfigMaps("default").List(tCtx, listOptions)
		if err != nil {
			return false, nil
		}
		return res.ResourceVersion == se.ResourceVersion, nil
	}); err != nil {
		t.Errorf("Failed to wait for configmaps watchcache synced: %v", err)
	}
	t.Logf("Waiting for events watchcache NOT synced to %s", se.ResourceVersion)
	if err := wait.Poll(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		res, err := c.CoreV1().Events("default").List(tCtx, listOptions)
		if err != nil {
			return false, nil
		}
		return res.ResourceVersion == se.ResourceVersion, nil
	}); err == nil || !wait.Interrupted(err) {
		t.Errorf("Events watchcache unexpected synced: %v", err)
	}
}

func BenchmarkListFromWatchCache(b *testing.B) {
	tCtx := ktesting.Init(b)
	c, _, tearDownFn := framework.StartTestServer(tCtx, b, framework.TestServerSetup{
		ModifyServerConfig: func(config *controlplane.Config) {
			// Switch off endpoints reconciler to avoid unnecessary operations.
			config.Extra.EndpointReconcilerType = reconcilers.NoneEndpointReconcilerType
		},
	})
	defer tearDownFn()

	namespaces, secretsPerNamespace := 100, 1000
	wg := sync.WaitGroup{}

	errCh := make(chan error, namespaces)
	for i := 0; i < namespaces; i++ {
		wg.Add(1)
		index := i
		go func() {
			defer wg.Done()

			ns := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("namespace-%d", index)},
			}
			ns, err := c.CoreV1().Namespaces().Create(tCtx, ns, metav1.CreateOptions{})
			if err != nil {
				errCh <- err
				return
			}

			for j := 0; j < secretsPerNamespace; j++ {
				secret := &v1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("secret-%d", j),
					},
				}
				_, err := c.CoreV1().Secrets(ns.Name).Create(tCtx, secret, metav1.CreateOptions{})
				if err != nil {
					errCh <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		b.Error(err)
	}

	b.ResetTimer()

	opts := metav1.ListOptions{
		ResourceVersion: "0",
	}
	for i := 0; i < b.N; i++ {
		secrets, err := c.CoreV1().Secrets("").List(tCtx, opts)
		if err != nil {
			b.Errorf("failed to list secrets: %v", err)
		}
		b.Logf("Number of secrets: %d", len(secrets.Items))
	}
}
