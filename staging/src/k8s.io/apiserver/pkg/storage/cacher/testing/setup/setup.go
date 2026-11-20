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

package setup

import (
	"context"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/clock"
)

type TearDownFunc func()

type SetupOptions struct {
	ResourcePrefix  string
	KeyFunc         func(runtime.Object) (string, error)
	ReverseKeyFunc  storage.ReverseKeyFunc
	IndexerFuncs    map[string]storage.IndexerFunc
	Indexers        cache.Indexers
	Clock           clock.WithTicker
	Codec           runtime.Codec
	Transformer     value.Transformer
	MaxResyncEvents int
}

type SetupOption func(*SetupOptions)

func WithDefaults(options *SetupOptions) {
	prefix := "/pods/"

	options.ResourcePrefix = prefix
	options.KeyFunc = func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) }
	options.ReverseKeyFunc = storagetesting.PodReverseKeyFunc(prefix, true)
	options.Clock = clock.RealClock{}
	options.Codec = cachertesting.Examplev1ProtoCodec
	options.Transformer = identity.NewEncryptCheckTransformer()
}

func WithClusterScopedKeyFunc(options *SetupOptions) {
	options.ReverseKeyFunc = storagetesting.PodReverseKeyFunc(options.ResourcePrefix, false)
	options.KeyFunc = func(obj runtime.Object) (string, error) {
		return storage.NoNamespaceKeyFunc(options.ResourcePrefix, obj)
	}
}

// mirror indexer configuration from pkg/registry/core/pod/strategy.go
func WithNodeNameAndNamespaceIndex(options *SetupOptions) {
	options.IndexerFuncs = map[string]storage.IndexerFunc{
		"spec.nodeName": func(obj runtime.Object) string {
			pod, ok := obj.(*example.Pod)
			if !ok {
				return ""
			}
			return pod.Spec.NodeName
		},
	}
	options.Indexers = map[string]cache.IndexFunc{
		"f:spec.nodeName": func(obj interface{}) ([]string, error) {
			pod := obj.(*example.Pod)
			return []string{pod.Spec.NodeName}, nil
		},
		"f:metadata.namespace": func(obj interface{}) ([]string, error) {
			pod := obj.(*example.Pod)
			return []string{pod.ObjectMeta.Namespace}, nil
		},
	}
}

func WithCodec(codec runtime.Codec) SetupOption {
	return func(options *SetupOptions) {
		options.Codec = codec
	}
}

func WithTransformer(transformer value.Transformer) SetupOption {
	return func(options *SetupOptions) {
		options.Transformer = transformer
	}
}

func WithClock(c clock.WithTicker) SetupOption {
	return func(options *SetupOptions) {
		options.Clock = c
	}
}

func WithMaxResyncEvents(maxResyncEvents int) SetupOption {
	return func(options *SetupOptions) {
		options.MaxResyncEvents = maxResyncEvents
	}
}

func Setup(t *testing.T, opts ...SetupOption) (context.Context, *cacher.CacheDelegator, TearDownFunc) {
	ctx, _, _, delegator, _, tearDown := SetupWithCacher(t, opts...)
	return ctx, delegator, tearDown
}

func SetupWithEtcdServer(t testing.TB, opts ...SetupOption) (context.Context, *cacher.CacheDelegator, *etcd3testing.EtcdTestServer, TearDownFunc) {
	ctx, _, _, delegator, server, tearDown := SetupWithCacher(t, opts...)
	return ctx, delegator, server, tearDown
}

func SetupWithCacher(t testing.TB, opts ...SetupOption) (context.Context, *cacher.Cacher, storage.Interface, *cacher.CacheDelegator, *etcd3testing.EtcdTestServer, TearDownFunc) {
	setupOpts := SetupOptions{}
	opts = append([]SetupOption{WithDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}

	server, etcdStorage := cachertesting.NewEtcdTestStorageWithOptions(t, etcd3testing.PathPrefix(), setupOpts.Codec, setupOpts.Transformer, setupOpts.ReverseKeyFunc)
	// Inject one list error to make sure we test the relist case.
	listErrors := 1
	if clientfeatures.FeatureGates().Enabled(clientfeatures.WatchListClient) {
		// The WatchListClient feature changes the reflector to use WATCH
		// instead of LIST, therefore we don't expect any errors
		listErrors = 0
	}
	wrappedStorage := &storagetesting.StorageInjectingListErrors{
		Interface: etcdStorage,
		Errors:    listErrors,
	}

	config := cacher.Config{
		Storage:             wrappedStorage,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       schema.GroupResource{Resource: "pods"},
		EventsHistoryWindow: cacher.DefaultEventFreshDuration,
		ResourcePrefix:      setupOpts.ResourcePrefix,
		KeyFunc:             setupOpts.KeyFunc,
		GetAttrsFunc:        cachertesting.GetPodAttrs,
		NewFunc:             cachertesting.NewPod,
		NewListFunc:         cachertesting.NewPodList,
		IndexerFuncs:        setupOpts.IndexerFuncs,
		Indexers:            &setupOpts.Indexers,
		Codec:               setupOpts.Codec,
		Clock:               setupOpts.Clock,
		CompactorPollPeriod: 5 * time.Millisecond,
		MaxResyncEvents:     setupOpts.MaxResyncEvents,
	}
	rawCacher, err := cacher.NewCacherFromConfig(config)
	if err != nil {
		t.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()

	// Since some tests depend on the fact that GetList shouldn't fail,
	// we wait until the error from the underlying storage is consumed.
	if err := wait.PollInfinite(5*time.Millisecond, wrappedStorage.ErrorsConsumed); err != nil {
		t.Fatalf("Failed to inject list errors: %v", err)
	}

	// The tests assume that Get/GetList/Watch calls shouldn't fail.
	// However, 429 error can now be returned if watchcache is under initialization.
	// To avoid rewriting all tests, we wait for watchcache to initialize.
	if err := rawCacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	delegator := cacher.NewCacheDelegator(rawCacher, etcdStorage)
	terminate := func() {
		delegator.Stop()
		rawCacher.Stop()
		server.Terminate(t)
	}

	return ctx, rawCacher, etcdStorage, delegator, server, terminate
}

func SetupWithEtcdAndCreateWrapper(t *testing.T, opts ...SetupOption) (storage.Interface, TearDownFunc) {
	_, delegator, _, tearDown := SetupWithEtcdServer(t, opts...)

	return &CreateWrapper{CacheDelegator: delegator}, tearDown
}

type CreateWrapper struct {
	*cacher.CacheDelegator
}

func (c *CreateWrapper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if err := c.CacheDelegator.Create(ctx, key, obj, out, ttl); err != nil {
		return err
	}
	return wait.PollUntilContextTimeout(ctx, 5*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		currentObj := cachertesting.NewPod()
		err := c.CacheDelegator.Get(ctx, key, storage.GetOptions{ResourceVersion: "0"}, currentObj)
		if err != nil {
			if storage.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		if !apiequality.Semantic.DeepEqual(currentObj, out) {
			return false, nil
		}
		return true, nil
	})
}

func CompactStore(c storage.Interface, client *clientv3.Client) storagetesting.Compaction {
	var currentVersion int64
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}
		var compactRev int64
		currentVersion, _, compactRev, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		if err == nil && compactRev != int64(rv) {
			currentVersion, _, compactRev, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		}
		if err != nil {
			t.Fatal(err)
		}
		if compactRev != int64(rv) {
			t.Fatalf("expected compact revision %d, got %d", rv, compactRev)
		}
		// Wait for compaction to be observed when snapshot compaction is enabled.
		if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
			for c.CompactRevision() < int64(rv) {
				select {
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				case <-time.After(5 * time.Millisecond):
				}
			}
		}
	}
}
