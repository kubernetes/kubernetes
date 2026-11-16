/*
Copyright 2015 The Kubernetes Authors.

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

package delegator_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	"k8s.io/apiserver/pkg/storage/cacher/testing/setup"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func checkStorageInvariants(ctx context.Context, t *testing.T, key string) {
	// No-op function since cacher simply passes object creation to the underlying storage.
}

func TestCreate(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreate(ctx, t, cacher, checkStorageInvariants)
}

func TestCreateWithTTL(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithTTL(ctx, t, cacher)
}

func TestCreateWithKeyExist(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithKeyExist(ctx, t, cacher)
}

func TestGet(t *testing.T) {
	for _, consistentGetFromCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("ConsistentGetFromCache=%v", consistentGetFromCache), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, consistentGetFromCache)
			cachertesting.ForceRequestWatchProgressSupport(t)
			ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
			t.Cleanup(terminate)
			storagetesting.RunTestGet(ctx, t, cacher, cachertesting.IncreaseRVFunc(server.V3Client.Client))
		})
	}
}

func TestDeleteOnce(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteOnce(ctx, t, cacher, etcd3testing.PathPrefix(), cachertesting.Examplev1ProtoCodec)
}

func TestUpdateOnce(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestUpdateOnce(ctx, t, cacher, etcd3testing.PathPrefix(), cachertesting.Examplev1ProtoCodec)
}

type testTransformer struct {
	value.Transformer
	fail atomic.Bool
}

func (tt *testTransformer) setFailing(c bool) {
	tt.fail.Store(c)
}

func (tt *testTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, stale bool, err error) {
	if tt.fail.Load() {
		return nil, false, errors.New("synthetic error")
	}
	return tt.Transformer.TransformFromStorage(ctx, data, dataCtx)
}

type testCodec struct {
	runtime.Codec
	fail atomic.Bool
}

func (tc *testCodec) setFailing(c bool) {
	tc.fail.Store(c)
}

func (tc *testCodec) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	if tc.fail.Load() {
		return nil, nil, errors.New("synthetic error")
	}
	return tc.Codec.Decode(data, defaults, into)
}

func TestDelete(t *testing.T) {
	for _, consistentGetFromCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("ConsistentGetFromCache=%v", consistentGetFromCache), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, consistentGetFromCache)
			cachertesting.ForceRequestWatchProgressSupport(t)

			t.Run("UnconditionalDelete", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestUnconditionalDelete(ctx, t, cacher)
			})
			t.Run("ConditionalDelete", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestConditionalDelete(ctx, t, cacher)
			})
			t.Run("DeleteWithSuggestion", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithSuggestion(ctx, t, cacher)
			})
			t.Run("DeleteWithSuggestionAndConflict", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithSuggestionAndConflict(ctx, t, cacher)
			})
			t.Run("DeleteWithSuggestionOfDeletedObject", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(ctx, t, cacher)
			})
			t.Run("ValidateDeletionWithSuggestion", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestValidateDeletionWithSuggestion(ctx, t, cacher)
			})
			t.Run("ValidateDeletionWithOnlySuggestionValid", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestValidateDeletionWithOnlySuggestionValid(ctx, t, cacher)
			})
			t.Run("DeleteWithConflict", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithConflict(ctx, t, cacher)
			})
			t.Run("DeleteWithConflictAndMissingExpectedTransformOrDecodeError", func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
				transformer := &testTransformer{Transformer: identity.NewEncryptCheckTransformer()}
				ctx, s, terminate := setup.Setup(t, setup.WithTransformer(transformer))
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithConflictAndMissingExpectedTransformOrDecodeError(ctx, t, s, transformer.setFailing)
			})
			t.Run("DeleteWithConflictAndExpectedTransformError", func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
				transformer := &testTransformer{Transformer: identity.NewEncryptCheckTransformer()}
				ctx, cacher, terminate := setup.Setup(t, setup.WithTransformer(transformer))
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteExpectedTransformOrDecodeError(ctx, t, cacher, transformer.setFailing)
			})
			t.Run("DeleteWithConflictAndExpectedDecodeError", func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
				// Decode errors cause a panic when they occur during tests. Disable for this particular
				// test since it is deliberately exercising a case where the stored bytes cannot be decoded.
				etcd3.TestOnlySetFatalOnDecodeError(t, false)
				codec := &testCodec{Codec: apitesting.TestCodec(cachertesting.Codecs, examplev1.SchemeGroupVersion)}
				ctx, cacher, terminate := setup.Setup(t, setup.WithCodec(codec))
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteExpectedTransformOrDecodeError(ctx, t, cacher, codec.setFailing)
			})
			t.Run("DeleteWithSuggestionAndMissingExpectedTransformOrDecodeError", func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteWithSuggestionAndMissingExpectedTransformOrDecodeError(ctx, t, cacher)
			})
			t.Run("PreconditionalDeleteWithSuggestion", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestPreconditionalDeleteWithSuggestion(ctx, t, cacher)
			})
			t.Run("PreconditionalDeleteWithSuggestionPass", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestPreconditionalDeleteWithOnlySuggestionPass(ctx, t, cacher)
			})
		})
	}
}

func TestListPaging(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaging(ctx, t, cacher)
}

func TestLists(t *testing.T) {
	for _, listFromCacheSnapshot := range []bool{true, false} {
		t.Run(fmt.Sprintf("ListFromCacheSnapshot=%v", listFromCacheSnapshot), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, listFromCacheSnapshot)
			t.Run("List", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t, setup.WithNodeNameAndNamespaceIndex)
				t.Cleanup(terminate)
				storagetesting.RunTestList(ctx, t, cacher, setup.CompactStore(cacher, server.V3Client.Client), true, server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder))
			})

			t.Run("ConsistentList", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
				t.Cleanup(terminate)
				storagetesting.RunTestConsistentList(ctx, t, cacher, cachertesting.IncreaseRVFunc(server.V3Client.Client), true, true, listFromCacheSnapshot)
			})

			t.Run("GetListNonRecursive", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
				t.Cleanup(terminate)
				storagetesting.RunTestGetListNonRecursive(ctx, t, cachertesting.IncreaseRVFunc(server.V3Client.Client), cacher)
			})
		})
	}
}

func TestCompactRevision(t *testing.T) {
	// Test requires store to observe extenal changes to compaction revision, requiring dedicated watch on compact key which is enabled by ListFromCacheSnapshot.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)
	cachertesting.ForceRequestWatchProgressSupport(t)
	ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCompactRevision(ctx, t, cacher, cachertesting.IncreaseRVFunc(server.V3Client.Client), setup.CompactStore(cacher, server.V3Client.Client))
}

func TestMarkConsistent(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)
	ctx, rawCacher, _, cacher, server, terminate := setup.SetupWithCacher(t)
	t.Cleanup(terminate)
	recorder := server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder)

	t.Log("New cache collects snapshots, list skips etcd")
	resourceVersion1 := createObject(t, ctx, cacher)
	etcdRequests := etcdListRequests(t, ctx, cacher, recorder, storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersion1,
		Recursive:            true,
	})
	if len(etcdRequests) != 0 {
		t.Errorf("Expected no requests to etcd, got: %+v", etcdRequests)
	}

	t.Log("Consistent cache marked consistent again keeps snapshots, list skips etcd")
	rawCacher.MarkConsistent(true)
	etcdRequests = etcdListRequests(t, ctx, cacher, recorder, storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersion1,
		Recursive:            true,
	})
	if len(etcdRequests) != 0 {
		t.Errorf("Expected no requests to etcd, got: %+v", etcdRequests)
	}

	t.Log("Inconsistent cache clears old snapshots, list hits etcd")
	rawCacher.MarkConsistent(false)
	etcdRequests = etcdListRequests(t, ctx, cacher, recorder, storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersion1,
		Recursive:            true,
	})
	if len(etcdRequests) != 1 {
		t.Errorf("Expected request to etcd, got: %+v", etcdRequests)
	}

	t.Log("Inconsistent cache doesn't collect new snapshot, list hits etcd")
	resourceVersion2 := createObject(t, ctx, cacher)
	etcdRequests = etcdListRequests(t, ctx, cacher, recorder, storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersion2,
		Recursive:            true,
	})
	if len(etcdRequests) != 1 {
		t.Errorf("Expected request to etcd, got: %+v", etcdRequests)
	}

	t.Log("Marking cache consistent allows it to collect new snapshots, list skips etcd")
	rawCacher.MarkConsistent(true)
	resourceVersion3 := createObject(t, ctx, cacher)
	etcdRequests = etcdListRequests(t, ctx, cacher, recorder, storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersion3,
		Recursive:            true,
	})
	if len(etcdRequests) != 0 {
		t.Errorf("Expected no requests to etcd, got: %+v", etcdRequests)
	}
}

func createObject(t *testing.T, ctx context.Context, store storage.Interface) string {
	var out example.Pod
	pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: rand.String(10)}}
	err := store.Create(ctx, cachertesting.ComputePodKey(pod), pod, &out, 0)
	if err != nil {
		t.Fatal(err)
	}
	return out.ResourceVersion
}

func etcdListRequests(t *testing.T, ctx context.Context, store storage.Interface, recorder *storagetesting.KubernetesRecorder, opts storage.ListOptions) []storagetesting.RecordedList {
	key := rand.String(10)
	listCtx := context.WithValue(ctx, storagetesting.RecorderContextKey, key)
	listOut := &example.PodList{}
	if err := store.GetList(listCtx, "/pods/", opts, listOut); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	return recorder.ListRequestForKey(key)
}

func TestCompactPastLastWrite(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)
	cachertesting.ForceRequestWatchProgressSupport(t)
	ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
	t.Cleanup(terminate)
	recorder := server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder)

	createObject(t, ctx, cacher)
	resourceVersionPastLastWrite := fmt.Sprint(cachertesting.IncreaseRVFunc(server.V3Client.Client)(ctx, t))
	list := storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersionPastLastWrite,
		Recursive:            true,
	}

	t.Log("List before compaction")
	require.Empty(t, etcdListRequests(t, ctx, cacher, recorder, list))

	t.Log("Same list after compaction to the listed resourceVersion")
	setup.CompactStore(cacher, server.V3Client.Client)(ctx, t, resourceVersionPastLastWrite)
	require.Empty(t, etcdListRequests(t, ctx, cacher, recorder, list))
}

func TestGetListRecursivePrefix(t *testing.T) {
	t.Parallel()
	ctx, store, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGetListRecursivePrefix(ctx, t, store)
}

func checkStorageCalls(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
	// No-op function for now, since cacher passes pagination calls to underlying storage.
}

func TestListContinuation(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuation(ctx, t, cacher, checkStorageCalls)
}

func TestListPaginationRareObject(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaginationRareObject(ctx, t, cacher, checkStorageCalls)
}

func TestListContinuationWithFilter(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuationWithFilter(ctx, t, cacher, checkStorageCalls)
}

func TestListInconsistentContinuation(t *testing.T) {
	// TODO(#109831): Enable use of this by setting compaction.
}

func TestListResourceVersionMatch(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestNamespaceScopedList(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t, setup.WithNodeNameAndNamespaceIndex)
	t.Cleanup(terminate)
	storagetesting.RunTestNamespaceScopedList(ctx, t, cacher)
}

type swappablePrefixTransformer struct {
	mu    sync.RWMutex
	inner value.Transformer
}

func newSwappablePrefixTransformer() *swappablePrefixTransformer {
	return &swappablePrefixTransformer{
		inner: storagetesting.NewPrefixTransformer([]byte("test!"), false),
	}
}

func (t *swappablePrefixTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inner.TransformFromStorage(ctx, data, dataCtx)
}

func (t *swappablePrefixTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inner.TransformToStorage(ctx, data, dataCtx)
}

type cacherWithPrefixTransformer struct {
	*cacher.CacheDelegator
	storage     storage.Interface
	transformer *swappablePrefixTransformer
}

func (c *cacherWithPrefixTransformer) UnderlyingStorage() storage.Interface {
	return c.storage
}

func (c *cacherWithPrefixTransformer) UpdatePrefixTransformer(modifier storagetesting.PrefixTransformerModifier) func() {
	c.transformer.mu.Lock()
	orig := c.transformer.inner
	copyTransformer := *orig.(*storagetesting.PrefixTransformer)
	c.transformer.inner = modifier(&copyTransformer)
	c.transformer.mu.Unlock()
	return func() {
		c.transformer.mu.Lock()
		c.transformer.inner = orig
		c.transformer.mu.Unlock()
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	for _, consistentGetFromCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("ConsistentGetFromCache=%v", consistentGetFromCache), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, consistentGetFromCache)
			cachertesting.ForceRequestWatchProgressSupport(t)

			t.Run("GuaranteedUpdate", func(t *testing.T) {
				t.Parallel()
				transformer := newSwappablePrefixTransformer()
				ctx, _, etcdStorage, cacher, _, terminate := setup.SetupWithCacher(t, setup.WithTransformer(transformer))
				t.Cleanup(terminate)
				storagetesting.RunTestGuaranteedUpdate(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: cacher, storage: etcdStorage, transformer: transformer}, checkStorageInvariants)
			})
			t.Run("GuaranteedUpdateWithTTL", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, cacher)
			})
			t.Run("GuaranteedUpdateChecksStoredData", func(t *testing.T) {
				t.Parallel()
				transformer := newSwappablePrefixTransformer()
				// Leading-whitespace normalization in RunTestGuaranteedUpdateChecksStoredData requires a JSON codec.
				ctx, _, etcdStorage, cacher, _, terminate := setup.SetupWithCacher(t, setup.WithCodec(apitesting.TestCodec(cachertesting.Codecs, examplev1.SchemeGroupVersion)), setup.WithTransformer(transformer))
				t.Cleanup(terminate)
				storagetesting.RunTestGuaranteedUpdateChecksStoredData(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: cacher, storage: etcdStorage, transformer: transformer})
			})
			t.Run("GuaranteedUpdateWithConflict", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestGuaranteedUpdateWithConflict(ctx, t, cacher)
			})
			t.Run("GuaranteedUpdateWithSuggestionAndConflict", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := setup.Setup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx, t, cacher)
			})
		})
	}
}

func TestTransformationFailure(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestStats(t *testing.T) {
	for _, sizeBasedListCostEstimate := range []bool{true, false} {
		t.Run(fmt.Sprintf("SizeBasedListCostEstimate=%v", sizeBasedListCostEstimate), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SizeBasedListCostEstimate, sizeBasedListCostEstimate)
			ctx, cacher, terminate := setup.Setup(t)
			t.Cleanup(terminate)
			storagetesting.RunTestStats(ctx, t, cacher, cachertesting.Examplev1ProtoCodec, identity.NewEncryptCheckTransformer(), sizeBasedListCostEstimate)
		})
	}
}
func TestKeySchema(t *testing.T) {
	t.Parallel()
	ctx, cacher, terminate := setup.Setup(t, func(options *setup.SetupOptions) {
		options.KeyFunc = func(obj runtime.Object) (string, error) {
			accessor, err := meta.Accessor(obj)
			if err != nil {
				return "", err
			}
			if accessor.GetNamespace() != "" {
				return options.ResourcePrefix + accessor.GetNamespace() + "/" + accessor.GetName(), nil
			}
			return options.ResourcePrefix + accessor.GetName(), nil
		}
	})
	t.Cleanup(terminate)
	storagetesting.RunTestKeySchema(ctx, t, cacher)
}

func TestOptimisticConcurrency(t *testing.T) {
	for _, consistentGetFromCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("ConsistentGetFromCache=%v", consistentGetFromCache), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, consistentGetFromCache)
			cachertesting.ForceRequestWatchProgressSupport(t)
			transformer := newSwappablePrefixTransformer()
			ctx, _, etcdStorage, delegator, server, terminate := setup.SetupWithCacher(t, setup.WithTransformer(transformer))
			t.Cleanup(terminate)
			recorder := server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder)
			storagetesting.RunTestOptimisticConcurrency(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: delegator, storage: etcdStorage, transformer: transformer}, true, recorder)
		})
	}
}

func TestCorrectness(t *testing.T) {
	for _, consistentGetFromCache := range []bool{true, false} {
		t.Run(fmt.Sprintf("ConsistentGetFromCache=%v", consistentGetFromCache), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, consistentGetFromCache)
			for _, watchCacheStallResume := range []bool{true, false} {
				t.Run(fmt.Sprintf("WatchCacheStallResume=%v", watchCacheStallResume), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, watchCacheStallResume)
					cachertesting.ForceRequestWatchProgressSupport(t)
					ctx, cacher, server, terminate := setup.SetupWithEtcdServer(t)
					t.Cleanup(terminate)
					keyFunc := func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc("/pods/", obj) }
					correctness.RunTestCorrectness(ctx, t, cacher, etcd3testing.PathPrefix(), keyFunc, setup.CompactStore(cacher, server.V3Client.Client))
				})
			}
		})
	}
}
