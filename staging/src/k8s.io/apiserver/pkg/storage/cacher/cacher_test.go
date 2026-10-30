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

package cacher

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

// GetPodAttrs returns labels and fields of a given object for filtering purposes.
func GetPodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*example.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), nil
}

// PodToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *example.Pod) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	podSpecificFieldsSet := make(fields.Set, 5)
	podSpecificFieldsSet["spec.nodeName"] = pod.Spec.NodeName
	podSpecificFieldsSet["spec.restartPolicy"] = string(pod.Spec.RestartPolicy)
	podSpecificFieldsSet["status.phase"] = string(pod.Status.Phase)
	return AddObjectMetaFieldsSet(podSpecificFieldsSet, &pod.ObjectMeta, true)
}

func AddObjectMetaFieldsSet(source fields.Set, objectMeta *metav1.ObjectMeta, hasNamespaceField bool) fields.Set {
	source["metadata.name"] = objectMeta.Name
	if hasNamespaceField {
		source["metadata.namespace"] = objectMeta.Namespace
	}
	return source
}

func checkStorageInvariants(ctx context.Context, t *testing.T, key string) {
	// No-op function since cacher simply passes object creation to the underlying storage.
}

func TestCreate(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreate(ctx, t, cacher, checkStorageInvariants)
}

func TestCreateWithTTL(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithTTL(ctx, t, cacher)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithKeyExist(ctx, t, cacher)
}

func TestGet(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGet(ctx, t, cacher)
}

func TestDeleteOnce(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteOnce(ctx, t, cacher, examplev1ProtoCodec)
}

func TestUpdateOnce(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestUpdateOnce(ctx, t, cacher, etcd3testing.PathPrefix(), examplev1ProtoCodec)
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
	t.Run("UnconditionalDelete", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestUnconditionalDelete(ctx, t, cacher)
	})
	t.Run("ConditionalDelete", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestConditionalDelete(ctx, t, cacher)
	})
	t.Run("DeleteWithSuggestion", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithSuggestion(ctx, t, cacher)
	})
	t.Run("DeleteWithSuggestionAndConflict", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithSuggestionAndConflict(ctx, t, cacher)
	})
	t.Run("DeleteWithSuggestionOfDeletedObject", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(ctx, t, cacher)
	})
	t.Run("ValidateDeletionWithSuggestion", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestValidateDeletionWithSuggestion(ctx, t, cacher)
	})
	t.Run("ValidateDeletionWithOnlySuggestionValid", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestValidateDeletionWithOnlySuggestionValid(ctx, t, cacher)
	})
	t.Run("DeleteWithConflict", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithConflict(ctx, t, cacher)
	})
	t.Run("DeleteWithConflictAndMissingExpectedTransformOrDecodeError", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
		transformer := &testTransformer{Transformer: identity.NewEncryptCheckTransformer()}
		ctx, s, terminate := testSetup(t, withTransformer(transformer))
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithConflictAndMissingExpectedTransformOrDecodeError(ctx, t, s, transformer.setFailing)
	})
	t.Run("DeleteWithConflictAndExpectedTransformError", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
		transformer := &testTransformer{Transformer: identity.NewEncryptCheckTransformer()}
		ctx, cacher, terminate := testSetup(t, withTransformer(transformer))
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteExpectedTransformOrDecodeError(ctx, t, cacher, transformer.setFailing)
	})
	t.Run("DeleteWithConflictAndExpectedDecodeError", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
		// Decode errors cause a panic when they occur during tests. Disable for this particular
		// test since it is deliberately exercising a case where the stored bytes cannot be decoded.
		etcd3.TestOnlySetFatalOnDecodeError(t, false)
		codec := &testCodec{Codec: apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)}
		ctx, cacher, terminate := testSetup(t, withCodec(codec))
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteExpectedTransformOrDecodeError(ctx, t, cacher, codec.setFailing)
	})
	t.Run("DeleteWithSuggestionAndMissingExpectedTransformOrDecodeError", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteWithSuggestionAndMissingExpectedTransformOrDecodeError(ctx, t, cacher)
	})
	t.Run("PreconditionalDeleteWithSuggestion", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestPreconditionalDeleteWithSuggestion(ctx, t, cacher)
	})
	t.Run("PreconditionalDeleteWithSuggestionPass", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestPreconditionalDeleteWithOnlySuggestionPass(ctx, t, cacher)
	})
}

func TestListPaging(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaging(ctx, t, cacher)
}

func TestLists(t *testing.T) {
	for _, listFromCacheSnapshot := range []bool{true, false} {
		t.Run(fmt.Sprintf("ListFromCacheSnapshot=%v", listFromCacheSnapshot), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, listFromCacheSnapshot)
			t.Run("List", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := testSetupWithEtcdServer(t, withNodeNameAndNamespaceIndex)
				t.Cleanup(terminate)
				storagetesting.RunTestList(ctx, t, cacher, compactStore(cacher, server.V3Client.Client), true, server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder))
			})

			t.Run("ConsistentList", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
				t.Cleanup(terminate)
				storagetesting.RunTestConsistentList(ctx, t, cacher, increaseRVFunc(server.V3Client.Client), true, true, listFromCacheSnapshot)
			})

			t.Run("GetListNonRecursive", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
				t.Cleanup(terminate)
				storagetesting.RunTestGetListNonRecursive(ctx, t, increaseRVFunc(server.V3Client.Client), cacher)
			})
		})
	}
}

func TestCompactRevision(t *testing.T) {
	// Test requires store to observe extenal changes to compaction revision, requiring dedicated watch on compact key which is enabled by ListFromCacheSnapshot.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)
	forceRequestWatchProgressSupport(t)
	ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCompactRevision(ctx, t, cacher, increaseRVFunc(server.V3Client.Client), compactStore(cacher, server.V3Client.Client))
}

func TestMarkConsistent(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)
	ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
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
	cacher.cacher.MarkConsistent(true)
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
	cacher.cacher.MarkConsistent(false)
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
	cacher.cacher.MarkConsistent(true)
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
	err := store.Create(ctx, computePodKey(pod), pod, &out, 0)
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
	forceRequestWatchProgressSupport(t)
	ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
	t.Cleanup(terminate)
	recorder := server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder)

	createObject(t, ctx, cacher)
	resourceVersionPastLastWrite := fmt.Sprint(increaseRVFunc(server.V3Client.Client)(ctx, t))
	list := storage.ListOptions{
		Predicate:            storage.Everything,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		ResourceVersion:      resourceVersionPastLastWrite,
		Recursive:            true,
	}

	t.Log("List before compaction")
	require.Empty(t, etcdListRequests(t, ctx, cacher, recorder, list))

	t.Log("Same list after compaction to the listed resourceVersion")
	compactStore(cacher, server.V3Client.Client)(ctx, t, resourceVersionPastLastWrite)
	require.Empty(t, etcdListRequests(t, ctx, cacher, recorder, list))
}

func TestGetListRecursivePrefix(t *testing.T) {
	ctx, store, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGetListRecursivePrefix(ctx, t, store)
}

func checkStorageCalls(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
	// No-op function for now, since cacher passes pagination calls to underlying storage.
}

func TestListContinuation(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuation(ctx, t, cacher, checkStorageCalls)
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaginationRareObject(ctx, t, cacher, checkStorageCalls)
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
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
	ctx, cacher, terminate := testSetup(t, withNodeNameAndNamespaceIndex)
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
	*CacheDelegator
	transformer *swappablePrefixTransformer
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
	t.Run("GuaranteedUpdate", func(t *testing.T) {
		transformer := newSwappablePrefixTransformer()
		ctx, cacher, terminate := testSetup(t, withTransformer(transformer))
		t.Cleanup(terminate)
		storagetesting.RunTestGuaranteedUpdate(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: cacher, transformer: transformer}, checkStorageInvariants)
	})
	t.Run("GuaranteedUpdateWithTTL", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, cacher)
	})
	t.Run("GuaranteedUpdateChecksStoredData", func(t *testing.T) {
		transformer := newSwappablePrefixTransformer()
		// Leading-whitespace normalization in RunTestGuaranteedUpdateChecksStoredData requires a JSON codec.
		ctx, cacher, terminate := testSetup(t, withCodec(apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)), withTransformer(transformer))
		t.Cleanup(terminate)
		storagetesting.RunTestGuaranteedUpdateChecksStoredData(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: cacher, transformer: transformer})
	})
	t.Run("GuaranteedUpdateWithConflict", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestGuaranteedUpdateWithConflict(ctx, t, cacher)
	})
	t.Run("GuaranteedUpdateWithSuggestionAndConflict", func(t *testing.T) {
		ctx, cacher, terminate := testSetup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx, t, cacher)
	})
}

func TestTransformationFailure(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestStats(t *testing.T) {
	for _, sizeBasedListCostEstimate := range []bool{true, false} {
		t.Run(fmt.Sprintf("SizeBasedListCostEstimate=%v", sizeBasedListCostEstimate), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SizeBasedListCostEstimate, sizeBasedListCostEstimate)
			ctx, cacher, terminate := testSetup(t)
			t.Cleanup(terminate)
			storagetesting.RunTestStats(ctx, t, cacher, examplev1ProtoCodec, identity.NewEncryptCheckTransformer(), sizeBasedListCostEstimate)
		})
	}
}
func TestKeySchema(t *testing.T) {
	ctx, cacher, terminate := testSetup(t, func(options *setupOptions) {
		options.keyFunc = func(obj runtime.Object) (string, error) {
			accessor, err := meta.Accessor(obj)
			if err != nil {
				return "", err
			}
			if accessor.GetNamespace() != "" {
				return options.resourcePrefix + accessor.GetNamespace() + "/" + accessor.GetName(), nil
			}
			return options.resourcePrefix + accessor.GetName(), nil
		}
	})
	t.Cleanup(terminate)
	storagetesting.RunTestKeySchema(ctx, t, cacher)
}

func TestWatch(t *testing.T) {
	for _, watchWithoutPrevKV := range []bool{true, false} {
		t.Run(fmt.Sprintf("WatchWithoutPrevKV=%v", watchWithoutPrevKV), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchFromStorageWithoutPrevKV, watchWithoutPrevKV)
			t.Run("Watch", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatch(ctx, t, cacher)
			})
			t.Run("WatchFromZero", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatchFromZero(ctx, t, cacher, compactWatch(cacher, server.V3Client.Client))
			})
			t.Run("DeleteTriggerWatch", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDeleteTriggerWatch(ctx, t, cacher)
			})
			t.Run("WatchFromNonZero", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatchFromNonZero(ctx, t, cacher)
			})
			t.Run("DelayedWatchDelivery", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestDelayedWatchDelivery(ctx, t, cacher)
			})
			t.Run("WatchError", func(t *testing.T) {
				// TODO(#109831): Enable use of this test and run it.
			})
			t.Run("WatchContextCancel", func(t *testing.T) {
				// TODO(#109831): Enable use of this test and run it.
			})
			t.Run("WatcherTimeout", func(t *testing.T) {
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatcherTimeout(ctx, t, cacher)
			})
			t.Run("WatchDeleteEventObjectHaveLatestRV", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, cacher, true, false, nil)
			})
			t.Run("WatchInitializationSignal", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatchInitializationSignal(ctx, t, cacher)
			})
			t.Run("ClusterScopedWatch", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t, withClusterScopedKeyFunc, withNodeNameAndNamespaceIndex)
				t.Cleanup(terminate)
				storagetesting.RunTestClusterScopedWatch(ctx, t, cacher)
			})
			t.Run("NamespaceScopedWatch", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t, withNodeNameAndNamespaceIndex)
				t.Cleanup(terminate)
				storagetesting.RunTestNamespaceScopedWatch(ctx, t, cacher)
			})
			t.Run("WatchDispatchBookmarkEvents", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestWatchDispatchBookmarkEvents(ctx, t, cacher, true)
			})
			t.Run("WatchBookmarksWithCorrectResourceVersion", func(t *testing.T) {
				t.Parallel()
				ctx, cacher, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunTestOptionalWatchBookmarksWithCorrectResourceVersion(ctx, t, cacher)
			})
			t.Run("SendInitialEventsBackwardCompatibility", func(t *testing.T) {
				t.Parallel()
				ctx, store, terminate := testSetup(t)
				t.Cleanup(terminate)
				storagetesting.RunSendInitialEventsBackwardCompatibility(ctx, t, store)
			})
			t.Run("WatchResync", func(t *testing.T) {
				t.Parallel()
				fakeClock := testingclock.NewFakeClock(time.Now())
				ctx, cacher, server, terminate := testSetupWithEtcdServer(t, withClock(fakeClock))
				t.Cleanup(terminate)
				store := &createWrapper{CacheDelegator: cacher}
				resync := func() {
					time.Sleep(50 * time.Millisecond)
					fakeClock.Step(100 * time.Millisecond)
					time.Sleep(50 * time.Millisecond)
					fakeClock.Step(2 * time.Second)
					time.Sleep(50 * time.Millisecond)
				}
				triggerBookmarks := func() {
					time.Sleep(50 * time.Millisecond)
					fakeClock.Step(defaultBookmarkFrequency + time.Second)
					time.Sleep(50 * time.Millisecond)
				}
				storagetesting.RunTestWatchResync(ctx, t, store, true, compactWatch(cacher, server.V3Client.Client), resync, triggerBookmarks)
			})
			// These tests override WatchList and must finish before the parallel subtests start.
			t.Run("WatchSemantics", func(t *testing.T) {
				store, terminate := testSetupWithEtcdAndCreateWrapper(t)
				t.Cleanup(terminate)
				storagetesting.RunWatchSemantics(context.TODO(), t, store)
			})
			t.Run("WatchSemanticInitialEventsExtended", func(t *testing.T) {
				store, terminate := testSetupWithEtcdAndCreateWrapper(t)
				t.Cleanup(terminate)
				storagetesting.RunWatchSemanticInitialEventsExtended(context.TODO(), t, store)
			})
			t.Run("WatchListMatchSingle", func(t *testing.T) {
				store, terminate := testSetupWithEtcdAndCreateWrapper(t)
				t.Cleanup(terminate)
				storagetesting.RunWatchListMatchSingle(context.TODO(), t, store)
			})
		})
	}
}

func TestOptimisticConcurrency(t *testing.T) {
	transformer := newSwappablePrefixTransformer()
	ctx, delegator, server, terminate := testSetupWithEtcdServer(t, withTransformer(transformer))
	t.Cleanup(terminate)
	recorder := server.V3Client.Kubernetes.(*storagetesting.KubernetesRecorder)
	storagetesting.RunTestOptimisticConcurrency(ctx, t, &cacherWithPrefixTransformer{CacheDelegator: delegator, transformer: transformer}, true, recorder)
}

// ===================================================
// Test-setup related function are following.
// ===================================================

type tearDownFunc func()

type setupOptions struct {
	resourcePrefix string
	keyFunc        func(runtime.Object) (string, error)
	reverseKeyFunc storage.ReverseKeyFunc
	indexerFuncs   map[string]storage.IndexerFunc
	indexers       cache.Indexers
	clock          clock.WithTicker
	codec          runtime.Codec
	transformer    value.Transformer
}

type setupOption func(*setupOptions)

func withDefaults(options *setupOptions) {
	prefix := "/pods/"

	options.resourcePrefix = prefix
	options.keyFunc = func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) }
	options.reverseKeyFunc = storagetesting.PodReverseKeyFunc(prefix, true)
	options.clock = clock.RealClock{}
	options.codec = examplev1ProtoCodec
	options.transformer = identity.NewEncryptCheckTransformer()
}

func withClusterScopedKeyFunc(options *setupOptions) {
	options.reverseKeyFunc = storagetesting.PodReverseKeyFunc(options.resourcePrefix, false)
	options.keyFunc = func(obj runtime.Object) (string, error) {
		return storage.NoNamespaceKeyFunc(options.resourcePrefix, obj)
	}
}

// mirror indexer configuration from pkg/registry/core/pod/strategy.go
func withNodeNameAndNamespaceIndex(options *setupOptions) {
	options.indexerFuncs = map[string]storage.IndexerFunc{
		"spec.nodeName": func(obj runtime.Object) string {
			pod, ok := obj.(*example.Pod)
			if !ok {
				return ""
			}
			return pod.Spec.NodeName
		},
	}
	options.indexers = map[string]cache.IndexFunc{
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

func withCodec(codec runtime.Codec) setupOption {
	return func(options *setupOptions) {
		options.codec = codec
	}
}

func withTransformer(transformer value.Transformer) setupOption {
	return func(options *setupOptions) {
		options.transformer = transformer
	}
}

func withClock(c clock.WithTicker) setupOption {
	return func(options *setupOptions) {
		options.clock = c
	}
}

func testSetup(t *testing.T, opts ...setupOption) (context.Context, *CacheDelegator, tearDownFunc) {
	ctx, cacher, _, tearDown := testSetupWithEtcdServer(t, opts...)
	return ctx, cacher, tearDown
}

func testSetupWithEtcdServer(t testing.TB, opts ...setupOption) (context.Context, *CacheDelegator, *etcd3testing.EtcdTestServer, tearDownFunc) {
	setupOpts := setupOptions{}
	opts = append([]setupOption{withDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}

	server, etcdStorage := newEtcdTestStorageWithOptions(t, etcd3testing.PathPrefix(), setupOpts.codec, setupOpts.transformer, setupOpts.reverseKeyFunc)
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

	config := Config{
		Storage:             wrappedStorage,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       schema.GroupResource{Resource: "pods"},
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      setupOpts.resourcePrefix,
		KeyFunc:             setupOpts.keyFunc,
		GetAttrsFunc:        GetPodAttrs,
		NewFunc:             newPod,
		NewListFunc:         newPodList,
		IndexerFuncs:        setupOpts.indexerFuncs,
		Indexers:            &setupOpts.indexers,
		Codec:               setupOpts.codec,
		Clock:               setupOpts.clock,
	}
	cacher, err := NewCacherFromConfig(config)
	if err != nil {
		t.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()

	// Since some tests depend on the fact that GetList shouldn't fail,
	// we wait until the error from the underlying storage is consumed.
	if err := wait.PollInfinite(100*time.Millisecond, wrappedStorage.ErrorsConsumed); err != nil {
		t.Fatalf("Failed to inject list errors: %v", err)
	}

	// The tests assume that Get/GetList/Watch calls shouldn't fail.
	// However, 429 error can now be returned if watchcache is under initialization.
	// To avoid rewriting all tests, we wait for watchcache to initialize.
	if err := cacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	delegator := NewCacheDelegator(cacher, etcdStorage)
	terminate := func() {
		delegator.Stop()
		cacher.Stop()
		server.Terminate(t)
	}

	return ctx, delegator, server, terminate
}

func testSetupWithEtcdAndCreateWrapper(t *testing.T, opts ...setupOption) (storage.Interface, tearDownFunc) {
	_, cacher, _, tearDown := testSetupWithEtcdServer(t, opts...)

	return &createWrapper{CacheDelegator: cacher}, tearDown
}

type createWrapper struct {
	*CacheDelegator
}

func (c *createWrapper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if err := c.CacheDelegator.Create(ctx, key, obj, out, ttl); err != nil {
		return err
	}
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		currentObj := c.CacheDelegator.cacher.newFunc()
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

func benchmarkSetup(b *testing.B) (context.Context, *CacheDelegator) {
	server, etcdStorage := benchmarkEtcdTestStorage(b)
	config := benchmarkConfig(etcdStorage)
	cacher, err := NewCacherFromConfig(config)
	if err != nil {
		b.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()
	if err := cacher.Wait(ctx); err != nil {
		b.Fatal(err)
	}
	delegator := NewCacheDelegator(cacher, etcdStorage)
	b.Cleanup(func() {
		delegator.Stop()
		cacher.Stop()
		server.Terminate(b)
	})
	return ctx, delegator
}

func benchmarkConfig(etcdStorage storage.Interface) Config {
	config := storagetesting.StoreConfigForBenchmarks()
	return Config{
		Storage:             etcdStorage,
		Versioner:           config.Versioner,
		GroupResource:       config.GroupResource,
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      config.ResourcePrefix,
		KeyFunc:             config.KeyFunc,
		GetAttrsFunc:        config.GetAttrsFunc,
		NewFunc:             config.NewFunc,
		NewListFunc:         config.NewListFunc,
		IndexerFuncs: map[string]storage.IndexerFunc{
			"spec.nodeName": func(obj runtime.Object) string {
				pod, ok := obj.(*corev1.Pod)
				if !ok {
					return ""
				}
				return pod.Spec.NodeName
			},
		},
		Indexers: &cache.Indexers{
			"f:spec.nodeName": func(obj interface{}) ([]string, error) {
				pod, ok := obj.(*corev1.Pod)
				if !ok {
					return nil, fmt.Errorf("not a pod")
				}
				return []string{pod.Spec.NodeName}, nil
			},
		},
		Codec: config.Codec,
		Clock: clock.RealClock{},
	}
}

func BenchmarkStoreWriteThroughput(b *testing.B) {
	klog.SetLogger(logr.Discard())
	dimensions := []struct {
		namespaceCount       int
		podPerNamespaceCount int
		nodeCount            int
	}{
		{
			namespaceCount:       50,
			podPerNamespaceCount: 3_000,
			nodeCount:            5_000,
		},
	}
	for _, dims := range dimensions {
		b.Run(fmt.Sprintf("Namespaces=%d/Pods=%d/Nodes=%d", dims.namespaceCount, dims.namespaceCount*dims.podPerNamespaceCount, dims.nodeCount), func(b *testing.B) {
			ctx, cacher := benchmarkSetup(b)
			data := storagetesting.PrepareBenchmarkData(dims.namespaceCount, dims.podPerNamespaceCount, dims.nodeCount)
			tracker := storagetesting.NewWatchLatencyTracker(clock.RealClock{})
			originalHandler := cacher.cacher.watchCache.config.eventHandler
			cacher.cacher.watchCache.config.eventHandler = func(event *watchCacheEvent) {
				if originalHandler != nil {
					originalHandler(event)
				}
				tracker.HandleEvent(event.Object)
			}
			b.Cleanup(func() {
				cacher.cacher.watchCache.config.eventHandler = originalHandler
			})
			storagetesting.RunBenchmarkWriteThroughput(ctx, b, cacher, data, true, tracker)
		})
	}
}

func BenchmarkStoreList(b *testing.B) {
	klog.SetLogger(logr.Discard())
	// Based on https://github.com/kubernetes/community/blob/master/sig-scalability/configs-and-limits/thresholds.md
	dimensions := []struct {
		namespaceCount       int
		podPerNamespaceCount int
		nodeCount            int
	}{
		{
			namespaceCount:       10_000,
			podPerNamespaceCount: 15,
			nodeCount:            5_000,
		},
		{
			namespaceCount:       50,
			podPerNamespaceCount: 3_000,
			nodeCount:            5_000,
		},
		{
			namespaceCount:       100,
			podPerNamespaceCount: 1_100,
			nodeCount:            1000,
		},
	}
	for _, dims := range dimensions {
		b.Run(fmt.Sprintf("Namespaces=%d/Pods=%d/Nodes=%d", dims.namespaceCount, dims.namespaceCount*dims.podPerNamespaceCount, dims.nodeCount), func(b *testing.B) {
			data := storagetesting.PrepareBenchmarkData(dims.namespaceCount, dims.podPerNamespaceCount, dims.nodeCount)
			ctx, cacher := benchmarkSetup(b)
			require.NoError(b, storagetesting.PrecreateBenchmarkPods(ctx, cacher, data))
			for _, useIndex := range []bool{true, false} {
				b.Run(fmt.Sprintf("Indexed=%v", useIndex), func(b *testing.B) {
					storagetesting.RunBenchmarkStoreList(ctx, b, cacher, data, useIndex)
				})
			}
		})
	}
}

func BenchmarkStoreStats(b *testing.B) {
	klog.SetLogger(logr.Discard())
	data := storagetesting.PrepareBenchmarkData(50, 3_000, 5_000)
	ctx, cacher := benchmarkSetup(b)
	require.NoError(b, storagetesting.PrecreateBenchmarkPods(ctx, cacher, data))
	storagetesting.RunBenchmarkStoreStats(ctx, b, cacher)
}

func TestCorrectness(t *testing.T) {
	forceRequestWatchProgressSupport(t)
	ctx, cacher, server, terminate := testSetupWithEtcdServer(t)
	t.Cleanup(terminate)
	correctness.RunTestCorrectness(ctx, t, cacher, etcd3testing.PathPrefix(), cacher.cacher.watchCache.config.keyFunc, compactStore(cacher, server.V3Client.Client))
}
