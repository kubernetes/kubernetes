/*
Copyright 2016 The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"
	"google.golang.org/grpc/grpclog"

	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

const defaultTestPrefix = "test!"

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))

	grpclog.SetLoggerV2(grpclog.NewLoggerV2(ioutil.Discard, ioutil.Discard, os.Stderr))
}

func newPod() runtime.Object {
	return &example.Pod{}
}

func checkStorageInvariants(etcdClient *clientv3.Client, codec runtime.Codec) storagetesting.KeyValidation {
	return func(ctx context.Context, t *testing.T, key string) {
		getResp, err := etcdClient.KV.Get(ctx, key)
		if err != nil {
			t.Fatalf("etcdClient.KV.Get failed: %v", err)
		}
		if len(getResp.Kvs) == 0 {
			t.Fatalf("expecting non empty result on key: %s", key)
		}
		decoded, err := runtime.Decode(codec, getResp.Kvs[0].Value[len(defaultTestPrefix):])
		if err != nil {
			t.Fatalf("expecting successful decode of object from %v\n%v", err, string(getResp.Kvs[0].Value))
		}
		obj := decoded.(*example.Pod)
		if obj.ResourceVersion != "" {
			t.Errorf("stored object should have empty resource version")
		}
		if obj.SelfLink != "" {
			t.Errorf("stored output should have empty selfLink")
		}
	}
}

func TestCreate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	storagetesting.RunTestCreate(ctx, t, store, checkStorageInvariants(etcdClient, store.codec))
}

func TestCreateWithTTL(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestCreateWithTTL(ctx, t, store)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestCreateWithKeyExist(ctx, t, store)
}

func TestGet(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGet(ctx, t, store)
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestUnconditionalDelete(ctx, t, store)
}

func TestConditionalDelete(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestConditionalDelete(ctx, t, store)
}

func TestDeleteWithSuggestion(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestDeleteWithSuggestion(ctx, t, store)
}

func TestDeleteWithSuggestionAndConflict(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestDeleteWithSuggestionAndConflict(ctx, t, store)
}

func TestDeleteWithSuggestionOfDeletedObject(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(ctx, t, store)
}

func TestValidateDeletionWithSuggestion(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestValidateDeletionWithSuggestion(ctx, t, store)
}

func TestPreconditionalDeleteWithSuggestion(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestPreconditionalDeleteWithSuggestion(ctx, t, store)
}

func TestGetListNonRecursive(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGetListNonRecursive(ctx, t, store)
}

type storeWithPrefixTransformer struct {
	*store
}

func (s *storeWithPrefixTransformer) UpdatePrefixTransformer(modifier storagetesting.PrefixTransformerModifier) func() {
	originalTransformer := s.transformer.(*storagetesting.PrefixTransformer)
	transformer := *originalTransformer
	s.transformer = modifier(&transformer)
	return func() {
		s.transformer = originalTransformer
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	storagetesting.RunTestGuaranteedUpdate(ctx, t, &storeWithPrefixTransformer{store}, checkStorageInvariants(etcdClient, store.codec))
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, store)
}

func TestGuaranteedUpdateChecksStoredData(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGuaranteedUpdateChecksStoredData(ctx, t, &storeWithPrefixTransformer{store})
}

func TestGuaranteedUpdateWithConflict(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGuaranteedUpdateWithConflict(ctx, t, store)
}

func TestGuaranteedUpdateWithSuggestionAndConflict(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx, t, store)
}

func TestTransformationFailure(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestTransformationFailure(ctx, t, &storeWithPrefixTransformer{store})
}

func TestList(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestList(ctx, t, store, false)
}

func TestListWithoutPaging(t *testing.T) {
	ctx, store, _ := testSetup(t, withoutPaging())
	storagetesting.RunTestListWithoutPaging(ctx, t, store)
}

func checkStorageCallsInvariants(transformer *storagetesting.PrefixTransformer, recorder *clientRecorder) storagetesting.CallsValidation {
	return func(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
		t.Helper()

		if reads := transformer.GetReadsAndReset(); reads != estimatedProcessedObjects {
			t.Errorf("unexpected reads: %d, expected: %d", reads, estimatedProcessedObjects)
		}
		estimatedGetCalls := uint64(1)
		if pageSize != 0 {
			// We expect that kube-apiserver will be increasing page sizes
			// if not full pages are received, so we should see significantly less
			// than 1000 pages (which would be result of talking to etcd with page size
			// copied from pred.Limit).
			// The expected number of calls is n+1 where n is the smallest n so that:
			// pageSize + pageSize * 2 + pageSize * 4 + ... + pageSize * 2^n >= podCount.
			// For pageSize = 1, podCount = 1000, we get n+1 = 10, 2 ^ 10 = 1024.
			currLimit := pageSize
			for sum := uint64(1); sum < estimatedProcessedObjects; {
				currLimit *= 2
				if currLimit > maxLimit {
					currLimit = maxLimit
				}
				sum += currLimit
				estimatedGetCalls++
			}
		}
		if reads := recorder.GetReadsAndReset(); reads != estimatedGetCalls {
			t.Errorf("unexpected reads: %d", reads)
		}
	}
}

func TestListContinuation(t *testing.T) {
	ctx, store, etcdClient := testSetup(t, withRecorder())
	validation := checkStorageCallsInvariants(
		store.transformer.(*storagetesting.PrefixTransformer), etcdClient.KV.(*clientRecorder))
	storagetesting.RunTestListContinuation(ctx, t, store, validation)
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, store, etcdClient := testSetup(t, withRecorder())
	validation := checkStorageCallsInvariants(
		store.transformer.(*storagetesting.PrefixTransformer), etcdClient.KV.(*clientRecorder))
	storagetesting.RunTestListPaginationRareObject(ctx, t, store, validation)
}

func TestListSlowReadTransformer(t *testing.T) {
	ctx, store, etcdClient := testSetup(t, withRecorder())
	validation := checkStorageCallsInvariants(
		store.transformer.(*storagetesting.PrefixTransformer), etcdClient.KV.(*clientRecorder))

	// fail the test if it runs for too long
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	t.Cleanup(cancel)

	// simulate a transformer that takes some time to perform decryption
	store.transformer = &slowReadTransformer{sleep: 50 * time.Millisecond, transformer: store.transformer}

	// this transforms 1000 objects which takes ~50 seconds without parallelism
	// with our hardcoded parallelism of 4, it takes ~12.5 seconds which is enough for the test to complete in time
	storagetesting.RunTestListPaginationRareObject(ctx, t, store, validation)
}

var _ value.Transformer = &slowReadTransformer{}

type slowReadTransformer struct {
	sleep       time.Duration
	transformer value.Transformer
}

func (s *slowReadTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, stale bool, err error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err // simulate network request cancellation
	}
	time.Sleep(s.sleep)
	return s.transformer.TransformFromStorage(ctx, data, dataCtx)
}

func (s *slowReadTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, err error) {
	return s.transformer.TransformToStorage(ctx, data, dataCtx)
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, store, etcdClient := testSetup(t, withRecorder())
	validation := checkStorageCallsInvariants(
		store.transformer.(*storagetesting.PrefixTransformer), etcdClient.KV.(*clientRecorder))
	storagetesting.RunTestListContinuationWithFilter(ctx, t, store, validation)
}

func compactStorage(etcdClient *clientv3.Client) storagetesting.Compaction {
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := etcdClient.KV.Compact(ctx, int64(rv), clientv3.WithCompactPhysical()); err != nil {
			t.Fatalf("Unable to compact, %v", err)
		}
	}
}

func TestListInconsistentContinuation(t *testing.T) {
	ctx, store, client := testSetup(t)
	storagetesting.RunTestListInconsistentContinuation(ctx, t, store, compactStorage(client))
}

func TestConsistentList(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestConsistentList(ctx, t, &storeWithPrefixTransformer{store})
}

func TestCount(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestCount(ctx, t, store)
}

// =======================================================================
// Implementation-specific tests are following.
// The following tests are exercising the details of the implementation
// not the actual user-facing contract of storage interface.
// As such, they may focus e.g. on non-functional aspects like performance
// impact.
// =======================================================================

func TestPrefix(t *testing.T) {
	testcases := map[string]string{
		"custom/prefix":     "/custom/prefix/",
		"/custom//prefix//": "/custom/prefix/",
		"/registry":         "/registry/",
	}
	for configuredPrefix, effectivePrefix := range testcases {
		_, store, _ := testSetup(t, withPrefix(configuredPrefix))
		if store.pathPrefix != effectivePrefix {
			t.Errorf("configured prefix of %s, expected effective prefix of %s, got %s", configuredPrefix, effectivePrefix, store.pathPrefix)
		}
	}
}

func Test_growSlice(t *testing.T) {
	type args struct {
		initialCapacity int
		initialLen      int
		v               reflect.Value
		maxCapacity     int
		sizes           []int
	}
	tests := []struct {
		name string
		args args
		cap  int
		len  int
	}{
		{
			name: "empty",
			args: args{v: reflect.ValueOf([]example.Pod{})},
			cap:  0,
		},
		{
			name: "no sizes",
			args: args{v: reflect.ValueOf([]example.Pod{}), maxCapacity: 10},
			cap:  10,
		},
		{
			name: "above maxCapacity",
			args: args{v: reflect.ValueOf([]example.Pod{}), maxCapacity: 10, sizes: []int{1, 12}},
			cap:  10,
		},
		{
			name: "takes max",
			args: args{v: reflect.ValueOf([]example.Pod{}), maxCapacity: 10, sizes: []int{8, 4}},
			cap:  8,
		},
		{
			name: "with existing capacity above max",
			args: args{initialCapacity: 12, maxCapacity: 10, sizes: []int{8, 4}},
			cap:  12,
		},
		{
			name: "with existing capacity below max",
			args: args{initialCapacity: 5, maxCapacity: 10, sizes: []int{8, 4}},
			cap:  8,
		},
		{
			name: "with existing capacity and length above max",
			args: args{initialCapacity: 12, initialLen: 5, maxCapacity: 10, sizes: []int{8, 4}},
			cap:  12,
			len:  5,
		},
		{
			name: "with existing capacity and length below max",
			args: args{initialCapacity: 5, initialLen: 3, maxCapacity: 10, sizes: []int{8, 4}},
			cap:  8,
			len:  3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.args.initialCapacity > 0 {
				val := make([]example.Pod, tt.args.initialLen, tt.args.initialCapacity)
				for i := 0; i < tt.args.initialLen; i++ {
					val[i].Name = fmt.Sprintf("test-%d", i)
				}
				tt.args.v = reflect.ValueOf(val)
			}
			// reflection requires that the value be addressable in order to call set,
			// so we must ensure the value we created is available on the heap (not a problem
			// for normal usage)
			if !tt.args.v.CanAddr() {
				x := reflect.New(tt.args.v.Type())
				x.Elem().Set(tt.args.v)
				tt.args.v = x.Elem()
			}
			growSlice(tt.args.v, tt.args.maxCapacity, tt.args.sizes...)
			if tt.cap != tt.args.v.Cap() {
				t.Errorf("Unexpected capacity: got=%d want=%d", tt.args.v.Cap(), tt.cap)
			}
			if tt.len != tt.args.v.Len() {
				t.Errorf("Unexpected length: got=%d want=%d", tt.args.v.Len(), tt.len)
			}
			for i := 0; i < tt.args.v.Len(); i++ {
				nameWanted := fmt.Sprintf("test-%d", i)
				val := tt.args.v.Index(i).Interface()
				pod, ok := val.(example.Pod)
				if !ok || pod.Name != nameWanted {
					t.Errorf("Unexpected element value: got=%s, want=%s", pod.Name, nameWanted)
				}
			}
		})
	}
}

func TestLeaseMaxObjectCount(t *testing.T) {
	ctx, store, _ := testSetup(t, withLeaseConfig(LeaseManagerConfig{
		ReuseDurationSeconds: defaultLeaseReuseDurationSeconds,
		MaxObjectCount:       2,
	}))

	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	out := &example.Pod{}

	testCases := []struct {
		key                 string
		expectAttachedCount int64
	}{
		{
			key:                 "testkey1",
			expectAttachedCount: 1,
		},
		{
			key:                 "testkey2",
			expectAttachedCount: 2,
		},
		{
			key: "testkey3",
			// We assume each time has 1 object attached to the lease
			// so after granting a new lease, the recorded count is set to 1
			expectAttachedCount: 1,
		},
	}

	for _, tc := range testCases {
		err := store.Create(ctx, tc.key, obj, out, 120)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
		if store.leaseManager.leaseAttachedObjectCount != tc.expectAttachedCount {
			t.Errorf("Lease manager recorded count %v should be %v", store.leaseManager.leaseAttachedObjectCount, tc.expectAttachedCount)
		}
	}
}

// ===================================================
// Test-setup related function are following.
// ===================================================

func newTestLeaseManagerConfig() LeaseManagerConfig {
	cfg := NewDefaultLeaseManagerConfig()
	// As 30s is the default timeout for testing in global configuration,
	// we cannot wait longer than that in a single time: change it to 1s
	// for testing purposes. See wait.ForeverTestTimeout
	cfg.ReuseDurationSeconds = 1
	return cfg
}

func newTestTransformer() value.Transformer {
	return storagetesting.NewPrefixTransformer([]byte(defaultTestPrefix), false)
}

type clientRecorder struct {
	reads uint64
	clientv3.KV
}

func (r *clientRecorder) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	atomic.AddUint64(&r.reads, 1)
	return r.KV.Get(ctx, key, opts...)
}

func (r *clientRecorder) GetReadsAndReset() uint64 {
	return atomic.SwapUint64(&r.reads, 0)
}

type setupOptions struct {
	client        func(*testing.T) *clientv3.Client
	codec         runtime.Codec
	newFunc       func() runtime.Object
	prefix        string
	groupResource schema.GroupResource
	transformer   value.Transformer
	pagingEnabled bool
	leaseConfig   LeaseManagerConfig

	recorderEnabled bool
}

type setupOption func(*setupOptions)

func withClientConfig(config *embed.Config) setupOption {
	return func(options *setupOptions) {
		options.client = func(t *testing.T) *clientv3.Client {
			return testserver.RunEtcd(t, config)
		}
	}
}

func withPrefix(prefix string) setupOption {
	return func(options *setupOptions) {
		options.prefix = prefix
	}
}

func withoutPaging() setupOption {
	return func(options *setupOptions) {
		options.pagingEnabled = false
	}
}

func withLeaseConfig(leaseConfig LeaseManagerConfig) setupOption {
	return func(options *setupOptions) {
		options.leaseConfig = leaseConfig
	}
}

func withRecorder() setupOption {
	return func(options *setupOptions) {
		options.recorderEnabled = true
	}
}

func withDefaults(options *setupOptions) {
	options.client = func(t *testing.T) *clientv3.Client {
		return testserver.RunEtcd(t, nil)
	}
	options.codec = apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	options.newFunc = newPod
	options.prefix = ""
	options.groupResource = schema.GroupResource{Resource: "pods"}
	options.transformer = newTestTransformer()
	options.pagingEnabled = true
	options.leaseConfig = newTestLeaseManagerConfig()
}

var _ setupOption = withDefaults

func testSetup(t *testing.T, opts ...setupOption) (context.Context, *store, *clientv3.Client) {
	setupOpts := setupOptions{}
	opts = append([]setupOption{withDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}
	client := setupOpts.client(t)
	if setupOpts.recorderEnabled {
		client.KV = &clientRecorder{KV: client.KV}
	}
	store := newStore(
		client,
		setupOpts.codec,
		setupOpts.newFunc,
		setupOpts.prefix,
		setupOpts.groupResource,
		setupOpts.transformer,
		setupOpts.pagingEnabled,
		setupOpts.leaseConfig,
	)
	ctx := context.Background()
	return ctx, store, client
}

func TestValidateKey(t *testing.T) {
	validKeys := []string{
		"/foo/bar/baz/a.b.c/",
		"/foo",
		"foo/bar/baz",
		"/foo/bar..baz/",
		"/foo/bar..",
		"foo",
		"foo/bar",
		"/foo/bar/",
	}
	invalidKeys := []string{
		"/foo/bar/../a.b.c/",
		"..",
		"/..",
		"../",
		"/foo/bar/..",
		"../foo/bar",
		"/../foo",
		"/foo/bar/../",
		".",
		"/.",
		"./",
		"/./",
		"/foo/.",
		"./bar",
		"/foo/./bar/",
	}
	const (
		pathPrefix   = "/first/second"
		expectPrefix = pathPrefix + "/"
	)
	_, store, _ := testSetup(t, withPrefix(pathPrefix))

	for _, key := range validKeys {
		k, err := store.prepareKey(key)
		if err != nil {
			t.Errorf("key %q should be valid; unexpected error: %v", key, err)
		} else if !strings.HasPrefix(k, expectPrefix) {
			t.Errorf("key %q should have prefix %q", k, expectPrefix)
		}
	}

	for _, key := range invalidKeys {
		_, err := store.prepareKey(key)
		if err == nil {
			t.Errorf("key %q should be invalid", key)
		}
	}
}

func TestInvalidKeys(t *testing.T) {
	const invalidKey = "/foo/bar/../baz"
	expectedError := fmt.Sprintf("invalid key: %q", invalidKey)

	expectInvalidKey := func(methodName string, err error) {
		if err == nil {
			t.Errorf("[%s] expected invalid key error; got nil", methodName)
		} else if err.Error() != expectedError {
			t.Errorf("[%s] expected invalid key error; got %v", methodName, err)
		}
	}

	ctx, store, _ := testSetup(t)
	expectInvalidKey("Create", store.Create(ctx, invalidKey, nil, nil, 0))
	expectInvalidKey("Delete", store.Delete(ctx, invalidKey, nil, nil, nil, nil))
	_, watchErr := store.Watch(ctx, invalidKey, storage.ListOptions{})
	expectInvalidKey("Watch", watchErr)
	expectInvalidKey("Get", store.Get(ctx, invalidKey, storage.GetOptions{}, nil))
	expectInvalidKey("GetList", store.GetList(ctx, invalidKey, storage.ListOptions{}, nil))
	expectInvalidKey("GuaranteedUpdate", store.GuaranteedUpdate(ctx, invalidKey, nil, true, nil, nil, nil))
	_, countErr := store.Count(invalidKey)
	expectInvalidKey("Count", countErr)
}

func TestSplitChunks(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	var (
		kv1 = &mvccpb.KeyValue{Key: []byte("one")}
		kv2 = &mvccpb.KeyValue{Key: []byte("two")}
		kv3 = &mvccpb.KeyValue{Key: []byte("three")}
		kv4 = &mvccpb.KeyValue{Key: []byte("four")}
		kv5 = &mvccpb.KeyValue{Key: []byte("five")}
	)

	testCases := []struct {
		name      string
		kvs       []*mvccpb.KeyValue
		chunkSize int
		want      [][]*mvccpb.KeyValue
	}{
		{
			name:      "nil input results in nil - can't split an nil slice",
			kvs:       nil,
			chunkSize: 4,
			want:      nil,
		},
		{
			name:      "empty input results in nil - can't split an empty slice",
			kvs:       []*mvccpb.KeyValue{},
			chunkSize: 4,
			want:      nil,
		},
		{
			name:      "input slice on len one",
			kvs:       []*mvccpb.KeyValue{kv1},
			chunkSize: 4,
			want:      [][]*mvccpb.KeyValue{{kv1}},
		},
		{
			name:      "input slice of len two",
			kvs:       []*mvccpb.KeyValue{kv1, kv2},
			chunkSize: 4,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2}},
		},
		{
			name:      "input slice on len three",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3},
			chunkSize: 4,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3}},
		},
		{
			name:      "input slice on len four",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4},
			chunkSize: 4,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3, kv4}},
		},
		{
			name:      "input slice on len five with chunk 1",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 1,
			want:      [][]*mvccpb.KeyValue{{kv1}, {kv2}, {kv3}, {kv4}, {kv5}},
		},
		{
			name:      "input slice on len five with chunk 2",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 2,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2}, {kv3, kv4}, {kv5}},
		},
		{
			name:      "input slice on len five with chunk 3",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 3,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3}, {kv4, kv5}},
		},
		{
			name:      "input slice on len five with chunk 4",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 4,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3, kv4}, {kv5}},
		},
		{
			name:      "input slice on len five with chunk 5",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 5,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3, kv4, kv5}},
		},
		{
			name:      "input slice on len five with chunk 6",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 6,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3, kv4, kv5}},
		},
		{
			name:      "input slice on len five with chunk 7",
			kvs:       []*mvccpb.KeyValue{kv1, kv2, kv3, kv4, kv5},
			chunkSize: 7,
			want:      [][]*mvccpb.KeyValue{{kv1, kv2, kv3, kv4, kv5}},
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got := splitChunks(ctx, tt.kvs, tt.chunkSize)
			if diff := cmp.Diff(tt.want, kvChanToList(got)); len(diff) != 0 {
				t.Errorf("incorrect chunks (-want, +got):\n%s", diff)
			}
		})
	}
}

func kvChanToList(ch chan []*mvccpb.KeyValue) [][]*mvccpb.KeyValue {
	var out [][]*mvccpb.KeyValue
	for kv := range ch {
		out = append(out, kv)
	}
	return out
}
