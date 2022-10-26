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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"
	"google.golang.org/grpc/grpclog"

	"k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
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
	storagetesting.RunTestList(ctx, t, store)
}

func TestListWithoutPaging(t *testing.T) {
	ctx, store, _ := testSetup(t, withoutPaging())
	storagetesting.RunTestListWithoutPaging(ctx, t, store)
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
	result := atomic.LoadUint64(&r.reads)
	atomic.StoreUint64(&r.reads, 0)
	return result
}

func checkStorageCallsInvariants(transformer *storagetesting.PrefixTransformer, recorder *clientRecorder) storagetesting.CallsValidation {
	return func(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
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
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*storagetesting.PrefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder
	validation := checkStorageCallsInvariants(transformer, recorder)

	storagetesting.RunTestListContinuation(ctx, t, store, validation)
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*storagetesting.PrefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder
	validation := checkStorageCallsInvariants(transformer, recorder)

	storagetesting.RunTestListPaginationRareObject(ctx, t, store, validation)
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*storagetesting.PrefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder
	validation := checkStorageCallsInvariants(transformer, recorder)

	storagetesting.RunTestListContinuationWithFilter(ctx, t, store, validation)
}

func TestListInconsistentContinuation(t *testing.T) {
	ctx, store, client := testSetup(t)

	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//               - 1/
	//              |   - test
	//              |
	//               - 2/
	//                  - test
	//
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/one-level/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/1/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/2/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	pred := func(limit int64, continueValue string) storage.SelectionPredicate {
		return storage.SelectionPredicate{
			Limit:    limit,
			Continue: continueValue,
			Label:    labels.Everything(),
			Field:    fields.Everything(),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		}
	}

	out := &example.PodList{}
	options := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, ""),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	storagetesting.ExpectNoDiff(t, "incorrect first page", []example.Pod{*preset[0].storedObj}, out.Items)

	continueFromSecondItem := out.Continue

	// update /two-level/2/test/bar
	oldName := preset[2].obj.Name
	newPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: oldName,
			Labels: map[string]string{
				"state": "new",
			},
		},
	}
	if err := store.GuaranteedUpdate(ctx, preset[2].key, preset[2].storedObj, false, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return newPod, nil, nil
		}, newPod); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	// compact to latest revision.
	versioner := storage.APIObjectVersioner{}
	lastRVString := preset[2].storedObj.ResourceVersion
	lastRV, err := versioner.ParseResourceVersion(lastRVString)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.KV.Compact(ctx, int64(lastRV), clientv3.WithCompactPhysical()); err != nil {
		t.Fatalf("Unable to compact, %v", err)
	}

	// The old continue token should have expired
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(0, continueFromSecondItem),
		Recursive:       true,
	}
	err = store.GetList(ctx, "/", options, out)
	if err == nil {
		t.Fatalf("unexpected no error")
	}
	if !strings.Contains(err.Error(), inconsistentContinue) {
		t.Fatalf("unexpected error message %v", err)
	}
	status, ok := err.(apierrors.APIStatus)
	if !ok {
		t.Fatalf("expect error of implements the APIStatus interface, got %v", reflect.TypeOf(err))
	}
	inconsistentContinueFromSecondItem := status.Status().ListMeta.Continue
	if len(inconsistentContinueFromSecondItem) == 0 {
		t.Fatalf("expect non-empty continue token")
	}

	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, inconsistentContinueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	validateResourceVersion := storagetesting.ResourceVersionNotOlderThan(lastRVString)
	storagetesting.ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj}, out.Items)
	if err := validateResourceVersion(out.ResourceVersion); err != nil {
		t.Fatal(err)
	}
	continueFromThirdItem := out.Continue
	resolvedResourceVersionFromThirdItem := out.ResourceVersion
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, continueFromThirdItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	storagetesting.ExpectNoDiff(t, "incorrect third page", []example.Pod{*preset[2].storedObj}, out.Items)
	if out.ResourceVersion != resolvedResourceVersionFromThirdItem {
		t.Fatalf("Expected list resource version to be %s, got %s", resolvedResourceVersionFromThirdItem, out.ResourceVersion)
	}
}

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

type setupOptions struct {
	client        func(*testing.T) *clientv3.Client
	codec         runtime.Codec
	newFunc       func() runtime.Object
	prefix        string
	groupResource schema.GroupResource
	transformer   value.Transformer
	pagingEnabled bool
	leaseConfig   LeaseManagerConfig
}

type setupOption func(*setupOptions)

func withClient(client *clientv3.Client) setupOption {
	return func(options *setupOptions) {
		options.client = func(t *testing.T) *clientv3.Client {
			return client
		}
	}
}

func withClientConfig(config *embed.Config) setupOption {
	return func(options *setupOptions) {
		options.client = func(t *testing.T) *clientv3.Client {
			return testserver.RunEtcd(t, config)
		}
	}
}

func withCodec(codec runtime.Codec) setupOption {
	return func(options *setupOptions) {
		options.codec = codec
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

func withTransformer(transformer value.Transformer) setupOption {
	return func(options *setupOptions) {
		options.transformer = transformer
	}
}

func withLeaseConfig(leaseConfig LeaseManagerConfig) setupOption {
	return func(options *setupOptions) {
		options.leaseConfig = leaseConfig
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

func TestPrefix(t *testing.T) {
	testcases := map[string]string{
		"custom/prefix":     "/custom/prefix",
		"/custom//prefix//": "/custom/prefix",
		"/registry":         "/registry",
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

// fancyTransformer creates next object on each call to
// TransformFromStorage call.
type fancyTransformer struct {
	transformer value.Transformer
	store       *store

	lock  sync.Mutex
	index int
}

func (t *fancyTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if err := t.createObject(ctx); err != nil {
		return nil, false, err
	}
	return t.transformer.TransformFromStorage(ctx, data, dataCtx)
}

func (t *fancyTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	return t.transformer.TransformToStorage(ctx, data, dataCtx)
}

func (t *fancyTransformer) createObject(ctx context.Context) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	t.index++
	key := fmt.Sprintf("pod-%d", t.index)
	obj := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: key,
			Labels: map[string]string{
				"even": strconv.FormatBool(t.index%2 == 0),
			},
		},
	}
	out := &example.Pod{}
	return t.store.Create(ctx, key, obj, out, 0)
}

func TestConsistentList(t *testing.T) {
	transformer := &fancyTransformer{
		transformer: newTestTransformer(),
	}
	ctx, store, _ := testSetup(t, withTransformer(transformer))
	transformer.store = store

	for i := 0; i < 5; i++ {
		if err := transformer.createObject(ctx); err != nil {
			t.Fatalf("failed to create object: %v", err)
		}
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod, ok := obj.(*example.Pod)
		if !ok {
			return nil, nil, fmt.Errorf("invalid object")
		}
		return labels.Set(pod.Labels), nil, nil
	}
	predicate := storage.SelectionPredicate{
		Label:    labels.Set{"even": "true"}.AsSelector(),
		GetAttrs: getAttrs,
		Limit:    4,
	}

	result1 := example.PodList{}
	options := storage.ListOptions{
		Predicate: predicate,
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", options, &result1); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	// List objects from the returned resource version.
	options = storage.ListOptions{
		Predicate:            predicate,
		ResourceVersion:      result1.ResourceVersion,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		Recursive:            true,
	}

	result2 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result2); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	storagetesting.ExpectNoDiff(t, "incorrect lists", result1, result2)

	// Now also verify the  ResourceVersionMatchNotOlderThan.
	options.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan

	result3 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result3); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	options.ResourceVersion = result3.ResourceVersion
	options.ResourceVersionMatch = metav1.ResourceVersionMatchExact

	result4 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result4); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	storagetesting.ExpectNoDiff(t, "incorrect lists", result3, result4)
}

func TestCount(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestCount(ctx, t, store)
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
