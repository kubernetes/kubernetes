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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
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
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	utilpointer "k8s.io/utils/pointer"
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

// prefixTransformer adds and verifies that all data has the correct prefix on its way in and out.
type prefixTransformer struct {
	prefix []byte
	stale  bool
	err    error
	reads  uint64
}

func (p *prefixTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	atomic.AddUint64(&p.reads, 1)
	if dataCtx == nil {
		panic("no context provided")
	}
	if !bytes.HasPrefix(data, p.prefix) {
		return nil, false, fmt.Errorf("value does not have expected prefix %q: %s,", p.prefix, string(data))
	}
	return bytes.TrimPrefix(data, p.prefix), p.stale, p.err
}
func (p *prefixTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	if dataCtx == nil {
		panic("no context provided")
	}
	if len(data) > 0 {
		return append(append([]byte{}, p.prefix...), data...), p.err
	}
	return data, p.err
}

func (p *prefixTransformer) resetReads() {
	p.reads = 0
}

func newPod() runtime.Object {
	return &example.Pod{}
}

func TestCreate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	storagetesting.RunTestCreate(ctx, t, store, func(ctx context.Context, t *testing.T, key string) {
		checkStorageInvariants(ctx, t, etcdClient, store.codec, key)
	})
}

func checkStorageInvariants(ctx context.Context, t *testing.T, etcdClient *clientv3.Client, codec runtime.Codec, key string) {
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

// The following set of Delete tests are testing the logic of adding `suggestion`
// as a parameter with probably value of the current state.
// Introducing it for GuaranteedUpdate cause a number of issues, so we're addressing
// all of those upfront by adding appropriate tests:
// - https://github.com/kubernetes/kubernetes/pull/35415
//   [DONE] Lack of tests originally - added TestDeleteWithSuggestion.
// - https://github.com/kubernetes/kubernetes/pull/40664
//   [DONE] Irrelevant for delete, as Delete doesn't write data (nor compare it).
// - https://github.com/kubernetes/kubernetes/pull/47703
//   [DONE] Irrelevant for delete, because Delete doesn't persist data.
// - https://github.com/kubernetes/kubernetes/pull/48394/
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/43152
//   [DONE] Added TestDeleteWithSuggestionAndConflict
// - https://github.com/kubernetes/kubernetes/pull/54780
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/58375
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/77619
//   [DONE] Added TestValidateDeletionWithSuggestion for corresponding delete checks.
// - https://github.com/kubernetes/kubernetes/pull/78713
//   [DONE] Bug was in getState function which is shared with the new code.
// - https://github.com/kubernetes/kubernetes/pull/78713
//   [DONE] Added TestPreconditionalDeleteWithSuggestion

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

func TestGuaranteedUpdate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	key := "/testkey"

	tests := []struct {
		name                string
		key                 string
		ignoreNotFound      bool
		precondition        *storage.Preconditions
		expectNotFoundErr   bool
		expectInvalidObjErr bool
		expectNoUpdate      bool
		transformStale      bool
		hasSelfLink         bool
	}{{
		name:                "non-existing key, ignoreNotFound=false",
		key:                 "/non-existing",
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   true,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "non-existing key, ignoreNotFound=true",
		key:                 "/non-existing",
		ignoreNotFound:      true,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "existing key",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "same data",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, {
		name:                "same data, a selfLink",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
		hasSelfLink:         true,
	}, {
		name:                "same data, stale",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
		transformStale:      true,
	}, {
		name:                "UID match",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("A"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, {
		name:                "UID mismatch",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("B"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: true,
		expectNoUpdate:      true,
	}}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, storeObj := storagetesting.TestPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})

			out := &example.Pod{}
			name := fmt.Sprintf("foo-%d", i)
			if tt.expectNoUpdate {
				name = storeObj.Name
			}
			originalTransformer := store.transformer.(*prefixTransformer)
			if tt.transformStale {
				transformer := *originalTransformer
				transformer.stale = true
				store.transformer = &transformer
			}
			version := storeObj.ResourceVersion
			err := store.GuaranteedUpdate(ctx, tt.key, out, tt.ignoreNotFound, tt.precondition,
				storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					if tt.expectNotFoundErr && tt.ignoreNotFound {
						if pod := obj.(*example.Pod); pod.Name != "" {
							t.Errorf("%s: expecting zero value, but get=%#v", tt.name, pod)
						}
					}
					pod := *storeObj
					if tt.hasSelfLink {
						pod.SelfLink = "testlink"
					}
					pod.Name = name
					return &pod, nil
				}), nil)
			store.transformer = originalTransformer

			if tt.expectNotFoundErr {
				if err == nil || !storage.IsNotFound(err) {
					t.Errorf("%s: expecting not found error, but get: %v", tt.name, err)
				}
				return
			}
			if tt.expectInvalidObjErr {
				if err == nil || !storage.IsInvalidObj(err) {
					t.Errorf("%s: expecting invalid UID error, but get: %s", tt.name, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("%s: GuaranteedUpdate failed: %v", tt.name, err)
			}
			if out.ObjectMeta.Name != name {
				t.Errorf("%s: pod name want=%s, get=%s", tt.name, name, out.ObjectMeta.Name)
			}
			if out.SelfLink != "" {
				t.Errorf("%s: selfLink should not be set", tt.name)
			}

			// verify that kv pair is not empty after set and that the underlying data matches expectations
			checkStorageInvariants(ctx, t, etcdClient, store.codec, key)

			switch tt.expectNoUpdate {
			case true:
				if version != out.ResourceVersion {
					t.Errorf("%s: expect no version change, before=%s, after=%s", tt.name, version, out.ResourceVersion)
				}
			case false:
				if version == out.ResourceVersion {
					t.Errorf("%s: expect version change, but get the same version=%s", tt.name, version)
				}
			}
		})
	}
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, store)
}

func TestGuaranteedUpdateChecksStoredData(t *testing.T) {
	ctx, store, _ := testSetup(t)
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	// serialize input into etcd with data that would be normalized by a write - in this case, leading
	// and trailing whitespace
	codec := codecs.LegacyCodec(examplev1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, input)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := store.client.Put(ctx, key, "test! "+string(data)+" ")
	if err != nil {
		t.Fatal(err)
	}

	// this update should write the canonical value to etcd because the new serialization differs
	// from the stored serialization
	input.ResourceVersion = strconv.FormatInt(resp.Header.Revision, 10)
	out := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion == strconv.FormatInt(resp.Header.Revision, 10) {
		t.Errorf("guaranteed update should have updated the serialized data, got %#v", out)
	}

	lastVersion := out.ResourceVersion

	// this update should not write to etcd because the input matches the stored data
	input = out
	out = &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion != lastVersion {
		t.Errorf("guaranteed update should have short-circuited write, got %#v", out)
	}

	store.transformer = &prefixTransformer{prefix: []byte(defaultTestPrefix), stale: true}

	// this update should write to etcd because the transformer reported stale
	err = store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion == lastVersion {
		t.Errorf("guaranteed update should have written to etcd when transformer reported stale, got %#v", out)
	}
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

	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{{
		key: "/one-level/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Spec:       storagetesting.DeepEqualSafePodSpec(),
		},
	}, {
		key: "/two-level/1/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "baz"},
			Spec:       storagetesting.DeepEqualSafePodSpec(),
		},
	}}
	for i, ps := range preset[:1] {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[:1][i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	// create a second resource with an invalid prefix
	oldTransformer := store.transformer
	store.transformer = &prefixTransformer{prefix: []byte("otherprefix!")}
	for i, ps := range preset[1:] {
		preset[1:][i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[1:][i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}
	store.transformer = oldTransformer

	// List should fail
	var got example.PodList
	storageOpts := storage.ListOptions{
		Predicate: storage.Everything,
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", storageOpts, &got); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error %v", err)
	}

	// Get should fail
	if err := store.Get(ctx, preset[1].key, storage.GetOptions{}, &example.Pod{}); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate without suggestion should return an error
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, nil
	}, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate with suggestion should return an error if we don't change the object
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, nil
	}, preset[1].obj); err == nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete fails with internal error.
	if err := store.Delete(ctx, preset[1].key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if err := store.Get(ctx, preset[1].key, storage.GetOptions{}, &example.Pod{}); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestList(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RemainingItemCount, true)()
	ctx, store, client := testSetup(t)
	_, disablePagingStore, _ := testSetup(t, withoutPaging(), withClient(client))

	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//  |            - 1/
	//  |           |   - test
	//  |           |
	//  |            - 2/
	//  |               - test
	//  |
	//   - z-level/
	//               - 3/
	//              |   - test
	//              |
	//               - 3/
	//                  - test-2
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
		{
			key: "/z-level/3/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "fourth"}},
		},
		{
			key: "/z-level/3/test-2",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}

	// we want to figure out the resourceVersion before we create anything
	initialList := &example.PodList{}
	if err := store.GetList(ctx, "/", storage.ListOptions{Predicate: storage.Everything, Recursive: true}, initialList); err != nil {
		t.Errorf("Unexpected List error: %v", err)
	}
	initialRV := initialList.ResourceVersion

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	list := &example.PodList{}
	storageOpts := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       storage.Everything,
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/two-level", storageOpts, list); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	continueRV, _ := strconv.Atoi(list.ResourceVersion)
	secondContinuation, err := encodeContinue("/two-level/2", "/two-level/", int64(continueRV))
	if err != nil {
		t.Fatal(err)
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod := obj.(*example.Pod)
		return nil, fields.Set{"metadata.name": pod.Name}, nil
	}

	tests := []struct {
		name                       string
		disablePaging              bool
		rv                         string
		rvMatch                    metav1.ResourceVersionMatch
		prefix                     string
		pred                       storage.SelectionPredicate
		expectedOut                []*example.Pod
		expectContinue             bool
		expectedRemainingItemCount *int64
		expectError                bool
		expectRVTooLarge           bool
		expectRV                   string
		expectRVFunc               func(string) error
	}{
		{
			name:        "rejects invalid resource version",
			prefix:      "/",
			pred:        storage.Everything,
			rv:          "abc",
			expectError: true,
		},
		{
			name:   "rejects resource version and continue token",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			rv:          "1",
			expectError: true,
		},
		{
			name:             "rejects resource version set too high",
			prefix:           "/",
			rv:               strconv.FormatInt(math.MaxInt64, 10),
			expectRVTooLarge: true,
		},
		{
			name:        "test List on existing key",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
		},
		{
			name:        "test List on existing key with resource version set to 0",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          "0",
		},
		{
			name:        "test List on existing key with resource version set before first write, match=Exact",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{},
			rv:          initialRV,
			rvMatch:     metav1.ResourceVersionMatchExact,
			expectRV:    initialRV,
		},
		{
			name:        "test List on existing key with resource version set to 0, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          "0",
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on existing key with resource version set to 0, match=Invalid",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			rv:          "0",
			rvMatch:     "Invalid",
			expectError: true,
		},
		{
			name:        "test List on existing key with resource version set before first write, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          initialRV,
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on existing key with resource version set before first write, match=Invalid",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			rv:          initialRV,
			rvMatch:     "Invalid",
			expectError: true,
		},
		{
			name:        "test List on existing key with resource version set to current resource version",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          list.ResourceVersion,
		},
		{
			name:        "test List on existing key with resource version set to current resource version, match=Exact",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          list.ResourceVersion,
			rvMatch:     metav1.ResourceVersionMatchExact,
			expectRV:    list.ResourceVersion,
		},
		{
			name:        "test List on existing key with resource version set to current resource version, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          list.ResourceVersion,
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on non-existing key",
			prefix:      "/non-existing/",
			pred:        storage.Everything,
			expectedOut: nil,
		},
		{
			name:   "test List with pod name matching",
			prefix: "/one-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.ParseSelectorOrDie("metadata.name!=foo"),
			},
			expectedOut: nil,
		},
		{
			name:   "test List with limit",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1].storedObj},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
		},
		{
			name:   "test List with limit at current resource version",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1].storedObj},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         list.ResourceVersion,
			expectRV:                   list.ResourceVersion,
		},
		{
			name:   "test List with limit at current resource version and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1].storedObj},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         list.ResourceVersion,
			rvMatch:                    metav1.ResourceVersionMatchExact,
			expectRV:                   list.ResourceVersion,
		},
		{
			name:   "test List with limit at resource version 0",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1].storedObj},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         "0",
			expectRVFunc:               resourceVersionNotOlderThan(list.ResourceVersion),
		},
		{
			name:   "test List with limit at resource version 0 match=NotOlderThan",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1].storedObj},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         "0",
			rvMatch:                    metav1.ResourceVersionMatchNotOlderThan,
			expectRVFunc:               resourceVersionNotOlderThan(list.ResourceVersion),
		},
		{
			name:   "test List with limit at resource version before first write and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{},
			expectContinue: false,
			rv:             initialRV,
			rvMatch:        metav1.ResourceVersionMatchExact,
			expectRV:       initialRV,
		},
		{
			name:          "test List with limit when paging disabled",
			disablePaging: true,
			prefix:        "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{preset[1].storedObj, preset[2].storedObj},
			expectContinue: false,
		},
		{
			name:   "test List with pregenerated continue token",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			expectedOut: []*example.Pod{preset[2].storedObj},
		},
		{
			name:   "ignores resource version 0 for List with pregenerated continue token",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			rv:          "0",
			expectedOut: []*example.Pod{preset[2].storedObj},
		},
		{
			name:        "test List with multiple levels of directories and expect flattened result",
			prefix:      "/two-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[1].storedObj, preset[2].storedObj},
		},
		{
			name:   "test List with filter returning only one item, ensure only a single page returned",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{preset[3].storedObj},
			expectContinue: true,
		},
		{
			name:   "test List with filter returning only one item, covers the entire list",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectedOut:    []*example.Pod{preset[3].storedObj},
			expectContinue: false,
		},
		{
			name:   "test List with filter returning only one item, covers the entire list, with resource version 0",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 2,
			},
			rv:             "0",
			expectedOut:    []*example.Pod{preset[3].storedObj},
			expectContinue: false,
		},
		{
			name:   "test List with filter returning two items, more pages possible",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "foo"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectContinue: true,
			expectedOut:    []*example.Pod{preset[0].storedObj, preset[1].storedObj},
		},
		{
			name:   "filter returns two items split across multiple pages",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "bar"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectedOut: []*example.Pod{preset[2].storedObj, preset[4].storedObj},
		},
		{
			name:   "filter returns one item for last page, ends on last item, not full",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    2,
				Continue: encodeContinueOrDie("meta.k8s.io/v1", int64(continueRV), "z-level/3"),
			},
			expectedOut: []*example.Pod{preset[4].storedObj},
		},
		{
			name:   "filter returns one item for last page, starts on last item, full",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    1,
				Continue: encodeContinueOrDie("meta.k8s.io/v1", int64(continueRV), "z-level/3/test-2"),
			},
			expectedOut: []*example.Pod{preset[4].storedObj},
		},
		{
			name:   "filter returns one item for last page, starts on last item, partial page",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    2,
				Continue: encodeContinueOrDie("meta.k8s.io/v1", int64(continueRV), "z-level/3/test-2"),
			},
			expectedOut: []*example.Pod{preset[4].storedObj},
		},
		{
			name:   "filter returns two items, page size equal to total list size",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "bar"),
				Label: labels.Everything(),
				Limit: 5,
			},
			expectedOut: []*example.Pod{preset[2].storedObj, preset[4].storedObj},
		},
		{
			name:   "filter returns one item, page size equal to total list size",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 5,
			},
			expectedOut: []*example.Pod{preset[3].storedObj},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pred.GetAttrs == nil {
				tt.pred.GetAttrs = getAttrs
			}

			out := &example.PodList{}
			storageOpts := storage.ListOptions{
				ResourceVersion:      tt.rv,
				ResourceVersionMatch: tt.rvMatch,
				Predicate:            tt.pred,
				Recursive:            true,
			}
			var err error
			if tt.disablePaging {
				err = disablePagingStore.GetList(ctx, tt.prefix, storageOpts, out)
			} else {
				err = store.GetList(ctx, tt.prefix, storageOpts, out)
			}
			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Fatalf("expecting resource version too high error, but get: %s", err)
				}
				return
			}

			if err != nil {
				if !tt.expectError {
					t.Fatalf("GetList failed: %v", err)
				}
				return
			}
			if tt.expectError {
				t.Fatalf("expected error but got none")
			}
			if (len(out.Continue) > 0) != tt.expectContinue {
				t.Errorf("unexpected continue token: %q", out.Continue)
			}

			// If a client requests an exact resource version, it must be echoed back to them.
			if tt.expectRV != "" {
				if tt.expectRV != out.ResourceVersion {
					t.Errorf("resourceVersion in list response want=%s, got=%s", tt.expectRV, out.ResourceVersion)
				}
			}
			if tt.expectRVFunc != nil {
				if err := tt.expectRVFunc(out.ResourceVersion); err != nil {
					t.Errorf("resourceVersion in list response invalid: %v", err)
				}
			}
			if len(tt.expectedOut) != len(out.Items) {
				t.Fatalf("length of list want=%d, got=%d", len(tt.expectedOut), len(out.Items))
			}
			if diff := cmp.Diff(tt.expectedRemainingItemCount, out.ListMeta.GetRemainingItemCount()); diff != "" {
				t.Errorf("incorrect remainingItemCount: %s", diff)
			}
			for j, wantPod := range tt.expectedOut {
				getPod := &out.Items[j]
				storagetesting.ExpectNoDiff(t, fmt.Sprintf("%s: incorrect pod", tt.name), wantPod, getPod)
			}
		})
	}
}

func TestListContinuation(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*prefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder

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

	// test continuations
	out := &example.PodList{}
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
	if transformer.reads != 1 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()

	continueFromSecondItem := out.Continue

	// no limit, should get two items
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(0, continueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	key, rv, err := decodeContinue(continueFromSecondItem, "/")
	t.Logf("continue token was %d %s %v", rv, key, err)
	storagetesting.ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj, *preset[2].storedObj}, out.Items)
	if transformer.reads != 2 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()

	// limit, should get two more pages
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, continueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	storagetesting.ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj}, out.Items)
	if transformer.reads != 1 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()

	continueFromThirdItem := out.Continue

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
	if transformer.reads != 1 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*prefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder

	podCount := 1000
	var pods []*example.Pod
	for i := 0; i < podCount; i++ {
		key := fmt.Sprintf("/one-level/pod-%d", i)
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i)}}
		storedObj := &example.Pod{}
		err := store.Create(ctx, key, obj, storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
		pods = append(pods, storedObj)
	}

	out := &example.PodList{}
	options := storage.ListOptions{
		Predicate: storage.SelectionPredicate{
			Limit: 1,
			Label: labels.Everything(),
			Field: fields.OneTermEqualSelector("metadata.name", "pod-999"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Errorf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], pods[999]) {
		t.Fatalf("Unexpected first page: %#v", out.Items)
	}
	if transformer.reads != uint64(podCount) {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	// We expect that kube-apiserver will be increasing page sizes
	// if not full pages are received, so we should see significantly less
	// than 1000 pages (which would be result of talking to etcd with page size
	// copied from pred.Limit).
	// The expected number of calls is n+1 where n is the smallest n so that:
	// pageSize + pageSize * 2 + pageSize * 4 + ... + pageSize * 2^n >= podCount.
	// For pageSize = 1, podCount = 1000, we get n+1 = 10, 2 ^ 10 = 1024.
	if recorder.reads != 10 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()
}

type clientRecorder struct {
	reads uint64
	clientv3.KV
}

func (r *clientRecorder) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	atomic.AddUint64(&r.reads, 1)
	return r.KV.Get(ctx, key, opts...)
}

func (r *clientRecorder) resetReads() {
	r.reads = 0
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	transformer := store.transformer.(*prefixTransformer)
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder

	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/1",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/2",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}, // this should not match
		},
		{
			key: "/3",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/4",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	// the first list call should try to get 2 items from etcd (and only those items should be returned)
	// the field selector should result in it reading 3 items via the transformer
	// the chunking should result in 2 etcd Gets
	// there should be a continueValue because there is more data
	out := &example.PodList{}
	pred := func(limit int64, continueValue string) storage.SelectionPredicate {
		return storage.SelectionPredicate{
			Limit:    limit,
			Continue: continueValue,
			Label:    labels.Everything(),
			Field:    fields.OneTermNotEqualSelector("metadata.name", "bar"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		}
	}
	options := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(2, ""),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Errorf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Errorf("No continuation token set")
	}
	storagetesting.ExpectNoDiff(t, "incorrect first page", []example.Pod{*preset[0].storedObj, *preset[2].storedObj}, out.Items)
	if transformer.reads != 3 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 2 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()

	// the rest of the test does not make sense if the previous call failed
	if t.Failed() {
		return
	}

	cont := out.Continue

	// the second list call should try to get 2 more items from etcd
	// but since there is only one item left, that is all we should get with no continueValue
	// both read counters should be incremented for the singular calls they make in this case
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(2, cont),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Errorf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Errorf("Unexpected continuation token set")
	}
	storagetesting.ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[3].storedObj}, out.Items)
	if transformer.reads != 1 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
		t.Errorf("unexpected reads: %d", recorder.reads)
	}
	transformer.resetReads()
	recorder.resetReads()
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
	validateResourceVersion := resourceVersionNotOlderThan(lastRVString)
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

func newTestTransformer() *prefixTransformer {
	return &prefixTransformer{prefix: []byte(defaultTestPrefix)}
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

func encodeContinueOrDie(apiVersion string, resourceVersion int64, nextKey string) string {
	out, err := json.Marshal(&continueToken{APIVersion: apiVersion, ResourceVersion: resourceVersion, StartKey: nextKey})
	if err != nil {
		panic(err)
	}
	return base64.RawURLEncoding.EncodeToString(out)
}

func Test_decodeContinue(t *testing.T) {
	type args struct {
		continueValue string
		keyPrefix     string
	}
	tests := []struct {
		name        string
		args        args
		wantFromKey string
		wantRv      int64
		wantErr     bool
	}{
		{name: "valid", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "key"), keyPrefix: "/test/"}, wantRv: 1, wantFromKey: "/test/key"},
		{name: "root path", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "/"), keyPrefix: "/test/"}, wantRv: 1, wantFromKey: "/test/"},

		{name: "empty version", args: args{continueValue: encodeContinueOrDie("", 1, "key"), keyPrefix: "/test/"}, wantErr: true},
		{name: "invalid version", args: args{continueValue: encodeContinueOrDie("v1", 1, "key"), keyPrefix: "/test/"}, wantErr: true},

		{name: "path traversal - parent", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "../key"), keyPrefix: "/test/"}, wantErr: true},
		{name: "path traversal - local", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "./key"), keyPrefix: "/test/"}, wantErr: true},
		{name: "path traversal - double parent", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "./../key"), keyPrefix: "/test/"}, wantErr: true},
		{name: "path traversal - after parent", args: args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "key/../.."), keyPrefix: "/test/"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotFromKey, gotRv, err := decodeContinue(tt.args.continueValue, tt.args.keyPrefix)
			if (err != nil) != tt.wantErr {
				t.Errorf("decodeContinue() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotFromKey != tt.wantFromKey {
				t.Errorf("decodeContinue() gotFromKey = %v, want %v", gotFromKey, tt.wantFromKey)
			}
			if gotRv != tt.wantRv {
				t.Errorf("decodeContinue() gotRv = %v, want %v", gotRv, tt.wantRv)
			}
		})
	}
}

func Test_growSlice(t *testing.T) {
	type args struct {
		initialCapacity int
		v               reflect.Value
		maxCapacity     int
		sizes           []int
	}
	tests := []struct {
		name string
		args args
		cap  int
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.args.initialCapacity > 0 {
				tt.args.v = reflect.ValueOf(make([]example.Pod, 0, tt.args.initialCapacity))
			}
			// reflection requires that the value be addressible in order to call set,
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
