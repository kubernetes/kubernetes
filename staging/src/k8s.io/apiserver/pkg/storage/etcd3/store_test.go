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
	"errors"
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
	"google.golang.org/grpc/grpclog"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
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

func (p *prefixTransformer) TransformFromStorage(b []byte, ctx value.Context) ([]byte, bool, error) {
	atomic.AddUint64(&p.reads, 1)
	if ctx == nil {
		panic("no context provided")
	}
	if !bytes.HasPrefix(b, p.prefix) {
		return nil, false, fmt.Errorf("value does not have expected prefix %q: %s,", p.prefix, string(b))
	}
	return bytes.TrimPrefix(b, p.prefix), p.stale, p.err
}
func (p *prefixTransformer) TransformToStorage(b []byte, ctx value.Context) ([]byte, error) {
	if ctx == nil {
		panic("no context provided")
	}
	if len(b) > 0 {
		return append(append([]byte{}, p.prefix...), b...), p.err
	}
	return b, p.err
}

func (p *prefixTransformer) resetReads() {
	p.reads = 0
}

func newPod() runtime.Object {
	return &example.Pod{}
}

func TestCreate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)

	key := "/testkey"
	out := &example.Pod{}
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}

	// verify that kv pair is empty before set
	getResp, err := etcdClient.KV.Get(ctx, key)
	if err != nil {
		t.Fatalf("etcdClient.KV.Get failed: %v", err)
	}
	if len(getResp.Kvs) != 0 {
		t.Fatalf("expecting empty result on key: %s", key)
	}

	err = store.Create(ctx, key, obj, out, 0)
	if err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	// basic tests of the output
	if obj.ObjectMeta.Name != out.ObjectMeta.Name {
		t.Errorf("pod name want=%s, get=%s", obj.ObjectMeta.Name, out.ObjectMeta.Name)
	}
	if out.ResourceVersion == "" {
		t.Errorf("output should have non-empty resource version")
	}
	if out.SelfLink != "" {
		t.Errorf("output should have empty self link")
	}

	checkStorageInvariants(ctx, t, etcdClient, store, key)
}

func checkStorageInvariants(ctx context.Context, t *testing.T, etcdClient *clientv3.Client, store *store, key string) {
	getResp, err := etcdClient.KV.Get(ctx, key)
	if err != nil {
		t.Fatalf("etcdClient.KV.Get failed: %v", err)
	}
	if len(getResp.Kvs) == 0 {
		t.Fatalf("expecting non empty result on key: %s", key)
	}
	decoded, err := runtime.Decode(store.codec, getResp.Kvs[0].Value[len(defaultTestPrefix):])
	if err != nil {
		t.Fatalf("expecting successful decode of object from %v\n%v", err, string(getResp.Kvs[0].Value))
	}
	obj := decoded.(*example.Pod)
	if obj.ResourceVersion != "" {
		t.Errorf("stored object should have empty resource version")
	}
	if obj.SelfLink != "" {
		t.Errorf("stored output should have empty self link")
	}
}

func TestCreateWithTTL(t *testing.T) {
	ctx, store, _ := testSetup(t)

	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	if err := store.Create(ctx, key, input, out, 1); err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: out.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, store, _ := testSetup(t)
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key, _ := testPropogateStore(ctx, t, store, obj)
	out := &example.Pod{}
	err := store.Create(ctx, key, obj, out, 0)
	if err == nil || !storage.IsNodeExist(err) {
		t.Errorf("expecting key exists error, but get: %s", err)
	}
}

func TestGet(t *testing.T) {
	ctx, store, _ := testSetup(t)
	// create an object to test
	key, createdObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	// update the object once to allow get by exact resource version to be tested
	updateObj := createdObj.DeepCopy()
	updateObj.Annotations = map[string]string{"test-annotation": "1"}
	storedObj := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, storedObj, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return updateObj, &ttl, nil
		}, nil)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	// create an additional object to increment the resource version for pods above the resource version of the foo object
	lastUpdatedObj := &example.Pod{}
	if err := store.Create(ctx, "bar", &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}, lastUpdatedObj, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}

	currentRV, _ := strconv.Atoi(storedObj.ResourceVersion)
	lastUpdatedCurrentRV, _ := strconv.Atoi(lastUpdatedObj.ResourceVersion)

	// TODO(jpbetz): Add exact test cases
	tests := []struct {
		name              string
		key               string
		ignoreNotFound    bool
		expectNotFoundErr bool
		expectRVTooLarge  bool
		expectedOut       *example.Pod
		rv                string
	}{{ // test get on existing item
		name:              "get existing",
		key:               key,
		ignoreNotFound:    false,
		expectNotFoundErr: false,
		expectedOut:       storedObj,
	}, { // test get on existing item with resource version set to 0
		name:        "resource version 0",
		key:         key,
		expectedOut: storedObj,
		rv:          "0",
	}, { // test get on existing item with resource version set to the resource version is was created on
		name:        "object created resource version",
		key:         key,
		expectedOut: storedObj,
		rv:          createdObj.ResourceVersion,
	}, { // test get on existing item with resource version set to current resource version of the object
		name:        "current object resource version, match=NotOlderThan",
		key:         key,
		expectedOut: storedObj,
		rv:          fmt.Sprintf("%d", currentRV),
	}, { // test get on existing item with resource version set to latest pod resource version
		name:        "latest resource version",
		key:         key,
		expectedOut: storedObj,
		rv:          fmt.Sprintf("%d", lastUpdatedCurrentRV),
	}, { // test get on existing item with resource version set too high
		name:             "too high resource version",
		key:              key,
		expectRVTooLarge: true,
		rv:               fmt.Sprintf("%d", lastUpdatedCurrentRV+1),
	}, { // test get on non-existing item with ignoreNotFound=false
		name:              "get non-existing",
		key:               "/non-existing",
		ignoreNotFound:    false,
		expectNotFoundErr: true,
	}, { // test get on non-existing item with ignoreNotFound=true
		name:              "get non-existing, ignore not found",
		key:               "/non-existing",
		ignoreNotFound:    true,
		expectNotFoundErr: false,
		expectedOut:       &example.Pod{},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &example.Pod{}
			err := store.Get(ctx, tt.key, storage.GetOptions{IgnoreNotFound: tt.ignoreNotFound, ResourceVersion: tt.rv}, out)
			if tt.expectNotFoundErr {
				if err == nil || !storage.IsNotFound(err) {
					t.Errorf("expecting not found error, but get: %v", err)
				}
				return
			}
			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Errorf("expecting resource version too high error, but get: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Get failed: %v", err)
			}
			if !reflect.DeepEqual(tt.expectedOut, out) {
				t.Errorf("pod want=\n%#v\nget=\n%#v", tt.expectedOut, out)
			}
		})
	}
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, store, _ := testSetup(t)
	key, storedObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key               string
		expectedObj       *example.Pod
		expectNotFoundErr bool
	}{{ // test unconditional delete on existing key
		key:               key,
		expectedObj:       storedObj,
		expectNotFoundErr: false,
	}, { // test unconditional delete on non-existing key
		key:               "/non-existing",
		expectedObj:       nil,
		expectNotFoundErr: true,
	}}

	for i, tt := range tests {
		out := &example.Pod{} // reset
		err := store.Delete(ctx, tt.key, out, nil, storage.ValidateAllObjectFunc, nil)
		if tt.expectNotFoundErr {
			if err == nil || !storage.IsNotFound(err) {
				t.Errorf("#%d: expecting not found error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if !reflect.DeepEqual(tt.expectedObj, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, tt.expectedObj, out)
		}
	}
}

func TestConditionalDelete(t *testing.T) {
	ctx, store, _ := testSetup(t)
	key, storedObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})

	tests := []struct {
		precondition        *storage.Preconditions
		expectInvalidObjErr bool
	}{{ // test conditional delete with UID match
		precondition:        storage.NewUIDPreconditions("A"),
		expectInvalidObjErr: false,
	}, { // test conditional delete with UID mismatch
		precondition:        storage.NewUIDPreconditions("B"),
		expectInvalidObjErr: true,
	}}

	for i, tt := range tests {
		out := &example.Pod{}
		err := store.Delete(ctx, key, out, tt.precondition, storage.ValidateAllObjectFunc, nil)
		if tt.expectInvalidObjErr {
			if err == nil || !storage.IsInvalidObj(err) {
				t.Errorf("#%d: expecting invalid UID error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if !reflect.DeepEqual(storedObj, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, storedObj, out)
		}
		key, storedObj = testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})
	}
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

	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func TestDeleteWithSuggestionAndConflict(t *testing.T) {
	ctx, store, _ := testSetup(t)

	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.Labels = map[string]string{"foo": "bar"}
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func TestDeleteWithSuggestionOfDeletedObject(t *testing.T) {
	ctx, store, _ := testSetup(t)

	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First delete, so originalPod is outdated.
	deletedPod := &example.Pod{}
	if err := store.Delete(ctx, key, deletedPod, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	// Now try deleting with stale object.
	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error during deletion: %v, expected not-found", err)
	}
}

func TestValidateDeletionWithSuggestion(t *testing.T) {
	ctx, store, _ := testSetup(t)

	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// Check that validaing fresh object fails is called once and fails.
	validationCalls := 0
	validationError := fmt.Errorf("validation error")
	validateNothing := func(_ context.Context, _ runtime.Object) error {
		validationCalls++
		return validationError
	}
	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, validateNothing, originalPod); err != validationError {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}
	if validationCalls != 1 {
		t.Errorf("validate function should have been called once, called %d", validationCalls)
	}

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.Labels = map[string]string{"foo": "bar"}
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	calls := 0
	validateFresh := func(_ context.Context, obj runtime.Object) error {
		calls++
		pod := obj.(*example.Pod)
		if pod.ObjectMeta.Labels == nil || pod.ObjectMeta.Labels["foo"] != "bar" {
			return fmt.Errorf("stale object")
		}
		return nil
	}

	if err := store.Delete(ctx, key, out, nil, validateFresh, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if calls != 2 {
		t.Errorf("validate function should have been called twice, called %d", calls)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func TestPreconditionalDeleteWithSuggestion(t *testing.T) {
	ctx, store, _ := testSetup(t)

	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.UID = "myUID"
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	prec := storage.NewUIDPreconditions("myUID")

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, prec, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func TestGetToList(t *testing.T) {
	ctx, store, _ := testSetup(t)
	prevKey, prevStoredObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "prev"}})

	prevRV, _ := strconv.Atoi(prevStoredObj.ResourceVersion)

	key, storedObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	currentRV, _ := strconv.Atoi(storedObj.ResourceVersion)

	tests := []struct {
		key              string
		pred             storage.SelectionPredicate
		expectedOut      []*example.Pod
		rv               string
		rvMatch          metav1.ResourceVersionMatch
		expectRVTooLarge bool
	}{{ // test GetToList on existing key
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
	}, { // test GetToList on existing key with minimum resource version set to 0
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          "0",
	}, { // test GetToList on existing key with minimum resource version set to 0, match=minimum
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          "0",
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, { // test GetToList on existing key with minimum resource version set to current resource version
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
	}, { // test GetToList on existing key with minimum resource version set to current resource version, match=minimum
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, { // test GetToList on existing key with minimum resource version set to previous resource version, match=minimum
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", prevRV),
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, { // test GetToList on existing key with resource version set to current resource version, match=exact
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
		rvMatch:     metav1.ResourceVersionMatchExact,
	}, { // test GetToList on existing key with resource version set to previous resource version, match=exact
		key:         prevKey,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{prevStoredObj},
		rv:          fmt.Sprintf("%d", prevRV),
		rvMatch:     metav1.ResourceVersionMatchExact,
	}, { // test GetToList on existing key with minimum resource version set too high
		key:              key,
		pred:             storage.Everything,
		expectedOut:      []*example.Pod{storedObj},
		rv:               fmt.Sprintf("%d", currentRV+1),
		expectRVTooLarge: true,
	}, { // test GetToList on non-existing key
		key:         "/non-existing",
		pred:        storage.Everything,
		expectedOut: nil,
	}, { // test GetToList with matching pod name
		key: "/non-existing",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		expectedOut: nil,
	}}

	for i, tt := range tests {
		out := &example.PodList{}
		err := store.GetToList(ctx, tt.key, storage.ListOptions{ResourceVersion: tt.rv, ResourceVersionMatch: tt.rvMatch, Predicate: tt.pred}, out)

		if tt.expectRVTooLarge {
			if err == nil || !storage.IsTooLargeResourceVersion(err) {
				t.Errorf("#%d: expecting resource version too high error, but get: %s", i, err)
			}
			continue
		}

		if err != nil {
			t.Fatalf("GetToList failed: %v", err)
		}
		if len(out.ResourceVersion) == 0 {
			t.Errorf("#%d: unset resourceVersion", i)
		}
		if len(out.Items) != len(tt.expectedOut) {
			t.Errorf("#%d: length of list want=%d, get=%d", i, len(tt.expectedOut), len(out.Items))
			continue
		}
		for j, wantPod := range tt.expectedOut {
			getPod := &out.Items[j]
			if !reflect.DeepEqual(wantPod, getPod) {
				t.Errorf("#%d: pod want=%#v, get=%#v", i, wantPod, getPod)
			}
		}
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	ctx, store, etcdClient := testSetup(t)
	key := "/testkey"

	tests := []struct {
		key                 string
		ignoreNotFound      bool
		precondition        *storage.Preconditions
		expectNotFoundErr   bool
		expectInvalidObjErr bool
		expectNoUpdate      bool
		transformStale      bool
		hasSelfLink         bool
	}{{ // GuaranteedUpdate on non-existing key with ignoreNotFound=false
		key:                 "/non-existing",
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   true,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate on non-existing key with ignoreNotFound=true
		key:                 "/non-existing",
		ignoreNotFound:      true,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate on existing key
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate with same data
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, { // GuaranteedUpdate with same data AND a self link
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
		hasSelfLink:         true,
	}, { // GuaranteedUpdate with same data but stale
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
		transformStale:      true,
	}, { // GuaranteedUpdate with UID match
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("A"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, { // GuaranteedUpdate with UID mismatch
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("B"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: true,
		expectNoUpdate:      true,
	}}

	for i, tt := range tests {
		key, storeObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})

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
						t.Errorf("#%d: expecting zero value, but get=%#v", i, pod)
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
				t.Errorf("#%d: expecting not found error, but get: %v", i, err)
			}
			continue
		}
		if tt.expectInvalidObjErr {
			if err == nil || !storage.IsInvalidObj(err) {
				t.Errorf("#%d: expecting invalid UID error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("GuaranteedUpdate failed: %v", err)
		}
		if out.ObjectMeta.Name != name {
			t.Errorf("#%d: pod name want=%s, get=%s", i, name, out.ObjectMeta.Name)
		}
		if out.SelfLink != "" {
			t.Errorf("#%d: selflink should not be set", i)
		}

		// verify that kv pair is not empty after set and that the underlying data matches expectations
		checkStorageInvariants(ctx, t, etcdClient, store, key)

		switch tt.expectNoUpdate {
		case true:
			if version != out.ResourceVersion {
				t.Errorf("#%d: expect no version change, before=%s, after=%s", i, version, out.ResourceVersion)
			}
		case false:
			if version == out.ResourceVersion {
				t.Errorf("#%d: expect version change, but get the same version=%s", i, version)
			}
		}
	}
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, store, _ := testSetup(t)

	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return input, &ttl, nil
		}, nil)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: out.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
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

	store.transformer = &prefixTransformer{prefix: []byte(defaultTestPrefix)}

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
	key, _ := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	errChan := make(chan error, 1)
	var firstToFinish sync.WaitGroup
	var secondToEnter sync.WaitGroup
	firstToFinish.Add(1)
	secondToEnter.Add(1)

	go func() {
		err := store.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
			storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				pod := obj.(*example.Pod)
				pod.Name = "foo-1"
				secondToEnter.Wait()
				return pod, nil
			}), nil)
		firstToFinish.Done()
		errChan <- err
	}()

	updateCount := 0
	err := store.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			if updateCount == 0 {
				secondToEnter.Done()
				firstToFinish.Wait()
			}
			updateCount++
			pod := obj.(*example.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}), nil)
	if err != nil {
		t.Fatalf("Second GuaranteedUpdate error %#v", err)
	}
	if err := <-errChan; err != nil {
		t.Fatalf("First GuaranteedUpdate error %#v", err)
	}

	if updateCount != 2 {
		t.Errorf("Should have conflict and called update func twice")
	}
}

func TestGuaranteedUpdateWithSuggestionAndConflict(t *testing.T) {
	ctx, store, _ := testSetup(t)
	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	// First, update without a suggestion so originalPod is outdated
	updatedPod := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}),
		nil,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Second, update using the outdated originalPod as the suggestion. Return a conflict error when
	// passed originalPod, and make sure that SimpleUpdate is called a second time after a live lookup
	// with the value of updatedPod.
	sawConflict := false
	updatedPod2 := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, updatedPod2, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			if pod.Name != "foo-2" {
				if sawConflict {
					t.Fatalf("unexpected second conflict")
				}
				sawConflict = true
				// simulated stale object - return a conflict
				return nil, apierrors.NewConflict(example.SchemeGroupVersion.WithResource("pods").GroupResource(), "name", errors.New("foo"))
			}
			pod.Name = "foo-3"
			return pod, nil
		}),
		originalPod,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if updatedPod2.Name != "foo-3" {
		t.Errorf("unexpected pod name: %q", updatedPod2.Name)
	}

	// Third, update using a current version as the suggestion.
	// Return an error and make sure that SimpleUpdate is NOT called a second time,
	// since the live lookup shows the suggestion was already up to date.
	attempts := 0
	updatedPod3 := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, updatedPod3, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			if pod.Name != updatedPod2.Name || pod.ResourceVersion != updatedPod2.ResourceVersion {
				t.Errorf(
					"unexpected live object (name=%s, rv=%s), expected name=%s, rv=%s",
					pod.Name,
					pod.ResourceVersion,
					updatedPod2.Name,
					updatedPod2.ResourceVersion,
				)
			}
			attempts++
			return nil, fmt.Errorf("validation or admission error")
		}),
		updatedPod2,
	)
	if err == nil {
		t.Fatalf("expected error, got none")
	}
	if attempts != 1 {
		t.Errorf("expected 1 attempt, got %d", attempts)
	}
}

func TestTransformationFailure(t *testing.T) {
	client := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, false, NewDefaultLeaseManagerConfig())
	ctx := context.Background()

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
	if err := store.List(ctx, "/", storage.ListOptions{Predicate: storage.Everything}, &got); !storage.IsInternalError(err) {
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
	client := testserver.RunEtcd(t, nil)
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RemainingItemCount, true)()
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, true, NewDefaultLeaseManagerConfig())
	disablePagingStore := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, false, NewDefaultLeaseManagerConfig())
	ctx := context.Background()

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

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	list := &example.PodList{}
	store.List(ctx, "/two-level", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything}, list)
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
			rv:               fmt.Sprintf("%d", continueRV+1),
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
			name:        "test List on existing key with resource version set to 1, match=Exact",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{},
			rv:          "1",
			rvMatch:     metav1.ResourceVersionMatchExact,
			expectRV:    "1",
		},
		{
			name:        "test List on existing key with resource version set to 1, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
			rv:          "0",
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on existing key with resource version set to 1, match=Invalid",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			rv:          "0",
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
			expectRV:                   list.ResourceVersion,
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
			expectRV:                   list.ResourceVersion,
		},
		{
			name:   "test List with limit at resource version 1 and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{},
			expectContinue: false,
			rv:             "1",
			rvMatch:        metav1.ResourceVersionMatchExact,
			expectRV:       "1",
		},
		{
			name:   "test List with limit at old resource version and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{},
			expectContinue: false,
			rv:             "1",
			rvMatch:        metav1.ResourceVersionMatchExact,
			expectRV:       "1",
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
			storageOpts := storage.ListOptions{ResourceVersion: tt.rv, ResourceVersionMatch: tt.rvMatch, Predicate: tt.pred}
			var err error
			if tt.disablePaging {
				err = disablePagingStore.List(ctx, tt.prefix, storageOpts, out)
			} else {
				err = store.List(ctx, tt.prefix, storageOpts, out)
			}
			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Fatalf("expecting resource version too high error, but get: %s", err)
				}
				return
			}

			if err != nil {
				if !tt.expectError {
					t.Fatalf("List failed: %v", err)
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
			if len(tt.expectedOut) != len(out.Items) {
				t.Fatalf("length of list want=%d, got=%d", len(tt.expectedOut), len(out.Items))
			}
			if e, a := tt.expectedRemainingItemCount, out.ListMeta.GetRemainingItemCount(); (e == nil) != (a == nil) || (e != nil && a != nil && *e != *a) {
				t.Errorf("remainingItemCount want=%#v, got=%#v", e, a)
			}
			for j, wantPod := range tt.expectedOut {
				getPod := &out.Items[j]
				if !reflect.DeepEqual(wantPod, getPod) {
					t.Errorf("pod want=%#v, got=%#v", wantPod, getPod)
				}
			}
		})
	}
}

func TestListContinuation(t *testing.T) {
	etcdClient := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	transformer := &prefixTransformer{prefix: []byte(defaultTestPrefix)}
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder
	store := newStore(etcdClient, codec, newPod, "", schema.GroupResource{Resource: "pods"}, transformer, true, NewDefaultLeaseManagerConfig())
	ctx := context.Background()

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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, "")}, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[0].storedObj) {
		t.Fatalf("Unexpected first page: %#v", out.Items)
	}
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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(0, continueFromSecondItem)}, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	if !reflect.DeepEqual(out.Items, []example.Pod{*preset[1].storedObj, *preset[2].storedObj}) {
		key, rv, err := decodeContinue(continueFromSecondItem, "/")
		t.Logf("continue token was %d %s %v", rv, key, err)
		t.Fatalf("Unexpected second page: %#v", out.Items)
	}
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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, continueFromSecondItem)}, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[1].storedObj) {
		t.Fatalf("Unexpected second page: %#v", out.Items)
	}
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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, continueFromThirdItem)}, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[2].storedObj) {
		t.Fatalf("Unexpected third page: %#v", out.Items)
	}
	if transformer.reads != 1 {
		t.Errorf("unexpected reads: %d", transformer.reads)
	}
	if recorder.reads != 1 {
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
	etcdClient := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	transformer := &prefixTransformer{prefix: []byte(defaultTestPrefix)}
	recorder := &clientRecorder{KV: etcdClient.KV}
	etcdClient.KV = recorder
	store := newStore(etcdClient, codec, newPod, "", schema.GroupResource{Resource: "pods"}, transformer, true, NewDefaultLeaseManagerConfig())
	ctx := context.Background()

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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(2, "")}, out); err != nil {
		t.Errorf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Errorf("No continuation token set")
	}
	if len(out.Items) != 2 || !reflect.DeepEqual(&out.Items[0], preset[0].storedObj) || !reflect.DeepEqual(&out.Items[1], preset[2].storedObj) {
		t.Errorf("Unexpected first page, len=%d: %#v", len(out.Items), out.Items)
	}
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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(2, cont)}, out); err != nil {
		t.Errorf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Errorf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[3].storedObj) {
		t.Errorf("Unexpected second page, len=%d: %#v", len(out.Items), out.Items)
	}
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
	client := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, true, NewDefaultLeaseManagerConfig())
	ctx := context.Background()

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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, "")}, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[0].storedObj) {
		t.Fatalf("Unexpected first page: %#v", out.Items)
	}

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
	versioner := APIObjectVersioner{}
	lastRVString := preset[2].storedObj.ResourceVersion
	lastRV, err := versioner.ParseResourceVersion(lastRVString)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.KV.Compact(ctx, int64(lastRV), clientv3.WithCompactPhysical()); err != nil {
		t.Fatalf("Unable to compact, %v", err)
	}

	// The old continue token should have expired
	err = store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(0, continueFromSecondItem)}, out)
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
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, inconsistentContinueFromSecondItem)}, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[1].storedObj) {
		t.Fatalf("Unexpected second page: %#v", out.Items)
	}
	if out.ResourceVersion != lastRVString {
		t.Fatalf("Expected list resource version to be %s, got %s", lastRVString, out.ResourceVersion)
	}
	continueFromThirdItem := out.Continue
	out = &example.PodList{}
	if err := store.List(ctx, "/", storage.ListOptions{ResourceVersion: "0", Predicate: pred(1, continueFromThirdItem)}, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[2].storedObj) {
		t.Fatalf("Unexpected third page: %#v", out.Items)
	}
	if out.ResourceVersion != lastRVString {
		t.Fatalf("Expected list resource version to be %s, got %s", lastRVString, out.ResourceVersion)
	}
}

func testSetup(t *testing.T) (context.Context, *store, *clientv3.Client) {
	client := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	// As 30s is the default timeout for testing in glboal configuration,
	// we cannot wait longer than that in a single time: change it to 10
	// for testing purposes. See apimachinery/pkg/util/wait/wait.go
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, true, LeaseManagerConfig{
		ReuseDurationSeconds: 1,
		MaxObjectCount:       defaultLeaseMaxObjectCount,
	})
	ctx := context.Background()
	return ctx, store, client
}

// testPropogateStore helps propagates store with objects, automates key generation, and returns
// keys and stored objects.
func testPropogateStore(ctx context.Context, t *testing.T, store *store, obj *example.Pod) (string, *example.Pod) {
	// Setup store with a key and grab the output for returning.
	key := "/testkey"
	return key, testPropogateStoreWithKey(ctx, t, store, key, obj)
}

// testPropogateStoreWithKey helps propagate store with objects, the given object will be stored at the specified key.
func testPropogateStoreWithKey(ctx context.Context, t *testing.T, store *store, key string, obj *example.Pod) *example.Pod {
	// Setup store with the specified key and grab the output for returning.
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		panic("unable to convert output object to pointer")
	}
	err = store.conditionalDelete(ctx, key, &example.Pod{}, v, nil, storage.ValidateAllObjectFunc, nil)
	if err != nil && !storage.IsNotFound(err) {
		t.Fatalf("Cleanup failed: %v", err)
	}
	setOutput := &example.Pod{}
	if err := store.Create(ctx, key, obj, setOutput, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	return setOutput
}

func TestPrefix(t *testing.T) {
	client := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	transformer := &prefixTransformer{prefix: []byte(defaultTestPrefix)}
	testcases := map[string]string{
		"custom/prefix":     "/custom/prefix",
		"/custom//prefix//": "/custom/prefix",
		"/registry":         "/registry",
	}
	for configuredPrefix, effectivePrefix := range testcases {
		store := newStore(client, codec, nil, configuredPrefix, schema.GroupResource{Resource: "widgets"}, transformer, true, NewDefaultLeaseManagerConfig())
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

func (t *fancyTransformer) TransformFromStorage(b []byte, ctx value.Context) ([]byte, bool, error) {
	if err := t.createObject(); err != nil {
		return nil, false, err
	}
	return t.transformer.TransformFromStorage(b, ctx)
}

func (t *fancyTransformer) TransformToStorage(b []byte, ctx value.Context) ([]byte, error) {
	return t.transformer.TransformToStorage(b, ctx)
}

func (t *fancyTransformer) createObject() error {
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
	return t.store.Create(context.TODO(), key, obj, out, 0)
}

func TestConsistentList(t *testing.T) {
	client := testserver.RunEtcd(t, nil)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)

	transformer := &fancyTransformer{
		transformer: &prefixTransformer{prefix: []byte(defaultTestPrefix)},
	}
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, transformer, true, NewDefaultLeaseManagerConfig())
	transformer.store = store

	for i := 0; i < 5; i++ {
		if err := transformer.createObject(); err != nil {
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
	if err := store.List(context.TODO(), "/", storage.ListOptions{Predicate: predicate}, &result1); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	// List objects from the returned resource version.
	options := storage.ListOptions{
		Predicate:            predicate,
		ResourceVersion:      result1.ResourceVersion,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
	}

	result2 := example.PodList{}
	if err := store.List(context.TODO(), "/", options, &result2); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	if !reflect.DeepEqual(result1, result2) {
		t.Errorf("inconsistent lists: %#v, %#v", result1, result2)
	}

	// Now also verify the  ResourceVersionMatchNotOlderThan.
	options.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan

	result3 := example.PodList{}
	if err := store.List(context.TODO(), "/", options, &result3); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	options.ResourceVersion = result3.ResourceVersion
	options.ResourceVersionMatch = metav1.ResourceVersionMatchExact

	result4 := example.PodList{}
	if err := store.List(context.TODO(), "/", options, &result4); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	if !reflect.DeepEqual(result3, result4) {
		t.Errorf("inconsistent lists: %#v, %#v", result3, result4)
	}
}

func TestCount(t *testing.T) {
	ctx, store, _ := testSetup(t)

	resourceA := "/foo.bar.io/abc"

	// resourceA is intentionally a prefix of resourceB to ensure that the count
	// for resourceA does not include any objects from resourceB.
	resourceB := fmt.Sprintf("%sdef", resourceA)

	resourceACountExpected := 5
	for i := 1; i <= resourceACountExpected; i++ {
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("foo-%d", i)}}

		key := fmt.Sprintf("%s/%d", resourceA, i)
		testPropogateStoreWithKey(ctx, t, store, key, obj)
	}

	resourceBCount := 4
	for i := 1; i <= resourceBCount; i++ {
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("foo-%d", i)}}

		key := fmt.Sprintf("%s/%d", resourceB, i)
		testPropogateStoreWithKey(ctx, t, store, key, obj)
	}

	resourceACountGot, err := store.Count(resourceA)
	if err != nil {
		t.Fatalf("store.Count failed: %v", err)
	}

	// count for resourceA should not include the objects for resourceB
	// even though resourceA is a prefix of resourceB.
	if int64(resourceACountExpected) != resourceACountGot {
		t.Fatalf("store.Count for resource %s: expected %d but got %d", resourceA, resourceACountExpected, resourceACountGot)
	}
}

func TestLeaseMaxObjectCount(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	client := testserver.RunEtcd(t, nil)
	store := newStore(client, codec, newPod, "", schema.GroupResource{Resource: "pods"}, &prefixTransformer{prefix: []byte(defaultTestPrefix)}, true, LeaseManagerConfig{
		ReuseDurationSeconds: defaultLeaseReuseDurationSeconds,
		MaxObjectCount:       2,
	})
	ctx := context.Background()

	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}
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
