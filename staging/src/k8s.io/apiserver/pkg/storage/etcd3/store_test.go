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
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/pkg/capnslog"
	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd"
	storagetests "k8s.io/apiserver/pkg/storage/tests"
	"k8s.io/apiserver/pkg/storage/value"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

const defaultTestPrefix = "test!"

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))

	capnslog.SetGlobalLogLevel(capnslog.CRITICAL)
}

// prefixTransformer adds and verifies that all data has the correct prefix on its way in and out.
type prefixTransformer struct {
	prefix []byte
	stale  bool
	err    error
}

func (p prefixTransformer) TransformFromStorage(b []byte, ctx value.Context) ([]byte, bool, error) {
	if ctx == nil {
		panic("no context provided")
	}
	if !bytes.HasPrefix(b, p.prefix) {
		return nil, false, fmt.Errorf("value does not have expected prefix: %s", string(b))
	}
	return bytes.TrimPrefix(b, p.prefix), p.stale, p.err
}
func (p prefixTransformer) TransformToStorage(b []byte, ctx value.Context) ([]byte, error) {
	if ctx == nil {
		panic("no context provided")
	}
	if len(b) > 0 {
		return append(append([]byte{}, p.prefix...), b...), p.err
	}
	return b, p.err
}

func TestCreate(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	etcdClient := cluster.RandClient()

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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	if err := store.Create(ctx, key, input, out, 1); err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, out.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key, _ := testPropogateStore(ctx, t, store, obj)
	out := &example.Pod{}
	err := store.Create(ctx, key, obj, out, 0)
	if err == nil || !storage.IsNodeExist(err) {
		t.Errorf("expecting key exists error, but get: %s", err)
	}
}

func TestGet(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key               string
		ignoreNotFound    bool
		expectNotFoundErr bool
		expectedOut       *example.Pod
	}{{ // test get on existing item
		key:               key,
		ignoreNotFound:    false,
		expectNotFoundErr: false,
		expectedOut:       storedObj,
	}, { // test get on non-existing item with ignoreNotFound=false
		key:               "/non-existing",
		ignoreNotFound:    false,
		expectNotFoundErr: true,
	}, { // test get on non-existing item with ignoreNotFound=true
		key:               "/non-existing",
		ignoreNotFound:    true,
		expectNotFoundErr: false,
		expectedOut:       &example.Pod{},
	}}

	for i, tt := range tests {
		out := &example.Pod{}
		err := store.Get(ctx, tt.key, "", out, tt.ignoreNotFound)
		if tt.expectNotFoundErr {
			if err == nil || !storage.IsNotFound(err) {
				t.Errorf("#%d: expecting not found error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}
		if !reflect.DeepEqual(tt.expectedOut, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, tt.expectedOut, out)
		}
	}
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
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
		err := store.Delete(ctx, tt.key, out, nil)
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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
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
		err := store.Delete(ctx, key, out, tt.precondition)
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

func TestGetToList(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key         string
		pred        storage.SelectionPredicate
		expectedOut []*example.Pod
	}{{ // test GetToList on existing key
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
	}, { // test GetToList on non-existing key
		key:         "/non-existing",
		pred:        storage.Everything,
		expectedOut: nil,
	}, { // test GetToList with matching pod name
		key: "/non-existing",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, pod.Initializers != nil, nil
			},
		},
		expectedOut: nil,
	}}

	for i, tt := range tests {
		out := &example.PodList{}
		err := store.GetToList(ctx, tt.key, "", tt.pred, out)
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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	etcdClient := cluster.RandClient()
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
		originalTransformer := store.transformer.(prefixTransformer)
		if tt.transformStale {
			transformer := originalTransformer
			transformer.stale = true
			store.transformer = transformer
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
			}))
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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return input, &ttl, nil
		})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, out.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func TestGuaranteedUpdateChecksStoredData(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

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

	store.transformer = prefixTransformer{prefix: []byte(defaultTestPrefix)}

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

	store.transformer = prefixTransformer{prefix: []byte(defaultTestPrefix), stale: true}

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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
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
			}))
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
		}))
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
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, originalPod := testPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	// First, update without a suggestion so originalPod is outdated
	updatedPod := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}),
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
}

func TestTransformationFailure(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	store := newStore(cluster.RandClient(), false, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
	ctx := context.Background()

	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{{
		key: "/one-level/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Spec:       storagetests.DeepEqualSafePodSpec(),
		},
	}, {
		key: "/two-level/1/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "baz"},
			Spec:       storagetests.DeepEqualSafePodSpec(),
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
	store.transformer = prefixTransformer{prefix: []byte("otherprefix!")}
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
	if err := store.List(ctx, "/", "", storage.Everything, &got); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error %v", err)
	}

	// Get should fail
	if err := store.Get(ctx, preset[1].key, "", &example.Pod{}, false); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate without suggestion should return an error
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, nil
	}); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate with suggestion should return an error if we don't change the object
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, nil
	}, preset[1].obj); err == nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete succeeds but reports an error because we cannot access the body
	if err := store.Delete(ctx, preset[1].key, &example.Pod{}, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	if err := store.Get(ctx, preset[1].key, "", &example.Pod{}, false); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestList(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	store := newStore(cluster.RandClient(), true, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
	disablePagingStore := newStore(cluster.RandClient(), false, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
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
	store.List(ctx, "/two-level", "0", storage.Everything, list)
	continueRV, _ := strconv.Atoi(list.ResourceVersion)
	secondContinuation, err := encodeContinue("/two-level/2", "/two-level/", int64(continueRV))
	if err != nil {
		t.Fatal(err)
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
		pod := obj.(*example.Pod)
		return nil, fields.Set{"metadata.name": pod.Name}, pod.Initializers != nil, nil
	}

	tests := []struct {
		name           string
		disablePaging  bool
		rv             string
		prefix         string
		pred           storage.SelectionPredicate
		expectedOut    []*example.Pod
		expectContinue bool
		expectError    bool
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
			name:        "test List on existing key",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0].storedObj},
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
			expectedOut:    []*example.Pod{preset[1].storedObj},
			expectContinue: true,
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
		if tt.pred.GetAttrs == nil {
			tt.pred.GetAttrs = getAttrs
		}

		out := &example.PodList{}
		var err error
		if tt.disablePaging {
			err = disablePagingStore.List(ctx, tt.prefix, tt.rv, tt.pred, out)
		} else {
			err = store.List(ctx, tt.prefix, tt.rv, tt.pred, out)
		}
		if (err != nil) != tt.expectError {
			t.Errorf("(%s): List failed: %v", tt.name, err)
		}
		if err != nil {
			continue
		}
		if (len(out.Continue) > 0) != tt.expectContinue {
			t.Errorf("(%s): unexpected continue token: %q", tt.name, out.Continue)
		}
		if len(tt.expectedOut) != len(out.Items) {
			t.Errorf("(%s): length of list want=%d, got=%d", tt.name, len(tt.expectedOut), len(out.Items))
			continue
		}
		for j, wantPod := range tt.expectedOut {
			getPod := &out.Items[j]
			if !reflect.DeepEqual(wantPod, getPod) {
				t.Errorf("(%s): pod want=%#v, got=%#v", tt.name, wantPod, getPod)
			}
		}
	}
}

func TestListContinuation(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	store := newStore(cluster.RandClient(), true, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
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
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, pod.Initializers != nil, nil
			},
		}
	}
	if err := store.List(ctx, "/", "0", pred(1, ""), out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[0].storedObj) {
		t.Fatalf("Unexpected first page: %#v", out.Items)
	}

	continueFromSecondItem := out.Continue

	// no limit, should get two items
	out = &example.PodList{}
	if err := store.List(ctx, "/", "0", pred(0, continueFromSecondItem), out); err != nil {
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

	// limit, should get two more pages
	out = &example.PodList{}
	if err := store.List(ctx, "/", "0", pred(1, continueFromSecondItem), out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[1].storedObj) {
		t.Fatalf("Unexpected second page: %#v", out.Items)
	}
	continueFromThirdItem := out.Continue
	out = &example.PodList{}
	if err := store.List(ctx, "/", "0", pred(1, continueFromThirdItem), out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], preset[2].storedObj) {
		t.Fatalf("Unexpected third page: %#v", out.Items)
	}

}

func TestListInconsistentContinuation(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	store := newStore(cluster.RandClient(), true, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
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
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, pod.Initializers != nil, nil
			},
		}
	}

	out := &example.PodList{}
	if err := store.List(ctx, "/", "0", pred(1, ""), out); err != nil {
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
	versioner := etcd.APIObjectVersioner{}
	lastRVString := preset[2].storedObj.ResourceVersion
	lastRV, err := versioner.ParseResourceVersion(lastRVString)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cluster.Client(0).KV.Compact(ctx, int64(lastRV), clientv3.WithCompactPhysical()); err != nil {
		t.Fatalf("Unable to compact, %v", err)
	}

	// The old continue token should have expired
	err = store.List(ctx, "/", "0", pred(0, continueFromSecondItem), out)
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
	if err := store.List(ctx, "/", "0", pred(1, inconsistentContinueFromSecondItem), out); err != nil {
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
	if err := store.List(ctx, "/", "0", pred(1, continueFromThirdItem), out); err != nil {
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

func testSetup(t *testing.T) (context.Context, *store, *integration.ClusterV3) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	store := newStore(cluster.RandClient(), true, codec, "", prefixTransformer{prefix: []byte(defaultTestPrefix)})
	ctx := context.Background()
	// As 30s is the default timeout for testing in glboal configuration,
	// we cannot wait longer than that in a single time: change it to 10
	// for testing purposes. See apimachinery/pkg/util/wait/wait.go
	store.leaseManager.setLeaseReuseDurationSeconds(1)
	return ctx, store, cluster
}

// testPropogateStore helps propagates store with objects, automates key generation, and returns
// keys and stored objects.
func testPropogateStore(ctx context.Context, t *testing.T, store *store, obj *example.Pod) (string, *example.Pod) {
	// Setup store with a key and grab the output for returning.
	key := "/testkey"
	err := store.unconditionalDelete(ctx, key, &example.Pod{})
	if err != nil && !storage.IsNotFound(err) {
		t.Fatalf("Cleanup failed: %v", err)
	}
	setOutput := &example.Pod{}
	if err := store.Create(ctx, key, obj, setOutput, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	return key, setOutput
}

func TestPrefix(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	transformer := prefixTransformer{prefix: []byte(defaultTestPrefix)}
	testcases := map[string]string{
		"custom/prefix":     "/custom/prefix",
		"/custom//prefix//": "/custom/prefix",
		"/registry":         "/registry",
	}
	for configuredPrefix, effectivePrefix := range testcases {
		store := newStore(cluster.RandClient(), true, codec, configuredPrefix, transformer)
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
		t               reflect.Type
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
