// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors.

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

package master

import (
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/integration/framework"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
)

func TestCreate(t *testing.T) {
	client := framework.NewEtcdClient()
	keysAPI := etcd.NewKeysAPI(client)
	etcdStorage := etcdstorage.NewEtcdStorage(client, testapi.Default.Codec(), "", false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()
	framework.WithEtcdKey(func(key string) {
		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		if err := etcdStorage.Create(ctx, key, &testObject, nil, 0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp, err := keysAPI.Get(ctx, key, nil)
		if err != nil || resp.Node == nil {
			t.Fatalf("unexpected error: %v %v", err, resp)
		}
		decoded, err := runtime.Decode(testapi.Default.Codec(), []byte(resp.Node.Value))
		if err != nil {
			t.Fatalf("unexpected response: %#v", resp.Node)
		}
		result := *decoded.(*api.ServiceAccount)
		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected: %#v got: %#v", testObject, result)
		}
	})
}

func TestGet(t *testing.T) {
	client := framework.NewEtcdClient()
	keysAPI := etcd.NewKeysAPI(client)
	etcdStorage := etcdstorage.NewEtcdStorage(client, testapi.Default.Codec(), "", false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()
	framework.WithEtcdKey(func(key string) {
		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		coded, err := runtime.Encode(testapi.Default.Codec(), &testObject)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, err = keysAPI.Set(ctx, key, string(coded), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		result := api.ServiceAccount{}
		if err := etcdStorage.Get(ctx, key, &result, false); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Propagate ResourceVersion (it is set automatically).
		testObject.ObjectMeta.ResourceVersion = result.ObjectMeta.ResourceVersion
		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected: %#v got: %#v", testObject, result)
		}
	})
}

func TestWriteTTL(t *testing.T) {
	client := framework.NewEtcdClient()
	keysAPI := etcd.NewKeysAPI(client)
	etcdStorage := etcdstorage.NewEtcdStorage(client, testapi.Default.Codec(), "", false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()
	framework.WithEtcdKey(func(key string) {
		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		if err := etcdStorage.Create(ctx, key, &testObject, nil, 0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		result := &api.ServiceAccount{}
		err := etcdStorage.GuaranteedUpdate(ctx, key, result, false, nil, func(obj runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
			if in, ok := obj.(*api.ServiceAccount); !ok || in.Name != "foo" {
				t.Fatalf("unexpected existing object: %v", obj)
			}
			if res.TTL != 0 {
				t.Fatalf("unexpected TTL: %#v", res)
			}
			ttl := uint64(10)
			out := &api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "out"}}
			return out, &ttl, nil
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result.Name != "out" {
			t.Errorf("unexpected response: %#v", result)
		}
		if res, err := keysAPI.Get(ctx, key, nil); err != nil || res == nil || res.Node.TTL != 10 {
			t.Fatalf("unexpected get: %v %#v", err, res)
		}

		result = &api.ServiceAccount{}
		err = etcdStorage.GuaranteedUpdate(ctx, key, result, false, nil, func(obj runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
			if in, ok := obj.(*api.ServiceAccount); !ok || in.Name != "out" {
				t.Fatalf("unexpected existing object: %v", obj)
			}
			if res.TTL <= 1 {
				t.Fatalf("unexpected TTL: %#v", res)
			}
			out := &api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "out2"}}
			return out, nil, nil
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result.Name != "out2" {
			t.Errorf("unexpected response: %#v", result)
		}
		if res, err := keysAPI.Get(ctx, key, nil); err != nil || res == nil || res.Node.TTL <= 1 {
			t.Fatalf("unexpected get: %v %#v", err, res)
		}
	})
}

func TestWatch(t *testing.T) {
	client := framework.NewEtcdClient()
	keysAPI := etcd.NewKeysAPI(client)
	etcdStorage := etcdstorage.NewEtcdStorage(client, testapi.Default.Codec(), etcdtest.PathPrefix(), false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()
	framework.WithEtcdKey(func(key string) {
		key = etcdtest.AddPrefix(key)
		resp, err := keysAPI.Set(ctx, key, runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expectedVersion := resp.Node.ModifiedIndex

		// watch should load the object at the current index
		w, err := etcdStorage.Watch(ctx, key, "0", storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		event := <-w.ResultChan()
		if event.Type != watch.Added || event.Object == nil {
			t.Fatalf("expected first value to be set to ADDED, got %#v", event)
		}

		// version should match what we set
		pod := event.Object.(*api.Pod)
		if pod.ResourceVersion != strconv.FormatUint(expectedVersion, 10) {
			t.Errorf("expected version %d, got %#v", expectedVersion, pod)
		}

		// should be no events in the stream
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatalf("channel closed unexpectedly")
			}
			t.Fatalf("unexpected object in channel: %#v", event)
		default:
		}

		// should return the previously deleted item in the watch, but with the latest index
		resp, err = keysAPI.Delete(ctx, key, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expectedVersion = resp.Node.ModifiedIndex
		event = <-w.ResultChan()
		if event.Type != watch.Deleted {
			t.Errorf("expected deleted event %#v", event)
		}
		pod = event.Object.(*api.Pod)
		if pod.ResourceVersion != strconv.FormatUint(expectedVersion, 10) {
			t.Errorf("expected version %d, got %#v", expectedVersion, pod)
		}
	})
}
