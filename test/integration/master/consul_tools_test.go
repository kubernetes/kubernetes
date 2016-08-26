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
	"math/rand"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/consul/consultest"
	storagebackend "k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/storage/storagebackend/factory"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/integration/framework"

	consulapi "github.com/hashicorp/consul/api"
	"golang.org/x/net/context"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func getConsulTestConfig() storagebackend.Config {
	serverList := []string{"http://localhost:8500"}

	return storagebackend.Config{
		Type:       storagebackend.StorageTypeConsul,
		ServerList: serverList,
		Prefix:     consultest.PathPrefix(),
		DeserializationCacheSize: consultest.DeserializationCacheSize,
		Codec: testapi.Default.Codec(),
	}
}

func TestConsulCreate(t *testing.T) {
	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	cstorage, _, err := factory.Create(getConsulTestConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		prefixedKey := consultest.AddPrefix(key)

		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}

		//Create
		err = cstorage.Create(ctx, key, &testObject, nil, 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		//GET
		kvPair, _, err := consulClient.KV().Get(prefixedKey, nil)
		if kvPair == nil {
			t.Fatalf("Key %v not found", prefixedKey)
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		decoded, err := runtime.Decode(testapi.Default.Codec(), kvPair.Value)
		if err != nil {
			t.Fatalf("unexpected response: %#v", kvPair)
		}
		result := *decoded.(*api.ServiceAccount)

		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected:\n%#v\ngot:\n%#v", testObject, result)
		}
	})
}

func TestConsulGet(t *testing.T) {
	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	cstorage, _, err := factory.Create(getConsulTestConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		prefixedKey := consultest.AddPrefix(key)

		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		coded, err := runtime.Encode(testapi.Default.Codec(), &testObject)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		testKV := &consulapi.KVPair{
			Key:   prefixedKey,
			Value: coded,
		}

		//Set
		_, err = consulClient.KV().Put(testKV, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		result := api.ServiceAccount{}
		//Get
		if err = cstorage.Get(ctx, key, &result, false); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Propagate ResourceVersion (it is not set automatically).
		testObject.ObjectMeta.ResourceVersion = result.ObjectMeta.ResourceVersion
		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected:\n%#v\ngot:\n%#v", testObject, result)
		}
	})
}

func TestConsulDelete(t *testing.T) {
	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	cstorage, _, err := factory.Create(getConsulTestConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		prefixedKey := consultest.AddPrefix(key)

		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		coded, err := runtime.Encode(testapi.Default.Codec(), &testObject)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		testKV := &consulapi.KVPair{
			Key:   prefixedKey,
			Value: coded,
		}

		//Set
		_, err = consulClient.KV().Put(testKV, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		result := api.ServiceAccount{}
		//Delete
		if err = cstorage.Delete(ctx, key, &result, nil); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Propagate ResourceVersion (it is not set automatically).
		testObject.ObjectMeta.ResourceVersion = result.ObjectMeta.ResourceVersion
		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected:\n%#v\ngot:\n%#v", testObject, result)
		}
	})
}

func TestConsulDeleteWithPrecondition(t *testing.T) {
	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	cstorage, _, err := factory.Create(getConsulTestConfig())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		prefixedKey := consultest.AddPrefix(key)

		testObject := api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		coded, err := runtime.Encode(testapi.Default.Codec(), &testObject)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		testKV := &consulapi.KVPair{
			Key:   prefixedKey,
			Value: coded,
		}

		//Set
		_, err = consulClient.KV().Put(testKV, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		//Get (to retrieve meta info)
		result := api.ServiceAccount{}
		if err = cstorage.Get(ctx, key, &result, false); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		//retrieve meta info
		objMeta, err := api.ObjectMetaFor(&result)
		preCon := storage.NewUIDPreconditions(string(objMeta.UID))

		//Delete with preconditon set
		result = api.ServiceAccount{}
		if err = cstorage.Delete(ctx, key, &result, preCon); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Propagate ResourceVersion (it is not set automatically).
		testObject.ObjectMeta.ResourceVersion = result.ObjectMeta.ResourceVersion
		if !api.Semantic.DeepEqual(testObject, result) {
			t.Errorf("expected:\n%#v\ngot:\n%#v", testObject, result)
		}
	})
}

func TestConsulWatch(t *testing.T) {
	config := getConsulTestConfig()
	cstorage, _, err := factory.Create(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		prefixedKey := consultest.AddPrefix(key)

		testObject := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		testObjectOut := &api.Pod{}

		testKVPAir := &consulapi.KVPair{
			Key:   prefixedKey,
			Value: []byte(runtime.EncodeOrDie(testapi.Default.Codec(), testObject)),
		}

		_, err = consulClient.KV().Put(testKVPAir, &consulapi.WriteOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		//GET (to retrieve ModifiyIndex)
		kvPair, _, err := consulClient.KV().Get(prefixedKey, nil)
		if kvPair == nil {
			t.Fatalf("Key %v not found", prefixedKey)
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expectedVersion := kvPair.ModifyIndex

		w, err := cstorage.Watch(ctx, key, "0", storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		//create event
		event := <-w.ResultChan()
		if event.Type != watch.Added {
			t.Fatalf("expected: %#v got: %#v", watch.Added, event.Type)
		}

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

		//DELETE
		err = cstorage.Delete(ctx, key, testObjectOut, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		event = <-w.ResultChan()
		if event.Type != watch.Deleted {
			t.Fatalf("expected: %#v got: %#v", watch.Deleted, event.Type)
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

		w.Stop()
	})
}

func TestConsulWatchList(t *testing.T) {
	config := getConsulTestConfig()
	cstorage, _, err := factory.Create(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	consulClient, err := consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		t.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	ctx := context.TODO()

	framework.WithConsulKey(cstorage, func(key string) {
		keyDeep := key + "/deep"
		prefixedKey := consultest.AddPrefix(key)
		prefixedKeyDeep := consultest.AddPrefix(keyDeep)

		w, err := cstorage.WatchList(ctx, key, "0", storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		testObject := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		testObjectOut := &api.Pod{}

		testKVPAir := &consulapi.KVPair{
			Key:   prefixedKeyDeep,
			Value: []byte(runtime.EncodeOrDie(testapi.Default.Codec(), testObject)),
		}

		_, err = consulClient.KV().Put(testKVPAir, &consulapi.WriteOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		//GET (to retrieve ModifiyIndex)
		kvPair, _, err := consulClient.KV().Get(prefixedKeyDeep, nil)
		if kvPair == nil {
			t.Fatalf("Key %v not found", prefixedKey)
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expectedVersion := kvPair.ModifyIndex

		//create event
		event := <-w.ResultChan()
		if event.Type != watch.Added {
			t.Fatalf("expected: %#v got: %#v", watch.Added, event.Type)
		}

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

		//DELETE
		err = cstorage.Delete(ctx, keyDeep, testObjectOut, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		event = <-w.ResultChan()
		if event.Type != watch.Deleted {
			t.Fatalf("expected: %#v got: %#v", watch.Deleted, event.Type)
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

		w.Stop()
	})
}
