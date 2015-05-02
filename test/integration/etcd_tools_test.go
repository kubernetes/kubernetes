// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func init() {
	requireEtcd()
}

type stringCodec struct{}

type fakeAPIObject string

func (*fakeAPIObject) IsAnAPIObject() {}

func (c stringCodec) Encode(obj runtime.Object) ([]byte, error) {
	return []byte(*obj.(*fakeAPIObject)), nil
}

func (c stringCodec) Decode(data []byte) (runtime.Object, error) {
	o := fakeAPIObject(data)
	return &o, nil
}

func (c stringCodec) DecodeInto(data []byte, obj runtime.Object) error {
	o := obj.(*fakeAPIObject)
	*o = fakeAPIObject(data)
	return nil
}

func TestSetObj(t *testing.T) {
	client := newEtcdClient()
	helper := tools.EtcdHelper{Client: client, Codec: stringCodec{}}
	withEtcdKey(func(key string) {
		fakeObject := fakeAPIObject("object")
		if err := helper.SetObj(key, &fakeObject, nil, 0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp, err := client.Get(key, false, false)
		if err != nil || resp.Node == nil {
			t.Fatalf("unexpected error: %v %v", err, resp)
		}
		if resp.Node.Value != "object" {
			t.Errorf("unexpected response: %#v", resp.Node)
		}
	})
}

func TestExtractObj(t *testing.T) {
	client := newEtcdClient()
	helper := tools.EtcdHelper{Client: client, Codec: stringCodec{}}
	withEtcdKey(func(key string) {
		_, err := client.Set(key, "object", 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		s := fakeAPIObject("")
		if err := helper.ExtractObj(key, &s, false); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s != "object" {
			t.Errorf("unexpected response: %#v", s)
		}
	})
}

func TestWatch(t *testing.T) {
	client := newEtcdClient()
	helper := tools.NewEtcdHelper(client, testapi.Codec(), etcdtest.PathPrefix())
	withEtcdKey(func(key string) {
		key = etcdtest.AddPrefix(key)
		resp, err := client.Set(key, runtime.EncodeOrDie(testapi.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expectedVersion := resp.Node.ModifiedIndex

		// watch should load the object at the current index
		w, err := helper.Watch(key, 0, tools.Everything)
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
		resp, err = client.Delete(key, false)
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
