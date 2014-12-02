/*
Copyright 2014 Google Inc. All rights reserved.

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

package boundpods

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) Registry {
	return NewEtcdRegistry(tools.EtcdHelper{client, testapi.Codec(), tools.RuntimeVersionAdapter{testapi.MetadataAccessor()}})
}

func TestEtcdGetBoundPods(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/registry/nodes/foo/boundpods"
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &api.BoundPods{Host: "foo"}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	pod, err := registry.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pod.(*api.BoundPods).Host != "foo" {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}

func TestEtcdGetBoundPodsNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/registry/nodes/foo/boundpods"
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.Get(ctx, "foo")
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdWatchBoundPods(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.Watch(ctx,
		labels.Everything(),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchBoundPodsByHost(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.Watch(ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"host": "foo"}),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	select {
	case obj, ok := <-watching.ResultChan():
		if !ok {
			t.Fatalf("watching channel should be open")
		}
		if obj.Type != watch.Added {
			t.Errorf("unexpected type: %#v", obj)
		}
		if obj.Object.(*api.BoundPods).Host != "foo" {
			t.Errorf("unexpected object: %#v", obj)
		}
	default:
	}

	go func() {
		fakeClient.Lock() // memory barrier for the channels
		fakeClient.WatchResponse <- &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), &api.BoundPods{Host: "foo"}),
				CreatedIndex:  2,
				ModifiedIndex: 2,
			},
		}
		fakeClient.Unlock()

		fakeClient.WaitForWatchCompletion()
		fakeClient.WatchInjectError <- nil
	}()

	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchBoundPodsInvalidOptions(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)

	// unrecognized field name
	_, err := registry.Watch(
		ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"selector": "foo"}),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	// labels
	_, err = registry.Watch(
		ctx,
		labels.SelectorFromSet(labels.Set{"label": "foo"}),
		labels.Everything(),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	// more than one field
	_, err = registry.Watch(
		ctx,
		labels.SelectorFromSet(labels.Set{"host": "foo", "other": "bar"}),
		labels.Everything(),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
}
