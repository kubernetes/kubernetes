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

package config

import (
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/coreos/go-etcd/etcd"
)

// TODO(lavalamp): Use the etcd watcher from the tools package, and make sure all test cases here are tested there.

func TestGetEtcdData(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{})
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: api.EncodeOrDie(&api.ContainerManifestList{
					Items: []api.ContainerManifest{{ID: "foo"}},
				}),
				ModifiedIndex: 1,
			},
		},
		E: nil,
	}
	NewSourceEtcd("/registry/hosts/machine/kubelet", fakeClient, ch)

	//TODO: update FakeEtcdClient.Watch to handle receiver=nil with a given index
	//returns an infinite stream of updates
	for i := 0; i < 2; i++ {
		update := (<-ch).(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.Pod{Name: "foo", Manifest: api.ContainerManifest{ID: "foo"}})
		if !reflect.DeepEqual(expected, update) {
			t.Errorf("Expected %#v, Got %#v", expected, update)
		}
	}
}

func TestGetEtcdNoData(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: nil,
	}
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	_, err := c.fetchNextState(0)
	if err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestGetEtcd(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	manifest := api.ContainerManifest{ID: "foo", Version: "v1beta1", Containers: []api.Container{{Name: "1", Image: "foo"}}}
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: api.EncodeOrDie(&api.ContainerManifestList{
					Items: []api.ContainerManifest{manifest},
				}),
				ModifiedIndex: 1,
			},
		},
		E: nil,
	}
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	lastIndex, err := c.fetchNextState(0)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if lastIndex != 2 {
		t.Errorf("Expected %#v, Got %#v", 2, lastIndex)
	}
	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.Pod{Name: "foo", Manifest: manifest})
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
	for i := range update.Pods {
		if errs := kubelet.ValidatePod(&update.Pods[i]); len(errs) != 0 {
			t.Errorf("Expected no validation errors on %#v, Got %#v", update.Pods[i], errs)
		}
	}
}

func TestWatchEtcd(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         api.EncodeOrDie(&api.ContainerManifestList{}),
				ModifiedIndex: 2,
			},
		},
		E: nil,
	}
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	lastIndex, err := c.fetchNextState(1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if lastIndex != 3 {
		t.Errorf("Expected %d, Got %d", 3, lastIndex)
	}
	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET)
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}

func TestGetEtcdNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	_, err := c.fetchNextState(0)
	if err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestGetEtcdError(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: 200, // non not found error
		},
	}
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	_, err := c.fetchNextState(0)
	if err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}
