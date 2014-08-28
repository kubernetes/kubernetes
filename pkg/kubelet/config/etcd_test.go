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
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/coreos/go-etcd/etcd"
)

// TODO(lavalamp): Use the etcd watcher from the tools package, and make sure all test cases here are tested there.

func TestNewSourceEtcd(t *testing.T) {
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

func TestHandleResponseParseFailure(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ch := make(chan interface{}, 1)
	c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, ch, time.Millisecond, time.Minute}
	err := c.handleResponse(&etcd.Response{})
	if err == nil {
		t.Errorf("Expected non-nil error")
	}
	expectEmptyChannel(t, ch)
}

func TestGetCurrentState(t *testing.T) {
	tests := []struct {
		clientResponse *etcd.Response
		clientError    error

		expectResponse *etcd.Response
		expectError    error
	}{
		{
			clientResponse: &etcd.Response{Node: &etcd.Node{ModifiedIndex: 12}},
			clientError:    nil,
			expectResponse: &etcd.Response{Node: &etcd.Node{ModifiedIndex: 12}},
			expectError:    nil,
		},
		{
			clientResponse: &etcd.Response{},
			clientError:    tools.EtcdErrorNotFound,
			expectResponse: nil,
			expectError:    nil,
		},
		{
			clientResponse: &etcd.Response{},
			clientError:    errors.New("unrecognized error"),
			expectResponse: &etcd.Response{},
			expectError:    errors.New("unrecognized error"),
		},
	}

	for i, tt := range tests {
		fakeClient := tools.NewFakeEtcdClient(t)
		fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
			R: tt.clientResponse,
			E: tt.clientError,
		}

		c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, nil, time.Millisecond, time.Minute}
		response, err := c.getCurrentState()
		if !reflect.DeepEqual(tt.expectResponse, response) {
			t.Errorf("case %d: expected response=%#v, got %#v", i, tt.expectResponse, response)
		}
		if !reflect.DeepEqual(tt.expectError, err) {
			t.Errorf("case %d: expected error=%v, got %v", i, tt.expectError, err)
		}
	}
}

func TestWatchForNextState(t *testing.T) {
	tests := []struct {
		clientResponse *etcd.Response
		clientError    error

		expectResponse *etcd.Response
		expectError    error
	}{
		{
			clientResponse: &etcd.Response{Node: &etcd.Node{ModifiedIndex: 12}},
			clientError:    nil,
			expectResponse: &etcd.Response{Node: &etcd.Node{ModifiedIndex: 12}},
			expectError:    nil,
		},
		{
			clientResponse: &etcd.Response{},
			clientError:    tools.EtcdErrorNotFound,
			expectResponse: nil,
			expectError:    nil,
		},
		{
			clientResponse: &etcd.Response{},
			clientError:    errors.New("unrecognized error"),
			expectResponse: &etcd.Response{},
			expectError:    errors.New("unrecognized error"),
		},
		{
			clientResponse: &etcd.Response{},
			clientError:    etcd.ErrWatchStoppedByUser,
			expectResponse: nil,
			expectError:    nil,
		},
	}

	for i, tt := range tests {
		fakeClient := tools.NewFakeEtcdClient(t)
		fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
			R: tt.clientResponse,
			E: tt.clientError,
		}

		c := SourceEtcd{"/registry/hosts/machine/kubelet", fakeClient, nil, time.Millisecond, time.Minute}
		response, err := c.watchForNextState(42)
		if !reflect.DeepEqual(tt.expectResponse, response) {
			t.Errorf("case %d: expected response=%#v, got %#v", i, tt.expectResponse, response)
		}
		if !reflect.DeepEqual(tt.expectError, err) {
			t.Errorf("case %d: expected error=%v, got %v", i, tt.expectError, err)
		}
	}
}
