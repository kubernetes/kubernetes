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

package apiserver

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

func TestListEventsForPodNotFound(t *testing.T) {
	fakeClient := tools.MakeFakeEtcdClient(t)
	events := []api.Event{}
	fakeClient.Data["/events/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	helper := tools.EtcdHelper{fakeClient}
	eventStore := EtcdEventStore{
		etcdHelper: helper,
	}
	list, err := eventStore.ListEventsForPod("foo")
	expectNoError(t, err)
	if list != nil {
		t.Errorf("Expected %#v, Got %#v", events, list)
	}
}

func TestListEventsForPod(t *testing.T) {
	fakeClient := tools.MakeFakeEtcdClient(t)
	events := []api.Event{
		{Event: "zero"},
		{Event: "one"},
	}
	fakeClient.Data["/events/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(events[0]),
					},
					{
						Value: util.MakeJSONString(events[1]),
					},
				},
			},
		},
	}
	helper := tools.EtcdHelper{fakeClient}
	eventStore := EtcdEventStore{
		etcdHelper: helper,
	}
	list, err := eventStore.ListEventsForPod("foo")
	expectNoError(t, err)
	if !reflect.DeepEqual(events, list) {
		t.Errorf("Expected %#v, Got %#v", events, list)
	}
}

func TestListEvents(t *testing.T) {
	fakeClient := tools.MakeFakeEtcdClient(t)
	fakeClient.Data["/events"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Key: "foo",
					},
					{
						Key: "bar",
					},
				},
			},
		},
	}
	events := []api.Event{
		{Event: "zero"},
		{Event: "one"},
		{Event: "two"},
	}
	fakeClient.Data["/events/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(events[0]),
					},
					{
						Value: util.MakeJSONString(events[1]),
					},
				},
			},
		},
	}
	fakeClient.Data["/events/bar"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(events[2]),
					},
				},
			},
		},
	}

	helper := tools.EtcdHelper{fakeClient}
	eventStore := EtcdEventStore{
		etcdHelper: helper,
	}
	list, err := eventStore.ListEvents()
	expectNoError(t, err)
	if !reflect.DeepEqual(events, list) {
		t.Errorf("Expected %#v, Got %#v", events, list)
	}
}

func TestListEventsEmpty(t *testing.T) {
	fakeClient := tools.MakeFakeEtcdClient(t)
	fakeClient.Data["/events"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
	}
	helper := tools.EtcdHelper{fakeClient}
	events := EtcdEventStore{
		etcdHelper: helper,
	}
	list, err := events.ListEvents()
	expectNoError(t, err)
	if len(list) != 0 {
		t.Errorf("Unexpected non-empty list: %#v", list)
	}
}
