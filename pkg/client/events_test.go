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

package client

import (
	"net/url"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestEventSearch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/events",
			Query: url.Values{
				"fields": []string{"involvedObject.kind=Pod,involvedObject.name=foo,involvedObject.namespace=baz"},
				"labels": []string{},
			},
		},
		Response: Response{StatusCode: 200, Body: &api.EventList{}},
	}
	eventList, err := c.Setup().Events("").Search(
		&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "baz",
				SelfLink:  testapi.SelfLink("pods", ""),
			},
		},
	)
	c.Validate(t, eventList, err)
}

func TestEventCreate(t *testing.T) {
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       "nm",
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	event := &api.Event{
		//namespace: namespace{"default"},
		InvolvedObject: *objReference,
		Timestamp:      timeStamp,
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   "/events",
			Body:   event,
		},
		Response: Response{StatusCode: 200, Body: event},
	}

	response, err := c.Setup().Events("").Create(event)

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if e, a := *objReference, response.InvolvedObject; !reflect.DeepEqual(e, a) {
		t.Errorf("%#v != %#v.", e, a)
	}
}

func TestEventGet(t *testing.T) {
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       "nm",
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	event := &api.Event{
		InvolvedObject: *objReference,
		Timestamp:      timeStamp,
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/events/1",
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: event},
	}

	response, err := c.Setup().Events("").Get("1")

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if e, r := event.InvolvedObject, response.InvolvedObject; !reflect.DeepEqual(e, r) {
		t.Errorf("%#v != %#v.", e, r)
	}
}

func TestEventList(t *testing.T) {
	ns := api.NamespaceDefault
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       ns,
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	eventList := &api.EventList{
		Items: []api.Event{
			{
				InvolvedObject: *objReference,
				Timestamp:      timeStamp,
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/events",
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: eventList},
	}
	response, err := c.Setup().Events(ns).List(labels.Everything(),
		labels.Everything())

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if len(response.Items) != 1 {
		t.Errorf("%#v response.Items should have len 1.", response.Items)
	}

	responseEvent := response.Items[0]
	if e, r := eventList.Items[0].InvolvedObject,
		responseEvent.InvolvedObject; !reflect.DeepEqual(e, r) {
		t.Errorf("%#v != %#v.", e, r)
	}
}
